"""LB-ADMM binary factorization for NanoQuant (Algorithm 1)."""

import torch
from typing import Tuple


def _clamp_signed_scales(scales: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Preserve sign while keeping scale magnitudes away from zero."""
    scales = scales.float()
    sign = torch.sign(scales)
    sign[sign == 0] = 1.0
    return sign * torch.clamp(scales.abs(), min=eps)


def _fit_balanced_scales(
    W_target: torch.Tensor,
    U_bin: torch.Tensor,
    V_bin: torch.Tensor,
    n_iters: int = 8,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit row/column scales to the signed target using fixed binary factors."""
    core = U_bin.float() @ V_bin.float().T
    target = W_target.float()

    s1 = torch.ones(target.shape[0], device=target.device, dtype=target.dtype)
    s2 = torch.ones(target.shape[1], device=target.device, dtype=target.dtype)

    for _ in range(n_iters):
        row_basis = core * s2.unsqueeze(0)
        s1 = (target * row_basis).sum(dim=1) / (row_basis.pow(2).sum(dim=1) + eps)

        col_basis = core * s1.unsqueeze(1)
        s2 = (target * col_basis).sum(dim=0) / (col_basis.pow(2).sum(dim=0) + eps)

    return s1, s2

def svid(P: torch.Tensor, rank: int = 1) -> torch.Tensor:
    """Sign-Value Independent Decomposition.

    For each rank-1 component, compute SVD of P, extract sign(u)*sign(v)^T
    as a binary approximation. For rank > 1, deflate iteratively.

    Args:
        P: (m, n) matrix to approximate.
        rank: number of rank-1 binary components to sum.

    Returns:
        B: (m, n) binary matrix in {-1, +1}.
    """
    B = torch.zeros_like(P)
    residual = P.clone()
    for _ in range(rank):
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        u1 = U[:, 0]  # (m,)
        v1 = Vh[0, :]  # (n,)
        b = torch.sign(u1).unsqueeze(1) * torch.sign(v1).unsqueeze(0)  # (m, n)
        # Fix zeros (rare but possible)
        b[b == 0] = 1.0
        B += b
        residual = residual - S[0] * u1.unsqueeze(1) * v1.unsqueeze(0)
    # Final binarization
    B = torch.sign(B)
    B[B == 0] = 1.0
    return B


def _cholesky_solve(A: torch.Tensor, B: torch.Tensor, device, dtype, rank: int) -> torch.Tensor:
    """Solve X @ A = B via Cholesky with jitter fallback for numerical stability."""
    for jitter_scale in [0, 1e-6, 1e-5, 1e-4, 1e-3]:
        try:
            A_reg = A if jitter_scale == 0 else A + jitter_scale * torch.eye(rank, device=device, dtype=dtype)
            L = torch.linalg.cholesky(A_reg)
            return torch.linalg.solve_triangular(
                L.T,
                torch.linalg.solve_triangular(L, B.T, upper=False),
                upper=True,
            ).T
        except torch._C._LinAlgError:
            continue
    # Final fallback: use least-squares solve
    return torch.linalg.lstsq(A.T, B.T).solution.T


def lb_admm(
    W_target: torch.Tensor,
    rank: int = 8,
    rho: float = 0.5,
    lam: float = 1e-4,
    n_iters: int = 50,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-Bit ADMM binary factorization.

    Solves: min ||W_target - U @ V^T||^2_F + lambda*(||U||^2 + ||V||^2)
            s.t. U, V in {-1, +1}

    Uses stabilized Cholesky solves for continuous updates and SVID for
    binary projection.

    Args:
        W_target: (m, n) target weight matrix.
        rank: factorization rank.
        rho: ADMM penalty parameter.
        lam: regularization strength.
        n_iters: number of ADMM iterations.
        eps: convergence epsilon.

    Returns:
        U_bin: (m, rank) binary matrix in {-1, +1}.
        V_bin: (n, rank) binary matrix in {-1, +1}.
        s1: (m,) row scale vector.
        s2: (n,) column scale vector.
    """
    m, n = W_target.shape
    device = W_target.device
    dtype = torch.float32

    W = W_target.float()

    # Guard against non-finite values (overflow from Hessian preconditioning)
    if not torch.isfinite(W).all():
        finite_mask = torch.isfinite(W)
        max_abs = W[finite_mask].abs().max() if finite_mask.any() else torch.tensor(1.0)
        W = torch.nan_to_num(W, nan=0.0, posinf=max_abs.item(), neginf=-max_abs.item())

    # Initialize via SVD (fall back to CPU if CUDA SVD fails on ill-conditioned matrices)
    try:
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W, full_matrices=False)
    except torch._C._LinAlgError:
        # Add jitter to break degeneracy for ill-conditioned matrices
        jitter = torch.randn_like(W) * (W.norm() * 1e-6 + 1e-8)
        try:
            U_svd, S_svd, Vh_svd = torch.linalg.svd(W + jitter, full_matrices=False)
        except torch._C._LinAlgError:
            U_svd, S_svd, Vh_svd = torch.linalg.svd((W + jitter).cpu(), full_matrices=False)
            U_svd, S_svd, Vh_svd = U_svd.to(device), S_svd.to(device), Vh_svd.to(device)
    r = min(rank, len(S_svd))

    # Continuous latent variables
    U_cont = U_svd[:, :r] * S_svd[:r].sqrt().unsqueeze(0)  # (m, r)
    V_cont = Vh_svd[:r, :].T * S_svd[:r].sqrt().unsqueeze(0)  # (n, r)

    # Pad if rank > available singular values
    if r < rank:
        U_cont = torch.cat(
            [U_cont, torch.randn(m, rank - r, device=device, dtype=dtype) * 0.01], dim=1
        )
        V_cont = torch.cat(
            [V_cont, torch.randn(n, rank - r, device=device, dtype=dtype) * 0.01], dim=1
        )

    # Binary variables (SVID initialization)
    U_bin = torch.sign(U_cont)
    U_bin[U_bin == 0] = 1.0
    V_bin = torch.sign(V_cont)
    V_bin[V_bin == 0] = 1.0

    # Dual variables (Lagrange multipliers)
    Lambda_U = torch.zeros(m, rank, device=device, dtype=dtype)
    Lambda_V = torch.zeros(n, rank, device=device, dtype=dtype)

    prev_obj = float("inf")

    for it in range(n_iters):
        # --- U continuous update ---
        # min ||W - U V^T||^2 + lam*||U||^2 + (rho/2)*||U - U_bin + Lambda_U||^2
        # Derivative w.r.t. U = 0:
        # U (V^T V + (lam + rho/2) I) = W V + (rho/2)(U_bin - Lambda_U)
        VtV = V_cont.T @ V_cont  # (r, r)
        A_u = VtV + (lam + rho / 2.0) * torch.eye(rank, device=device, dtype=dtype)
        B_u = W @ V_cont + (rho / 2.0) * (U_bin - Lambda_U)
        # Solve U @ A_u = B_u => U = B_u @ A_u^{-1}
        # Equivalent: A_u^T @ U^T = B_u^T
        U_cont = _cholesky_solve(A_u, B_u, device, dtype, rank)

        # --- V continuous update ---
        UtU = U_cont.T @ U_cont  # (r, r)
        A_v = UtU + (lam + rho / 2.0) * torch.eye(rank, device=device, dtype=dtype)
        B_v = W.T @ U_cont + (rho / 2.0) * (V_bin - Lambda_V)
        V_cont = _cholesky_solve(A_v, B_v, device, dtype, rank)

        # --- Binary projection via SVID ---
        P_U = U_cont + Lambda_U
        P_V = V_cont + Lambda_V
        U_bin = torch.sign(P_U)
        U_bin[U_bin == 0] = 1.0
        V_bin = torch.sign(P_V)
        V_bin[V_bin == 0] = 1.0

        # --- Dual update ---
        Lambda_U = Lambda_U + (U_cont - U_bin)
        Lambda_V = Lambda_V + (V_cont - V_bin)

        # --- Check convergence ---
        recon = U_cont @ V_cont.T
        obj = (W - recon).norm().item()
        if abs(prev_obj - obj) < eps and it > 5:
            break
        prev_obj = obj

    # --- Magnitude balancing ---
    # Fit positive row/column scales against the target matrix instead of
    # deriving them only from binary factor norms.
    s1, s2 = _fit_balanced_scales(W, U_bin, V_bin, eps=eps)

    # Clamp magnitudes without destroying the fitted sign pattern.
    s1 = _clamp_signed_scales(s1, eps=1e-8)
    s2 = _clamp_signed_scales(s2, eps=1e-8)

    return U_bin, V_bin, s1, s2
