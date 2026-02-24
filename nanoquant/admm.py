"""LB-ADMM binary factorization for NanoQuant (Algorithm 1)."""

import torch
from typing import Tuple


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
        L_u = torch.linalg.cholesky(A_u)
        U_cont = torch.linalg.solve_triangular(
            L_u.T,
            torch.linalg.solve_triangular(L_u, B_u.T, upper=False),
            upper=True,
        ).T

        # --- V continuous update ---
        UtU = U_cont.T @ U_cont  # (r, r)
        A_v = UtU + (lam + rho / 2.0) * torch.eye(rank, device=device, dtype=dtype)
        B_v = W.T @ U_cont + (rho / 2.0) * (V_bin - Lambda_V)
        L_v = torch.linalg.cholesky(A_v)
        V_cont = torch.linalg.solve_triangular(
            L_v.T,
            torch.linalg.solve_triangular(L_v, B_v.T, upper=False),
            upper=True,
        ).T

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
    u_norm = U_bin.float().norm()
    v_norm = V_bin.float().norm()
    eta = (v_norm / (u_norm + 1e-12)).sqrt()

    # s1_i = mean(|eta * u_i|) for each row i
    s1 = (eta * U_bin.float()).abs().mean(dim=1)  # (m,)
    # s2_j = mean(|eta^{-1} * v_j|) for each row j of V
    s2 = (V_bin.float() / (eta + 1e-12)).abs().mean(dim=1)  # (n,)

    # Clamp scales to avoid degenerate values
    s1 = torch.clamp(s1, min=1e-8)
    s2 = torch.clamp(s2, min=1e-8)

    return U_bin, V_bin, s1, s2
