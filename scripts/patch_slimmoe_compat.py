#!/usr/bin/env python3
"""Patch microsoft/Phi-tiny-MoE-instruct modeling_slimmoe.py for transformers>=5.x compatibility.

The SlimMoE modeling file was written for transformers~4.43.3. When running with
transformers>=5.x, several APIs changed. This script patches the cached HF module
file to restore compatibility.

Usage:
    python scripts/patch_slimmoe_compat.py

This must be run once after the model is first downloaded. It patches:
  1. flash_attn import made optional (flash_attn not required for CPU/SDPA inference)
  2. is_torch_fx_available import stub (removed from transformers 5.x)
  3. rope_scaling 'rope_type' vs 'type' key difference (transformers 5.x sets 'rope_type')
  4. DynamicCache API shims (from_legacy_cache, to_legacy_cache, get_usable_length, seen_tokens)
  5. DynamicCache.from_legacy_cache call made conditional
  6. _tied_weights_keys changed from list to dict (transformers 5.x save_pretrained)
"""

import os
import re


def find_cached_slimmoe() -> list[str]:
    """Find all copies of modeling_slimmoe.py in the HF cache."""
    cache_root = os.path.expanduser("~/.cache/huggingface")
    found = []
    for root, _, files in os.walk(cache_root):
        for f in files:
            if f == "modeling_slimmoe.py" and "Phi" in root and "tiny" in root.lower():
                found.append(os.path.join(root, f))
    return found


def patch_file(path: str) -> bool:
    """Apply all compatibility patches to a modeling_slimmoe.py file.

    Returns True if any patches were applied.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    original = content
    changed = False

    # Patch 1: Make flash_attn optional
    OLD1 = "from einops import rearrange\nfrom flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding"
    NEW1 = (
        "from einops import rearrange\n"
        "try:\n"
        "    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding\n"
        "except ImportError:\n"
        "    FlashRotaryEmbedding = None"
    )
    if OLD1 in content:
        content = content.replace(OLD1, NEW1)
        changed = True
        print(f"  [OK] Patch 1: flash_attn optional import")
    elif "FlashRotaryEmbedding = None" in content:
        print(f"  [--] Patch 1: already applied")
    else:
        print(f"  [??] Patch 1: pattern not found (may not need patching)")

    # Patch 2: is_torch_fx_available stub
    OLD2 = "from transformers.utils.import_utils import is_torch_fx_available"
    NEW2 = (
        "try:\n"
        "    from transformers.utils.import_utils import is_torch_fx_available\n"
        "except ImportError:\n"
        "    def is_torch_fx_available():\n"
        "        return False"
    )
    if OLD2 in content:
        content = content.replace(OLD2, NEW2)
        changed = True
        print(f"  [OK] Patch 2: is_torch_fx_available stub")
    elif "def is_torch_fx_available():" in content:
        print(f"  [--] Patch 2: already applied")
    else:
        print(f"  [??] Patch 2: pattern not found")

    # Patch 3: rope_scaling 'type' vs 'rope_type' key
    OLD3 = (
        "        if getattr(config, 'rope_scaling', None) is None:\n"
        "            self.rotary_emb = PhiMoERotaryEmbedding(\n"
        "                self.head_dim,\n"
        "                max_position_embeddings=self.max_position_embeddings,\n"
        "                base=self.rope_theta,\n"
        "            )\n"
        "        else:\n"
        "            scaling_type = self.config.rope_scaling[\"type\"]\n"
        "            if scaling_type == \"longrope\":\n"
        "                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(self.head_dim, self.config)\n"
        "            else:\n"
        "                raise ValueError(f\"Unknown RoPE scaling type {scaling_type}\")"
    )
    NEW3 = (
        "        _rope_scaling = getattr(config, 'rope_scaling', None)\n"
        "        # transformers>=5.x auto-sets rope_scaling={'rope_type': 'default'} when none is specified.\n"
        "        # Treat 'default' rope_type (or missing 'type' key) the same as no rope_scaling.\n"
        "        _rope_type = None\n"
        "        if _rope_scaling is not None:\n"
        "            _rope_type = _rope_scaling.get(\"type\") or _rope_scaling.get(\"rope_type\")\n"
        "            if _rope_type in (None, \"default\"):\n"
        "                _rope_scaling = None\n"
        "        if _rope_scaling is None:\n"
        "            self.rotary_emb = PhiMoERotaryEmbedding(\n"
        "                self.head_dim,\n"
        "                max_position_embeddings=self.max_position_embeddings,\n"
        "                base=self.rope_theta,\n"
        "            )\n"
        "        else:\n"
        "            scaling_type = _rope_type\n"
        "            if scaling_type == \"longrope\":\n"
        "                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(self.head_dim, self.config)\n"
        "            else:\n"
        "                raise ValueError(f\"Unknown RoPE scaling type {scaling_type}\")"
    )
    if OLD3 in content:
        content = content.replace(OLD3, NEW3)
        changed = True
        print(f"  [OK] Patch 3: rope_scaling type key compatibility")
    elif "_rope_type = None" in content:
        print(f"  [--] Patch 3: already applied")
    else:
        print(f"  [??] Patch 3: pattern not found")

    # Patch 4: DynamicCache compatibility shims
    SHIMS_MARKER = "# --- transformers>=5.x compatibility shims ---"
    if SHIMS_MARKER not in content:
        SHIMS_INSERT_AFTER = "from transformers.cache_utils import Cache, DynamicCache"
        SHIMS = (
            "\n\n"
            "# --- transformers>=5.x compatibility shims ---\n"
            "# Patch DynamicCache to restore methods removed in transformers 5.x\n"
            "if not hasattr(DynamicCache, 'from_legacy_cache'):\n"
            "    @classmethod\n"
            "    def _from_legacy_cache(cls, past_key_values=None):\n"
            "        return cls()\n"
            "    DynamicCache.from_legacy_cache = _from_legacy_cache\n"
            "\n"
            "if not hasattr(DynamicCache, 'to_legacy_cache'):\n"
            "    def _to_legacy_cache(self):\n"
            "        if not hasattr(self, 'key_cache') or not self.key_cache:\n"
            "            return ()\n"
            "        return tuple(zip(self.key_cache, self.value_cache))\n"
            "    DynamicCache.to_legacy_cache = _to_legacy_cache\n"
            "\n"
            "if not hasattr(DynamicCache, 'get_usable_length'):\n"
            "    def _get_usable_length(self, new_seq_length, layer_idx=0):\n"
            "        return self.get_seq_length(layer_idx)\n"
            "    DynamicCache.get_usable_length = _get_usable_length\n"
            "\n"
            "if not hasattr(DynamicCache, 'seen_tokens'):\n"
            "    _orig_init = DynamicCache.__init__\n"
            "    def _new_init(self, *args, **kwargs):\n"
            "        _orig_init(self, *args, **kwargs)\n"
            "        if not hasattr(self, '_seen_tokens'):\n"
            "            self._seen_tokens = 0\n"
            "    DynamicCache.__init__ = _new_init\n"
            "    DynamicCache.seen_tokens = property(lambda self: self._seen_tokens)\n"
            "# --- end compatibility shims ---"
        )
        if SHIMS_INSERT_AFTER in content:
            content = content.replace(
                SHIMS_INSERT_AFTER,
                SHIMS_INSERT_AFTER + SHIMS,
                1,
            )
            changed = True
            print(f"  [OK] Patch 4: DynamicCache compatibility shims")
    else:
        print(f"  [--] Patch 4: already applied")

    # Patch 5: DynamicCache.from_legacy_cache conditional call
    OLD5 = "                past_key_values = DynamicCache.from_legacy_cache(past_key_values)"
    NEW5 = (
        "                # from_legacy_cache removed in transformers>=5.x; use DynamicCache() directly\n"
        "                if hasattr(DynamicCache, 'from_legacy_cache'):\n"
        "                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n"
        "                else:\n"
        "                    past_key_values = DynamicCache()"
    )
    if OLD5 in content:
        content = content.replace(OLD5, NEW5)
        changed = True
        print(f"  [OK] Patch 5: DynamicCache.from_legacy_cache conditional")
    elif "from_legacy_cache removed in transformers>=5.x" in content:
        print(f"  [--] Patch 5: already applied")
    else:
        print(f"  [??] Patch 5: pattern not found")

    # Patch 6: _tied_weights_keys list -> dict
    OLD6 = '    _tied_weights_keys = ["lm_head.weight"]'
    NEW6 = (
        '    # transformers>=5.x expects _tied_weights_keys to be a dict {key: tied_key}\n'
        '    # Convert list to dict for compatibility\n'
        '    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}'
    )
    if OLD6 in content:
        content = content.replace(OLD6, NEW6)
        changed = True
        print(f"  [OK] Patch 6: _tied_weights_keys list -> dict")
    elif '"lm_head.weight": "model.embed_tokens.weight"' in content:
        print(f"  [--] Patch 6: already applied")
    else:
        print(f"  [??] Patch 6: pattern not found")

    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Wrote patched file: {path}")

    return changed


def main():
    print("Searching for modeling_slimmoe.py in HF cache...")
    paths = find_cached_slimmoe()

    if not paths:
        print("No cached modeling_slimmoe.py found.")
        print("Run: python -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('microsoft/Phi-tiny-MoE-instruct', trust_remote_code=True)\"")
        print("Then re-run this script.")
        return

    for path in paths:
        print(f"\nPatching: {path}")
        patch_file(path)

    print("\nDone. Patches applied for transformers>=5.x compatibility.")


if __name__ == "__main__":
    main()
