#!/usr/bin/env python3
"""Patch microsoft/Phi-tiny-MoE-instruct modeling_slimmoe.py for transformers>=5.x compatibility."""

import os


def find_cached_slimmoe() -> list[str]:
    cache_root = os.path.expanduser("~/.cache/huggingface")
    found = []
    for root, _, files in os.walk(cache_root):
        for f in files:
            if f == "modeling_slimmoe.py" and "Phi" in root and "tiny" in root.lower():
                found.append(os.path.join(root, f))
    return found


def patch_file(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    original = content
    changed = False

    old1 = "from einops import rearrange\nfrom flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding"
    new1 = (
        "from einops import rearrange\n"
        "try:\n"
        "    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding\n"
        "except ImportError:\n"
        "    FlashRotaryEmbedding = None"
    )
    if old1 in content:
        content = content.replace(old1, new1)
        changed = True
        print("  [OK] Patch 1: flash_attn optional import")
    elif "FlashRotaryEmbedding = None" in content:
        print("  [--] Patch 1: already applied")
    else:
        print("  [??] Patch 1: pattern not found")

    old2 = "from transformers.utils.import_utils import is_torch_fx_available"
    new2 = (
        "try:\n"
        "    from transformers.utils.import_utils import is_torch_fx_available\n"
        "except ImportError:\n"
        "    def is_torch_fx_available():\n"
        "        return False"
    )
    broken2 = (
        "try:\n"
        "    try:\n"
        "    from transformers.utils.import_utils import is_torch_fx_available\n"
        "except ImportError:\n"
        "    def is_torch_fx_available():\n"
        "        return False\n"
        "except ImportError:\n"
        "    def is_torch_fx_available():\n"
        "        return False"
    )
    if broken2 in content:
        content = content.replace(broken2, new2)
        changed = True
        print("  [OK] Patch 2: repaired broken nested stub")
    elif new2 in content:
        print("  [--] Patch 2: already applied")
    elif old2 in content:
        content = content.replace(old2, new2)
        changed = True
        print("  [OK] Patch 2: is_torch_fx_available stub")
    else:
        print("  [??] Patch 2: pattern not found")

    old3 = (
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
    new3 = (
        "        _rope_scaling = getattr(config, 'rope_scaling', None)\n"
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
    if old3 in content:
        content = content.replace(old3, new3)
        changed = True
        print("  [OK] Patch 3: rope_scaling compatibility")
    elif "_rope_type = None" in content:
        print("  [--] Patch 3: already applied")
    else:
        print("  [??] Patch 3: pattern not found")

    shims_marker = "# --- transformers>=5.x compatibility shims ---"
    if shims_marker not in content:
        insert_after = "from transformers.cache_utils import Cache, DynamicCache"
        shims = (
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
        if insert_after in content:
            content = content.replace(insert_after, insert_after + shims, 1)
            changed = True
            print("  [OK] Patch 4: DynamicCache shims")
    else:
        print("  [--] Patch 4: already applied")

    old5 = "                past_key_values = DynamicCache.from_legacy_cache(past_key_values)"
    new5 = (
        "                # from_legacy_cache removed in transformers>=5.x; use DynamicCache() directly\n"
        "                if hasattr(DynamicCache, 'from_legacy_cache'):\n"
        "                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n"
        "                else:\n"
        "                    past_key_values = DynamicCache()"
    )
    broken5 = (
        "                # from_legacy_cache removed in transformers>=5.x; use DynamicCache() directly\n"
        "                if hasattr(DynamicCache, 'from_legacy_cache'):\n"
        "                    # from_legacy_cache removed in transformers>=5.x; use DynamicCache() directly\n"
        "                if hasattr(DynamicCache, 'from_legacy_cache'):\n"
        "                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n"
        "                else:\n"
        "                    past_key_values = DynamicCache()\n"
        "                else:\n"
        "                    past_key_values = DynamicCache()"
    )
    if broken5 in content:
        content = content.replace(broken5, new5)
        changed = True
        print("  [OK] Patch 5: repaired broken nested conditional")
    elif new5 in content:
        print("  [--] Patch 5: already applied")
    elif old5 in content:
        content = content.replace(old5, new5)
        changed = True
        print("  [OK] Patch 5: conditional cache conversion")
    else:
        print("  [??] Patch 5: pattern not found")

    old6 = '    _tied_weights_keys = ["lm_head.weight"]'
    new6 = (
        '    # transformers>=5.x expects _tied_weights_keys to be a dict {key: tied_key}\n'
        '    # Convert list to dict for compatibility\n'
        '    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}'
    )
    if old6 in content:
        content = content.replace(old6, new6)
        changed = True
        print("  [OK] Patch 6: tied weights dict")
    elif '"lm_head.weight": "model.embed_tokens.weight"' in content:
        print("  [--] Patch 6: already applied")
    else:
        print("  [??] Patch 6: pattern not found")

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
        return
    for path in paths:
        print(f"\nPatching: {path}")
        patch_file(path)
    print("\nDone. Patches applied for transformers>=5.x compatibility.")


if __name__ == "__main__":
    main()
