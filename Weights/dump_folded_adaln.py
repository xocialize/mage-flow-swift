#!/usr/bin/env python
"""Extract the folded adaLN constants from the constructed PyTorch MageVAE.

Upstream's `_freeze_adaln_cache()` evaluates `adaLN_modulation(t_embedder(0))`
once at construction, in the PARAMETER dtype (bf16, since the safetensors are
bf16 on disk), and installs the result as a `_ConstAdaLN` buffer.

Reproducing that bit-for-bit in another framework is a losing game — bf16
matmul accumulation order differs, and being off by 0.039 in a single gate
value produced a 1.2 absolute error in that channel's output. So don't
reproduce it: BAKE IT. Dump the real buffers here and ship them as weights.
This is what the MLX/Swift conversion should do too, and it drops both
`t_embedder`s and all 42 `adaLN_modulation` Linears from the runtime graph
(~37.7M params).

Output: folded_adaln.safetensors, keys matching the checkpoint prefixes
(`student.dconv_encoder.blocks.N.` / `pipeline.blocks.N.`).
"""
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))
os.environ.setdefault("HF_HOME", "/Volumes/Satechi/hf-cache")
os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")

from mage_flow.pipeline import load_from_repo  # noqa: E402
from mage_flow.models.modules.mage_vae import DiCoBlock, _ConstAdaLN  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folded_adaln.safetensors")


def main():
    model = load_from_repo("microsoft/Mage-Flow-Edit-Turbo", device="cpu")
    vae = model.vae

    tensors = {}
    for root, prefix in ((vae.dconv_encoder, "student.dconv_encoder."),
                         (vae.decoder_model, "pipeline.")):
        for name, mod in root.named_modules():
            if not isinstance(mod, DiCoBlock):
                continue
            adaln = mod.adaLN_modulation
            assert isinstance(adaln, _ConstAdaLN), \
                f"{prefix}{name} adaLN was not folded — did _freeze_adaln_cache run?"
            buf = adaln.modulation
            # stored bf16; keep fp32 for a lossless carrier
            tensors[f"fold:{prefix}{name}."] = buf.detach().float().reshape(-1).clone()

    save_file(tensors, OUT)
    print(f"wrote {len(tensors)} folded adaLN constants -> {OUT}")
    for k in sorted(tensors)[:3]:
        print(f"  {k}  shape {tuple(tensors[k].shape)}")
    n = sum(t.numel() for t in tensors.values())
    print(f"  total {n} values ({n * 4 / 1e6:.2f} MB fp32)")


if __name__ == "__main__":
    main()
