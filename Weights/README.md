# Weights

`folded_adaln.safetensors` — the MageVAE adaLN constants baked at t=0, the one
genuine weight artifact of this port (0.39 MB, replacing 37.7 M params of
`t_embedder` + `adaLN_modulation` Linears). Regenerate with
`dump_folded_adaln.py` against a `microsoft/Mage-Flow-Edit-*` snapshot.

The large component weights (transformer, vae, text_encoder) are loaded from a
`microsoft/Mage-Flow-Edit-*` snapshot and sanitized at runtime — they are not
re-hosted here. Place `folded_adaln.safetensors` at the snapshot root and point
`--repo` at it.
