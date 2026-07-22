# mage-flow-swift

MLX-Swift port of Microsoft's **[Mage-Flow](https://huggingface.co/microsoft/Mage-Flow)**
NR-MMDiT (4B) — a native-resolution multimodal diffusion transformer for
text-to-image generation and instruction-based image editing (MIT licence).

**Status: work in progress.** The transformer is ported and parity-locked; the
VAE and pipeline are not yet in Swift.

| component | state |
|---|---|
| NR-MMDiT (12-block dual-stream, 4.1B) | ported, parity-locked (worst rel **6.8e-6**) |
| MageVAE | verified MLX-**Python** rung only; Swift port pending |
| Qwen3-VL-4B conditioner | via [qwen3vl-mlx-swift](https://github.com/xocialize/qwen3vl-mlx-swift) |
| pipeline (Turbo + CFG paths) | pending |

The transformer is adapted from `qwen-image-edit-swift`: Mage-Flow's
`MageFlowEmbedRope` is the same scaled 3-axis RoPE (theta 10000,
axesDim `[16,56,56]`, `scaleRope`, 4096 pos/neg tables) and its block is the same
dual-stream MMDiT. Deltas — text is never rotated, no `zero_cond_t` modulation
index, no patchify (`patch_size=1` on 128-channel latents), `contextInDim` 2560,
12 layers — are documented inline in `Sources/MageFlow/Transformer.swift`.

## Parity gate

```
swift build -c release
.build/release/MageFlowGate <transformerDir> <dit_goldens_fp32.safetensors>
```

Two gate-design notes that cost real debugging time:

* **Gate on RELATIVE error.** Activations reach |x| ~ 1.2e5 by block 11, so
  0.35 absolute there is 2.8e-6 relative. An absolute tolerance fails a correct
  port.
* **The oracle's default goldens are bf16** (upstream runs the DiT in bf16), so
  an fp32 port differs from them by 1-4 bf16 ULPs — which looks catastrophic in
  absolute terms. Gate against an fp32 reference generated from identical
  starting activations.

Full porting notes, including the upstream traps, live in the oracle's
`PORTING-SPEC.md`.
