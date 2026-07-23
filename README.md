# mage-flow-swift

MLX-Swift port of Microsoft's **[Mage-Flow-Edit](https://huggingface.co/microsoft/Mage-Flow-Edit-Turbo)**
— native-resolution instruction-based image editing on Apple Silicon (MIT).

**Working end-to-end.** Image + instruction → edited image, with live Qwen3-VL
conditioning, the mandatory Responsible-AI content filter, and the
Gaussian-Shading provenance watermark, all reproduced.

```bash
swift build -c release
.build/release/mage-flow-edit \
  --repo <Mage-Flow-Edit-Turbo snapshot dir> \
  --ref dog.jpg --prompt "make the background a snowy forest" \
  --out edit.png
```

`<snapshot dir>` is a downloaded `microsoft/Mage-Flow-Edit-*` repo plus a
`folded_adaln.safetensors` at its root. The baked artifact + a ready model card
live at **[xocialize/Mage-Flow-Edit-Turbo-mlx](https://huggingface.co/xocialize/Mage-Flow-Edit-Turbo-mlx)**
(`Weights/folded_adaln.safetensors` here is the same file); regenerate with
`Weights/dump_folded_adaln.py`.

## Scope — full family supported

All six checkpoints run and are validated (numeric per-step gates vs the
PyTorch oracle + a clean render each at real defaults):

| checkpoint | mode | defaults | validation |
|---|---|---|---|
| Mage-Flow-Edit-Turbo | edit | 4 / cfg 1.0 | full parity suite + e2e |
| Mage-Flow-Edit (RL) | edit | 20 / cfg 5.0 | per-step CFG gate 1.3e-2 + render |
| Mage-Flow-Edit-Base | edit | 30 / cfg 5.0 | render |
| Mage-Flow (RL) | t2i | 20 / cfg 5.0 | t2i gate 5.5e-2, CFG per-step 2.5e-2 + render |
| Mage-Flow-Base | t2i | 30 / cfg 5.0 | render |
| Mage-Flow-Turbo | t2i | 4 / cfg 1.0 | render |

CFG is implemented as two forwards (upstream `batch_cfg=False`) — mathematically
identical to the fused varlen pack, since rotary attention depends only on
relative positions. Upstream's default negative prompt is a single space `" "`.

**Gate note:** at cfg 5.0 an accumulated-trajectory latent comparison is the
wrong metric — guidance multiplies each step's ~1e-2 cross-dtype noise into the
next step's input, and trajectories diverge chaotically while remaining equally
valid. Gate the per-step map (oracle input → one step) plus decoded-image
validity instead.

## Components (all parity-locked vs the PyTorch oracle, CPU stream)

| component | worst rel error | gate |
|---|---|---|
| MageFlow NR-MMDiT (12-block, 4.1B) | 6.8e-6 | `MageFlowGate` |
| MageVAE (encode + decode) | 1.08e-5 | `MageVAEGate` |
| Gaussian-Shading watermark | 0 (bit-exact) | `GSGate` |
| FlowMatchEuler schedule | exact | Turbo 4-step to the digit |
| end-to-end 4-step denoise | 2.8e-2 | `E2EGate` (bf16-oracle vs fp32) |

Qwen3-VL-4B conditioning + the content filter come from
[qwen3vl-mlx-swift](https://github.com/xocialize/qwen3vl-mlx-swift).

## Notes that cost real debugging (full detail in `PORTING-SPEC.md`)

- **The bf16 "grid garbage at ≥512²" was the mlx-swift NAX split-K GEMM bug**
  ([ml-explore/mlx#3797](https://github.com/ml-explore/mlx/issues/3797), fixed
  upstream by [#3810](https://github.com/ml-explore/mlx/pull/3810) but not yet
  in an mlx-swift release ≤0.31.6): mlx-swift JIT-compiles
  `steel_gemm_splitk_axpby_nax` and the 26.x/27.x Metal toolchain miscompiles it
  on M5-class GPUs. The DiT's FFN `proj_out` (K=12288, N=3072) enters the
  dispatch window at 1366 image tokens — a 512² edit packs 2048.
  `MageFeedForward.downProjected` row-chunks below the boundary (exact), so the
  DiT now **runs bf16 by default** (~2× faster than fp32, half the memory);
  `MAGEFLOW_FP32` remains for parity work. Same root cause as Boogu-Image's
  `useFP32DiT` and qwen3vl's `down_proj` chunking.
- **The timestep embedding is bf16-rounded twice** — the sigma *and* the
  frequency table. At scale-1000 sinusoid arguments a 0.2% bf16 shift moves
  cos/sin by radians. Layer parity was perfect (6.8e-6) yet the sampler was
  105% wrong until this was found — visible only end-to-end.
- **VL conditioning needs PIL BICUBIC**, not CoreGraphics — the ViT is
  resampling-sensitive (cos 0.93 → garbage vs 0.98 → clean).
- **The VAE adaLN fold must be baked, not recomputed** (bf16 fold, one channel
  off by 0.039 → 1.2 error while cosine read 1.00000000).
- **`sample_posterior` is True at runtime** despite the config saying false.

## Gates

Each gate drives one component against captured oracle goldens on the CPU stream,
gating on *relative* error (this network's activations reach ~1e5, so absolute
tolerances mislead).
