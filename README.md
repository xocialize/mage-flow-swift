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
| **full resolution range 512–2048** | 2048² @ **34.2 dB PSNR** vs oracle | decoded-render gate, every tier eyeballed |

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

## Run-phase progress

Both packages report into the engine's `RunProgress` plane (contract 1.18.0), so a
consumer can show a real seam instead of one indeterminate "Generating…". The phases,
in order:

| phase | covers | steps? |
|---|---|---|
| `screen` | the mandatory Responsible-AI filter (AR verdict, fail-closed); carries the 8.3 GB conditioner load on the evicted tier. Absent when the caller bypasses the filter. | no |
| `encode` | Qwen3-VL conditioning, the negative pass under CFG, and the MageVAE ref encode on the edit path | no |
| `denoise` | the flow-matching Euler loop | **yes** — `step i/N`, 1-based |
| `decode` | MageVAE decode to pixels — one eval, so no per-chunk cadence is claimed | no |
| `postprocess` | PNG encode (reported by the wrapper, after the model is done) | no |

`screen` is Mage's own phase name rather than one of the canonical constants: the filter
is a gating classifier pass, not conditioning, and it is **not a rounding error in the
run**. Measured, Turbo @512² on an M5 Max, warm:

```
edit  screen 4.1 s → encode 0.6 s → denoise 4 steps 1.5 s → decode 0.3 s   (10.0 s incl. 3.4 s load)
t2i   screen 1.2 s → encode 0.1 s → denoise 4 steps 0.4 s → decode 0.1 s   ( 3.7 s incl. 1.9 s load)
```

On the 4-step Turbo tiers the content filter is the **largest** stage — a consumer that
renders only denoise counters still shows a stalled-looking run for the first seconds. The
20–30-step Base/RL tiers invert that balance (2 DiT forwards per step under CFG).

The core stays engine-free: `MageFlow`/`MageFlowEdit` report into their own `MageProgress`
task-local sink, and `MLXMageFlow` forwards those events into `RunProgress`
(`MageProgressBridge`). Nothing bound ⇒ no-op, so the CLI and the gates run unchanged;
`mage-pkg-smoke` binds a print sink the way the engine does and **fails** if the per-step
denoise counters do not cover the loop.

## Gates

Each gate drives one component against captured oracle goldens on the CPU stream,
gating on *relative* error (this network's activations reach ~1e5, so absolute
tolerances mislead).
