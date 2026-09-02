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

# multi-reference (1–3 refs, in prompt order: Image 1, Image 2, Image 3)
.build/release/mage-flow-edit --repo <snapshot dir> \
  --ref headphones.png --ref marble.png \
  --prompt "Place the headphones from Image 1 on the marble table from Image 2" --out edit.png

# runtime DiT-LoRA (activation-path adapter; works on the int8/int4 tiers too)
.build/release/mage-flow-edit --repo <snapshot dir> --ref skeleton.png --ref identity.png \
  --lora refcontrol-pose.safetensors --lora-strength 1.0 \
  --prompt "refcontrol apply pose from image 1 with reference from image 2" --out cell.png
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
| Mage-Flow-Edit-Turbo | edit | 4 / cfg 1.0 | full parity suite + e2e + **multi-ref (2 & 3 refs) e2e gate** |
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
| **multi-reference edit (2 and 3 refs)** | per-step DiT 1.6e-2 (2-ref) · 1.4e-2 (3-ref) | `E2EGate --edit-refs N` + conditioning cosine + decoded render (see below) |
| **runtime DiT-LoRA** | zero-delta adapter bit-identical (bf16 + int8); 144/144 targets | `mage-lora-smoke synth` + CLI inertness pair (see below) |

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

## Multi-reference edit (1–3 references) — v0.6.0, AB-A-0047

`MageFlowEditPackage` accepts `IEditRequest.images.count ∈ 1…3` (upstream's *trained*
range; 4+ is rejected legibly, never truncated). References are in prompt order —
Image 1 is the primary — and the mechanism is upstream's exactly: one
`Image j: <|vision_start|><|image_pad|><|vision_end|>` placeholder per reference in the
VL prompt (`_edit_prompt_body`), every reference VAE-encoded at the target size, and
the latents sequence-concatenated as `[target, ref_1 … ref_N]` with RoPE frame index
*j* per reference. The content filter screens all references together (`There are N
source image(s) above.`), at original resolution, fail-closed.

**Gates (Turbo, 512², seed 42, cfg 1.0, PNG fixtures = upstream's `multiref_000000_{0,1}` +
`dog.jpg`, oracle on CPU via `Weights/capture_multiref.py`):**

| gate | 2 refs | 3 refs |
|---|---|---|
| per-step DiT parity, packed `[target, ref…]` (`E2EGate --edit-refs N`, fp32 vs bf16 oracle) | worst rel **1.6e-2** PASS | worst rel **1.4e-2** PASS |
| Qwen3-VL conditioning cosine vs oracle `txt_norm` input (`MAGEFLOW_DUMP_FEATS`) | 0.971 global (text tokens 0.999) | 0.975 global (text tokens 0.999) |
| decoded render vs oracle (`goldens/multiref/N/render.png`) | 24.3 dB, same composition | 23.0 dB, same composition |
| content-filter verdict vs oracle `screen_edit` | pass / pass | refuse / refuse |
| filter M-RoPE positions vs HF `get_rope_index` (`MAGEFLOW_DUMP_FILTER`) | exact (3348 tokens) | exact (5402 tokens) |

Reading the conditioning number honestly: the same two fixtures encoded **alone**
score 0.969 / 0.971 on their image tokens, identical to their joint scores, and the
causal-prefix invariant holds (Image 1's tokens are unchanged when Image 2 follows) —
so the residual is the pre-existing per-image bf16 vision-path precision of the
single-ref port on these particular images (the dog fixture scores 0.996), not the
multi-reference logic. Two findings that fell out of the gate:

- **JPEG decoding, not resizing, was the larger conditioning error.** ImageIO and
  libjpeg decode the same JPEG differently (max 44 / mean 1.0 levels on the landscape
  fixture); the PIL-BICUBIC resize path matches PIL to within 2 levels. Feed the port
  PNGs when parity matters (the ModelSheet consumer already does).
- Single-ref path is **bit-identical** to v0.5.0 (same seed, same input) — the N loop
  is a pure generalisation.

Measured peaks (M5 Max, filter on, bf16 DiT unless noted): 2-ref @512² 19.4 GB ·
3-ref @768² 19.1 GB (int8 DiT 16.8 GB) · 3-ref @1024² 19.8 GB · vs 1-ref @768² 18.9 GB —
each extra reference adds one clean latent (≈ +2 % at 1024²), well inside the declared
1024² envelope. 4-step Turbo composes three references without collapsing (the
consumer's "if Turbo multi-ref turns out weak" question): see the gate renders.

## Runtime DiT-LoRA (1…N adapters) — v0.7.0, AB-A-0050

`MageFlowConfiguration.loraPath` (+ `loraStrength`, default 1.0) applies a local safetensors
adapter to the resident DiT in `load()`, as an **activation-path** term (`LoRALinear` /
`QLoRALinear` from MLXLMCommon — never fused into the base, so it survives bf16 AND the
int8/int4 tiers). It is a runtime override like klein's: not persisted, nil on decode. On the
CLI: `--lora <adapter> [--lora <second> …] [--lora-strength s]` — several adapters rank-stack
into one term per module (their exact sum). First consumer: LTX Studio's RefControl **pose**
LoRA (ai-toolkit `mageflow_edit`, rank 32 / alpha 32, base Mage-Flow-Edit-Base) — a pose cell
is a two-reference edit, hence the pairing with the multi-ref work above.

**Key dialect** (ai-toolkit comfy prefix, diffusers module names): `diffusion_model.
transformer_blocks.<i>.attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,
to_add_out}` and `…{img_mlp,txt_mlp}.{net.0.proj,net.2}` `.lora_{A,B}.weight` → 144 modules /
288 tensors (~85 MB fp16 at rank 32). `img_mod.1` / `txt_mod.1` are accepted if present; the
`transformer.` and `base_model.model.` prefixes too. **Any other key is an error naming the
keys** — a partially-landing adapter would otherwise be a silent no-op.

**Gates** (`mage-lora-smoke synth <dir>` writes the synthetic adapters, then the CLI):

| gate | result |
|---|---|
| structural, bf16 DiT: synthetic rank-32 adapter with the trainer's key set | 144 `LoRALinear`, zero unused keys |
| structural, int8 DiT (in-memory `MageQuantConfig.int8`, `keepHiBlocks [11]`) | 132 `QLoRALinear` + 12 `LoRALinear` — both dispatch branches |
| unknown key (`attn.to_qkv`, klein's fused dialect) | rejected loudly, key named |
| **inertness**, real weights, 2-ref Turbo edit, seed-locked: zero-B adapter vs no LoRA | **bit-identical** on bf16 AND int8 |
| non-inertness (the gate's complement, AB-L-0026): random-B adapter vs no LoRA | max 255 / mean 21.8 (bf16), 254 / 20.8 (int8) — the term is live |
| wrapper path (`MAGE_LORA_PATH=… mage-pkg-smoke … edit`) | applied in `load()`, render valid |

The first real checkpoint (the pod's 250-step save) is the live applicator test:
`mage-lora-smoke apply <ckpt.safetensors> [int8]` prints the landed-target summary and flags a
count ≠ 144. No licence-clean community Mage LoRA exists yet to gate against.

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
