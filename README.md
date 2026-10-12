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
  upstream by [#3810](https://github.com/ml-explore/mlx/pull/3810), shipped in
  mlx-swift 0.32.3): mlx-swift ≤0.31.6 JIT-compiles
  `steel_gemm_splitk_axpby_nax` and the 26.x/27.x Metal toolchain miscompiles it
  on M5-class GPUs. The DiT's FFN `proj_out` (K=12288, N=3072) enters the
  dispatch window at 1366 image tokens — a 512² edit packs 2048.
  It was row-chunked below the boundary until mlx-swift 0.32.3 (now the floor)
  fixed it; the DiT **runs bf16 by default** (~2× faster than fp32, half the memory);
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

## Image-to-image start — `initSigma` (creative upscale) — v0.10.0, G-5

`MageFlowEditConfig.initSigma` (σ in (0, 1]) starts the target from **Image 1's latent noised to σ** instead of
pure noise, and spends all `steps` below it: `FlowMatchEulerScheduler(startSigma:)` scales the shifted base grid
so its first sigma is exactly σ. Absent or 1.0 is the unchanged edit (identical sigma array, no mix). Lower σ keeps
more of Image 1. The Gaussian-Shading watermark noise is then only partly present in the result.

- **Package:** `IEditRequest.metaData["initSigma"]` (`MageFlowMetaKeys.initSigmaKey`), validated before any model
  work; outside (0, 1] or non-numeric → `unsupportedRequestFeature`. Reset after every request.
- **CLI:** `--init-sigma S`.
- **ComfyUI equivalence:** `denoise d` on this shift-6 schedule starts at σ = 6d / (1 + 5d); the CivitAI Mage
  restoration workflow's `denoise 0.88` is σ ≈ 0.978.

First consumer: the creative upscale (Gigapixel gap G-5, `mlxengine-todo/CREATIVE-UPSCALE-EVAL.md`) — precision
upscale to the target, then Mage-Edit-Turbo over ≤ 1024² tiles at σ ≈ 0.9 with a content-free instruction. There
σ 0.85–0.93 keeps framing, glyphs and eye colour; σ 1.0 at ×4 rebuilds tile fragments of faces and letters;
below 0.85 it smooths more than it adds. Tests: `SchedulerStartSigmaTests` (3), `InitSigmaMetaTests` (2).

## Tiles in one call, and the small-Mac tiers — v0.11.0, G-5

The creative upscale edits 6–25 tiles per image. One `imageEdit` call per tile is fine while everything stays loaded,
but a small Mac can't hold the Qwen3-VL conditioner (8.3 GB) and the DiT (5.6 GB at int8) at once, and reloading the
conditioner per tile reads ~8 GB per tile.

- **The pipeline is two phases.** `editConditioning` (content filter + Qwen3-VL features: the only part that needs the
  conditioner) and `editRender` (reference latents, start, denoise, decode at `cfg`'s size, seed and start sigma);
  `edit` is the two in a row. T2I splits the same way (`t2iConditioning` / `t2iRender`).
- **`editBatch(_ jobs:)`** runs every job's conditioning in one conditioner load, drops the conditioner
  (`releaseConditioner`), then renders every job. Each `MageEditJob` keeps its own references, seed, size and start
  sigma, so job k's output is the one `edit` makes alone with them.
- **Package mode `tiles`** (`MageFlowModes.tiles`, declared on the `imageEdit` surface): each image of the request is
  edited on its own with the same instruction, as Image 1 of its own edit, at the request's square `width` and
  `metaData["initSigma"]`, image k at `seed + k`. MLXToolKit's `IEditResponse` carries one image, so the edits come back
  **stacked top to bottom in input order**: `width` wide, `count × width` tall. Up to 64 tiles per request
  (`MageFlowModes.maxTiles`); progress reports tile k as `stage` k of n. No contract change.
- **Residency tiers** (`MageFlowConfiguration`):

  | tier | conditioner | DiT | peak (Turbo int8, measured) |
  |---|---|---|---|
  | resident (default) | stays | stays | 16.9–17.1 GB MLX |
  | `evictConditioner` | leaves after each encode, before the denoise | stays | 15.1 GB |
  | `evictTransformer` (implies the above) | takes turns with the DiT | out while the conditioner is in, and between requests | **10.3–10.7 GB** |

  The light tiers reload what they dropped on every request: a plain edit per tile reloads both models per tile, a
  `tiles` request once. `MageFlowEditPackage.modelLoads` counts the loads and their time. Before 0.11.0
  `evictConditioner` dropped the conditioner only after the whole request, so its peak was still conditioner +
  denoise (16.1 GB measured).

Measured (M5 Max, Release, weights in the page cache; a 1024² own render upscaled ×2 → 9 tiles of 768² and
×4 → 25 tiles of 928²; σ 0.9; MLX memory limit 12.7 GB on the light arms, a 16 GB Mac's GPU budget):

| tier · path | ×2 total (per tile) | ×4 total (per tile) | MLX peak ×2 / ×4 | process peak ×2 / ×4 | loads at ×4 (cond. / DiT) |
|---|---|---|---|---|---|
| resident · per tile (v0.10.0, today's ≥ 32 GB) | 22.9 s (2.54) | 79.3 s (3.17) | 16.9 / 17.1 GB | 21.8 / 23.0 GB | 1 / 1 |
| resident · `tiles` | 16.9 s (1.88) | 72.6 s (2.90) | 16.9 / 17.4 GB | 21.6 / 23.0 GB | 1 / 1 |
| `evictConditioner` · per tile (v0.10.0) | 30.7 s (3.41) | 126.4 s (5.06) | 16.1 / 16.5 GB | 16.5 / 17.0 GB | 25 / 1 |
| `evictConditioner` · `tiles` | 18.6 s (2.07) | 81.5 s (3.26) | 15.1 / 15.2 GB | 15.4 / 15.5 GB | 1 / 1 |
| `evictTransformer` · per tile | 23.5 s (2.61) | 83.7 s (3.35) | 10.7 / 10.7 GB | 11.7 / 11.9 GB | 25 / 25 |
| **`evictTransformer` · `tiles`** (the small-Mac path) | **19.7 s (2.19)** | **72.6 s (2.90)** | **10.3 / 10.3 GB** | **11.3 / 11.5 GB** | **1 / 1** |

Each arm is its own process, so its total includes the first call's warm-up. Warm (one edit first), per ×4 tile on
the `tiles` path: conditioner encode (Qwen3-VL features + the reference's VAE encode) 0.10 s, denoise 2.60 s, decode
0.15 s; the lightest tier adds its two loads once per request (0.22 s + 0.15 s here). This Mac keeps every weight in
its page cache, so those reloads are a floor; one plain edit per tile on the lightest tier still cost 0.45–1.34 s more
per tile than `tiles` (two runs: 25 loads of each model, and a slower first denoise after every fresh DiT). A 16 GB Mac
can't cache 13.9 GB of weights, so there every reload is a disk read: 13.9 GB per tile (~350 GB for a ×4)
against 13.9 GB per request with `tiles`.

Every arm returns **the same bytes for every tile** at the same seeds — v0.10.0 per tile, the new per-tile path and
the `tiles` mode, on every tier. Tests: `TilesModeTests` (8 offline: the mode's validation, the light tiers'
configuration and footprints, stage progress; 1 live, `MAGEFLOW_LIVE_STORE=<store>`: three 512² tiles on the
resident and lightest tiers, each band byte-identical to its plain edit, conditioner and DiT loaded once per `tiles`
request against once per tile).

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

## GPU numerics: MageVAE conditioning decoder (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

MageVAE hits it only in `codDecoder`: conv_in 128→384, six 384→384 resnet convs, and conv_out, all
at latent resolution. It starts at 1024² output (latent 64²). The encoder is depthwise and 1×1
throughout. The oracle golden is 512² (latent 32², below the window), and `MageVAEGate` pins the
CPU, so neither could see this.

`cv(...)` now routes in-window shapes through conv3d with kT = 1. This is the default, since it is
eight convs at 64².

Measurements: 1024² DIV2K photo, against the CPU lane.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| `codDecoder` conditioning | 2.1e-3 · max 1.8e-2 | 1.2e-4 · max 6.7e-4 |
| Decode | 1.5e-3 · 68.7 dB | 5.0e-4 · 78.2 dB |
| Decode time, 1024² | 579 ms | +14 ms |

The residual after the route is TF32 in the patch attention and the one-step decoder's matmuls.

Controls and tests:

- Set `MAGEFLOW_VAE_CONV_ROUTE=winograd` to restore raw conv2d (read on every call).
- `swift test --filter MageVAEGPULaneTests/testProbe` is weight-free.
- `MAGE_PARITY=1 MAGE_VAE=… MAGE_FOLD=… swift test -c release -Xswiftc -enable-testing --filter
  MageVAEGPULaneTests` compares the GPU and CPU lanes.

## Gates

Each gate drives one component against captured oracle goldens on the CPU stream,
gating on *relative* error (this network's activations reach ~1e5, so absolute
tolerances mislead).
