# Mage-Flow → MLX porting spec

Derived from upstream `microsoft/Mage` @ `main` (cloned in `upstream/`) and the
actual `Mage-Flow-Edit-Turbo` safetensors key inventory. Everything here is
**verified against weights or source**, not inferred from the model card.

Regenerate the inventory with `.venv/bin/python inspect_keys.py`, the dead-weight
accounting with `.venv/bin/python measure_dead.py`.

## Family

Six checkpoints, **one architecture**. `text_encoder/` and `vae/` are
byte-identical across all six (verified by LFS SHA256); only `transformer/`
differs. Published 2026-07-21, MIT, 0 downloads, no MLX port anywhere.

| | Base | RL | Turbo |
|---|---|---|---|
| T2I | `Mage-Flow-Base` | `Mage-Flow` (family primary) | `Mage-Flow-Turbo` |
| Edit | `Mage-Flow-Edit-Base` | `Mage-Flow-Edit` | `Mage-Flow-Edit-Turbo` |
| steps / cfg | 30 / 5.0 | 20–30 / 5.0 | 4 / 1.0 |

Unique bytes for the whole family: 8.9 GB (shared VL) + 0.345 GB (shared VAE)
+ 6 × 8.2 GB (transformers) ≈ 58 GB. Fetch shared components once; pull only
`transformer/` for the other five.

## Routing

**Tier 3** — multi-component pipeline (DiT + MageVAE + Qwen3-VL + scheduler),
no single `model_type` slot. Per the hybrid decision: DiT and text encoder go
**PyTorch → MLX-Swift directly** (strong Swift donors exist); MageVAE gets a
**throwaway MLX-Python rung** (large + novel, no MLX donor anywhere).

## Donors

| Component | Donor | Fit |
|---|---|---|
| DiT | `mlxengine-image/PROD/qwen-image-edit-swift` | `QwenEmbedRope` is an exact match for `MageFlowEmbedRope` — same theta 10000, `axesDim [16,56,56]`, `scaleRope true`, 4096-entry pos/neg tables. Same `Timesteps(256, flipSinToCos, scale 1000)` → `TimestepEmbedding`. Same `txtNorm RMSNorm(eps 1e-6)` → `txtIn Linear`. |
| Text encoder | `mlxengine-think/PROD/qwen3vl-mlx-swift` | `lastHiddenState` already covers text-only (T2I) *and* vision-merged + DeepStack (Edit), returns post-final-norm / pre-`lm_head` — exactly Mage's `_skip_lm_head=True` output. `lm_head` + full `KVCache` threading intact. |
| MageVAE | *none* | Fully novel. The real work. |

## Config: most of `transformer/config.json` is dead metadata

`pipeline.load_from_repo` strips it. `MageFlowParams` accepts only
`in_channels, out_channels, context_in_dim, hidden_size, num_heads, depth,
axes_dim, checkpoint, patch_size`. There is exactly one code path — no branching.

Stripped and meaningless: `rope_type`, `time_type`, `double_block_type`,
`vec_type`, `mlp_ratio`, `theta`, `qkv_bias`, `guidance_embed`,
`depth_single_blocks`, `apply_text_rotary_emb`, `schedule_mode`,
`use_time_shift`, `vec_in_dim`.

Do **not** build dispatch for `"msrope"` / `"qwen_proj"` — they name the only
implementation. `"qwen_proj"` in particular is a misnomer: no text-encoder
feature enters the timestep embedding (the `hidden_states` arg is used only for
dtype), and the weight inventory confirms it — `time_text_embed` has only
`timestep_embedder.linear_1 [3072,256]` / `linear_2 [3072,3072]`.

## DiT — verified against 397 tensors / 4.116B params

```
img_in            Linear(128 → 3072)          # patch_size=1, NO patchify
txt_norm          RMSNorm(2560, eps 1e-6)
txt_in            Linear(2560 → 3072)
time_text_embed   Timesteps(256, flip_sin_to_cos, downscale_freq_shift 0, scale 1000)
                  → TimestepEmbedding(256 → 3072)   # linear_1, SiLU, linear_2
transformer_blocks × 12                        # double-stream only, zero single-stream
norm_out          AdaLayerNormContinuous(3072, elementwise_affine=False, eps 1e-6)
proj_out          Linear(3072 → 128, bias=True)
```

`inner_dim 3072`, `heads 24`, `head_dim 128`, `assert sum(axes_dim) == 128`.
No guidance embedder, no `vec` weights (`vec` is hard-zeroed upstream).

**Per block** (all eps 1e-6):
- `img_mod` / `txt_mod` = `Sequential(SiLU, Linear(3072 → 18432))` = 6×3072.
  Computed from `temb` only. `chunk(2)` → attn-half / mlp-half; each half
  chunks **(shift, scale, gate)** in that order — verified at
  [mage_layers.py:561](upstream/mage_flow/models/modules/mage_layers.py:561).
  **⚠ The source comment two lines up (`# For scale, shift, gate for norm1 and
  norm2`, lines 532/552) contradicts the code and is WRONG.** Trust
  `_modulate`, not the comment; believing it swaps shift and scale.
- `img_norm1/2`, `txt_norm1/2` = **non-affine `LayerNorm`**, eps 1e-6 — *not*
  RMSNorm.
- Modulation is FLUX-style `x*(1+scale)+shift`; gate applied additively on the
  residual. `scale/shift/gate` are `repeat_interleave`d per-token by `cu_lens`.
- MLP: `Linear(3072→12288)` → **tanh-approx GELU** (`gelu-approximate`) →
  `Linear(12288→3072)`.
- fp16 overflow clamp to ±65504, applied to **both** streams and **guarded on
  `dtype == torch.float16`** — genuinely inert in bf16, but its existence is
  evidence about magnitudes (see "Publish dtype" below).
- Returns `(encoder_hidden_states, hidden_states)` = **(txt, img)** — confirmed
  at [mage_layers.py:666](upstream/mage_flow/models/modules/mage_layers.py:666).
- Returns **`(txt, img)`** — the loop assigns `txt, img = block(...)`.

**Attention** — true dual-stream, all biases `True`:
- image `to_q/to_k/to_v` + `to_out.0`; text `add_q_proj/add_k_proj/add_v_proj`
  + `to_add_out`.
- **qk-norm always on, always RMSNorm**, `elementwise_affine=True`, dim 128,
  eps 1e-6: `norm_q/norm_k` (image), `norm_added_q/norm_added_k` (text).
  Confirmed by the four `[128]` tensors per block.
- `softmax_scale = 128^-0.5`, `causal=False`.
- Joint order is **[text, image] per sample**, assembled by scatter into a
  preallocated buffer (varlen segments interleave txt-then-img per sample).

**`AdaLayerNormContinuous` chunks scale-first, shift-second** — the opposite of
the usual diffusers order. `norm_out.linear [6144, 3072]` = 2×3072. Verified:
```python
emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
scale, shift = torch.chunk(emb, 2, dim=-1)      # scale FIRST
x = self.norm(x) * (1 + scale) + shift          # per-token repeat_interleave
```
So within one model the two modulation sites use **opposite** orders:
blocks are `(shift, scale, gate)`, the final norm is `(scale, shift)`.

**Joint sequence assembly** (verified, mage_layers.py:450–494) — scatter into a
preallocated buffer, not `cat`:
```python
txt_dest = joint_cu_lens[txt_sample_ids] + txt_intra_pos
img_dest = joint_cu_lens[img_sample_ids] + txt_lens[img_sample_ids] + img_intra_pos
```
The `+ txt_lens[...]` offset on the image side is what puts **text before image
within each sample**. Outputs are read back with the same index arrays.

## RoPE (`MageFlowEmbedRope`)

Qwen-Image-style scaled 3-axis RoPE over **(frame, height, width)**, applied to
**image tokens only**.

- Two complex tables of 4096 positions: `pos = arange(4096)`,
  `neg = arange(4096).flip(0) * -1 - 1`. `polar(1, outer(index, theta^(-2i/dim)))`,
  theta 10000.
- Column split `[d//2 for d in axes_dim] = [8, 28, 28]` → 64 complex = 128 real.
  So **`axes_dim[0]=16` → frame, `[1]=56` → height, `[2]=56` → width**.
- `scale_rope=True` is hard-coded → H and W are **centered around 0**:
  positions run `−(H − H//2) … −1, 0 … H//2 − 1`. This is what makes RoPE
  resolution-agnostic across 512–2048.
- **Frame axis = the enumeration index in `img_shapes`.** In an edit pack:
  target = 0, ref_1 = 1, ref_2 = 2. This is the *only* thing distinguishing
  target from reference — no mask, no channel concat.

  The nesting matters. `forward()` opens with a two-step normalization:
  ```python
  if isinstance(video_fhw, list):      video_fhw = video_fhw[0]   # drop batch dim
  if not isinstance(video_fhw, list):  video_fhw = [video_fhw]    # wrap a bare tuple
  ```
  and the pipeline always passes the **nested** form `img_shapes = [shape_seq]`
  (pipeline.py:524), where `shape_seq` is a *list of `(1,gh,gw)` tuples* built as
  ```python
  shape_seq.append((1, gh, gw))                 # target      -> frame idx 0
  shape_seq.extend(s[0] for s in ref_shapes)    # ref_j       -> frame idx j
  ```
  Pass a flat `[(1,h,w), (1,h,w)]` instead and step 1 silently collapses it to
  just the first shape — every ref would land on frame index 0 and become
  indistinguishable from the target. **Port the nesting, not just the values.**

- **The outer batch dimension of `img_shapes` is always discarded** (`[0]`).
  With `n > 1` samples, `shape_seq` is one flat run across *all* samples, so the
  frame index simply keeps incrementing: sample0-target=0, sample0-ref=1,
  sample1-target=2, … There is no per-sample reset.
- **Trap, confirmed:** `batch_cfg=True` builds
  `"d_img_shapes": [img_shapes[0] + img_shapes[0]]` (pipeline.py:167) — the
  shape list concatenated with **itself**. The uncond copies therefore continue
  the frame numbering rather than repeating it, so cond and uncond copies of the
  same image get **different** frame-axis RoPE. Reproduce exactly; do not "fix"
  it. Only reachable at cfg > 1, i.e. the Base/RL checkpoints, not Turbo.
- `img_ids` / `txt_ids` are built in the pipeline but **never reach the DiT**.
  Dead code — skip it.
- **Text is never rotated.** Text position info survives only through the causal
  Qwen encoder.
- Application is the **adjacent-pair** complex convention
  (`reshape(..., D/2, 2)` → complex), computed in **fp32**, freqs broadcast over
  heads.

## Publish dtype: bf16, NOT fp16 — measured

Activations blow past fp16's 65504 ceiling. Measured per-block std of the
**text** stream from the oracle at 512² (step 0):

| block | 0 | 1 | 4 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|
| std | 16.7 | 81.7 | 107.5 | 187.6 | 300.5 | **1089.0** | 916.8 | 916.8 |

The image stream is larger still — dumping it as fp16 turned **every** tensor
over 1M elements into `inf`.

Read the upstream clamp precisely: it is `clip(-65504, 65504)` on both streams,
**guarded on `dtype == torch.float16`**. So in fp16 upstream *saturates* rather
than producing `inf` — output degrades, it does not NaN — and in bf16 the clamp
never fires at all. The clamp is therefore not active in the shipping dtype;
what it tells us is that the authors expected these magnitudes to exceed fp16
range, and the measurements above confirm they do.

**Consequences:**
- Publish `-bf16`, never `-fp16`. This is the skill's "high-magnitude-activation
  net collapses under fp16" case, now evidence-backed rather than assumed.
- Parity goldens must be stored **fp32**. An fp16 `.npy` golden silently
  becomes `inf` and every downstream comparison is meaningless.
- Expect the same sensitivity in the Swift port. Compare against the sibling
  note in `mlxengine-image/CLAUDE.md` — Boogu needs `useFP32DiT: true` because
  its bf16 DiT NaNs ≥384² on the edit path. Gate this DiT at bf16 across the
  full resolution range before trusting it; be ready to fall back to fp32.

## The timestep embedding is bf16-rounded — TWICE (the subtle one)

The single hardest bug of the port, and exactly the skill's warning: **layer
parity was perfect (DiT 6.8e-6) yet the sampler path was wrong, visible only
end-to-end.** The DiT velocity was 105% wrong at σ=0.947, non-uniformly across
steps (fine at σ=1.0, worst at the middle steps).

Upstream **vendors its own** `get_timestep_embedding` (`mage_layers.py:24`, not
diffusers'), for one reason stated in its docstring: the frequency table is
downcast to `timesteps.dtype`.

Two bf16 rounds, both load-bearing:

1. **The frequency table:** `emb = torch.exp(exponent).to(timesteps.dtype)`
   (bf16). Rounding to fp32 instead is a silent no-op that passes small tests.
2. **The sigma:** `timesteps = timesteps.to(img.dtype)` (`mage_flow.py:112`,
   bf16) before `get_timestep_embedding` scales it by 1000.

Why it matters and why it's per-sigma: the sinusoid argument is `1000·σ·freq`,
i.e. ~1000. A bf16 rounding of σ (0.947368 → 0.949219) or of the freq table
shifts the argument by ~1–2 radians, which flips `cos`/`sin` entirely. At
σ = 1.0 the round is exact (→ step 0 always looks correct), but at the Turbo
schedule's middle steps it dominates. Rounding only one of the two still fails
(step 2 stayed at rel 0.77 with only the sigma rounded).

Op order also matters and matches upstream: **`σ · freqs`, then `· 1000`, then
`sin/cos`** — not `(σ·1000)·freqs`.

The whole timestep path runs in the model's dtype (bf16 in production), so a
bf16 DiT rounds naturally. An **fp32 parity gate must round to bf16 explicitly**
(both the σ and the table), else it diverges from the real model — which is
correct behaviour, since the model was *trained* with the rounding.

After the fix (fp32 port vs bf16 oracle, CPU stream): temb worst rel 2.6e-2,
velocity worst rel 5.0e-2, full 4-step denoise latent rel 2.8e-2 — all
bf16-vs-fp32 noise.

Gate: `E2EGate <transformerDir> goldens/e2e`. Capture with `capture_e2e.py`
(per-step DiT inputs) + `capture_velocity.py` (per-step velocity + temb).

## Precision landmines

Four places where "upgrading" to fp32 silently changes output:

1. **The sinusoid table is deliberately downcast to bf16 before the multiply.**
   Upstream comments that the model was trained with this exact rounding and
   that diffusers' fp32 variant degrades output.
2. RoPE is computed in fp32.
3. `RMSNorm` in `NerfFinalLayer` is fp32-internal.
4. VAE decode runs under `autocast(bfloat16)`.

Also: **the DiT is fed the raw sigma ∈ [0,1]**, not `t` ∈ [0,1000]. The ×1000
happens inside `Timesteps`.

## Packing / varlen attention

Single varlen sequence, batch dim 1. Image: each sample's latent
`rearrange(b c h w -> b (h w) c)`, all samples (and per sample,
`[target, ref_1…ref_N]`) concatenated on dim 1. `img_cu_seqlens` int32.
Text tokenized separately and `cat`'d, `txt_cu_seqlens` likewise — **no padding,
no attention mask** (the `txt_mask` built in the pipeline never reaches the
transformer). `_modulate` asserts `x.shape[0] == 1` when `cu_lens` is given.

`_attn_backend.py` already ships an **`sdpa` fallback**: a Python loop over
`zip(cu[:-1], cu[1:])` doing per-segment SDPA, `(s,h,d)→(1,h,s,d)` with explicit
GQA `repeat_interleave` on k/v. **That is the reference for the MLX port** —
mirror the per-segment loop; don't build a block-diagonal mask.

### Getting off flash-attn on macOS takes THREE aligned settings

Cost me three failed oracle launches. All of these are required:

1. **`VF_HF_ATTN_IMPL=sdpa`** — only picks `attn_implementation` when the HF
   Qwen3-VL is constructed. Covers the stock HF attention and nothing else.
2. **`_attn_backend.set_attn_backend("sdpa")`** — upstream monkey-patches
   *both* `Qwen3VLTextAttention.forward` and the DiT (`mage_layers.py:480`) to
   call `_attn_backend.flash_attn_varlen_func` for the packed varlen path,
   which bypasses (1) entirely and defaults to `_BACKEND = "flash2"`.
   Symptom: `ModuleNotFoundError: No module named 'flash_attn'` raised from
   `text_encoder.py:344` even with (1) set.
3. **…and (2) must be applied AFTER the model is built.**
   `MageFlowModel.__init__` calls `set_attn_backend(config.attn_type)`
   ([mage_flow.py:160](upstream/mage_flow/models/mage_flow.py:160)) with
   `attn_type` defaulting to `"flash2"`, silently clobbering any earlier
   setting. `attn_type` is a plain pydantic `Field` with **no env alias**, and
   its docstring lists only `flash2`/`flash4` — but `_normalize` does accept
   `"sdpa"`. Post-load assignment is the only hook.

## Text path

`mage_text.py` is **not** the encoder — it's the mandatory Responsible-AI
content classifier. The encoder is `modules/text_encoder.py`.

- `CustomQwen3VLForConditionalGeneration`, bf16, `_skip_lm_head=True` →
  returns `last_hidden_state` = **`hidden_states[-1]` after the final norm**
  (not `[-2]`, not pre-norm). `output_hidden_states=False`.
- 2560-dim context → `txt_norm` → `txt_in`.
- **The reference image goes through the VL vision tower**, including the
  **DeepStack** injection into the first N decoder layers
  (`deepstack_visual_indexes [5, 11, 17]`). The port must include ViT + deepstack
  merge.
- **Two different resizes of the same reference image:** VL conditioning at
  **384 px long edge, BICUBIC** (`vl_cond_long_edge=384`, to match training's
  `max_pixels`); the VAE path uses **full target resolution**.
- Prompt templates, verbatim from `PROMPT_TEMPLATE` in `models/utils.py`. The
  `{}` is the user body; for edit that body is
  `_edit_prompt_body` = `"Image 1: <|vision_start|><|image_pad|><|vision_end|>Image 2: …"
  + instruction`.

  **`"mage-flow"` (T2I), `start_idx = 34`:**
  ```
  <|im_start|>system
  Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
  <|im_start|>user
  {}<|im_end|>
  <|im_start|>assistant
  ```
  Note there is **no space and no newline** between `background:` and
  `<|im_end|>` — the system line ends flush against the token.

  **`"mage-flow-edit"` (Edit), `start_idx = 64`:**
  ```
  <|im_start|>system
  Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>
  <|im_start|>user
  {}<|im_end|>
  <|im_start|>assistant
  ```
- **The first `start_idx` tokens are sliced off the hidden states** — 34 (t2i) /
  64 (edit). These counts are tied to the exact template text above; change one
  character of the template and the slice is wrong. `vec = h_valid.mean(0)` is
  computed but discarded by the DiT.
- **M-RoPE degenerates to plain 1-D.** `qwen3_patch_forward` passes
  `position_ids` as a flat per-sequence `arange` expanded to 3 identical rows.
  **Do not implement mrope deltas for the conditioning path.**
- `tokenizer_max_length = 2048`, truncation at `2048 + drop_idx`.

### Required change to `qwen3vl-mlx-swift`

`LanguageModel.hiddenState` hard-computes spatial M-RoPE via `getRopeIndex`
([Qwen3VLModel.swift:1273](../../../mlxengine-think/PROD/qwen3vl-mlx-swift/Sources/Qwen3VL/Qwen3VLModel.swift:1273)).
Mage needs flat positions. Plumb an optional `positionIds` through
`Qwen3VL.lastHiddenState` → `LanguageModel.hiddenState` →
`Model.callAsFunction` (which already accepts one).

**The AR content-filter path must keep real M-RoPE** — the two callers differ.

**DONE** — implemented as an additive optional `positionIds` (defaults to nil =
existing M-RoPE behaviour, so Boogu-Image is unaffected), plus
`Qwen3VL.flatPositionIds(sequenceLength:batch:)`. Verified by
`Qwen3VLGate --load-probe <weightsDir> [image.png]`:

| check | result |
|---|---|
| Qwen3-VL-**4B** config decodes | hidden 2560, inter 9728, 36 layers, 32/8 heads, headDim 128, ropeTheta 5e6, mropeSection [24,20,20]; vision depth 24, outHidden 2560, deepstack [5,11,17] |
| strict weight load (throws on any missing module key) | PASS |
| text-only forward | `[1,24,2560]`, all finite |
| override on text-only | `max_abs 0.0` — correctly a **no-op** (no image tokens ⇒ M-RoPE already degenerates to a flat arange) |
| **override on an image case** (grid 1×32×32, 256 img tokens, seq 266) | **`max_abs 106.5`, `cos 0.877`** |

That last row is the point: the two paths diverge **enormously**. Using the
default spatial M-RoPE for Mage-Flow conditioning would have produced
cos-0.877 features — badly degraded edits — and at a mere 32×32 grid. The gap
widens with grid size, so it would have looked "nearly right" in a small smoke
test and collapsed at 1024–2048 px.

The backbone was previously gated only at 8B; 4B loads unchanged because the
config is fully `Codable`-driven.

## Scheduler / sampler

`FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
use_dynamic_shifting=False)`. Base sigmas `linspace(1.0, 1/steps, steps)`, then
static shift `σ' = 6σ / (1 + 5σ)`, terminal 0 appended.

**Turbo (4 steps) gate — exact values:**

```
sigmas    [1.0, 0.947368, 0.857143, 0.666667, 0.0]
timesteps [1000, 947.37, 857.14, 666.67]
```

Loop: `pred = velocity(img, ctx, sigmas[i])` → `x += (σ_{i+1} − σ_i) · v`.

**Turbo runs cfg=1.0 → CFG disabled, one forward per step, 4 forwards total.**
Base/RL run cfg=5.0 → `batch_cfg` duplicate-and-pack with a negative encode plus
renormalization. **Build the CFG path anyway** — skipping it strands four of six
checkpoints and it is expensive to retrofit into the packing logic later.

## THREE RNG streams must be reproduced (or injected)

None of these has an MLX equivalent. For parity, **dump each from PyTorch and
inject on both sides** — the skill's standing rule for cross-framework RNG.
Order of draws matters; they interleave within one generation.

| # | stream | seeded by | consumed by |
|---|---|---|---|
| 1 | NumPy **PCG64** (`default_rng(key)`) | GS key (default `20260720`) | Gaussian-Shading XOR pad + index map |
| 2 | torch **CPU `Generator`** | `seed & 0x7FFFFFFF` | GS uniforms `u`, then `ndtri((half+u)/2)` |
| 3 | torch **GLOBAL** CPU RNG | `torch.manual_seed(seeds[i])` per sample | `randn_like` inside `MageVAE.encode` posterior sampling — **once per reference image** |

Stream 3 is the easy one to miss: it is not passed a generator, it silently
uses the process-global RNG, and it fires once per ref-image encode.

## Latent init is watermarked, not `randn`

`get_noise()` is computed and then **immediately overwritten** by
`mage_latent.encode_noise` — a **Gaussian-Shading** watermarked initial noise:
SHA-256-derived 256-bit payload (`"MageFlow"`), a NumPy `default_rng(key)` (PCG64)
XOR pad + index map over all C·H·W entries, `u ~ U(0,1)` from a **CPU torch**
generator seeded `seed & 0x7FFFFFFF`, then `z = ndtri((half + u)/2)`.

Default key `20260720`, overridable via `MAGEFLOW_GS_KEY` / `~/.mageflow/gs_key`.
**There is no toggle to disable it.** `invert_to_noise` is the detector.

Marginally N(0,1), so output *quality* does not depend on reproducing it — but
**bit-exact parity does**, and neither PCG64 nor torch's CPU `rand` has an MLX
equivalent. **For parity testing, dump this tensor from PyTorch and inject it on
both sides** (standard practice per the skill's RNG rule). `ndtri` itself maps to
`sqrt(2) · erfinv(2p − 1)`.

## Edit conditioning

**Sequence concat, not channel concat** (`in_channels == out_channels == 128`,
no extra conditioning channels). Per sample, refs are resized to the *target*
H/W, VAE-encoded, flattened, and packed as `[target, ref_1 … ref_N]`. Refs are
**clean latents rebuilt and held fixed at every step**; only the target slice of
the velocity is stepped:

```python
vel     = _velocity(...)
pred_t  = vel[:, target_idx, :]
stepped = scheduler.step(pred_t, t, cat(targets, dim=1))[0]
```

## Resolution

All sizes floored to a multiple of 16 (floor 16). Latent grid
`ceil(H/16) × ceil(W/16)`. RoPE tables cover 4096 positions/axis → any size up
to 65 536 px. T2I defaults 1024². Edit target-size precedence: explicit
`height`+`width` > `max_size` (longest side, aspect-preserved) > the source
image's own size. Different samples in one pack may differ in resolution — that
is the point of the packing.

**Gate decoded output across the full 512–2048 range**, not just at one small
size (skill pitfall #32 / largest-production-grid rule). A cosine that sags
monotonically with grid size is a structural bug, not noise.

## MageVAE — the novel component

Latent `[B, 128, H/16, W/16]`. **No `quant_conv`/`post_quant_conv`, no conv
up/down pyramid, no ConvTranspose, no scaling/shift factor.** Nothing like
`AutoencoderKL`.

Two top-level weight trees: `student.dconv_encoder.*` (encoder) and
`pipeline.*` (decoder).

**Encoder** — a one-step diffusion encoder run at `t=0` with `z_t = zeros`:
- `patch_cond_embed = Conv2d(3 → 768, k16 s16)` — the **entire** 16× downsample
  in one strided conv.
- 2 × `head_blocks` at 768 (`_EncoderDiCoBlock`, affine LayerNorm, no adaLN)
- `proj_down Conv2d(768 → 384, 1)`, `z_proj Conv2d(128 → 384, 1)`,
  `fuse_proj Conv2d(768 → 384, 1)`
- 21 × `DiCoBlock(384)`
- `norm_out LayerNorm2d(384, eps 1e-6)`, `proj_out Conv2d(384 → 256, 1)`
- `encode()`: `logvar.clamp(-20, 10)`. H, W must be exact multiples of 16.

  **⚠ `sample_posterior` is TRUE at runtime — encode is STOCHASTIC.**
  `vae/config.json` says `"sample_posterior": false`, and that value is
  **never read**. `load_from_repo` builds `ModelConfig` from
  `transformer/config.json` only; `MageVAE` is constructed with
  `sample_posterior=self.config.vae_sample_posterior`, whose `Field` default is
  **`True`** ([mage_flow.py:35](upstream/mage_flow/models/mage_flow.py:35),
  [:268](upstream/mage_flow/models/mage_flow.py:268)). Verified live:
  `m.vae.sample_posterior == True`. Textbook constructor-default-beats-config.

  So encode returns
  ```python
  mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
  ```
  **The logvar half of `proj_out` is LIVE. Do NOT drop those 128 channels.**
  (An earlier draft of this spec said to drop them — wrong.)

  Renders are still bit-reproducible because the pipeline seeds the **global**
  torch RNG per sample immediately before encoding, with an upstream comment
  saying exactly why:
  ```python
  torch.manual_seed(seeds[i])   # MageVAE.encode samples the posterior (global RNG)
  ```
  (pipeline.py:499 for edit, :302 for t2i). Empirically confirmed: two
  independent 512² runs with identical settings are **100.00% pixel-identical**.

**Decoder** — genuinely a denoiser net, but **deterministic at inference**:
```python
cond  = decoder_model.y_embedder.decoder(z)
noise = torch.zeros(B, 3, H*16, W*16)   # zeros, not randn
t     = torch.zeros(B)
return decoder_model.forward(noise, t, cond)
```
**No RNG needed in the port.**
- `y_embedder.decoder`: `conv_in Conv2d(128→384, 3×3)` → `ResnetBlock, AttnBlock,
  ResnetBlock, AttnBlock, ResnetBlock` all at 384 and at latent resolution →
  `Normalize` → swish → `conv_out Conv2d(384→384, 3×3)`.
- `_DConvDenoiser`: `s_embedder` = `Conv2d(3→128, k16 s16, bias=False)` on the
  zero image, concat with cond, `Conv2d(512→384, 1)`; 21 × `DiCoBlock(384)`;
  then `unfold(16, stride=16)` on pixels, concat `y_embedder_x
  Conv2d(384 → 32·256, 1)` features, `NerfEmbedder` (2-D DCT positional
  features), `SimpleMLPAdaLN` with **3** `_MLPResBlock`s at width 32,
  `NerfFinalLayer(RMSNorm(32, eps 1e-6) + Linear(32→3))`, then `fold` back.
- **Upsampling is per-patch pixel regression + fold — no ConvTranspose, no
  nearest-neighbour upsample.** Since stride == kernel, `unfold`/`fold` are
  **pure reshapes** in MLX.
- `x_embedder.embedder.0` is `[32, 99]` → NerfEmbedder input is
  **99 = 3 (pixel) + 32 (cond) + 64 (DCT)**. Read the exact concat order from
  source before wiring.

**`DiCoBlock`** (×42 total across encoder + decoder):
`conv1 1×1` → `conv2` **depthwise 3×3** (`groups=C`, weight `[384,1,3,3]` —
confirmed in the inventory) → **exact GELU** (not tanh-approx) → channel
attention (`AdaptiveAvgPool2d(1)` → 1×1 → Sigmoid) → `conv3 1×1`; then a 1×1
FFN at 4× (`conv4 384→1536`, `conv5 1536→384`) with adaLN
`x*(1+scale)+shift` and gates.

**Norms and activations:**
- All `Conv2d` — **no Conv3d anywhere**.
- `Normalize = GroupNorm(32, eps=1e-6, affine=True)` in ResnetBlock / AttnBlock /
  `_Decoder`. ← the skill's #1 silent VAE killer; get this exact.
- `LayerNorm2d = LayerNorm(C, eps 1e-6)` via NHWC permute; DiCoBlock norms are
  `affine=False`, encoder-side `_EncoderLayerNorm2d` is `affine=True`.
- Custom `RMSNorm(eps 1e-6)`, fp32-internal, in `NerfFinalLayer`.
- swish (`x·sigmoid(x)`) in Resnet/`_Decoder`; **exact `F.gelu`** inside
  DiCoBlock; SiLU in adaLN/timestep MLPs; Sigmoid in the channel-attention branch.

**`AttnBlock` is not global attention.** It is **patch-local 32×32
self-attention** with replicate padding, `bmm`-based, single-head, softmax scale
`c^-0.5` where **c is the channel count (384), not head_dim**. Mathematically
odd — replicate it exactly. Crops padding afterwards.

### 41.9% of the VAE is dead at inference

Measured with `measure_dead.py` against the real file:

| group | params | share |
|---|---|---|
| `y_embedder.encoder.*` / `y_embedder.bottleneck.*` | 34.43 M | 20.0 % |
| `adaLN_modulation` (constant-folded at t=0) | 37.26 M | 21.6 % |
| `t_embedder` (dead after the fold) | 0.49 M | 0.3 % |
| **live** | **100.30 M** | **58.1 %** |

*(An earlier version of this table also listed the `proj_out` logvar half,
0.05 M, as droppable. It is NOT — `sample_posterior` is `True` at runtime, so
logvar is used. Corrected above.)*

- `y_embedder.encoder.*` / `bottleneck.*` — an embedded AutoencoderKL-style conv
  encoder plus a BatchNorm, the training-time anchor encoder. **Explicitly
  `continue`d at load** ([mage_vae.py:588](upstream/mage_flow/models/modules/mage_vae.py:588)).
  Never executed.
- `adaLN_modulation` — `_freeze_adaln_cache()` runs at construction: `t` is
  always 0, so every `DiCoBlock`'s `SiLU + Linear(384→2304)` is evaluated once
  and replaced with a `_ConstAdaLN` buffer.
  **Exception:** `dec_net.res_blocks.*.adaLN_modulation` is deliberately *not*
  folded — it is conditioned on a per-position latent. Keep it.

  **⚠ BAKE the constants; do NOT recompute the fold.** The fold runs in
  `dtype = next(self.parameters()).dtype` — **bf16**, because the safetensors are
  bf16 on disk and the fold happens at construction before any upcast.
  Reproducing that bit-for-bit in another framework fails: bf16 matmul
  accumulation order differs. Measured cost of getting it wrong —

  | fold | worst block max_abs vs golden |
  |---|---|
  | recomputed in fp32 | **1.20** (one gate value off by 0.039 → 1.2 in that channel) |
  | recomputed in bf16 (MLX) | **1.98** — closer on some blocks, worse on others |
  | **baked from the constructed model** | **8.5e-4 — PASS** |

  The failure is brutally localized and easy to misread: with fp32 folding,
  `blocks.0` was wrong in **exactly one channel of 384** (154), uniform across
  all spatial positions, while cosine stayed at **1.00000000**. A cosine gate
  alone would have passed it.

  `dump_folded_adaln.py` extracts all 42 buffers (21 encoder + 21 decoder) to
  `folded_adaln.safetensors` — 96 768 values, **0.39 MB fp32**, replacing 37.7 M
  params. The Swift conversion must do the same.

**Total VAE: 345 MB bf16 → 201 MB live.** Drop the dead groups at conversion.

Empirically confirmed at load: of 839 tensors, `CoDEncoder` loads 342 and
`CoDDecoder` loads 386 (= 728); **111 are never loaded**.

### VAE details verified line-by-line against source

Read from `mage_vae.py`; these are the ones an MLX port gets wrong by writing
the *natural* thing instead of the *upstream* thing.

- **`NerfEmbedder` frequency grid is `linspace`, not `arange`.**
  `torch.linspace(0, max_freqs, max_freqs)` with `max_freqs=8` gives
  **8 points from 0 to 8 inclusive** — `[0, 1.1429, 2.2857, 3.4286, 4.5714,
  5.7143, 6.8571, 8.0]`, step 8/7. Writing `arange(8)` = `[0..7]` is the
  obvious-but-wrong port and would silently shift every decoded pixel.
  ```python
  coeffs = (1 + fx * fy) ** -1
  dct    = cos(pos_x * fx * pi) * cos(pos_y * fy * pi) * coeffs   # (1, ps^2, 64)
  ```
  `pos = linspace(0, 1, patch_size)`, `meshgrid(..., indexing="ij")` → `pos_y,
  pos_x` **in that order**. Concat is `cat([x, dct], dim=-1)` — **features
  first, DCT last** → 35 + 64 = 99, matching `x_embedder.embedder.0 [32, 99]`.
- **The VAE's `TimestepEmbedder` is not the DiT's `Timesteps`.** It builds
  `cat([cos(args), sin(args)], -1)` directly — **cos first, no
  `flip_sin_to_cos` mechanism** — with `max_period=10000`,
  `frequency_embedding_size=256`, freqs in fp32. Dead after adaLN folding, but
  needed if you ever fold at runtime rather than at conversion.
- **`AttnBlock` exact formulation** (LDM/taming lineage):
  ```python
  w_ = bmm(Q.permute(0,2,1), K) * (c ** -0.5)   # c = 384 CHANNELS, not head_dim
  w_ = softmax(w_, dim=2).permute(0, 2, 1)
  h_ = bmm(V, w_)
  ```
  Patches are formed as `reshape(b,c,nph,d,npw,d).permute(0,2,4,1,3,5)
  .reshape(b*np,c,d*d)` with `d=32`, **`mode="replicate"` padding**, and the pad
  is cropped off after. Residual is `x + proj_out(h_)`.
- **`DiCoBlock.forward`** — modulation chunks **(shift_msa, scale_msa, gate_msa,
  shift_mlp, scale_mlp, gate_mlp)** in one `chunk(6, dim=1)`:
  ```python
  x = modulate(norm1(inp), shift_msa, scale_msa)
  x = gelu(conv2(conv1(x)));  x = x * ca(x);  x = conv3(x)
  x = inp + gate_msa[...,None,None] * x
  x = x + gate_mlp[...,None,None] * conv5(gelu(conv4(modulate(norm2(x), shift_mlp, scale_mlp))))
  ```
  `modulate` is `x*(1+scale)+shift`. `norm1/norm2` are **non-affine**
  `LayerNorm2d`; the encoder variant `_EncoderDiCoBlock` uses **affine**
  `_EncoderLayerNorm2d` and has **no adaLN and no gates** (plain residuals).
- **`LayerNorm2d` is NCHW→NHWC→NCHW** around `F.layer_norm`. In MLX, which is
  NHWC-native, this is just a plain LayerNorm on the last axis — do not
  replicate the permutes.
- **`_MLPResBlock`**: `chunk(3, dim=-1)` → (shift, scale, gate);
  `h = in_ln(x)*(1+scale)+shift`; `x + gate*mlp(h)`. `LayerNorm(eps 1e-6)`,
  `mlp = Linear → SiLU → Linear`. `SimpleMLPAdaLN.cond_embed` reshapes to
  `(B, patch_size², -1)` = `(B, 256, 32)`.
- **`ResnetBlock`** is swish-before-conv: `conv1(swish(norm1(x)))` then
  `conv2(swish(norm2(h)))`, residual `x + h`, with `nin_shortcut` only when
  channels differ. `nonlinearity = x*sigmoid(x)`.
- **`RMSNorm`** casts to fp32, computes `mean(x²)` over the last axis,
  `rsqrt(var + eps)`, then casts back **before** multiplying by `weight`.

**"Anchor-latent KL" at inference** is not a separate tensor: the "anchor" is the
zero latent `z_t` with `t=0` fed to the one-step encoder, and the KL head is the
packed `(mean, logvar)` in `proj_out`'s 256 channels. The KL itself exists only
in training.

## MageVAE MLX rung — DONE, parity-locked

`mage_vae_mlx.py` (+ `gate_vae.py`, `gate_vae_dec.py`), CPU stream, fp32:

| stage | max_abs | cosine |
|---|---|---|
| encoder head_blocks / proj_down / fuse_proj | ≤5.6e-4 | 1.0000000 |
| encoder 21× DiCoBlock (deepest, block 20) | 8.5e-4 | 1.0000002 |
| encoder `proj_out` / mean vs `moments.0` | 5.8e-5 | 0.9999999 |
| `y_embedder.decoder` (cond) | 3.1e-5 | 1.0000000 |
| **full decode → pixels** | **4.2e-6** | **1.0000000** |

Round-trip encode→decode renders a clean, sharp reconstruction — no tint, no
checkerboard.

### Two layout traps the port hit

**1. `y_embedder_x` is J-MAJOR; `dec_net.cond_embed` is P-MAJOR.** Adjacent
tensors, opposite layouts. Upstream builds the first as
`y_embedder_x(cond).flatten(2)` → `reshape(b, -1, patch**2, length)`, so its
8192 channels read as `(32, 256)` with `index = j*256 + p`. But `cond_embed`
reshapes *directly* to `(N, patch**2, -1)` = `(256, 32)`, p-major. Reshaping
both the same way is the obvious port and gives `max_abs 35.8` at `x_embedder`
and a visibly wrong image. Only the cond channels matter here — the "noise"
image is zeros, so the pixel channels contribute nothing and cannot reveal the
bug.

**2. `unfold`/`fold` are pure reshapes** (stride == kernel), and in NHWC the
whole per-patch pixel-regression path is
`reshape(B, lh, p, lw, p, C).transpose(0,1,3,2,4,5)` and its inverse. No
ConvTranspose, no interpolation anywhere in the decoder.

Confirmed correct on the first try, worth noting as *non*-traps: patch-local
32×32 attention with `c**-0.5` (c = 384 channels), replicate padding,
`GroupNorm(32, eps=1e-6)`, the `linspace(0,8,8)` DCT grid, and the
fp32-internal RMSNorm in `NerfFinalLayer`.

## MageFlow DiT (Swift) — DONE, parity-locked

`mlxengine-image/WIP/mage-flow-swift`, adapted from the qwen-image-edit donor.
`MageFlowGate <transformerDir> <goldens>`, CPU stream, fp32:

| stage | max_abs | **relative** | cosine |
|---|---|---|---|
| rope cos / sin | 4.8e-7 | 4.8e-7 | 1.00000000 |
| block 0 txt / img | 2.0e-3 | 3.9e-7 | 0.9999999 |
| block 11 txt | 3.5e-1 | 2.8e-6 | 0.9999999 |
| block 11 img | 3.1e-2 | 6.8e-6 | 1.00000000 |
| **proj_out** (model output) | **2.0e-5** | **3.8e-6** | **1.00000000** |

397 weight tensors load with **no missing module keys**, so `sanitize` is right
(`_mlp.net.0.proj`→`proj_in`, `_mlp.net.2`→`proj_out`, `{img,txt}_mod.1`→`{img,txt}_mod`).

### The goldens are bf16; gate on RELATIVE error

Two gate-design mistakes worth not repeating:

1. **The stage-B goldens came from a bf16 forward** (upstream does
   `transformer.to(bfloat16)`), so an fp32 port compared against them shows
   1–4 bf16 ULPs per tensor — `worst max_abs 1.59e3`, which reads as a
   catastrophic bug. Proof it was dtype: `|golden − bf16(golden)| == 0.00`
   exactly (the goldens are already bf16-representable), and a PyTorch **fp32**
   re-run from the identical starting activations differs from the bf16 golden
   by **1.5888e3** — matching the Swift port's 1.5891e3. `capture_dit_fp32.py`
   generates the true fp32 reference; gate against that.
2. **Absolute tolerances are meaningless here.** Activations reach |x| ≈ 1.2e5
   by block 11, so 0.35 absolute there is 2.8e-6 relative — fp32 noise. An
   absolute 1e-2 gate fails a correct port. The gate now judges
   `max_abs / max|golden|` with a 1e-4 threshold.

### Verified-correct on the first try

RoPE (exact, 4.8e-7 — including `scaleRope` centring, the negative tables, and
frame-index-per-`imgShapes`-entry), the `(shift, scale, gate)` block chunk order
against the misleading comment, the opposite `(scale, shift)` order in
`AdaLayerNormContinuous`, the `(txt, img)` return order, text-not-rotated, and
tanh-approx GELU in the FFN.

**Not yet exercised:** varlen packing beyond a single sample. With n=1 the packed
sequence is one contiguous `[txt, img]` run, so the donor's dense attention is
exactly right; `cu_seqlens` only bite with multiple samples or `batch_cfg`.

## Content filter

Two greedy `.generate()` calls (≤192 new tokens) on the **same** Qwen3-VL weights
run **before every generation**, fail-closed, returning a plain white PIL image
on refusal. Structurally inseparable from the pipeline as written.

`qwen3vl-mlx-swift` retains `lm_head` (with the tied-embedding
`embedTokens.asLinear` fallback — and this config has
`tie_word_embeddings: true`, so the tied path is the live one, already handled in
`sanitize`) and **full `KVCache` threading with incremental rope-delta
bookkeeping**. The README's "generation machinery is dropped" refers to the
convenience wrapper, not the model surface.

**DONE** — added `Qwen3VL.generate(inputIds:pixelValues:imageGridTHW:maxTokens:eosTokenIds:)`:
greedy argmax, `KVCacheSimple` per layer, prefill → single-token decode steps,
EOS stop (default 151645). The vision-merge is factored into a shared
`prepareVisionInputs` rather than duplicated from `lastHiddenState`.

Deliberately takes **no** `positionIds` override — the filter wants real spatial
M-RoPE; only `lastHiddenState` uses the flat one.

Verified via `Qwen3VLGate --gen-probe <weightsDir> [prompt] [cap]` on the
Mage-Flow Qwen3-VL-4B weights:

| prompt | result |
|---|---|
| "capital of France, one word" | `"Paris"`, 1 tok, **ended on EOS** |
| "reply with exactly this JSON…" | `{"violates": false, "category": "none"}`, 12 tok, EOS, **36 tok/s** |

The second is the filter's actual output shape (`FilterVerdict` JSON) and came
back exactly right — and being multi-token, it exercises the decode-step
cache-offset / rope-delta path that a 1-token answer never reaches. Test both;
a single short answer silently skips the harder path.

## No CUDA blockers

No custom CUDA kernels, no Triton, no CUTLASS in the released inference code
(the README's "fused CUDA kernels" refers to training infra, not in this repo).
`torch.compile` usage is optional and inference-irrelevant. flash-attn is
avoidable via the shipped `sdpa` backend.

Ops needing hand-work in MLX:
- `view_as_complex` / `polar` → convert to (cos, sin) real adjacent-pair rotation
- `flash_attn_varlen_func` → per-segment loop
- scatter-assign into the preallocated joint buffer → per-segment concat
- `ndtri` → `sqrt(2)·erfinv(2p−1)` (but dump the noise tensor instead)
- `unfold`/`fold` → pure reshape (stride == kernel)
- depthwise `Conv2d(groups=C)` 3×3 and `AdaptiveAvgPool2d(1)` (→ mean over H,W)
- `torchvision` BICUBIC resize — use PIL on both sides; resampling differences
  shift edit results

## Oracle assets (captured, in `goldens/`)

Reproduce with `./run_capture.sh` (full) or `capture.py --stage B` +
`vae_capture.py`. Fixture: `upstream/mage_flow/assets/dog.jpg`, instruction
*"make the background a snowy forest"*, seed 42, steps 4, cfg 1.0, CPU.

| asset | what |
|---|---|
| `step0/*.npy` | 39 per-sub-op DiT goldens at 512², **fp32**, denoise step 0 — all 12 blocks as `.0`=txt / `.1`=img, attn at blocks 0 and 11, `img_in`, `txt_norm`, `txt_in`, `time_text_embed`, `pos_embed.real/.imag`, `norm_out`, `proj_out` |
| `vae/*.npy` | fine-grained MageVAE goldens (encoder + decoder submodules), fp32, plus fp32-vs-bf16 decode delta |
| `init_noise_gaussian_shading.npy` | the watermarked init noise `(1,128,32,32)` — **inject this on both sides**; the PCG64 + torch-CPU RNG has no MLX equivalent |
| `scheduler_sigmas.npy` / `_timesteps.npy` | gated exactly against the documented Turbo schedule |
| `render_{512,768,1024,1536}.png` | decoded reference renders, all verified coherent |
| `filter_verdict.json` | cached RAI verdict (the gate costs 370 s/call on CPU) |

**Renders are NOT comparable across sizes** — same seed at a different token
grid composes differently. Compare each only to its own MLX counterpart at the
same resolution.

**2048² is deliberately not captured.** Measured CPU cost scales at roughly
pixel^1.5, with the exponent climbing as attention's share grows:

| size | 512² | 768² | 1024² | 1536² | 2048² |
|---|---|---|---|---|---|
| CPU time | 4.1 min | 12.9 min | 34.9 min | 124.7 min | ~6–8 h (est.) |

2048² was cut so it wouldn't hold the machine for a full day ahead of the
port existing. Capture it before the final resolution gate — overnight, or
on GPU once an MLX path exists.

## Environment

`.venv` (uv, Python 3.12) pinned to upstream `requirements.txt`: torch 2.13.0,
transformers 5.5.0, diffusers 0.38.0, numpy 2.4.3, einops 0.8.2,
safetensors 0.8.0, accelerate 1.13.0. Version drift matters — upstream
monkey-patches `Qwen3VLTextModel.forward` and `Qwen3VLTextAttention.forward`.

Weights: `HF_HOME=/Volumes/Satechi/hf-cache` (the home volume is at 74%;
the dev volume has 3.6 TB free).
