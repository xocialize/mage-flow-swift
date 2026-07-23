// Mage-Flow NR-MMDiT — 12 dual-stream blocks, hidden 3072, 24 heads, headDim 128.
//
// Adapted from qwen-image-edit-swift (parity-tested). Structure is preserved 1:1
// with upstream `mage_flow/models/modules/mage_layers.py` so the two can be
// diffed op-for-op.
//
// Deltas vs the Qwen-Image-Edit donor:
//   * TEXT IS NOT ROTATED. Upstream comment: "Prepare vision RoPE (msrope); text
//     tokens are not rotated." Text position info survives only via the causal
//     Qwen encoder. The donor rotates both streams.
//   * No `zero_cond_t` per-token modulation index — Mage modulates from `temb`
//     alone (`modulateIndex` is always nil upstream).
//   * No patchify: patch_size=1 on 128-channel latents, so img_in is a plain
//     Linear(128 -> 3072).
//   * jointAttentionDim 2560 (Qwen3-VL-4B), numLayers 12.
//
// Traps encoded here (all verified against source, see PORTING-SPEC.md):
//   * Block modulation chunks (shift, scale, gate). The upstream COMMENT says
//     "scale, shift, gate" and is WRONG; `_modulate` unpacks shift first.
//   * AdaLayerNormContinuous chunks (scale, shift) — the OPPOSITE order, in the
//     same model.
//   * The block returns (txt, img).
//   * The DiT is fed the raw sigma in [0,1], not t in [0,1000]; the x1000 is
//     inside Timesteps.
//   * The sinusoid table is deliberately downcast to bf16 before the multiply.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum MageFlowConfig {
    public static let hiddenSize = 3072
    public static let numHeads = 24
    public static let headDim = 128           // sum(axesDim) == headDim
    public static let depth = 12
    public static let inChannels = 128
    public static let contextInDim = 2560
    public static let axesDim = [16, 56, 56]  // (frame, height, width)
    public static let ropeTheta = 10_000
    public static let eps: Float = 1e-6
}

// MARK: - RoPE

/// MageFlowEmbedRope: Qwen-Image-style scaled 3-axis RoPE over (frame, h, w),
/// applied to IMAGE tokens only.
///
/// The frame axis is the enumeration index in `imgShapes` — target = 0, ref_j = j.
/// That index is the ONLY thing distinguishing target from reference tokens.
public final class MageFlowEmbedRope {
    let theta: Int
    let axesDim: [Int]
    let scaleRope: Bool
    let posFreqs: MLXArray   // [4096, sum(axesDim)/2]
    let negFreqs: MLXArray

    public init(theta: Int = MageFlowConfig.ropeTheta,
                axesDim: [Int] = MageFlowConfig.axesDim,
                scaleRope: Bool = true) {
        self.theta = theta
        self.axesDim = axesDim
        self.scaleRope = scaleRope
        let pos = MLXArray(0 ..< 4096)
        // neg_index = arange(4096).flip(0) * -1 - 1  ->  -4096 ... -1
        let neg = MLXArray(stride(from: 4095, through: 0, by: -1).map { Int32($0) }) * -1 - 1
        self.posFreqs = concatenated(axesDim.map { Self.ropeParams(pos, $0, theta) }, axis: 1)
        self.negFreqs = concatenated(axesDim.map { Self.ropeParams(neg, $0, theta) }, axis: 1)
    }

    static func ropeParams(_ index: MLXArray, _ dim: Int, _ theta: Int) -> MLXArray {
        let invFreq = MLXArray(1.0).asType(.float32)
            / pow(MLXArray(Float(theta)),
                  MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim))
        return outer(index.asType(.float32), invFreq)
    }

    /// Returns (cos, sin) for the packed image sequence. Text gets nothing.
    ///
    /// `imgShapes` is one flat run across ALL samples and refs — the frame index
    /// keeps incrementing and never resets per sample.
    public func callAsFunction(imgShapes: [(frame: Int, height: Int, width: Int)])
        -> (MLXArray, MLXArray)
    {
        var freqs: [MLXArray] = []
        for (idx, fhw) in imgShapes.enumerated() {
            freqs.append(videoFreqs(frame: fhw.frame, height: fhw.height, width: fhw.width, idx: idx))
        }
        let vid = concatenated(freqs, axis: 0)
        return (cos(vid), sin(vid))
    }

    func videoFreqs(frame: Int, height: Int, width: Int, idx: Int) -> MLXArray {
        let seqLen = frame * height * width
        let splits = axesDim.map { $0 / 2 }        // [8, 28, 28] complex -> 128 real
        var bounds: [Int] = [0]
        for s in splits { bounds.append(bounds.last! + s) }
        let fp = (0 ..< splits.count).map { posFreqs[0..., bounds[$0] ..< bounds[$0 + 1]] }
        let fn = (0 ..< splits.count).map { negFreqs[0..., bounds[$0] ..< bounds[$0 + 1]] }

        // frame axis indexes by position in imgShapes
        let fFrame = broadcast(fp[0][idx ..< (idx + frame)].reshaped(frame, 1, 1, -1),
                               to: [frame, height, width, splits[0]])
        // scaleRope centres H and W around 0 -> resolution-agnostic across 512..2048
        let fH = broadcast(
            concatenated([fn[1][(4096 - (height - height / 2))...], fp[1][..<(height / 2)]], axis: 0)
                .reshaped(1, height, 1, -1),
            to: [frame, height, width, splits[1]])
        let fW = broadcast(
            concatenated([fn[2][(4096 - (width - width / 2))...], fp[2][..<(width / 2)]], axis: 0)
                .reshaped(1, 1, width, -1),
            to: [frame, height, width, splits[2]])
        return concatenated([fFrame, fH, fW], axis: -1).reshaped(seqLen, -1)
    }
}

/// Adjacent-pair complex rotation, computed in fp32.
public func applyRotary(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
    let shape = x.shape
    let v = x.asType(.float32).reshaped(shape.dropLast() + [shape.last! / 2, 2])
    let xR = v[.ellipsis, 0]
    let xI = v[.ellipsis, 1]
    let c = cosT[.newAxis, 0..., .newAxis, 0...]
    let s = sinT[.newAxis, 0..., .newAxis, 0...]
    return stacked([xR * c - xI * s, xR * s + xI * c], axis: -1)
        .reshaped(shape).asType(x.dtype)
}

// MARK: - Timestep

/// Vendored `get_timestep_embedding` (NOT diffusers'):
/// Timesteps(256, flip_sin_to_cos=true, downscale_freq_shift=0, scale=1000).
///
/// ⚠ The frequency table is downcast to `roundDtype` (bf16) BEFORE the multiply.
/// This is the whole reason upstream vendors its own copy — the model was trained
/// with that exact bf16 rounding, and diffusers' fp32 variant degrades output.
/// It is LOAD-BEARING and per-sigma sensitive: at scale-1000 arguments the bf16
/// rounding of the table flips cos/sin at some sigmas (step 2 of the Turbo
/// schedule most of all) and barely moves them at others — which reads as a
/// mysterious non-uniform per-step error if you round to fp32 instead.
///
/// Op order matches upstream exactly: `sigma * freqs`, THEN `* scale`, THEN the
/// sinusoid — not `(sigma*scale) * freqs`.
public func timestepEmbedding(_ t: MLXArray, dim: Int = 256, roundDtype: DType = .bfloat16)
    -> MLXArray
{
    let half = dim / 2
    let exponent = -log(10_000.0) * MLXArray((0 ..< half).map { Float($0) }) / Float(half)
    let freqs = exp(exponent).asType(roundDtype).asType(.float32)   // bf16-round the table
    var emb = t.asType(.float32)[0..., .newAxis] * freqs[.newAxis, 0...]
    emb = 1000.0 * emb
    // flip_sin_to_cos -> [cos, sin]
    return concatenated([cos(emb), sin(emb)], axis: -1)
}

public final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    public init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// `time_type="qwen_proj"` is a MISNOMER: no text feature enters here. The weight
/// inventory confirms it — time_text_embed holds only timestep_embedder.{1,2}.
public final class MageFlowTimestepEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedding
    public init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue =
            TimestepEmbedding(inChannels: 256, timeEmbedDim: embeddingDim)
        super.init()
    }
    /// `sigma` is the raw sigma in [0,1] — NOT t in [0,1000].
    ///
    /// ⚠ The sigma is rounded to the DiT compute `dtype` BEFORE the embedding —
    /// upstream `mage_flow.py:112` does `timesteps = timesteps.to(img.dtype)`
    /// and get_timestep_embedding then scales by 1000. This is LOAD-BEARING, not
    /// a precision detail: at scale-1000 arguments a bf16 rounding of sigma
    /// (e.g. 0.947368 -> 0.949219) shifts cos/sin by whole radians, changing the
    /// embedding entirely. At sigma == 1.0 the round is exact, which is why step 0
    /// alone looks correct if you skip this. In production dtype is bf16 so the
    /// round happens naturally; an fp32 parity run must round to bf16 explicitly.
    public func callAsFunction(sigma: MLXArray, dtype: DType) -> MLXArray {
        // Both the sigma AND the frequency table are bf16-rounded — upstream runs
        // the whole timestep path in the img dtype (bf16 in production). See
        // timestepEmbedding and mage_flow.py:112.
        let roundedSigma = sigma.asType(.bfloat16).asType(sigma.dtype)
        return timestepEmbedder(timestepEmbedding(roundedSigma).asType(dtype))
    }
}

// MARK: - FeedForward

/// diffusers FeedForward(activation_fn="gelu-approximate"): net.0.proj -> tanh-GELU
/// -> net.2. Keys remapped to proj_in / proj_out in `sanitize`.
public final class MageFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    public init(dim: Int, hiddenDim: Int) {
        self._projIn.wrappedValue = Linear(dim, hiddenDim, bias: true)
        self._projOut.wrappedValue = Linear(hiddenDim, dim, bias: true)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProjected(geluApproximate(projIn(x)))
    }

    /// mlx-swift ≤0.31.6 JIT-compiles `steel_gemm_splitk_axpby_nax` (dispatched at
    /// M·N ≥ 2048², K ≥ 10240, K ≥ 3·max(M,N)) and the 26.x/27.x Metal toolchain
    /// miscompiles it on M5-class GPUs → garbage/NaN. This proj_out
    /// (K=12288, N=3072) crosses the boundary at 1366 image tokens — a 512² edit
    /// packs 2048 — which is exactly the "bf16 DiT grid garbage on the edit path".
    /// Chunk rows below the threshold: output rows are independent, so this is
    /// mathematically exact. Same workaround as qwen3vl-mlx-swift's down_proj.
    /// Fixed upstream in ml-explore/mlx#3810 (2026-07-07); remove once an
    /// mlx-swift release ships it (latest 0.31.6 predates the fix).
    func downProjected(_ x: MLXArray) -> MLXArray {
        let tokens = x.dim(-2)
        let rowLimit = 896
        guard x.dtype != .float32, tokens > rowLimit else { return projOut(x) }
        var parts: [MLXArray] = []
        var start = 0
        while start < tokens {
            let end = min(start + rowLimit, tokens)
            parts.append(projOut(x[.ellipsis, start ..< end, 0...]))
            start = end
        }
        return concatenated(parts, axis: -2)
    }
}

// MARK: - Attention

/// MageDoubleStreamAttnProcessor: per-stream QKV, RMSNorm QK on both streams
/// (always on, always RMSNorm), joint [text, image] SDPA, per-stream out proj.
public final class MageJointAttention: Module {
    let heads: Int
    let dimHead: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    public init(queryDim: Int, heads: Int, dimHead: Int, eps: Float = MageFlowConfig.eps) {
        self.heads = heads
        self.dimHead = dimHead
        let inner = heads * dimHead
        self._toQ.wrappedValue = Linear(queryDim, inner, bias: true)
        self._toK.wrappedValue = Linear(queryDim, inner, bias: true)
        self._toV.wrappedValue = Linear(queryDim, inner, bias: true)
        self._addQProj.wrappedValue = Linear(queryDim, inner, bias: true)
        self._addKProj.wrappedValue = Linear(queryDim, inner, bias: true)
        self._addVProj.wrappedValue = Linear(queryDim, inner, bias: true)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._toOut.wrappedValue = [Linear(inner, queryDim, bias: true)]
        self._toAddOut.wrappedValue = Linear(inner, queryDim, bias: true)
        super.init()
    }

    /// Returns (imgOut, txtOut).
    public func callAsFunction(
        hiddenStates: MLXArray, encoderHiddenStates: MLXArray,
        rope: (MLXArray, MLXArray), mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let b = hiddenStates.dim(0)
        let sImg = hiddenStates.dim(1)
        let sTxt = encoderHiddenStates.dim(1)
        let (H, D) = (heads, dimHead)

        var iq = normQ(toQ(hiddenStates).reshaped(b, sImg, H, D))
        var ik = normK(toK(hiddenStates).reshaped(b, sImg, H, D))
        let iv = toV(hiddenStates).reshaped(b, sImg, H, D)
        let tq = normAddedQ(addQProj(encoderHiddenStates).reshaped(b, sTxt, H, D))
        let tk = normAddedK(addKProj(encoderHiddenStates).reshaped(b, sTxt, H, D))
        let tv = addVProj(encoderHiddenStates).reshaped(b, sTxt, H, D)

        // IMAGE ONLY — text is never rotated.
        let (rc, rs) = rope
        iq = applyRotary(iq, cos: rc[..<sImg], sin: rs[..<sImg])
        ik = applyRotary(ik, cos: rc[..<sImg], sin: rs[..<sImg])

        // Joint order [text, image].
        let q = concatenated([tq, iq], axis: 1).transposed(0, 2, 1, 3)
        let k = concatenated([tk, ik], axis: 1).transposed(0, 2, 1, 3)
        let v = concatenated([tv, iv], axis: 1).transposed(0, 2, 1, 3)

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(D)),
            mask: mask.map { .array($0) } ?? .none)
        out = out.transposed(0, 2, 1, 3).reshaped(b, sTxt + sImg, -1)
        return (toOut[0](out[0..., sTxt..., 0...]), toAddOut(out[0..., ..<sTxt, 0...]))
    }
}

// MARK: - Block

public final class MageFlowTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: MageJointAttention
    @ModuleInfo(key: "img_mod") var imgMod: Linear     // Sequential(SiLU, Linear) -> .1
    @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
    @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
    @ModuleInfo(key: "img_mlp") var imgMLP: MageFeedForward
    @ModuleInfo(key: "txt_mod") var txtMod: Linear
    @ModuleInfo(key: "txt_norm1") var txtNorm1: LayerNorm
    @ModuleInfo(key: "txt_norm2") var txtNorm2: LayerNorm
    @ModuleInfo(key: "txt_mlp") var txtMLP: MageFeedForward

    public init(dim: Int, heads: Int, headDim: Int, eps: Float = MageFlowConfig.eps) {
        self._attn.wrappedValue = MageJointAttention(
            queryDim: dim, heads: heads, dimHead: headDim, eps: eps)
        self._imgMod.wrappedValue = Linear(dim, 6 * dim, bias: true)
        self._imgNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._imgNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._imgMLP.wrappedValue = MageFeedForward(dim: dim, hiddenDim: 4 * dim)
        self._txtMod.wrappedValue = Linear(dim, 6 * dim, bias: true)
        self._txtNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._txtNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._txtMLP.wrappedValue = MageFeedForward(dim: dim, hiddenDim: 4 * dim)
        super.init()
    }

    /// (shift, scale, gate) — NOT the order the upstream comment claims.
    static func modulate(_ x: MLXArray, _ mod: MLXArray) -> (MLXArray, MLXArray) {
        let p = split(mod, parts: 3, axis: -1)
        return (x * (1 + p[1][0..., .newAxis, 0...]) + p[0][0..., .newAxis, 0...],
                p[2][0..., .newAxis, 0...])
    }

    /// Returns (txt, img) — matching upstream's `return encoder_hidden_states, hidden_states`.
    public func callAsFunction(
        hiddenStates: MLXArray, encoderHiddenStates: MLXArray, temb: MLXArray,
        rope: (MLXArray, MLXArray), mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        var img = hiddenStates
        var txt = encoderHiddenStates
        let iM = split(imgMod(silu(temb)), parts: 2, axis: -1)   // [attn-half, mlp-half]
        let tM = split(txtMod(silu(temb)), parts: 2, axis: -1)

        let (imgMod1, imgGate1) = Self.modulate(imgNorm1(img), iM[0])
        let (txtMod1, txtGate1) = Self.modulate(txtNorm1(txt), tM[0])
        let (imgAttn, txtAttn) = attn(
            hiddenStates: imgMod1, encoderHiddenStates: txtMod1, rope: rope, mask: mask)
        img = img + imgGate1 * imgAttn
        txt = txt + txtGate1 * txtAttn

        let (imgMod2, imgGate2) = Self.modulate(imgNorm2(img), iM[1])
        img = img + imgGate2 * imgMLP(imgMod2)
        let (txtMod2, txtGate2) = Self.modulate(txtNorm2(txt), tM[1])
        txt = txt + txtGate2 * txtMLP(txtMod2)
        return (txt, img)
    }
}

// MARK: - Final norm

/// NOTE the chunk order here is (scale, shift) — the OPPOSITE of the blocks.
public final class AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    public init(dim: Int, eps: Float = MageFlowConfig.eps) {
        self._linear.wrappedValue = Linear(dim, 2 * dim, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray, _ conditioning: MLXArray) -> MLXArray {
        let emb = linear(silu(conditioning))
        let p = split(emb, parts: 2, axis: -1)
        let scale = p[0][0..., .newAxis, 0...]
        let shift = p[1][0..., .newAxis, 0...]
        return norm(x) * (1 + scale) + shift
    }
}

// MARK: - Model

public final class MageFlowTransformer: Module {
    @ModuleInfo(key: "img_in") var imgIn: Linear
    @ModuleInfo(key: "txt_norm") var txtNorm: RMSNorm
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: MageFlowTimestepEmbeddings
    @ModuleInfo(key: "transformer_blocks") var blocks: [MageFlowTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public let posEmbed: MageFlowEmbedRope

    public init(
        inChannels: Int = MageFlowConfig.inChannels,
        hiddenSize: Int = MageFlowConfig.hiddenSize,
        contextInDim: Int = MageFlowConfig.contextInDim,
        depth: Int = MageFlowConfig.depth,
        heads: Int = MageFlowConfig.numHeads
    ) {
        let headDim = hiddenSize / heads
        // patch_size == 1 -> no patchify, img_in is a plain Linear
        self._imgIn.wrappedValue = Linear(inChannels, hiddenSize, bias: true)
        self._txtNorm.wrappedValue = RMSNorm(dimensions: contextInDim, eps: MageFlowConfig.eps)
        self._txtIn.wrappedValue = Linear(contextInDim, hiddenSize, bias: true)
        self._timeTextEmbed.wrappedValue = MageFlowTimestepEmbeddings(embeddingDim: hiddenSize)
        self._blocks.wrappedValue = (0 ..< depth).map { _ in
            MageFlowTransformerBlock(dim: hiddenSize, heads: heads, headDim: headDim)
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(dim: hiddenSize)
        self._projOut.wrappedValue = Linear(hiddenSize, inChannels, bias: true)
        self.posEmbed = MageFlowEmbedRope()
        super.init()
    }

    /// `img`: [1, L_img, 128] packed latent tokens. `txt`: [1, L_txt, 2560] VL features.
    /// `sigma`: raw sigma in [0,1].
    public func callAsFunction(
        img: MLXArray, txt: MLXArray, sigma: MLXArray,
        imgShapes: [(frame: Int, height: Int, width: Int)], mask: MLXArray? = nil
    ) -> MLXArray {
        var hidden = imgIn(img)
        var encoder = txtIn(txtNorm(txt))
        let temb = timeTextEmbed(sigma: sigma, dtype: hidden.dtype)
        let rope = posEmbed(imgShapes: imgShapes)
        for block in blocks {
            (encoder, hidden) = block(
                hiddenStates: hidden, encoderHiddenStates: encoder, temb: temb,
                rope: rope, mask: mask)
        }
        return projOut(normOut(hidden, temb))
    }

    /// Block-chain entry used by the parity gate: takes post-img_in / post-txt_in
    /// activations and a precomputed rope, so block math is isolated from the
    /// embedding and rope math.
    public func runBlocks(
        hidden hiddenIn: MLXArray, encoder encoderIn: MLXArray, temb: MLXArray,
        rope: (MLXArray, MLXArray), mask: MLXArray? = nil,
        onBlock: ((Int, MLXArray, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var hidden = hiddenIn
        var encoder = encoderIn
        for (i, block) in blocks.enumerated() {
            (encoder, hidden) = block(
                hiddenStates: hidden, encoderHiddenStates: encoder, temb: temb,
                rope: rope, mask: mask)
            onBlock?(i, encoder, hidden)
        }
        return projOut(normOut(hidden, temb))
    }

    /// Key remapping. All params are Linear/RMSNorm — no conv layout work.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            var key = k
            // diffusers FeedForward: net.0.proj -> proj_in, net.2 -> proj_out
            key = key.replacingOccurrences(of: "_mlp.net.0.proj", with: "_mlp.proj_in")
            key = key.replacingOccurrences(of: "_mlp.net.2", with: "_mlp.proj_out")
            // Sequential(SiLU, Linear): index 1 is the Linear
            key = key.replacingOccurrences(of: "img_mod.1.", with: "img_mod.")
            key = key.replacingOccurrences(of: "txt_mod.1.", with: "txt_mod.")
            out[key] = v
        }
        return out
    }
}

extension MageFlowTransformer {
    /// Debug: expose the timestep embedding for a given sigma.
    public func tembFor(sigma: Float) -> MLXArray {
        timeTextEmbed(sigma: MLXArray([sigma]), dtype: .float32)
    }
}
