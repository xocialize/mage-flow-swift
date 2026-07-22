// MageVAE in MLX-Swift — a one-step diffusion codec, nothing like AutoencoderKL.
//
// Language-port of the parity-locked MLX-Python rung (mage-flow-oracle/
// mage_vae_mlx.py). Same framework, so this is near-mechanical; the framework
// port (conv layouts, NHWC, lazy eval) was already debugged there.
//
// latent [B, H/16, W/16, 128]. No quant_conv, no conv pyramid, no ConvTranspose,
// no scaling/shift factor.
//
// Load-bearing details, each of which cost a debugging cycle in the Python rung:
//   * adaLN constants are BAKED (folded_adaln.safetensors), never recomputed.
//     Upstream folds at t=0 in the PARAMETER dtype (bf16, since the safetensors
//     are bf16 on disk and the fold runs at construction before any upcast).
//     Recomputing it in fp32 shifted ONE gate value by 0.039 -> 1.2 absolute
//     error in that channel, with cosine still reading 1.00000000.
//   * `y_embedder_x` is J-MAJOR (index = j*256 + p) but `dec_net.cond_embed` is
//     P-MAJOR. Adjacent tensors, opposite layouts.
//   * unfold/fold are PURE RESHAPES (stride == kernel).
//   * LayerNorm2d's NCHW->NHWC->NCHW permutes are a PyTorch memory-layout
//     workaround; here the data is already NHWC so it is a plain last-axis
//     LayerNorm and the permutes must NOT be reproduced.
//   * GroupNorm eps 1e-6 (a framework default of 1e-5 silently ruins output).
//   * AttnBlock is PATCH-LOCAL 32x32, softmax scale c**-0.5 where c is the
//     CHANNEL count (384), not head_dim.
//   * NerfEmbedder freqs are linspace(0, 8, 8) — 0..8 INCLUSIVE, step 8/7 —
//     not arange(8).

import Foundation
import MLX
import MLXNN

public enum MageVAEConfig {
    public static let latentChannels = 128
    public static let downsampleFactor = 16
    public static let hidden = 384
    public static let headSize = 768
    public static let numBlocks = 21
    public static let hiddenX = 32
    public static let eps: Float = 1e-6
}

// MARK: - primitives

@inline(__always) func swish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// Last-axis LayerNorm on NHWC — what upstream's LayerNorm2d achieves via its
/// permute sandwich.
@inline(__always)
func ln(_ x: MLXArray, _ w: MLXArray? = nil, _ b: MLXArray? = nil,
        eps: Float = MageVAEConfig.eps) -> MLXArray {
    MLXFast.layerNorm(x, weight: w, bias: b, eps: eps)
}

/// GroupNorm over channels for NHWC. eps 1e-6 — the #1 silent VAE killer.
func groupNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray,
               groups: Int = 32, eps: Float = MageVAEConfig.eps) -> MLXArray {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let g = x.reshaped(B, H * W, groups, C / groups)
    let mean = g.mean(axes: [1, 3], keepDims: true)
    let v = g.variance(axes: [1, 3], keepDims: true)
    let n = ((g - mean) * rsqrt(v + eps)).reshaped(B, H, W, C)
    return n * w + b
}

@inline(__always)
func cv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil,
        stride: Int = 1, padding: Int = 0, groups: Int = 1) -> MLXArray {
    let y = conv2d(x, w, stride: IntOrPair(stride), padding: IntOrPair(padding), groups: groups)
    return b.map { y + $0 } ?? y
}

/// Weight bag keyed by the checkpoint's own names (`student.` / `pipeline.`
/// prefixes, which do NOT appear in the live PyTorch module tree).
public struct VAEWeights {
    var t: [String: MLXArray]
    public subscript(_ k: String) -> MLXArray { t[k]! }
    public func has(_ k: String) -> Bool { t[k] != nil }
}

// MARK: - blocks

/// DiCoBlock with pre-folded adaLN `mod` (6*C):
/// 1x1 -> depthwise 3x3 -> EXACT gelu -> channel attention -> 1x1, then a 4x FFN.
/// Chunk order (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
func dicoBlock(_ x0: MLXArray, _ w: VAEWeights, _ p: String, _ mod: MLXArray) -> MLXArray {
    var x = x0
    let C = x.dim(-1)
    func part(_ i: Int) -> MLXArray { mod[(i * C) ..< ((i + 1) * C)] }
    let (sh1, sc1, g1) = (part(0), part(1), part(2))
    let (sh2, sc2, g2) = (part(3), part(4), part(5))

    var h = ln(x) * (1 + sc1) + sh1                       // norm1: non-affine
    h = cv(h, w[p + "conv1.weight"], w[p + "conv1.bias"])
    h = cv(h, w[p + "conv2.weight"], w[p + "conv2.bias"], padding: 1, groups: C)
    h = gelu(h)                                          // EXACT gelu, not tanh-approx
    var ca = h.mean(axes: [1, 2], keepDims: true)
    ca = sigmoid(cv(ca, w[p + "ca.1.weight"], w[p + "ca.1.bias"]))
    h = h * ca
    h = cv(h, w[p + "conv3.weight"], w[p + "conv3.bias"])
    x = x + g1 * h

    h = ln(x) * (1 + sc2) + sh2                           // norm2: non-affine
    h = cv(h, w[p + "conv4.weight"], w[p + "conv4.bias"])
    h = gelu(h)
    h = cv(h, w[p + "conv5.weight"], w[p + "conv5.bias"])
    return x + g2 * h
}

/// _EncoderDiCoBlock: no adaLN, no gates, and the norms ARE affine.
func encoderDicoBlock(_ x0: MLXArray, _ w: VAEWeights, _ p: String) -> MLXArray {
    var x = x0
    let C = x.dim(-1)
    var h = ln(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
    h = cv(h, w[p + "conv1.weight"], w[p + "conv1.bias"])
    h = cv(h, w[p + "conv2.weight"], w[p + "conv2.bias"], padding: 1, groups: C)
    h = gelu(h)
    var ca = h.mean(axes: [1, 2], keepDims: true)
    ca = sigmoid(cv(ca, w[p + "ca.1.weight"], w[p + "ca.1.bias"]))
    h = h * ca
    h = cv(h, w[p + "conv3.weight"], w[p + "conv3.bias"])
    x = x + h

    h = ln(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
    h = cv(h, w[p + "conv4.weight"], w[p + "conv4.bias"])
    h = gelu(h)
    h = cv(h, w[p + "conv5.weight"], w[p + "conv5.bias"])
    return x + h
}

/// swish BEFORE conv (LDM lineage). nin_shortcut only when channels differ.
func resnetBlock(_ x0: MLXArray, _ w: VAEWeights, _ p: String) -> MLXArray {
    var x = x0
    var h = cv(swish(groupNorm(x, w[p + "norm1.weight"], w[p + "norm1.bias"])),
               w[p + "conv1.weight"], w[p + "conv1.bias"], padding: 1)
    h = cv(swish(groupNorm(h, w[p + "norm2.weight"], w[p + "norm2.bias"])),
           w[p + "conv2.weight"], w[p + "conv2.bias"], padding: 1)
    if w.has(p + "nin_shortcut.weight") {
        x = cv(x, w[p + "nin_shortcut.weight"], w[p + "nin_shortcut.bias"])
    }
    return x + h
}

/// PATCH-LOCAL 32x32 self-attention, single head, replicate padding.
/// Softmax scale is `c ** -0.5` with c = CHANNEL count (384), not head_dim.
func attnBlock(_ x: MLXArray, _ w: VAEWeights, _ p: String, patch: Int = 32) -> MLXArray {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let h_ = groupNorm(x, w[p + "norm.weight"], w[p + "norm.bias"])
    var q = cv(h_, w[p + "q.weight"], w[p + "q.bias"])
    var k = cv(h_, w[p + "k.weight"], w[p + "k.bias"])
    var v = cv(h_, w[p + "v.weight"], w[p + "v.bias"])

    let ph = (patch - H % patch) % patch
    let pw = (patch - W % patch) % patch
    if ph != 0 || pw != 0 {
        let widths: [IntOrPair] = [IntOrPair(0), IntOrPair((0, ph)), IntOrPair((0, pw)), IntOrPair(0)]
        q = padded(q, widths: widths, mode: .edge)   // replicate
        k = padded(k, widths: widths, mode: .edge)
        v = padded(v, widths: widths, mode: .edge)
    }
    let (Hp, Wp) = (q.dim(1), q.dim(2))
    let (nph, npw) = (Hp / patch, Wp / patch)
    let d = patch * patch

    func toPatches(_ t: MLXArray) -> MLXArray {
        t.reshaped(B, nph, patch, npw, patch, C)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(B * nph * npw, d, C)
    }
    let (qp, kp, vp) = (toPatches(q), toPatches(k), toPatches(v))
    var wgt = matmul(qp, kp.transposed(0, 2, 1)) * (pow(Float(C), -0.5))
    wgt = softmax(wgt, axis: -1)
    var out = matmul(wgt, vp)

    out = out.reshaped(B, nph, npw, patch, patch, C)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(B, Hp, Wp, C)
    if ph != 0 || pw != 0 { out = out[0..., ..<H, ..<W, 0...] }
    return x + cv(out, w[p + "proj_out.weight"], w[p + "proj_out.bias"])
}

// MARK: - encoder

/// One-step diffusion encoder at t=0 with z_t = zeros. Returns packed
/// (mean, logvar) as 256 NHWC channels — BOTH halves are live, since
/// `sample_posterior` is TRUE at runtime (vae/config.json's `false` is never read).
public func encodeMoments(_ x: MLXArray, _ w: VAEWeights) -> MLXArray {
    let P = "student.dconv_encoder."
    let (B, H, W) = (x.dim(0), x.dim(1), x.dim(2))

    // the ENTIRE 16x downsample, in one strided conv
    var cond = cv(x, w[P + "patch_cond_embed.weight"], w[P + "patch_cond_embed.bias"], stride: 16)
    for i in 0 ..< 2 { cond = encoderDicoBlock(cond, w, "\(P)head_blocks.\(i).") }
    cond = cv(cond, w[P + "proj_down.weight"], w[P + "proj_down.bias"])

    let zt = MLXArray.zeros([B, H / 16, W / 16, MageVAEConfig.latentChannels], dtype: .float32)
    let zp = cv(zt, w[P + "z_proj.weight"], w[P + "z_proj.bias"])
    var s = cv(concatenated([cond, zp], axis: -1),
               w[P + "fuse_proj.weight"], w[P + "fuse_proj.bias"])

    for i in 0 ..< MageVAEConfig.numBlocks {
        s = dicoBlock(s, w, "\(P)blocks.\(i).", w["fold:\(P)blocks.\(i)."])
    }
    s = ln(s, w[P + "norm_out.weight"], w[P + "norm_out.bias"])
    return cv(s, w[P + "proj_out.weight"], w[P + "proj_out.bias"])
}

/// `sampledPosterior` defaults to TRUE to match runtime. Pass `noise` to inject
/// a captured tensor for parity (torch's global RNG has no MLX equivalent).
public func vaeEncode(_ x: MLXArray, _ w: VAEWeights,
                      samplePosterior: Bool = true, noise: MLXArray? = nil) -> MLXArray {
    let out = encodeMoments(x, w)
    let c = MageVAEConfig.latentChannels
    let mean = out[.ellipsis, ..<c]
    if !samplePosterior { return mean }
    let logvar = clip(out[.ellipsis, c...], min: -20.0, max: 10.0)
    let n = noise ?? MLXRandom.normal(mean.shape)
    return mean + exp(0.5 * logvar) * n
}

// MARK: - decoder

/// y_embedder.decoder: latent -> 384-ch conditioning at latent resolution.
public func codDecoder(_ z: MLXArray, _ w: VAEWeights) -> MLXArray {
    let P = "pipeline.y_embedder.decoder."
    var h = cv(z, w[P + "conv_in.weight"], w[P + "conv_in.bias"], padding: 1)
    h = resnetBlock(h, w, P + "block.0.")
    h = attnBlock(h, w, P + "block.1.")
    h = resnetBlock(h, w, P + "block.2.")
    h = attnBlock(h, w, P + "block.3.")
    h = resnetBlock(h, w, P + "block.4.")
    h = groupNorm(h, w[P + "norm_out.weight"], w[P + "norm_out.bias"])
    return cv(swish(h), w[P + "conv_out.weight"], w[P + "conv_out.bias"], padding: 1)
}

/// 2-D DCT positional features, (1, patch^2, maxFreqs^2).
/// freqs = linspace(0, maxFreqs, maxFreqs) -> 0..8 INCLUSIVE (step 8/7).
func nerfDCT(patch: Int, maxFreqs: Int = 8) -> MLXArray {
    let pos = MLX.linspace(Float(0), Float(1), count: patch)
    let grid = meshGrid([pos, pos], indexing: .ij)
    let posY = grid[0].reshaped(-1, 1, 1)
    let posX = grid[1].reshaped(-1, 1, 1)
    let freqs = MLX.linspace(Float(0), Float(maxFreqs), count: maxFreqs)
    let fx = freqs[.newAxis, 0..., .newAxis]
    let fy = freqs[.newAxis, .newAxis, 0...]
    let coeffs = pow(1 + fx * fy, -1)
    let dct = cos(posX * fx * Float.pi) * cos(posY * fy * Float.pi) * coeffs
    return dct.reshaped(1, -1, maxFreqs * maxFreqs)
}

/// One-step diffusion decoder. The noise input is ZEROS and t=0, so this is
/// deterministic — no RNG. Upsampling is per-patch pixel regression + fold.
public func vaeDecode(_ z: MLXArray, _ w: VAEWeights, patch: Int = 16) -> MLXArray {
    let P = "pipeline."
    let (B, lh, lw) = (z.dim(0), z.dim(1), z.dim(2))
    let (H, W) = (lh * patch, lw * patch)
    let length = lh * lw
    let hid = MageVAEConfig.hidden
    let hx = MageVAEConfig.hiddenX
    let p2 = patch * patch

    let cond = codDecoder(z, w)
    let zeros = MLXArray.zeros([B, H, W, 3], dtype: .float32)

    // s_embedder: Conv2d(3->128, k16 s16, NO bias) on the zero image, concat cond
    var s = cv(zeros, w[P + "s_embedder.proj1.weight"], nil, stride: patch)
    s = cv(concatenated([s, cond], axis: -1),
           w[P + "s_embedder.proj2.weight"], w[P + "s_embedder.proj2.bias"])
    for i in 0 ..< MageVAEConfig.numBlocks {
        s = dicoBlock(s, w, "\(P)blocks.\(i).", w["fold:\(P)blocks.\(i)."])
    }
    let sf = s.reshaped(B * length, hid)

    // unfold(patch, stride=patch): stride == kernel -> PURE RESHAPE
    let xp = zeros.reshaped(B, lh, patch, lw, patch, 3)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(B, length, p2, 3)

    // y_embedder_x is J-MAJOR: (B, L, 32, 256) -> transpose -> (B, L, 256, 32).
    // NOTE dec_net.cond_embed below is P-MAJOR. Two adjacent tensors, two layouts.
    let yx = cv(cond, w[P + "y_embedder_x.weight"], w[P + "y_embedder_x.bias"])
        .reshaped(B, length, hx, p2)
        .transposed(0, 1, 3, 2)

    // NerfEmbedder input is [pixel(3), cond(32), dct(64)] = 99 — features, then DCT
    var x = concatenated([xp, yx], axis: -1).reshaped(B * length, p2, 3 + hx)
    let dct = broadcast(nerfDCT(patch: patch), to: [B * length, p2, 64])
    x = concatenated([x, dct], axis: -1)
    x = matmul(x, w[P + "x_embedder.embedder.0.weight"].T) + w[P + "x_embedder.embedder.0.bias"]

    // SimpleMLPAdaLN — res_blocks adaLN is per-position and is NOT folded
    x = matmul(x, w[P + "dec_net.input_proj.weight"].T) + w[P + "dec_net.input_proj.bias"]
    let c = (matmul(sf, w[P + "dec_net.cond_embed.weight"].T)
        + w[P + "dec_net.cond_embed.bias"]).reshaped(B * length, p2, hx)
    for i in 0 ..< 3 {
        let q = "\(P)dec_net.res_blocks.\(i)."
        let mod = matmul(c * sigmoid(c), w[q + "adaLN_modulation.1.weight"].T)
            + w[q + "adaLN_modulation.1.bias"]
        let sh = mod[.ellipsis, ..<hx]
        let sc = mod[.ellipsis, hx ..< (2 * hx)]
        let g = mod[.ellipsis, (2 * hx)...]
        var h = MLXFast.layerNorm(x, weight: w[q + "in_ln.weight"], bias: w[q + "in_ln.bias"],
                                  eps: MageVAEConfig.eps) * (1 + sc) + sh
        h = matmul(h, w[q + "mlp.0.weight"].T) + w[q + "mlp.0.bias"]
        h = h * sigmoid(h)                                  // SiLU
        h = matmul(h, w[q + "mlp.2.weight"].T) + w[q + "mlp.2.bias"]
        x = x + g * h
    }

    // NerfFinalLayer: fp32-internal RMSNorm, then Linear(32 -> 3)
    let v = x.asType(.float32)
    var y = v * rsqrt(mean(v * v, axis: -1, keepDims: true) + MageVAEConfig.eps)
    y = y * w[P + "final_layer.norm.weight"]
    y = matmul(y, w[P + "final_layer.linear.weight"].T) + w[P + "final_layer.linear.bias"]

    // fold: inverse of the unfold reshape
    return y.reshaped(B, lh, lw, patch, patch, 3)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(B, H, W, 3)
}

// MARK: - loading

public enum MageVAELoader {
    /// `vaeSafetensors` is the upstream VAE file; `foldedAdaLN` is the BAKED
    /// constants (dump_folded_adaln.py). The fold must never be recomputed.
    public static func load(vae vaeURL: URL, foldedAdaLN foldURL: URL) throws -> VAEWeights {
        var t: [String: MLXArray] = [:]
        for (k, v) in try MLX.loadArrays(url: vaeURL) {
            // y_embedder.encoder/bottleneck are skipped at load upstream (dead:
            // a training-time anchor encoder, 34.4M params, never executed).
            if k.contains("y_embedder.encoder.") || k.contains("y_embedder.bottleneck.") { continue }
            // PyTorch conv (O, I, kH, kW) -> MLX (O, kH, kW, I). Depthwise
            // (C, 1, 3, 3) with groups=C becomes (C, 3, 3, 1) under the same
            // transpose.
            t[k] = (v.ndim == 4 ? v.transposed(0, 2, 3, 1) : v).asType(.float32)
        }
        for (k, v) in try MLX.loadArrays(url: foldURL) { t[k] = v.asType(.float32) }
        return VAEWeights(t: t)
    }
}
