// Mage-Flow denoise loop.
//
// Edit conditioning is SEQUENCE concat, not channel concat: per sample the
// packed image sequence is [target, ref_1 ... ref_N]. The refs are CLEAN latents
// held FIXED at every step — only the target slice is stepped. Target and ref
// are distinguished solely by the RoPE frame index (0 vs 1..N).

import Foundation
import MLX

public struct MageFlowPipeline {
    public let transformer: MageFlowTransformer

    public init(transformer: MageFlowTransformer) { self.transformer = transformer }

    /// One velocity evaluation over the packed sequence.
    public func velocity(img: MLXArray, txt: MLXArray, sigma: Float,
                         imgShapes: [(frame: Int, height: Int, width: Int)]) -> MLXArray {
        transformer(img: img, txt: txt, sigma: MLXArray([sigma]), imgShapes: imgShapes)
    }

    /// CFG-combined velocity: `unc + cfg * (cond - unc)`, optional per-token
    /// renormalization to the conditional velocity's norm (reduces
    /// oversaturation at high cfg).
    ///
    /// Implemented as TWO forwards (upstream `batch_cfg=False`). Upstream's
    /// default `batch_cfg=True` fuses them into one varlen pack, but the two are
    /// mathematically identical: rotary attention depends only on RELATIVE
    /// positions, so the fused pack's shifted uncond frame indices (2,3 vs 0,1)
    /// give the same geometry. Two forwards is exact for a dense single-sample
    /// port and needs no varlen multi-sample attention.
    public func velocityCFG(
        img: MLXArray, txt: MLXArray, negTxt: MLXArray, sigma: Float,
        imgShapes: [(frame: Int, height: Int, width: Int)],
        cfg: Float, renormalization: Bool
    ) -> MLXArray {
        let cond = velocity(img: img, txt: txt, sigma: sigma, imgShapes: imgShapes)
        let unc = velocity(img: img, txt: negTxt, sigma: sigma, imgShapes: imgShapes)
        let comb = unc + cfg * (cond - unc)
        guard renormalization else { return comb }
        let condNorm = sqrt(sum(cond.asType(.float32).square(), axis: -1, keepDims: true))
        let combNorm = sqrt(sum(comb.asType(.float32).square(), axis: -1, keepDims: true))
        return (comb.asType(.float32) * (condNorm / (combNorm + 1e-6))).asType(comb.dtype)
    }

    /// Turbo / Base / RL denoise. `img` is the packed [target, refs...] latent;
    /// `targetLen` is the target token count. Returns the packed sequence after
    /// each step (refs unchanged), so a caller can gate step-by-step.
    ///
    /// `negTxt` + `cfg > 1` enables classifier-free guidance (Base/RL
    /// checkpoints; upstream default negative prompt is a single space " ").
    public func denoise(
        img img0: MLXArray, txt: MLXArray, targetLen: Int,
        imgShapes: [(frame: Int, height: Int, width: Int)],
        scheduler: FlowMatchEulerScheduler,
        negTxt: MLXArray? = nil, cfg: Float = 1.0, renormalization: Bool = false,
        onStep: ((Int, MLXArray) -> Void)? = nil,
        shouldStop: (() -> Bool)? = nil
    ) -> MLXArray {
        var img = img0
        let refs = img0.dim(1) > targetLen ? img0[0..., targetLen..., 0...] : nil
        let useCFG = cfg > 1.0 && negTxt != nil
        for i in 0 ..< scheduler.timesteps.count {
            // cooperative-cancellation seam (CAN gate): break, caller classifies
            if shouldStop?() == true { break }
            let v: MLXArray
            if useCFG {
                v = velocityCFG(img: img, txt: txt, negTxt: negTxt!, sigma: scheduler.sigma(i),
                                imgShapes: imgShapes, cfg: cfg, renormalization: renormalization)
            } else {
                v = velocity(img: img, txt: txt, sigma: scheduler.sigma(i), imgShapes: imgShapes)
            }
            // step ONLY the target slice
            let predT = v[0..., ..<targetLen, 0...]
            let target = scheduler.step(predT, img[0..., ..<targetLen, 0...], i)
            // refs are rebuilt clean each iteration, never stepped
            img = refs.map { concatenated([target, $0], axis: 1) } ?? target
            eval(img)
            onStep?(i, img)
        }
        return img
    }
}
