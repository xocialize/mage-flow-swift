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

    /// Turbo / Base / RL denoise. `img` is the packed [target, refs...] latent;
    /// `targetLen` is the target token count. Returns the packed sequence after
    /// each step (refs unchanged), so a caller can gate step-by-step.
    public func denoise(
        img img0: MLXArray, txt: MLXArray, targetLen: Int,
        imgShapes: [(frame: Int, height: Int, width: Int)],
        scheduler: FlowMatchEulerScheduler,
        onStep: ((Int, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var img = img0
        let refs = img0.dim(1) > targetLen ? img0[0..., targetLen..., 0...] : nil
        for i in 0 ..< scheduler.timesteps.count {
            let v = velocity(img: img, txt: txt, sigma: scheduler.sigma(i), imgShapes: imgShapes)
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
