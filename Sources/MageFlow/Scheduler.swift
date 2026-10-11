// FlowMatchEulerDiscreteScheduler with a STATIC shift — Mage-Flow's whole
// sampler surface. `schedule_mode: "z-image"` / `use_time_shift: false` in
// transformer/config.json are stripped legacy names for exactly this.
//
// The sampler is a first-class port surface, not an afterthought: a placeholder
// integrator can pass latent-cosine parity and still emit resolution-dependent
// garbage. This one is simple and exactly checkable, so check it exactly.

import Foundation
import MLX

public struct FlowMatchEulerScheduler {
    public let sigmas: [Float]      // steps + 1, terminal 0 appended
    public let timesteps: [Float]   // steps
    public let shift: Float

    /// base sigmas = linspace(1.0, 1/steps, steps), then the static shift
    ///   sigma' = shift * sigma / (1 + (shift - 1) * sigma)
    /// and a terminal 0.
    ///
    /// `startSigma` < 1 starts the schedule part-way down (image-to-image / SDEdit): the base
    /// grid is scaled by the base value whose shifted sigma is `startSigma`, so the first
    /// sigma equals `startSigma` and all `steps` steps are spent below it. 1.0 is the
    /// unchanged schedule.
    public init(steps: Int, shift: Float = 6.0, numTrainTimesteps: Float = 1000,
                startSigma: Float = 1.0) {
        self.shift = shift
        let s0 = min(max(startSigma, 1e-4), 1.0)
        let b0 = s0 / (shift - (shift - 1) * s0)                // inverse of the shift map
        var s: [Float] = []
        for i in 0 ..< steps {
            let t = steps == 1 ? 0 : Float(i) / Float(steps - 1)
            let base = b0 * (1.0 + t * (1.0 / Float(steps) - 1.0))  // b0 · linspace(1, 1/steps, steps)
            s.append(shift * base / (1 + (shift - 1) * base))
        }
        self.timesteps = s.map { $0 * numTrainTimesteps }
        self.sigmas = s + [0]
    }

    /// Euler: x += (sigma_{i+1} - sigma_i) * v
    public func step(_ velocity: MLXArray, _ sample: MLXArray, _ i: Int) -> MLXArray {
        sample + (sigmas[i + 1] - sigmas[i]) * velocity
    }

    /// The DiT is fed the RAW SIGMA in [0,1], not t in [0,1000]. The x1000 lives
    /// inside Timesteps.
    public func sigma(_ i: Int) -> Float { sigmas[i] }
}
