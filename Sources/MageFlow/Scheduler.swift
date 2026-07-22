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
    public init(steps: Int, shift: Float = 6.0, numTrainTimesteps: Float = 1000) {
        self.shift = shift
        var s: [Float] = []
        for i in 0 ..< steps {
            let t = steps == 1 ? 0 : Float(i) / Float(steps - 1)
            let base = 1.0 + t * (1.0 / Float(steps) - 1.0)     // linspace(1, 1/steps, steps)
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
