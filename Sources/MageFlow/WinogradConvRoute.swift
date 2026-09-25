// Route for 3×3 convs inside mlx's Winograd conv2d window (mlx-swift ≤ 0.31.6 Metal numerics).
//
// mlx's Metal conv2d (mlx/backend/metal/conv.cpp `dispatch_conv_2D_gpu`) runs a Winograd
// F(6×6,3×3) kernel when ALL of these hold: kernel 3×3, stride 1, dilation 1, groups 1,
// C % 32 == 0, O % 32 == 0, C + O ≥ 256, N·H·W ≥ 4096. On M5 that path is lossy — relL2 per
// conv against an exact reference: fp32 6.4e-3 (its batched GEMM runs TF32, MLX_ENABLE_TF32
// defaults on, and the output transform amplifies that ~8×), bf16 5.8e-2. Every other conv path
// is exact-class. conv3d with kT = 1 is the same conv on the implicit-GEMM path: exact.
//
// MageVAE hits the window only in the conditioning decoder (`codDecoder`: conv_in 128→384, six
// 384→384 resnet convs, conv_out 384→384 — all at latent resolution), and only from 1024² output
// up (latent 64² = 4096 px); the encoder is depthwise / 1×1 throughout. Default `.conv3d` (eight
// convs at 64²: cheap). `MAGEFLOW_VAE_CONV_ROUTE=winograd|conv3d` (read per call) overrides.
// Removal: when a new mlx-swift pin reports raw conv2d exact (MageVAEGPULaneTests probe), drop it.

import Foundation
import MLX

public enum MageVAEConvRoute: String, Sendable {
    /// mlx's default Winograd kernel — fastest; lossy on M5 (~6.4e-3 relL2 per conv in fp32).
    case winograd
    /// conv3d with kT = 1 on the implicit-GEMM path — exact.
    case conv3d

    /// The active route: `MAGEFLOW_VAE_CONV_ROUTE` if set, else `.conv3d`.
    public static var current: MageVAEConvRoute {
        getenv("MAGEFLOW_VAE_CONV_ROUTE").flatMap { MageVAEConvRoute(rawValue: String(cString: $0)) }
            ?? .conv3d
    }

    /// mlx's Winograd dispatch predicate, evaluated on the actual input (NHWC).
    static func takesWinograd(_ x: MLXArray, _ w: MLXArray, stride: Int, groups: Int) -> Bool {
        guard x.ndim == 4, groups == 1, stride == 1, w.dim(1) == 3, w.dim(2) == 3 else { return false }
        let (c, o) = (x.dim(3), w.dim(0))
        return c % 32 == 0 && o % 32 == 0 && c + o >= 256 && x.dim(0) * x.dim(1) * x.dim(2) >= 4096
    }
}

/// `conv2d` that takes the conv3d route for in-window shapes when the route says so.
func routedConv2d(_ x: MLXArray, _ w: MLXArray, stride: Int, padding: Int, groups: Int,
                  route: MageVAEConvRoute = .current) -> MLXArray {
    guard route == .conv3d, MageVAEConvRoute.takesWinograd(x, w, stride: stride, groups: groups)
    else {
        return conv2d(x, w, stride: IntOrPair(stride), padding: IntOrPair(padding), groups: groups)
    }
    return conv3d(x.expandedDimensions(axis: 1), w.expandedDimensions(axis: 1),
                  stride: [1, 1, 1], padding: [0, padding, padding]).squeezed(axis: 1)
}
