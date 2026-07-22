// swift-tools-version: 6.2
// mage-flow-swift — MLX-Swift port of Microsoft's Mage-Flow NR-MMDiT (4B).
//
// The transformer is adapted from qwen-image-edit-swift's parity-tested
// QwenImageEdit/Transformer.swift: Mage-Flow's MageFlowEmbedRope is the same
// scaled 3-axis RoPE (theta 10000, axesDim [16,56,56], scaleRope, 4096 pos/neg
// tables) and its block is the same dual-stream MMDiT. Deltas are documented
// inline in Transformer.swift.
import PackageDescription

let package = Package(
    name: "MageFlow",
    platforms: [.macOS(.v26)],
    products: [.library(name: "MageFlow", targets: ["MageFlow"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
    ],
    targets: [
        .target(
            name: "MageFlow",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/MageFlow"),
        .executableTarget(
            name: "MageFlowGate", dependencies: ["MageFlow"], path: "Sources/MageFlowGate"),
    ]
)
