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
    products: [
        .library(name: "MageFlow", targets: ["MageFlow"]),
        .library(name: "MageFlowEdit", targets: ["MageFlowEdit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        .package(url: "https://github.com/xocialize/qwen3vl-mlx-swift.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
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
        .executableTarget(
            name: "MageVAEGate", dependencies: ["MageFlow"], path: "Sources/MageVAEGate"),
        .executableTarget(
            name: "GSGate", dependencies: ["MageFlow"], path: "Sources/GSGate"),
        .executableTarget(
            name: "E2EGate", dependencies: ["MageFlow"], path: "Sources/E2EGate"),
        .executableTarget(
            name: "MageFlowGen", dependencies: ["MageFlow"], path: "Sources/MageFlowGen"),
        .target(
            name: "MageFlowEdit",
            dependencies: [
                "MageFlow",
                .product(name: "Qwen3VL", package: "qwen3vl-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/MageFlowEdit"),
        .executableTarget(
            name: "mage-flow-edit", dependencies: ["MageFlowEdit"], path: "Sources/MageFlowEditCLI"),
    ]
)
