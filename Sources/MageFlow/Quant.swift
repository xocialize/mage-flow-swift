// Pre-quantized MageFlow DiT — the qwen-image-edit `QuantizedDiT` pattern: convert the
// bf16 transformer ONCE into a single self-describing safetensors (metadata carries
// bits/group_size), then consumers set up the QuantizedLinear structure with the same
// filter and load the pre-quantized file directly — no bf16 peak on the loading machine.
//
// Quant scope (keep_hi_precision doctrine, mlx-porting step 7): ONLY the per-block
// attn + MLP Linears inside `transformer_blocks`. Kept hi-precision: img_in / txt_in
// (I/O projections), proj_out, norm_out, img_mod / txt_mod (modulation), time embed,
// and all norms. The modulation MLPs feed x*(1+scale)+shift — precision-sensitive.
//
// NOTE the NAX row-chunk (`MageFeedForward.downProjected`) only guards *bf16* Linears;
// QuantizedLinear runs the QMM kernels, a different family. The chunk's dtype guard
// (`x.dtype != .float32`) is inert for quantized layers because quantize() replaces the
// Linear wholesale — downProjected is only reached on the bf16 path.

import Foundation
import MLX
import MLXNN

public struct MageQuantConfig: Sendable {
    public var ditBits: Int
    public var groupSize: Int
    /// Block indices kept bf16 wholesale. The LAST block feeds norm_out's modulated
    /// LayerNorm, which amplifies its quant error ~1.6x into proj_out (measured:
    /// int8 g64 block-10 img cos 0.999974 -> final 0.999835 when block 11 was
    /// quantized). Keeping it hi-precision is what moves the per-pass gate.
    public var keepHiBlocks: Set<Int>
    /// Blocks whose TXT-side layers (add_*_proj, to_add_out, txt_mlp) stay bf16.
    /// Cannot use keepHiBlocks for a MIDDLE block: a block contributing zero
    /// replacements leaves a hole in the transformer_blocks list container and
    /// MLXNN's update(modules:) fatals with mismatchedContainers. keepHiBlocks is
    /// safe only for a TRAILING block (11); partial protection keeps the block in
    /// the replacement map.
    public var keepTxtSideBlocks: Set<Int>
    public init(ditBits: Int, groupSize: Int, keepHiBlocks: Set<Int> = [11],
                keepTxtSideBlocks: Set<Int> = []) {
        self.ditBits = ditBits
        self.groupSize = groupSize
        self.keepHiBlocks = keepHiBlocks
        self.keepTxtSideBlocks = keepTxtSideBlocks
    }

    /// int4 g64 (small) and int8 g64 (near-lossless) — the two published tiers.
    /// int8 g128 measured 0.999796 per-pass (below the 0.9999 gate); g64 is the tier.
    public static let int4 = MageQuantConfig(ditBits: 4, groupSize: 64)
    public static let int8 = MageQuantConfig(ditBits: 8, groupSize: 32)

    /// Which Linears to quantize: attn + MLP inside transformer_blocks only,
    /// skipping keepHiBlocks entirely.
    public func spec(_ path: String, _ module: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
        guard module is Linear, path.contains("transformer_blocks") else { return nil }
        guard path.contains(".attn.") || path.contains("_mlp.") else { return nil }
        if keepHiBlocks.contains(where: { path.contains("transformer_blocks.\($0).") }) { return nil }
        if keepTxtSideBlocks.contains(where: { path.contains("transformer_blocks.\($0).") }),
           path.contains("add_") || path.contains("txt_mlp") { return nil }
        return (groupSize, ditBits, .affine)
    }

    public var metadata: [String: String] {
        ["format": "mageflow-dit-quant", "dit_bits": "\(ditBits)", "group_size": "\(groupSize)",
         "keep_hi_blocks": keepHiBlocks.sorted().map(String.init).joined(separator: ","),
         "keep_txt_side_blocks": keepTxtSideBlocks.sorted().map(String.init).joined(separator: ",")]
    }

    public static func from(metadata m: [String: String]) throws -> MageQuantConfig {
        guard m["format"] == "mageflow-dit-quant",
              let bits = m["dit_bits"].flatMap(Int.init),
              let gs = m["group_size"].flatMap(Int.init)
        else { throw MageQuantError.badMetadata }
        let keep = Set((m["keep_hi_blocks"] ?? "").split(separator: ",").compactMap { Int($0) })
        let keepTxt = Set((m["keep_txt_side_blocks"] ?? "").split(separator: ",").compactMap { Int($0) })
        return MageQuantConfig(ditBits: bits, groupSize: gs, keepHiBlocks: keep, keepTxtSideBlocks: keepTxt)
    }
}

public enum MageQuantError: Error, CustomStringConvertible {
    case badMetadata
    case missingKeys(Int, [String])
    public var description: String {
        switch self {
        case .badMetadata: return "quantized DiT: missing/invalid quant metadata"
        case .missingKeys(let n, let sample): return "quantized DiT: \(n) missing keys, e.g. \(sample)"
        }
    }
}

public enum MageQuant {
    /// One-time conversion: microsoft `transformer/` (bf16) -> one pre-quantized
    /// safetensors. Runs at the bf16 peak — do it once on a big-RAM box.
    public static func saveQuantizedDiT(
        fromTransformerDir dir: URL, to outURL: URL, config: MageQuantConfig
    ) throws {
        let model = MageFlowTransformer()
        var raw: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter({ $0.pathExtension == "safetensors" }) {
            raw.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        let keys = Set(model.parameters().flattened().map(\.0))
        let w = model.sanitize(weights: raw).mapValues { $0.asType(.bfloat16) }
        model.update(parameters: ModuleParameters.unflattened(w.filter { keys.contains($0.key) }))
        eval(model)
        quantize(model: model, filter: config.spec)
        eval(model)
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        // materialize before save — lazy tensors serialize as zeros (mlx-porting rule)
        for v in params.values { eval(v) }
        try MLX.save(arrays: params, metadata: config.metadata, url: outURL)
    }

    /// Load a pre-quantized DiT with no bf16 peak. Self-describing via metadata.
    public static func loadQuantizedDiT(from url: URL) throws -> MageFlowTransformer {
        let (weights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let config = try MageQuantConfig.from(metadata: metadata)
        let model = MageFlowTransformer()
        quantize(model: model, filter: config.spec)   // structure only; placeholders stay lazy
        let keys = Set(model.parameters().flattened().map(\.0))
        let missing = keys.subtracting(Set(weights.keys)).sorted()
        guard missing.isEmpty else {
            throw MageQuantError.missingKeys(missing.count, Array(missing.prefix(4)))
        }
        model.update(parameters: ModuleParameters.unflattened(
            weights.filter { keys.contains($0.key) }))
        eval(model)
        return model
    }
}
