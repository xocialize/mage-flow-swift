// Runtime DiT-LoRA for the Mage-Flow NR-MMDiT (MageFlowTransformer) — AB-A-0050.
//
// Applies one or more LoRAs to a resident `MageFlowTransformer` as an ACTIVATION-PATH adapter
// (never fused into the base weights), so the low-rank term survives bf16 AND the int8/int4
// tiers: `LoRALinear.from` dispatches to `QLoRALinear` when the base is a `QuantizedLinear`.
// The first consumer is LTX Studio's RefControl POSE LoRA (ai-toolkit `mageflow_edit`, base
// Mage-Flow-Edit-Base, rank 32 / alpha 32) — a pose cell is a two-reference edit (Image 1 =
// skeleton, Image 2 = identity), which is why this is the co-requisite of AB-A-0047.
//
// Donor: flux2-klein-swift `KleinLoRA` (LoRA.swift), minus the fused-qkv split — Mage keys
// arrive already diffusers-split. Why not `MLXLMCommon.LoRAContainer`: it is typed against
// `LanguageModel`; the primitives (`LoRALinear.from`, replace-children, `update(parameters:)`)
// are generic, so the container's logic is mirrored here, Mage-typed.
//
// KEY DIALECT (ai-toolkit comfy prefix, diffusers module names — from the vendored
// `src/transformer.py` and the trainer's `only_if_contains: [transformer_blocks]` /
// `ignore_if_contains: [img_mod, txt_mod]`):
//   diffusion_model.transformer_blocks.<i>.attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,
//     add_v_proj,to_out.0,to_add_out}.lora_{A,B}.weight
//   diffusion_model.transformer_blocks.<i>.{img_mlp,txt_mlp}.net.0.proj.lora_{A,B}.weight
//   diffusion_model.transformer_blocks.<i>.{img_mlp,txt_mlp}.net.2.lora_{A,B}.weight
// → 12 targets × 12 blocks = 144 modules / 288 tensors. `img_mod.1` / `txt_mod.1` (the
// Linear inside upstream's Sequential(SiLU, Linear)) are ACCEPTED if present (plain Linears in
// the port, keyed `img_mod` / `txt_mod`), as are the `transformer.` / `base_model.model.`
// prefixes. Anything else is an ERROR listing the offending keys — an adapter whose keys do not
// land is a silent no-op otherwise, and a gate that cannot fail measures nothing (AB-L-0026).
//
// FACTOR CONVENTION: torch `lora_A.weight` is [rank, in], `lora_B.weight` is [out, rank].
// MLXLMCommon's layer wants `lora_a` [in, rank] and `lora_b` [rank, out] and computes
// `y + scale · (x·a·b)`; the layer scale is fixed at 1.0 and the effective scale
// `strength · alpha/rank` (alpha-less adapters: `strength`) is baked into `b` — exact, and
// it lets several adapters rank-stack into one module as the SUM of their terms.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MageFlow

public enum MageLoRA {

    public enum LoRAError: Error, LocalizedError {
        case incompletePair(String)
        case noTargets(String)
        case unknownKeys(String, [String])
        case rankMismatch(String)

        public var errorDescription: String? {
            switch self {
            case .incompletePair(let p):
                return "LoRA layer \(p) is missing its lora_A or lora_B tensor."
            case .noTargets(let url):
                return "No Mage-Flow LoRA tensors found in \(url)."
            case .unknownKeys(let url, let keys):
                let sample = keys.prefix(6).joined(separator: ", ")
                return "\(keys.count) LoRA key(s) in \(url) do not map onto the Mage-Flow DiT "
                    + "(e.g. \(sample)) — refusing to apply a partially-landing adapter."
            case .rankMismatch(let p):
                return "LoRA layer \(p): lora_A rank axis does not match lora_B."
            }
        }
    }

    private static let aSuffixes = [
        ".lora_A.weight", ".lora_A.default.weight", ".lora.down.weight", ".lora_down.weight",
    ]
    private static let bSuffixes = [
        ".lora_B.weight", ".lora_B.default.weight", ".lora.up.weight", ".lora_up.weight",
    ]
    private static let alphaSuffix = ".alpha"
    private static let prefixes = ["base_model.model.", "diffusion_model.", "transformer."]

    /// Source (diffusers / ai-toolkit) block-relative submodule → port block-relative Linear
    /// path. The port's `sanitize` does the same `net.0.proj → proj_in`, `net.2 → proj_out`,
    /// `img_mod.1 → img_mod` renames for base weights (`Transformer.swift`).
    static let blockMap: [String: String] = [
        "attn.to_q": "attn.to_q",
        "attn.to_k": "attn.to_k",
        "attn.to_v": "attn.to_v",
        "attn.add_q_proj": "attn.add_q_proj",
        "attn.add_k_proj": "attn.add_k_proj",
        "attn.add_v_proj": "attn.add_v_proj",
        "attn.to_out.0": "attn.to_out.0",
        "attn.to_out": "attn.to_out.0",
        "attn.to_add_out": "attn.to_add_out",
        "img_mlp.net.0.proj": "img_mlp.proj_in",
        "img_mlp.net.2": "img_mlp.proj_out",
        "txt_mlp.net.0.proj": "txt_mlp.proj_in",
        "txt_mlp.net.2": "txt_mlp.proj_out",
        "img_mlp.proj_in": "img_mlp.proj_in",
        "img_mlp.proj_out": "img_mlp.proj_out",
        "txt_mlp.proj_in": "txt_mlp.proj_in",
        "txt_mlp.proj_out": "txt_mlp.proj_out",
        // modulation Linears — not in the trainer's target set, accepted if present
        "img_mod.1": "img_mod",
        "img_mod": "img_mod",
        "txt_mod.1": "txt_mod",
        "txt_mod": "txt_mod",
    ]

    /// The 12 trainer targets per block (attention + MLP), i.e. what the RefControl run emits.
    public static let trainedTargetsPerBlock = 12

    /// Map a source key base (suffix already stripped) onto the port's full module path
    /// `transformer_blocks.<i>.<rel>`, or nil when the key is not a Mage-Flow DiT target.
    public static func expand(base: String) -> String? {
        var s = base
        for p in prefixes where s.hasPrefix(p) { s.removeFirst(p.count) }
        let comps = s.split(separator: ".").map(String.init)
        guard comps.count >= 3, comps[0] == "transformer_blocks", let idx = Int(comps[1]),
              idx >= 0, idx < MageFlowConfig.depth
        else { return nil }
        guard let rel = blockMap[comps[2...].joined(separator: ".")] else { return nil }
        return "transformer_blocks.\(idx).\(rel)"
    }

    /// Block-relative key of a full path (`transformer_blocks.<i>.attn.to_q` → `attn.to_q`).
    static func blockRelative(_ path: String) -> String? {
        let prefix = "transformer_blocks."
        guard path.hasPrefix(prefix) else { return nil }
        let after = path.dropFirst(prefix.count)
        guard let dot = after.firstIndex(of: ".") else { return nil }
        return String(after[after.index(after: dot)...])
    }

    /// One adapter's per-target factors keyed by full port path: `a` [in, rank], `b` [rank, out]
    /// with the effective scale baked in.
    struct Factors { var a: MLXArray; var b: MLXArray }

    static func factors(from url: URL, dtype: DType, strength: Float) throws -> [String: Factors] {
        let raw = try MLX.loadArrays(url: url)
        func match(_ key: String, _ suffixes: [String]) -> String? {
            for s in suffixes where key.hasSuffix(s) { return String(key.dropLast(s.count)) }
            return nil
        }
        var aMats: [String: MLXArray] = [:]
        var bMats: [String: MLXArray] = [:]
        var alphas: [String: MLXArray] = [:]
        var unknown: [String] = []
        for (key, value) in raw {
            if let base = match(key, aSuffixes) { aMats[base] = value }
            else if let base = match(key, bSuffixes) { bMats[base] = value }
            else if key.hasSuffix(alphaSuffix) { alphas[String(key.dropLast(alphaSuffix.count))] = value }
            else { unknown.append(key) }
        }
        guard !aMats.isEmpty else { throw LoRAError.noTargets(url.path) }

        var out: [String: Factors] = [:]
        for (base, aMat) in aMats {
            guard let path = expand(base: base) else { unknown.append(base); continue }
            guard let bMat = bMats[base] else { throw LoRAError.incompletePair(base) }
            let rank = aMat.dim(0)                       // lora_A is [rank, in]
            guard bMat.dim(1) == rank else { throw LoRAError.rankMismatch(base) }
            let scale = strength * (alphas[base].map { $0.item(Float.self) / Float(rank) } ?? 1.0)
            out[path] = Factors(a: aMat.T.asType(dtype), b: (scale * bMat.T).asType(dtype))
        }
        for base in bMats.keys where aMats[base] == nil { throw LoRAError.incompletePair(base) }
        guard unknown.isEmpty else { throw LoRAError.unknownKeys(url.lastPathComponent, unknown.sorted()) }
        return out
    }

    /// Rank-stack several adapters into one set of per-module `lora_a`/`lora_b` parameters.
    static func combined(
        _ loras: [(url: URL, strength: Float)], dtype: DType
    ) throws -> (params: [String: MLXArray], ranks: [String: Int], targetKeys: Set<String>) {
        let perLoRA = try loras.map { try factors(from: $0.url, dtype: dtype, strength: $0.strength) }
        var paths = Set<String>()
        perLoRA.forEach { paths.formUnion($0.keys) }
        var params: [String: MLXArray] = [:]
        var ranks: [String: Int] = [:]
        var targetKeys = Set<String>()
        for path in paths {
            guard let rel = blockRelative(path) else { continue }
            let present = perLoRA.compactMap { $0[path] }
            let aCat = present.count == 1 ? present[0].a : concatenated(present.map(\.a), axis: 1)
            let bCat = present.count == 1 ? present[0].b : concatenated(present.map(\.b), axis: 0)
            params[path + ".lora_a"] = aCat
            params[path + ".lora_b"] = bCat
            ranks[path] = aCat.dim(1)
            targetKeys.insert(rel)
        }
        return (params, ranks, targetKeys)
    }

    /// What an `apply` did: adapted Linears by base kind. On the int8/int4 tiers the
    /// quantized blocks go `QLoRALinear` and the bf16-kept block(s) (`keepHiBlocks`, block 11)
    /// go `LoRALinear`, so both branches are visible here.
    public struct AppliedSummary: Sendable {
        public var bf16Targets: Int
        public var quantizedTargets: Int
        public var total: Int { bf16Targets + quantizedTargets }
    }

    /// Apply one or more LoRAs to the resident DiT in place. Every key must land
    /// (`.noUnusedKeys` + the unknown-key check in `factors`).
    @discardableResult
    public static func apply(
        loRAs loras: [(url: URL, strength: Float)],
        to model: MageFlowTransformer,
        dtype: DType = .bfloat16
    ) throws -> AppliedSummary {
        let (params, ranks, targetKeys) = try combined(loras, dtype: dtype)
        var bf16 = 0, quant = 0
        for (i, block) in model.transformerBlocks.enumerated() {
            var update: [(String, Module)] = []
            for (key, child) in block.namedModules() where targetKeys.contains(key) {
                let path = "transformer_blocks.\(i).\(key)"
                guard let linear = child as? Linear, let rank = ranks[path] else { continue }
                if linear is QuantizedLinear { quant += 1 } else { bf16 += 1 }
                update.append((key, LoRALinear.from(linear: linear, rank: rank, scale: 1.0)))
            }
            if !update.isEmpty { block.update(modules: .unflattened(update)) }
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
        return AppliedSummary(bf16Targets: bf16, quantizedTargets: quant)
    }

    @discardableResult
    public static func apply(
        loRA url: URL, to model: MageFlowTransformer, dtype: DType = .bfloat16, strength: Float = 1.0
    ) throws -> AppliedSummary {
        try apply(loRAs: [(url, strength)], to: model, dtype: dtype)
    }
}

extension MageFlowEditPipeline {
    /// Apply LoRA adapter(s) to this pipeline's resident DiT (edit AND t2i paths share it).
    /// Call once after construction; the adapter stays for the pipeline's lifetime.
    @discardableResult
    public func applyLoRA(loRAs: [(url: URL, strength: Float)]) throws -> MageLoRA.AppliedSummary {
        try MageLoRA.apply(loRAs: loRAs, to: transformer, dtype: ditDtype)
    }
}
