// Init-time configuration for the Mage-Flow MLXEngine packages (C9).
//
// A Mage checkpoint is a multi-component snapshot: transformer/ (NR-MMDiT, bf16)
// + text_encoder/ (Qwen3-VL-4B conditioner AND mandatory content filter) + vae/
// (MageVAE) from the upstream microsoft repo, plus the port's artifacts from the
// matching xocialize/<name>-mlx repo: folded_adaln.safetensors (baked MageVAE
// adaLN constants) and, for quant tiers, a pre-quantized DiT
// (transformer-int8/-int4.safetensors — loads with NO bf16 peak).

import Foundation
import MLXToolKit

/// The six Mage-Flow checkpoints. T2I and edit are DIFFERENT DiT weights
/// (same architecture); text_encoder/ and vae/ are byte-identical family-wide.
public enum MageFlowVariant: String, Codable, Sendable, CaseIterable {
    case flow = "Mage-Flow"                // t2i, RL-aligned (family primary)
    case base = "Mage-Flow-Base"           // t2i, quality
    case turbo = "Mage-Flow-Turbo"         // t2i, 4-step distilled
    case edit = "Mage-Flow-Edit"           // edit, RL-aligned
    case editBase = "Mage-Flow-Edit-Base"  // edit, quality
    case editTurbo = "Mage-Flow-Edit-Turbo" // edit, 4-step distilled

    public var isEdit: Bool {
        switch self {
        case .edit, .editBase, .editTurbo: return true
        default: return false
        }
    }

    /// Upstream component snapshot (MIT).
    public var componentsRepo: String { "microsoft/\(rawValue)" }
    /// The port's artifact repo (folded adaLN + pre-quantized DiTs).
    public var artifactsRepo: String { "xocialize/\(rawValue)-mlx" }

    /// Upstream generation defaults (family cards): Turbo 4/1.0 · RL 20/5.0 · Base 30/5.0.
    public var defaultSteps: Int {
        switch self {
        case .turbo, .editTurbo: return 4
        case .flow, .edit: return 20
        case .base, .editBase: return 30
        }
    }
    public var defaultGuidance: Float {
        switch self {
        case .turbo, .editTurbo: return 1.0
        default: return 5.0
        }
    }
}

public struct MageFlowConfiguration: PackageConfiguration, ModelStorable, QuantConfigured,
    FootprintConfigured {
    public var variant: MageFlowVariant
    /// DiT quant tier. bf16 loads transformer/ from the microsoft snapshot; int8/int4
    /// load the pre-quantized DiT from the artifacts repo (no bf16 peak, smaller download).
    /// Gate results (Edit-Turbo, per-pass vs fp32 golden): bf16 deficit 0.99e-4; int8 g32
    /// deficit 1.30e-4 (≤2× baseline — transparent); int4 g64 cos 0.9911 (e2e-validated).
    public var quant: Quant
    /// Explicit local snapshot root (components + folded_adaln [+ quant DiT]); nil ⇒
    /// resolve against the ModelStore after auto-materializing `weightSources`.
    public var snapshotPath: String?
    public var defaultSteps: Int
    public var guidanceScale: Float
    /// Default square output side (Mage is native-resolution but the port's validated
    /// envelope is square 512–2048; /16-floored).
    public var defaultSize: Int
    /// Light tier: drop the ~8.3 GB Qwen3-VL conditioner after each request's encode
    /// (it reloads on the next request) — resident set becomes DiT + VAE, so the
    /// low-RAM peak ≈ max(encode, denoise) instead of conditioner + denoise.
    public var evictConditioner: Bool
    public var modelsRootDirectory: URL?

    public init(
        variant: MageFlowVariant = .flow,
        quant: Quant = .bf16,
        snapshotPath: String? = nil,
        defaultSteps: Int? = nil,
        guidanceScale: Float? = nil,
        defaultSize: Int = 1024,
        evictConditioner: Bool = false,
        modelsRootDirectory: URL? = nil
    ) {
        self.variant = variant
        self.quant = quant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps ?? variant.defaultSteps
        self.guidanceScale = guidanceScale ?? variant.defaultGuidance
        self.defaultSize = defaultSize
        self.evictConditioner = evictConditioner
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant, quant, defaultSteps, guidanceScale, defaultSize, evictConditioner
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        variant = try c.decode(MageFlowVariant.self, forKey: .variant)
        quant = try c.decode(Quant.self, forKey: .quant)
        let v = variant
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? v.defaultSteps
        guidanceScale = try c.decodeIfPresent(Float.self, forKey: .guidanceScale) ?? v.defaultGuidance
        defaultSize = try c.decodeIfPresent(Int.self, forKey: .defaultSize) ?? 1024
        evictConditioner = try c.decodeIfPresent(Bool.self, forKey: .evictConditioner) ?? false
    }

    /// The pre-quantized DiT filename in the artifacts repo (nil for bf16).
    public var quantDiTFile: String? {
        switch quant {
        case .int8: return "transformer-int8.safetensors"
        case .int4: return "transformer-int4.safetensors"
        default: return nil
        }
    }

    // MARK: FootprintConfigured
    //
    // Static QuantFootprints declare the resident-conditioner path at the 1024²
    // envelope. The hints surface the two dynamic levers:
    //  - evictConditioner drops the ~8.3 GB Qwen3-VL from the resident set;
    //  - activation scales ~with latent token count ((size/16)² tokens; measured
    //    2.2 GB @512² → 3.7 GB @1024² → 8.8 GB @2048², edit path ≈ 2× tokens).
    public var residentBytesHint: UInt64? {
        guard evictConditioner else { return nil }
        // DiT (per tier: bf16 7.7 / int8-g32 5.6 / int4 4.3 GB) + VAE floor;
        // the conditioner is transient.
        switch quant {
        case .int8: return 6_000_000_000
        case .int4: return 4_700_000_000
        default: return 8_100_000_000
        }
    }
    public var peakActivationBytesHint: UInt64? {
        let tokens = Double(defaultSize / 16) * Double(defaultSize / 16)
        let scale = tokens / 4096.0   // measured baseline is the 1024² grid (64×64)
        let base = 3_700_000_000.0 * max(scale, 0.25)
        // evict mode: the transient conditioner encode (~8.3 GB weights + small
        // activation) becomes part of the activation envelope.
        return UInt64(evictConditioner ? Swift.max(base, 9_000_000_000) : base)
    }
}

extension MageFlowConfiguration: WeightSourcing {
    /// Two sources: upstream components (quant tiers EXCLUDE the 7.7 GB bf16
    /// transformer/) + the port's artifacts (folded adaLN, pre-quantized DiT).
    public var weightSources: [WeightSource] {
        let componentGlobs = quant == .bf16
            ? ["transformer/*", "text_encoder/*", "vae/*", "*.json"]
            : ["text_encoder/*", "vae/*", "*.json"]
        var artifactGlobs = ["folded_adaln.safetensors"]
        if let quantDiTFile { artifactGlobs.append(quantDiTFile) }
        return [
            WeightSource(role: "components", repo: variant.componentsRepo, revision: nil,
                         matching: componentGlobs),
            WeightSource(role: "mlx-artifacts", repo: variant.artifactsRepo, revision: nil,
                         matching: artifactGlobs),
        ]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func componentsSatisfied(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appendingPathComponent("text_encoder").path)
                && fm.fileExists(atPath: dir.appendingPathComponent("vae").path)
                && (quant != .bf16
                    || fm.fileExists(atPath: dir.appendingPathComponent("transformer").path))
        }
        func artifactsSatisfied(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appendingPathComponent("folded_adaln.safetensors").path)
                && (quantDiTFile.map { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
                    ?? true)
        }
        // Explicit local snapshot: everything (components + artifacts) in one root.
        if let snapshotPath {
            let root = URL(fileURLWithPath: snapshotPath)
            if componentsSatisfied(root), artifactsSatisfied(root) { return [] }
        }
        let store = ModelStore(root: storeRoot)
        var missing: [WeightSource] = []
        if !(store.directory(for: variant.componentsRepo).map(componentsSatisfied) ?? false) {
            missing.append(weightSources[0])
        }
        if !(store.directory(for: variant.artifactsRepo).map(artifactsSatisfied) ?? false) {
            missing.append(weightSources[1])
        }
        return missing
    }

    /// The components root (transformer/ text_encoder/ vae/). With an explicit
    /// `snapshotPath` the artifacts live in the same root; store-resolved snapshots
    /// keep them in `resolvedArtifactsDirectory`.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        return ModelStore(root: storeRoot).directory(for: variant.componentsRepo)
    }

    public func resolvedArtifactsDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        return ModelStore(root: storeRoot).directory(for: variant.artifactsRepo)
    }
}
