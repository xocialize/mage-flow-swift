// MLXEngine packages over the Mage-Flow family (MIT weights + MIT port):
// Microsoft's native-resolution NR-MMDiT (4.1B, 12 dual-stream blocks) + MageVAE
// one-step codec + Qwen3-VL-4B conditioner/content-filter.
//
// Two classes because t2i and edit are DIFFERENT DiT checkpoints (same
// architecture): `MageFlowT2IPackage` (Mage-Flow / -Base / -Turbo) and
// `MageFlowEditPackage` (Mage-Flow-Edit / -Base / -Turbo); the tier is
// configuration. Both keep the upstream trust features intact: the mandatory
// Responsible-AI screen (fail-closed → thrown refusal, never a silent bypass)
// and the bit-exact Gaussian-Shading provenance watermark in the initial noise.

import Foundation
import MLX
import MLXProfiling
import MLXToolKit
import MageFlow
import MageFlowEdit

public enum MageFlowPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    /// The mandatory upstream content filter refused the request (fail-closed).
    case contentRefused(String)
    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Mage-Flow snapshot not readable at \(p)."
        case .contentRefused(let v): return "Mage-Flow content filter refused the request: \(v)"
        }
    }
}

// MARK: - shared load/run engine (both packages delegate here)

/// Non-protocol core shared by the two package classes: resolve → load → dispatch.
@InferenceActor
final class MageFlowRuntime {
    let configuration: MageFlowConfiguration
    private var pipeline: MageFlowEditPipeline?

    nonisolated init(configuration: MageFlowConfiguration) { self.configuration = configuration }

    func load() async throws {
        guard pipeline == nil else { return }
        let cfg = configuration

        // First-run materialization is engine-executed since mlx-engine-swift 0.32.0
        // (contract 1.24.0): resident()/prepare() downloads missing WeightSourcing
        // sources before load() runs. The guard below stays as the offline backstop —
        // absent weights with no store root still fail legibly here.
        guard let snapshot = cfg.resolvedSnapshotDirectory(storeRoot: cfg.modelsRootDirectory),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("text_encoder").path)
        else { throw MageFlowPackageError.unreadableSnapshot(cfg.snapshotPath ?? cfg.variant.componentsRepo) }
        let artifacts = cfg.resolvedArtifactsDirectory(storeRoot: cfg.modelsRootDirectory) ?? snapshot

        var pcfg = MageFlowEditConfig()
        pcfg.steps = cfg.defaultSteps
        pcfg.cfg = cfg.guidanceScale
        pcfg.size = cfg.defaultSize
        pipeline = try await MageFlowEditPipeline(
            textEncoderDir: snapshot.appendingPathComponent("text_encoder"),
            transformerDir: snapshot.appendingPathComponent("transformer"),
            vaeSafetensors: snapshot.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"),
            foldedAdaLN: artifacts.appendingPathComponent("folded_adaln.safetensors"),
            ditQuant: cfg.quantDiTFile.map { artifacts.appendingPathComponent($0) },
            cfg: pcfg,
            deferConditioner: cfg.evictConditioner)
    }

    func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()
    }

    /// Apply per-request canonical overrides, run one generation, encode PNG.
    /// CancellationError from the pipeline seams propagates UNCHANGED; the
    /// upstream filter's refusal is re-thrown as a legible package error.
    func generate(
        prompt: String, negativePrompt: String?, width: Int?, steps: Int?,
        guidance: Double?, seed: UInt64?, refImage: Image?
    ) async throws -> Image {
        guard let pipeline else { throw PackageError.notLoaded }
        let cfg = configuration
        let base = pipeline.cfg
        defer { pipeline.cfg = base }
        pipeline.cfg.size = ((width ?? cfg.defaultSize) / 16) * 16
        pipeline.cfg.steps = steps ?? cfg.defaultSteps
        pipeline.cfg.cfg = Float(guidance ?? Double(cfg.guidanceScale))
        if let seed { pipeline.cfg.seed = seed }
        if let negativePrompt, !negativePrompt.isEmpty { pipeline.cfg.negPrompt = negativePrompt }

        let prof = MLXProfiler.shared
        prof.beginRun("mage-flow \(refImage == nil ? "t2i" : "edit") \(cfg.variant.rawValue) "
            + "q=\(cfg.quant) steps=\(pipeline.cfg.steps) g=\(pipeline.cfg.cfg) "
            + "\(pipeline.cfg.size)² evict=\(cfg.evictConditioner)")
        defer {
            prof.endRun(denominators: ["step": Double(pipeline.cfg.steps)])
            if cfg.evictConditioner { pipeline.dropConditioner() }
        }
        do {
            let nhwc: MLXArray
            if let refImage {
                let (rgb, w, h) = try MageFlowEditPipeline.decodeRGB(data: refImage.data)
                nhwc = try pipeline.edit(refRGB: rgb, width: w, height: h,
                                         instruction: prompt, screen: true,
                                         shouldStop: { Task.isCancelled })
            } else {
                nhwc = try pipeline.t2i(prompt: prompt, screen: true,
                                        shouldStop: { Task.isCancelled })
            }
            try Task.checkCancellation()
            let (png, w, h) = try MageFlowEditPipeline.encodePNG(nhwc)
            return Image(format: .png, data: png, width: w, height: h)
        } catch let e as MageFlowEditError {
            if case .refused(let verdict) = e { throw MageFlowPackageError.contentRefused(verdict) }
            throw e
        }
    }
}

// MARK: - shared manifest pieces

enum MageFlowManifest {
    // Split footprint (efficiency contract 1.14.0), resident-conditioner path at the
    // 1024² envelope. Resident = DiT tier + Qwen3-VL-4B (8.3 GB) + MageVAE (0.33 GB);
    // DiT tiers from the shipped artifacts: bf16 7.7 / int8 5.6 (g32) / int4 4.3 GB
    // (int8/int4 keep mods + block 11 bf16 — the gated recipe). Activation ~3.7 GB
    // at the 1024² envelope (edit path ≈ 2× t2i tokens; declared for edit).
    // Measured GPU peaks, t2i @1024²: bf16 19.59 / int8 17.15 / int4 16.08 GB ✓
    // (and @512² edit: 18.50 / 16.37 / 14.92). The conditioner-evict light tier is
    // surfaced dynamically via MageFlowConfiguration.FootprintConfigured hints.
    static let footprints = [
        QuantFootprint(quant: .bf16, residentBytes: 16_400_000_000, peakActivationBytes: 3_700_000_000),
        QuantFootprint(quant: .int8, residentBytes: 14_200_000_000, peakActivationBytes: 3_700_000_000),
        QuantFootprint(quant: .int4, residentBytes: 12_900_000_000, peakActivationBytes: 3_700_000_000),
    ]

    static func requirements() -> RequirementsManifest {
        RequirementsManifest(
            footprints: footprints,
            requiredBackends: [.metalGPU],
            os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
            chipFloor: nil
        )
    }
}

// MARK: - textToImage package (Mage-Flow / -Base / -Turbo)

@InferenceActor
public final class MageFlowT2IPackage: ModelPackage {
    public typealias Configuration = MageFlowConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Both layers MIT: microsoft weights + the xocialize port.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "microsoft/Mage-Flow", revision: "main", tier: 1),
            requirements: MageFlowManifest.requirements(),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "mage-flow-t2i",
                    summary: "Mage-Flow 4.1B text-to-image (MIT): native-resolution NR-MMDiT "
                        + "rectified flow, Qwen3-VL-4B conditioning, square 512–2048 validated. "
                        + "Tiers via configuration: RL-aligned (20 steps, CFG 5), Base (30/5), "
                        + "Turbo (4 steps, ~2.4 s @512²). Ships Microsoft's mandatory content "
                        + "filter (fail-closed) and bit-exact Gaussian-Shading provenance "
                        + "watermark.",
                    modes: []
                )
            ]
        )
    }

    let runtime: MageFlowRuntime
    public nonisolated var configuration: Configuration { runtime.configuration }

    public nonisolated init(configuration: Configuration) {
        // t2i package must carry a t2i checkpoint; a mismatched variant would load
        // edit weights under a t2i surface. Normalize rather than trap (C9).
        let v = configuration.variant
        var cfg = configuration
        if v.isEdit {
            cfg.variant = .flow
        }
        runtime = MageFlowRuntime(configuration: cfg)
    }

    public func load() async throws { try await runtime.load() }
    public func unload() async { await runtime.unload() }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint FIRST — before notLoaded validation (engine ≥ 0.27.0).
        try Task.checkCancellation()
        guard let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        // Mage's validated envelope is square; a mismatched aspect request is
        // rejected legibly (1.16.0) instead of silently squaring the output.
        if let w = t2i.width, let h = t2i.height, w != h {
            throw PackageError.unsupportedRequestFeature(
                "non-square output (\(w)×\(h)) — Mage-Flow's validated envelope is square 512–2048")
        }
        let image = try await runtime.generate(
            prompt: t2i.prompt, negativePrompt: t2i.negativePrompt, width: t2i.width ?? t2i.height,
            steps: t2i.steps, guidance: t2i.guidanceScale, seed: t2i.seed, refImage: nil)
        return T2IResponse(image: image)
    }
}

extension MageFlowT2IPackage {
    public nonisolated static var registration: PackageRegistration { .of(MageFlowT2IPackage.self) }
}

// MARK: - imageEdit package (Mage-Flow-Edit / -Base / -Turbo)

@InferenceActor
public final class MageFlowEditPackage: ModelPackage {
    public typealias Configuration = MageFlowConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "microsoft/Mage-Flow-Edit", revision: "main", tier: 1),
            requirements: MageFlowManifest.requirements(),
            specialties: [],
            surfaces: [
                IEditContract.descriptor(
                    name: "mage-flow-edit",
                    summary: "Mage-Flow-Edit 4.1B instruction-based image editing (MIT): "
                        + "native-resolution NR-MMDiT with sequence-concat reference "
                        + "conditioning (ref latent held clean, target denoised), Qwen3-VL-4B "
                        + "vision conditioning. Single reference image; square 512–2048. Tiers "
                        + "via configuration: RL (20/5), Base (30/5), Turbo (4 steps, ~2.4 s "
                        + "@512²). Mandatory fail-closed content filter + Gaussian-Shading "
                        + "watermark.",
                    modes: []
                )
            ]
        )
    }

    let runtime: MageFlowRuntime
    public nonisolated var configuration: Configuration { runtime.configuration }

    public nonisolated init(configuration: Configuration) {
        let v = configuration.variant
        var cfg = configuration
        if !v.isEdit {
            cfg.variant = .edit
        }
        runtime = MageFlowRuntime(configuration: cfg)
    }

    public func load() async throws { try await runtime.load() }
    public func unload() async { await runtime.unload() }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint FIRST — before notLoaded validation (engine ≥ 0.27.0).
        try Task.checkCancellation()
        guard let edit = request as? IEditRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        guard let ref = edit.images.first else {
            throw PackageError.unsupportedRequestFeature("imageEdit requires one reference image")
        }
        // The port's validated path is single-reference (upstream supports N refs;
        // multi-ref is a follow-on) — reject legibly rather than silently dropping refs.
        guard edit.images.count == 1 else {
            throw PackageError.unsupportedRequestFeature(
                "multiple reference images (\(edit.images.count)) — single-ref only in this port")
        }
        if let w = edit.width, let h = edit.height, w != h {
            throw PackageError.unsupportedRequestFeature(
                "non-square output (\(w)×\(h)) — Mage-Flow's validated envelope is square 512–2048")
        }
        let image = try await runtime.generate(
            prompt: edit.prompt, negativePrompt: edit.negativePrompt,
            width: edit.width ?? edit.height, steps: edit.steps,
            guidance: edit.guidanceScale, seed: edit.seed, refImage: ref)
        return IEditResponse(image: image)
    }
}

extension MageFlowEditPackage {
    public nonisolated static var registration: PackageRegistration { .of(MageFlowEditPackage.self) }
}
