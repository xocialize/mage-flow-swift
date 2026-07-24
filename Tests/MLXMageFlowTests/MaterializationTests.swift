// MLXMageFlow through the engine MAT gate (v0.19.0) + WeightSourcing shape. Offline.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXMageFlow

final class MaterializationTests: XCTestCase {

    /// A satisfied explicit snapshot: components + the port artifacts in one root.
    private func satisfiedSnapshot(quant: Quant) throws -> (dir: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "mage-mat-\(UUID().uuidString)")
        for sub in ["text_encoder", "vae"] + (quant == .bf16 ? ["transformer"] : []) {
            try FileManager.default.createDirectory(
                at: base.appending(path: sub), withIntermediateDirectories: true)
        }
        FileManager.default.createFile(
            atPath: base.appending(path: "folded_adaln.safetensors").path, contents: Data())
        if let f = MageFlowConfiguration(variant: .flow, quant: quant).quantDiTFile {
            FileManager.default.createFile(atPath: base.appending(path: f).path, contents: Data())
        }
        return (base, { try? FileManager.default.removeItem(at: base) })
    }

    func testMATGatePassesBF16() throws {
        let (dir, cleanup) = try satisfiedSnapshot(quant: .bf16); defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: MageFlowConfiguration(variant: .flow),
            satisfiedConfiguration: MageFlowConfiguration(variant: .flow, snapshotPath: dir.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testMATGatePassesInt4() throws {
        let (dir, cleanup) = try satisfiedSnapshot(quant: .int4); defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: MageFlowConfiguration(variant: .editTurbo, quant: .int4),
            satisfiedConfiguration: MageFlowConfiguration(
                variant: .editTurbo, quant: .int4, snapshotPath: dir.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesQuantTiering() {
        // bf16 pulls the transformer from the components repo…
        let bf16 = MageFlowConfiguration(variant: .editTurbo).weightSources
        XCTAssertEqual(bf16.map(\.role), ["components", "mlx-artifacts"])
        XCTAssertEqual(bf16[0].repo, "microsoft/Mage-Flow-Edit-Turbo")
        XCTAssertTrue(bf16[0].matching!.contains("transformer/*"))
        XCTAssertEqual(bf16[1].repo, "xocialize/Mage-Flow-Edit-Turbo-mlx")
        XCTAssertEqual(bf16[1].matching, ["folded_adaln.safetensors"])
        // …int4 EXCLUDES the 7.7 GB bf16 transformer and adds the pre-quantized DiT.
        let int4 = MageFlowConfiguration(variant: .editTurbo, quant: .int4).weightSources
        XCTAssertFalse(int4[0].matching!.contains("transformer/*"))
        XCTAssertTrue(int4[1].matching!.contains("transformer-int4.safetensors"))
    }

    func testStoreLayoutResolution() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "mage-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = MageFlowConfiguration(variant: .turbo, quant: .int8)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 2)
        // Materialize both sources in the store layout — paths from ModelStore so the
        // fixture tracks the engine's canonical convention (models--org--name since 1.22).
        let store = ModelStore(root: root)
        let comp = store.directory(for: "microsoft/Mage-Flow-Turbo")!
        for sub in ["text_encoder", "vae"] {
            try FileManager.default.createDirectory(
                at: comp.appending(path: sub), withIntermediateDirectories: true)
        }
        let arts = store.directory(for: "xocialize/Mage-Flow-Turbo-mlx")!
        try FileManager.default.createDirectory(at: arts, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: arts.appending(path: "folded_adaln.safetensors").path, contents: Data())
        FileManager.default.createFile(
            atPath: arts.appending(path: "transformer-int8.safetensors").path, contents: Data())
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(cfg.resolvedSnapshotDirectory(storeRoot: root)?.path, comp.path)
        XCTAssertEqual(cfg.resolvedArtifactsDirectory(storeRoot: root)?.path, arts.path)
    }

    func testManifestMITAndSurfaces() {
        XCTAssertEqual(MageFlowT2IPackage.manifest.license.weightLicense, .mit)
        XCTAssertEqual(MageFlowT2IPackage.manifest.license.portCodeLicense, .mit)
        XCTAssertEqual(MageFlowT2IPackage.manifest.surfaces.map(\.name), ["mage-flow-t2i"])
        XCTAssertEqual(MageFlowEditPackage.manifest.surfaces.map(\.name), ["mage-flow-edit"])
        // Distinct surface names so both packages co-register.
        XCTAssertNotEqual(MageFlowT2IPackage.manifest.surfaces[0].name,
                          MageFlowEditPackage.manifest.surfaces[0].name)
    }

    func testVariantNormalization() {
        // A t2i package handed an edit checkpoint normalizes to the t2i primary (and
        // vice versa) — the surface and the weights must agree.
        let t2i = MageFlowT2IPackage(configuration: MageFlowConfiguration(variant: .editTurbo))
        XCTAssertFalse(t2i.configuration.variant.isEdit)
        let edit = MageFlowEditPackage(configuration: MageFlowConfiguration(variant: .turbo))
        XCTAssertTrue(edit.configuration.variant.isEdit)
    }

    func testCodableRoundTrip() throws {
        let cfg = MageFlowConfiguration(variant: .editBase, quant: .int4, evictConditioner: true)
        let decoded = try JSONDecoder().decode(
            MageFlowConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.variant, .editBase)
        XCTAssertEqual(decoded.quant, .int4)
        XCTAssertEqual(decoded.defaultSteps, 30)     // Base tier default
        XCTAssertEqual(decoded.guidanceScale, 5.0)
        XCTAssertTrue(decoded.evictConditioner)
    }

    func testEvictHintsLowerFootprint() {
        // Non-evict ⇒ nil ⇒ the governor uses the static QuantFootprint.
        XCTAssertNil(MageFlowConfiguration(variant: .editTurbo, quant: .int4).residentBytesHint)
        // Evict ⇒ DiT+VAE floor (int4 ≈ 4.4 GB) — the 16 GB tier becomes admissible.
        let evict = MageFlowConfiguration(variant: .editTurbo, quant: .int4, evictConditioner: true)
        XCTAssertNotNil(evict.residentBytesHint)
        XCTAssertLessThan(evict.residentBytesHint!, 5_000_000_000)
    }

    func testVariantDefaults() {
        XCTAssertEqual(MageFlowConfiguration(variant: .turbo).defaultSteps, 4)
        XCTAssertEqual(MageFlowConfiguration(variant: .turbo).guidanceScale, 1.0)
        XCTAssertEqual(MageFlowConfiguration(variant: .flow).defaultSteps, 20)
        XCTAssertEqual(MageFlowConfiguration(variant: .flow).guidanceScale, 5.0)
        XCTAssertEqual(MageFlowConfiguration(variant: .editBase).defaultSteps, 30)
    }
}
