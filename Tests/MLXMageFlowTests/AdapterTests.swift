// Typed adapters on both Mage packages (contract 1.53.0, AB-A-0192): the ADP-1..4 gate over the
// manifests, the declarations, and how an IC adapter's conditioning reaches the references.
// Offline: no weights (the swapper's structural gates run in `mage-lora-smoke synth`).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXMageFlow

final class AdapterTests: XCTestCase {

    func testADPGate() {
        for (manifest, config) in [(MageFlowT2IPackage.manifest, MageFlowConfiguration(variant: .turbo)),
                                   (MageFlowEditPackage.manifest, MageFlowConfiguration(variant: .editTurbo))] {
            let report = AdapterConformance.check(manifest: manifest, configuration: config)
            XCTAssertTrue(report.passed, report.summary)
        }
    }

    /// t2i and edit are different checkpoints, so different base models; IC only on edit.
    func testDeclarations() throws {
        let t2i = try XCTUnwrap(MageFlowT2IPackage.manifest.surfaces.first?.adapters)
        let edit = try XCTUnwrap(MageFlowEditPackage.manifest.surfaces.first?.adapters)
        XCTAssertEqual(t2i.baseModel, "mage-flow")
        XCTAssertEqual(edit.baseModel, "mage-flow-edit")
        XCTAssertEqual(t2i.kinds, [.plain])
        XCTAssertEqual(edit.kinds, [.plain, .referenceConditioned])
        for support in [t2i, edit] {
            XCTAssertEqual(support.formats, [.diffusersPEFT])
            XCTAssertEqual(support.binding, .perRequest)
            XCTAssertEqual(support.maxCount, 4)
        }
    }

    private func resolved(_ id: String, kind: AdapterKind, conditioning: [ConditioningInput] = []) -> ResolvedAdapter {
        ResolvedAdapter(id: id, kind: kind, format: .diffusersPEFT, fileURL: URL(fileURLWithPath: "/dev/null"),
                        source: AdapterSource(repo: "o/r", revision: String(repeating: "a", count: 40),
                                              file: "f.safetensors", sha256: String(repeating: "b", count: 64)),
                        strength: 1, conditioning: conditioning,
                        footprint: AdapterFootprint(residentBytes: 1, activationMultiplier: kind == .plain ? 1 : 1.5))
    }

    func testICConditioningLeadsTheReferences() throws {
        let skeleton = Image(format: .png, data: Data([1]), width: 1, height: 1)
        let identity = Image(format: .png, data: Data([2]), width: 1, height: 1)
        let pose = resolved("pose", kind: .referenceConditioned, conditioning: [
            ConditioningInput(role: "pose", media: .image(skeleton)),
            ConditioningInput(role: "identity", media: .image(identity)),
        ])
        XCTAssertEqual(try MageAdapters.conditioningImages([pose, resolved("style", kind: .plain)]).map(\.data),
                       [Data([1]), Data([2])])
        XCTAssertThrowsError(try MageAdapters.conditioningImages([pose, resolved("pose2", kind: .referenceConditioned)]))
    }

    /// Refusals that need no weights: an IC adapter on text-to-image, and IC in the tiles mode.
    func testICRefusals() async {
        let ic = resolved("pose", kind: .referenceConditioned, conditioning: [
            ConditioningInput(role: "pose", media: .image(Image(format: .png, data: Data([1]), width: 1, height: 1))),
        ])
        let t2i = MageFlowT2IPackage(configuration: MageFlowConfiguration(variant: .turbo))
        do {
            _ = try await t2i.run(T2IRequest(prompt: "x").resolving([ic]))
            XCTFail("an IC adapter ran on text-to-image")
        } catch PackageError.unsupportedRequestFeature(let why) {
            XCTAssertTrue(why.contains("imageEdit"), why)
        } catch { XCTFail("\(error)") }

        let edit = MageFlowEditPackage(configuration: MageFlowConfiguration(variant: .editTurbo))
        let tile = Image(format: .png, data: Data([3]), width: 1, height: 1)
        do {
            _ = try await edit.run(IEditRequest(images: [tile], prompt: "x", mode: MageFlowModes.tiles).resolving([ic]))
            XCTFail("an IC adapter ran in the tiles mode")
        } catch PackageError.unsupportedRequestFeature(let why) {
            XCTAssertTrue(why.contains("tiles"), why)
        } catch { XCTFail("\(error)") }
    }
}
