// Multi-reference request validation (AB-A-0047, 2026-09-02) — offline, no MLX
// kernels, no weights. `MageFlowEditPackage.run()` validates the reference count
// BEFORE it touches the runtime, so an unloaded package is enough to prove:
//   - 0 images and >3 images are rejected legibly (`unsupportedRequestFeature`),
//     never silently truncated to the primary reference;
//   - 1…3 images pass the guard and reach the runtime (which reports `notLoaded`
//     here — the first thing past the guard, and the proof the guard let them through).
// The live 2-/3-ref renders are gated against the PyTorch oracle by
// `E2EGate --edit-refs N` + the decoded-render PSNR (see README §Gates).

import Foundation
import MLXToolKit
import XCTest

@testable import MLXMageFlow

final class MultiReferenceTests: XCTestCase {

    private func request(refs: Int) -> IEditRequest {
        IEditRequest(
            images: (0 ..< refs).map { _ in Image(format: .png, data: Data(), width: 1, height: 1) },
            prompt: "probe")
    }

    private func run(refs: Int) async -> PackageError? {
        let package = MageFlowEditPackage(
            configuration: MageFlowConfiguration(variant: .editTurbo, quant: .int8))
        do {
            _ = try await package.run(request(refs: refs))
            return nil
        } catch let e as PackageError {
            return e
        } catch {
            XCTFail("unexpected error type \(type(of: error)): \(error)")
            return nil
        }
    }

    func testTrainedRangePassesTheGuard() async {
        for n in 1 ... 3 {
            let e = await run(refs: n)
            XCTAssertEqual(e, .notLoaded, "\(n) refs must reach the runtime (notLoaded), got \(String(describing: e))")
        }
    }

    func testZeroReferencesRejected() async {
        guard case .unsupportedRequestFeature(let why)? = await run(refs: 0) else {
            return XCTFail("0 refs must be unsupportedRequestFeature")
        }
        XCTAssertTrue(why.contains("at least one"), why)
    }

    func testBeyondTrainedRangeRejectedLegibly() async {
        guard case .unsupportedRequestFeature(let why)? = await run(refs: 4) else {
            return XCTFail("4 refs must be unsupportedRequestFeature, not truncated")
        }
        XCTAssertTrue(why.contains("4 reference images"), why)
        XCTAssertTrue(why.contains("1…3"), why)
    }

    func testSurfaceDescriptorDeclaresTheRange() {
        let summary = MageFlowEditPackage.manifest.surfaces.first?.summary ?? ""
        XCTAssertTrue(summary.contains("1–3 reference images"), summary)
        XCTAssertFalse(summary.contains("Single reference"), summary)
    }
}
