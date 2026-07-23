// Mage-Flow packages through the engine's CAN gate (offline, no MLX kernels, no
// weights). CAN-1/2 drive the real run() pre-cancelled: the entry checkpoint
// (`try Task.checkCancellation()` as the FIRST act of run(), before notLoaded
// validation) fires before weights are touched, so a stub configuration suffices.
// CAN-3 documents the checkpoint cadence:
//   - post-screen seam — `shouldStop` check right after the content filter, before
//     conditioning encodes (MageFlowEditPipeline.t2i / .edit).
//   - denoise/step — `if shouldStop?() == true { break }` at the top of the
//     MageFlowPipeline.denoise loop (the one loop behind t2i AND edit, all tiers;
//     non-throwing core API — sanctioned break).
//   - pre-decode seam — a cancelled task throws CancellationError before the
//     monolithic MageVAE decode (ONE MLX eval; no per-chunk decode cadence claimed).
//   - the wrapper's post-generate `try Task.checkCancellation()` and the pipeline's
//     thrown CancellationError propagate UNCHANGED (the only catch in
//     MageFlowRuntime.generate matches MageFlowEditError, never CancellationError).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXMageFlow

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRunT2I() async {
        let package = MageFlowT2IPackage(
            configuration: MageFlowConfiguration(variant: .turbo))
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledRunEdit() async {
        let package = MageFlowEditPackage(
            configuration: MageFlowConfiguration(variant: .editTurbo, quant: .int4))
        let report = await CancellationConformance.checkRun(
            package: package,
            request: IEditRequest(
                images: [Image(format: .png, data: Data(), width: 1, height: 1)],
                prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    private var posture: CancellationConformance.CheckpointPosture {
        .cadence([
            .init(phase: .denoise, unit: .step)
        ])
    }

    func testCANCadenceDeclarationT2I() {
        // Multi-GB peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MageFlowT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: MageFlowT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclarationEdit() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MageFlowEditPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: MageFlowEditPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }
}
