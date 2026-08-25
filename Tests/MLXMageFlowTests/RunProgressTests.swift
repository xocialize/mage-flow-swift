// The Mage packages' end of the V2 run-phase progress plane (contract 1.18.0): the core reports
// into its own engine-free `MageProgress` sink, and the wrapper forwards those events into the
// engine's `RunProgress`. Offline — no kernels, no weights: `MageProgressBridge.forward` IS the
// wrapper's entire mapping, so exercising it here proves what `run()` does with a core event.
// The live per-step evidence is the `mage-pkg-smoke` gate, which binds a print sink the same way
// the engine does.

import Foundation
import MLXToolKit
import MageFlow
import XCTest

@testable import MLXMageFlow

final class RunProgressTests: XCTestCase {

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [RunPhaseReport] = []
        func append(_ r: RunPhaseReport) { lock.lock(); storage.append(r); lock.unlock() }
        var items: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return storage }
    }

    /// Bind both task-locals the way `run()` does — the engine's sink outermost, the core's
    /// bridge inside it — and return what the engine plane observed.
    private func forwarded(_ body: () -> Void) -> [RunPhaseReport] {
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            MageProgress.$sink.withValue(MageProgressBridge.forward, operation: body)
        }
        return box.items
    }

    // MARK: - forwarding

    func testCoreEventsReachTheEnginePlane() {
        let seen = forwarded {
            MageProgress.report(.screen)
            MageProgress.report(.encode)
            MageProgress.report(.denoise, step: 1, totalSteps: 4)
            MageProgress.report(.denoise, step: 4, totalSteps: 4)
            MageProgress.report(.decode)
        }
        XCTAssertEqual(seen.map(\.phase.rawValue),
                       ["screen", "encode", "denoise", "denoise", "decode"])
        XCTAssertEqual(seen[2], RunPhaseReport(phase: .denoise, step: 1, totalSteps: 4))
        XCTAssertEqual(seen[3].step, 4)
        XCTAssertNil(seen[0].step)   // phases without steps carry no invented counts
    }

    /// The forward is a raw-value pass-through with no mapping table — which is only correct
    /// while the core's names still ARE the canonical constants. Rename a core phase and this
    /// fails loudly instead of shipping a silently non-canonical phase name to consumers.
    func testCorePhaseNamesMatchCanonicalRunPhases() {
        XCTAssertEqual(RunPhase(rawValue: MageProgress.Phase.encode.rawValue), .encode)
        XCTAssertEqual(RunPhase(rawValue: MageProgress.Phase.denoise.rawValue), .denoise)
        XCTAssertEqual(RunPhase(rawValue: MageProgress.Phase.decode.rawValue), .decode)
        // `screen` is deliberately Mage's own phase, not one of the canonical constants: the
        // mandatory content filter is a gating AR classifier pass, not conditioning. Consumers
        // tolerate unknown phases by contract.
        let screen = RunPhase(rawValue: MageProgress.Phase.screen.rawValue)
        XCTAssertEqual(screen.rawValue, "screen")
        for canonical: RunPhase in [.encode, .denoise, .upsample, .decode, .generate, .postprocess] {
            XCTAssertNotEqual(screen, canonical)
        }
    }

    // MARK: - scoping

    func testUnboundCoreSinkIsANoOp() {
        // The CLI and the parity gates run with nothing bound — reporting must stay safe there.
        MageProgress.report(.denoise, step: 1, totalSteps: 4)
        // And a core event outside the wrapper's binding must not reach a bound engine plane.
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            MageProgress.report(.decode)
        }
        XCTAssertTrue(box.items.isEmpty)
    }
}
