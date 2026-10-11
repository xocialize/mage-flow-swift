import XCTest
@testable import MageFlow

/// `FlowMatchEulerScheduler(startSigma:)` — the image-to-image start behind `MageFlowEditConfig.initSigma`.
final class SchedulerStartSigmaTests: XCTestCase {
    func testDefaultStartIsTheUnchangedSchedule() {
        for steps in [1, 4, 20, 30] {
            let a = FlowMatchEulerScheduler(steps: steps), b = FlowMatchEulerScheduler(steps: steps, startSigma: 1.0)
            XCTAssertEqual(a.sigmas, b.sigmas)
            XCTAssertEqual(a.timesteps, b.timesteps)
        }
    }

    func testPartialStartBeginsAtTheRequestedSigma() {
        for s0: Float in [0.5, 0.85, 0.9, 0.933, 0.978] {
            let sched = FlowMatchEulerScheduler(steps: 4, shift: 6, startSigma: s0)
            XCTAssertEqual(sched.sigmas.count, 5)
            XCTAssertEqual(sched.sigmas[0], s0, accuracy: 1e-5)
            XCTAssertEqual(sched.sigmas.last, 0)
            for i in 0 ..< 4 { XCTAssertGreaterThan(sched.sigmas[i], sched.sigmas[i + 1]) }
        }
    }

    /// ComfyUI's `denoise d` on a shift-6 schedule starts at shift·d / (1 + (shift − 1)·d); the CivitAI Mage
    /// restoration workflow's 0.88 is σ ≈ 0.978 here.
    func testComfyDenoiseMapping() {
        let d: Float = 0.88, s0 = 6 * d / (1 + 5 * d)
        XCTAssertEqual(FlowMatchEulerScheduler(steps: 4, startSigma: s0).sigmas[0], 0.9778, accuracy: 1e-3)
    }
}
