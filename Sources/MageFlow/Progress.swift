// Progress.swift — core-owned ambient run-phase seam.
//
// The MageFlow core stays engine-free (MLXToolKit is a WRAPPER dependency by design — see
// Package.swift: only MLXMageFlow links it), so the core reports its phases into its OWN
// task-local sink and the MLXMageFlow wrapper forwards events into the engine contract's
// `RunProgress` (contract 1.18.0, ENGINE-NEEDS V2). Same shape as the contract plane: a coarse
// phase plus optional 1-based step/totalSteps within it. No-op when unbound, so the CLI and the
// parity gates run unchanged; a caller can bind a print sink for terminal progress.
//
// Task-local (not a pipeline property) so a binding scopes to exactly one generation — mirrors
// the engine's own WeightDownloadProgress/RunProgress pattern, and keeps a Sendable sink off the
// non-Sendable pipeline object.

public enum MageProgress {
    /// The phases a Mage run passes through. `encode`/`denoise`/`decode` match the engine
    /// contract's canonical `RunPhase` constants verbatim, so the wrapper forwards raw values
    /// with no mapping table.
    ///
    /// `screen` is Mage's own name, not a synonym for an existing constant: the mandatory
    /// upstream Responsible-AI classifier is an autoregressive VLM generation that gates the
    /// run, can refuse it outright, and on the evicted-conditioner tier carries the 8.3 GB
    /// conditioner reload — a different kind of stage from conditioning. Consumers tolerate
    /// unknown phases by contract (render the raw value or a generic label).
    public enum Phase: String, Sendable {
        /// Mandatory content filter (AR verdict, fail-closed). Skipped when the caller bypasses
        /// it; may carry a conditioner load on the light tier.
        case screen
        /// Qwen3-VL conditioning — the ref/text forward, the negative pass under CFG, and the
        /// MageVAE encode of the reference on the edit path.
        case encode
        /// The flow-matching Euler loop — reported per step, the run's dominant cost.
        case denoise
        /// MageVAE decode to pixels. One eval — no per-chunk cadence exists to report.
        case decode
    }

    /// One coarse observation: the phase, plus 1-based step counts when the phase has steps.
    /// Mage has a single denoise stage, so the contract's `stage`/`totalStages` have nothing to
    /// carry here; they can ride in additively if a staged path ever lands.
    public struct Event: Sendable, Equatable {
        public var phase: Phase
        public var step: Int?
        public var totalSteps: Int?

        public init(phase: Phase, step: Int? = nil, totalSteps: Int? = nil) {
            self.phase = phase
            self.step = step
            self.totalSteps = totalSteps
        }
    }

    public typealias Sink = @Sendable (Event) -> Void

    @TaskLocal public static var sink: Sink?

    /// Report a phase observation to the ambient sink (no-op if none is bound). Public because
    /// the reporting sites span both core targets (MageFlow's denoise loop, MageFlowEdit's
    /// pipeline seams).
    public static func report(_ phase: Phase, step: Int? = nil, totalSteps: Int? = nil) {
        sink?(Event(phase: phase, step: step, totalSteps: totalSteps))
    }
}
