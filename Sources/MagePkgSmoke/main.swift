// Live package smoke (the wrapper-level `--e2e-<surface>-pkg` gate): drives the real
// MLXEngine surface — registration factory → load() → run(request) → decoded PNG —
// against a local snapshot. This is where the silent-failure class shows up; the
// offline C/MAT/CAN suite never runs a kernel.
//
//   mage-pkg-smoke <snapshotRoot> t2i|edit [ref.png] [bf16|int8|int4] [out.png]
//
// The snapshot root needs transformer/ text_encoder/ vae/ folded_adaln.safetensors
// (+ transformer-<quant>.safetensors for quant tiers).

import Foundation
import MLXMageFlow
import MLXServeCore
import MLXToolKit

func die(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(2) }

let args = Array(CommandLine.arguments.dropFirst())
// probe-dl <repo> <glob> — download one source into a temp store, printing
// progress samples. Diagnoses single-file progress granularity. Since engine
// 0.32.0 this drives MLXServeCore.WeightMaterializer (the engine executor that
// replaced the package-local copy), so the probe exercises the shipping path.
if args.first == "probe-dl", args.count >= 3 {
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("mage-dl-probe-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tmp) }
    try await WeightDownloadProgress.$sink.withValue({ fraction, bps in
        FileHandle.standardError.write(Data(String(format: "[dl] frac=%.4f MB/s=%.1f\n",
            fraction, (bps ?? 0) / 1_048_576).utf8))
    }) {
        try await WeightMaterializer().materialize(
            [WeightSource(role: "probe", repo: args[1], revision: nil, matching: [args[2]])],
            into: tmp)
    }
    print("PROBE DONE")
    exit(0)
}

guard args.count >= 2 else { die("usage: mage-pkg-smoke <snapshotRoot> t2i|edit [ref.png] [quant] [out.png]") }
let root = args[0], surface = args[1]
let refPath = args.count > 2 ? args[2] : nil
let quant: Quant = args.count > 3 ? (args[3] == "int8" ? .int8 : args[3] == "int4" ? .int4 : .bf16) : .bf16
let outPath = args.count > 4 ? args[4] : "pkg_smoke.png"

let t0 = Date()

// Run-phase progress (contract 1.18.0, ENGINE-NEEDS V2): bind the plane the way
// MLXServeEngine binds it around run(), so this gate is the live evidence that the
// package reports phases — the offline suite can only prove the wrapper's mapping.
final class PhaseLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [RunPhaseReport] = []
    func append(_ r: RunPhaseReport) { lock.lock(); storage.append(r); lock.unlock() }
    var items: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return storage }
}
let phases = PhaseLog()
let progressSink: RunProgress.Sink = { r in
    phases.append(r)
    let counts = r.step.map { " \($0)/\(r.totalSteps ?? 0)" } ?? ""
    FileHandle.standardError.write(Data(String(
        format: "[phase %6.1fs] %@%@\n", Date().timeIntervalSince(t0), r.phase.rawValue, counts).utf8))
}

let response: any CapabilityResponse
if surface == "edit" {
    // ref.png may be a comma-separated list (up to 3) — the multi-reference wrapper path
    // (AB-A-0047): images land in IEditRequest.images in prompt order.
    let refPaths = (refPath ?? "").split(separator: ",").map(String.init)
    let refDatas = refPaths.compactMap { FileManager.default.contents(atPath: $0) }
    guard !refPaths.isEmpty, refDatas.count == refPaths.count else {
        die("edit needs readable ref image(s): \(refPaths)")
    }
    let refImages = refDatas.map { Image(format: .png, data: $0, width: 0, height: 0) }
    let editPrompt = refImages.count > 1
        ? "Place the object from Image 1 into the scene of Image 2"
        : "make the background a snowy forest"
    let config = MageFlowConfiguration(
        variant: .editTurbo, quant: quant, snapshotPath: root, defaultSize: 512)
    // Engine-shaped construction: registration factory (C13), not direct init.
    let package = try MageFlowEditPackage.registration.makePackage(config)
    try await package.load()
    print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    response = try await RunProgress.$sink.withValue(progressSink) {
        try await package.run(IEditRequest(images: refImages, prompt: editPrompt, seed: 42))
    }
} else {
    let config = MageFlowConfiguration(
        variant: .turbo, quant: quant, snapshotPath: root, defaultSize: 512)
    let package = try MageFlowT2IPackage.registration.makePackage(config)
    try await package.load()
    print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    response = try await RunProgress.$sink.withValue(progressSink) {
        try await package.run(T2IRequest(
            prompt: "a red fox sitting in a snowy forest, photorealistic", seed: 42))
    }
}

let image: Image
switch response {
case let r as IEditResponse: image = r.image
case let r as T2IResponse: image = r.image
default: die("unexpected response type \(type(of: response))")
}
// Quantify, don't trust eyes: a valid render is a real PNG with sane dims and
// non-degenerate pixel variance (all-flat = the silent-failure tell).
let (iw, ih) = (image.width ?? 0, image.height ?? 0)
guard image.format == .png, iw >= 256, ih >= 256, image.data.count > 10_000
else { die("SMOKE FAIL: degenerate image \(iw)x\(ih) \(image.data.count)B") }

// V2 progress plane: a silent run is a regression, not a cosmetic gap. Require the
// per-step denoise counter (the signal consumers actually render) to have covered the
// whole loop, plus the decode seam that follows it.
let steps = phases.items.filter { $0.phase == .denoise }.compactMap(\.step)
let declaredTotal = phases.items.first { $0.phase == .denoise }?.totalSteps ?? 0
guard !steps.isEmpty, steps == Array(1 ... max(declaredTotal, 1)),
      phases.items.contains(where: { $0.phase == .decode })
else {
    die("SMOKE FAIL: progress plane — denoise steps \(steps) of \(declaredTotal), "
        + "phases seen \(phases.items.map(\.phase.rawValue))")
}
try image.data.write(to: URL(fileURLWithPath: outPath))
print("SMOKE PASS \(surface) \(quant): \(iw)x\(ih) "
    + "\(image.data.count / 1024) KiB in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s -> \(outPath)")
// collapse the per-step repeats into the phase sequence a consumer would render
let phaseTrace = phases.items.map(\.phase.rawValue).reduce(into: [String]()) { acc, p in
    if acc.last != p { acc.append(p) }
}
print("  phases: \(phaseTrace.joined(separator: " → ")) "
    + "(denoise \(steps.count)/\(declaredTotal) steps reported)")
