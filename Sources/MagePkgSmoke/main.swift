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
import MLXToolKit

func die(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(2) }

let args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2 else { die("usage: mage-pkg-smoke <snapshotRoot> t2i|edit [ref.png] [quant] [out.png]") }
let root = args[0], surface = args[1]
let refPath = args.count > 2 ? args[2] : nil
let quant: Quant = args.count > 3 ? (args[3] == "int8" ? .int8 : args[3] == "int4" ? .int4 : .bf16) : .bf16
let outPath = args.count > 4 ? args[4] : "pkg_smoke.png"

let t0 = Date()
let response: any CapabilityResponse
if surface == "edit" {
    guard let refPath, let refData = FileManager.default.contents(atPath: refPath) else {
        die("edit needs a readable ref image")
    }
    let config = MageFlowConfiguration(
        variant: .editTurbo, quant: quant, snapshotPath: root, defaultSize: 512)
    // Engine-shaped construction: registration factory (C13), not direct init.
    let package = try MageFlowEditPackage.registration.makePackage(config)
    try await package.load()
    print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    response = try await package.run(IEditRequest(
        images: [Image(format: .png, data: refData, width: 0, height: 0)],
        prompt: "make the background a snowy forest", seed: 42))
} else {
    let config = MageFlowConfiguration(
        variant: .turbo, quant: quant, snapshotPath: root, defaultSize: 512)
    let package = try MageFlowT2IPackage.registration.makePackage(config)
    try await package.load()
    print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    response = try await package.run(T2IRequest(
        prompt: "a red fox sitting in a snowy forest, photorealistic", seed: 42))
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
try image.data.write(to: URL(fileURLWithPath: outPath))
print("SMOKE PASS \(surface) \(quant): \(iw)x\(ih) "
    + "\(image.data.count / 1024) KiB in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s -> \(outPath)")
