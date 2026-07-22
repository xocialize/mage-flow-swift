// Parity gate: MageVAE (Swift/MLX) vs the PyTorch oracle goldens.
//
//   MageVAEGate <vae.safetensors> <folded_adaln.safetensors> <vae_goldens.safetensors>
//
// CPU stream, fp32. Gates on relative error: this decoder's intermediates are
// modest, but the encoder's DiCoBlock chain reaches |x| ~ 700, so absolute
// tolerances mislead (see the DiT gate for the same lesson at 1e5).

import Foundation
import MLX
import MageFlow

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
    let n = sqrt((x * x).sum()).item(Float.self) * sqrt((y * y).sum()).item(Float.self)
    return n == 0 ? 1 : (x * y).sum().item(Float.self) / n
}

struct Row { let name: String; let maxAbs: Float; let rel: Float; let cos: Float }
final class Rows: @unchecked Sendable { var items: [Row] = [] }
let rows = Rows()

func check(_ name: String, _ a: MLXArray, _ g: MLXArray) {
    eval(a)
    let gf = g.asType(.float32)
    guard a.shape == gf.shape else {
        err("[gate] SHAPE \(name): \(a.shape) vs \(gf.shape)")
        rows.items.append(Row(name: name, maxAbs: .infinity, rel: .infinity, cos: 0)); return
    }
    let d = abs(a.asType(.float32) - gf).max().item(Float.self)
    let s = abs(gf).max().item(Float.self)
    rows.items.append(Row(name: name, maxAbs: d, rel: s > 0 ? d / s : d, cos: cosine(a, gf)))
}

let args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 3 else {
    err("usage: MageVAEGate <vae.safetensors> <folded_adaln.safetensors> <goldens.safetensors>")
    exit(2)
}
Device.setDefault(device: Device(.cpu))

let w = try MageVAELoader.load(vae: URL(fileURLWithPath: args[0]),
                               foldedAdaLN: URL(fileURLWithPath: args[1]))
let g = try MLX.loadArrays(url: URL(fileURLWithPath: args[2]))
err("[gate] loaded weights + \(g.count) goldens")

// --- encoder ----------------------------------------------------------------
let moments = encodeMoments(g["input"]!, w)
check("enc proj_out", moments, g["enc_proj_out"]!)
check("enc mean", moments[.ellipsis, ..<128], g["moments_mean"]!)

// --- decoder ----------------------------------------------------------------
let cond = codDecoder(g["latent_mean"]!, w)
check("cod decoder (cond)", cond, g["cond"]!)

let img = vaeDecode(g["latent_mean"]!, w)
check("decode -> pixels", img, g["decoded"]!)

// --- report -----------------------------------------------------------------
err("")
err(String(format: "%-22@ %13@ %11@ %13@",
           "stage" as NSString, "max_abs" as NSString, "rel" as NSString, "cosine" as NSString))
err(String(repeating: "-", count: 64))
var worst: Float = 0
for r in rows.items {
    worst = max(worst, r.rel)
    let flag = r.rel < 1e-4 ? "" : (r.rel < 1e-3 ? "  <-- high" : "  <-- BUG")
    err(String(format: "%-22@ %13.4e %11.2e %13.8f%@",
               r.name as NSString, r.maxAbs, r.rel, r.cos, flag as NSString))
}
err(String(repeating: "-", count: 64))
let pass = worst < 1e-4
err(String(format: "worst relative = %.3e  ->  %@", worst, (pass ? "PASS" : "FAIL") as NSString))

// --- scheduler: exact-value gate against the oracle's captured schedule ------
// Turbo (4 steps, shift 6.0) has a closed-form expected schedule; check it to
// the digit rather than trusting the implementation.
let sched = FlowMatchEulerScheduler(steps: 4, shift: 6.0)
let expectSigmas: [Float] = [1.0, 0.947368, 0.857143, 0.666667, 0.0]
let expectTs: [Float] = [1000.0, 947.37, 857.14, 666.67]
var schedOK = sched.sigmas.count == expectSigmas.count && sched.timesteps.count == expectTs.count
if schedOK {
    for (a, b) in zip(sched.sigmas, expectSigmas) where abs(a - b) > 1e-5 { schedOK = false }
    for (a, b) in zip(sched.timesteps, expectTs) where abs(a - b) > 1e-2 { schedOK = false }
}
err("")
err("[sched] sigmas    \(sched.sigmas.map { (($0 * 1e6).rounded() / 1e6) })")
err("[sched] timesteps \(sched.timesteps.map { (($0 * 100).rounded() / 100) })")
err("[sched] 4-step Turbo schedule -> \(schedOK ? "PASS" : "FAIL")")
if !schedOK { exit(1) }

exit(pass ? 0 : 1)
