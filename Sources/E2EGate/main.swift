// End-to-end gate: temb, per-step velocity, and the full denoise loop, against
// the oracle's per-step captures.  CPU stream.
//
//   E2EGate <transformerDir> <goldens/e2e dir>

import Foundation
import MLX
import MLXNN
import MageFlow

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
let args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2 else {
    err("usage: E2EGate <transformerDir> <e2eDir> [--t2i-a | --t2i-b]")
    exit(2)
}
Device.setDefault(device: Device(.cpu))
func npy(_ p: String) throws -> MLXArray { try MLX.loadArray(url: URL(fileURLWithPath: p)) }
let dir = args[1]

let model = MageFlowTransformer()
var raw: [String: MLXArray] = [:]
for f in try FileManager.default.contentsOfDirectory(
    at: URL(fileURLWithPath: args[0]), includingPropertiesForKeys: nil)
    .filter({ $0.pathExtension == "safetensors" }) {
    raw.merge(try MLX.loadArrays(url: f)) { a, _ in a }
}
// --bf16: run the port at the ORACLE's dtype. CFG (cfg * (cond - unc))
// amplifies cross-dtype noise — the difference term partially cancels, so its
// RELATIVE error inflates, then gets multiplied by cfg and compounded per step.
// bf16-vs-bf16 removes that axis: a surviving gap is a real path bug.
let gateDtype: DType = args.contains("--bf16") ? .bfloat16 : .float32
let w = model.sanitize(weights: raw).mapValues { $0.asType(gateDtype) }
let keys = Set(model.parameters().flattened().map(\.0))
model.update(parameters: ModuleParameters.unflattened(w.filter { keys.contains($0.key) }))
eval(model)
err("[e2e] model loaded")

let sched = FlowMatchEulerScheduler(steps: 4, shift: 6.0)
let pipe = MageFlowPipeline(transformer: model)

// --- T2I gates ---------------------------------------------------------------
// Gate A (--t2i-a): t2i path, cfg=1.0 — forwards [s0, s1, s2, s3];
//   start = img_in_00, txt = txt_00, compare post-step latent to img_in_{k+1}.
// Gate B (--t2i-b): + CFG (cfg=5.0, two forwards/step, batch_cfg=False):
//   img_in fires twice per step (cond, unc — identical), txt alternates pos/neg;
//   start = img_in_00, pos = txt_00, neg = txt_01, compare to img_in_{2(k+1)}.
if let mode = args.first(where: { $0 == "--t2i-a" || $0 == "--t2i-b" || $0 == "--edit-cfg" }) {
    let isCFG = mode != "--t2i-a"
    let isEdit = mode == "--edit-cfg"
    // inputs must be gateDtype too — fp32 inputs x bf16 weights type-promote
    // back to fp32 compute, silently undoing --bf16
    let img0 = try npy("\(dir)/img_in_00.npy").asType(gateDtype)
    let txtPos = try npy("\(dir)/txt_00.npy").asType(gateDtype)
    let txtNeg = isCFG ? try npy("\(dir)/txt_01.npy").asType(gateDtype) : nil
    let L = img0.dim(1)
    // edit: packed [target, ref], each side^2; t2i: whole sequence is the target
    let targetLen = isEdit ? L / 2 : L
    let side = Int(Double(targetLen).squareRoot())
    let shapes = isEdit
        ? [(frame: 1, height: side, width: side), (frame: 1, height: side, width: side)]
        : [(frame: 1, height: side, width: side)]
    err("[t2i] \(mode) packed \(img0.shape) grid \(side)x\(side)")

    final class W2: @unchecked Sendable { var worst: Float = 0 }
    let acc2 = W2()
    // PER-STEP RESET: feed the oracle's exact step-k input, one velocity+step,
    // compare to step k+1. Isolates the per-step map from trajectory compounding
    // (CFG at 5.0 amplifies any per-step noise into the next step's input).
    for k in 0 ..< 3 {
        let inIdx = isCFG ? 2 * k : k
        let outIdx = isCFG ? 2 * (k + 1) : (k + 1)
        let stepIn = try npy(String(format: "%@/img_in_%02d.npy", dir, inIdx)).asType(gateDtype)
        let v: MLXArray = isCFG
            ? pipe.velocityCFG(img: stepIn, txt: txtPos, negTxt: txtNeg!, sigma: sched.sigma(k),
                               imgShapes: shapes, cfg: 5.0, renormalization: false)
            : pipe.velocity(img: stepIn, txt: txtPos, sigma: sched.sigma(k), imgShapes: shapes)
        // edit: step ONLY the target slice; refs stay clean
        var stepped = sched.step(v[0..., ..<targetLen, 0...], stepIn[0..., ..<targetLen, 0...], k)
        if isEdit { stepped = concatenated([stepped, stepIn[0..., targetLen..., 0...]], axis: 1) }
        eval(stepped)
        let g = try npy(String(format: "%@/img_in_%02d.npy", dir, outIdx)).asType(.float32)
        let d = abs(stepped.asType(.float32) - g).max().item(Float.self)
        let sc = abs(g).max().item(Float.self)
        let rel = sc > 0 ? d / sc : d
        acc2.worst = max(acc2.worst, rel)
        err(String(format: "  step %d (oracle input) -> %d: max_abs %.4e  rel %.2e", k, k + 1, d, rel))
    }
    _ = img0
    let pass = acc2.worst < 8e-2
    err(String(format: "[t2i] worst relative = %.3e  ->  %@",
               acc2.worst, (pass ? "PASS" : "FAIL") as NSString))
    exit(pass ? 0 : 1)
}

let vl = try npy("\(dir)/vl_features_0.npy").asType(.float32)
let shapes = [(frame: 1, height: 32, width: 32), (frame: 1, height: 32, width: 32)]
let img0 = try npy("\(dir)/img_tokens_step0.npy").asType(.float32)
let targetLen = img0.dim(1) / 2

var worstTemb: Float = 0, worstVel: Float = 0
err("[temb + velocity] per step (my fp32 vs oracle bf16):")
for i in 0 ..< 4 {
    let sig = sched.sigma(i)
    let t = model.tembFor(sigma: sig); eval(t)
    let gt = try npy("\(dir)/temb_step\(i).npy").asType(.float32)
    let dt = abs(t - gt).max().item(Float.self); let st = abs(gt).max().item(Float.self)
    let relT = st > 0 ? dt / st : dt; worstTemb = max(worstTemb, relT)

    let stepIn = try npy("\(dir)/img_tokens_step\(i).npy").asType(.float32)
    let v = pipe.velocity(img: stepIn, txt: vl, sigma: sig, imgShapes: shapes); eval(v)
    let gv = try npy("\(dir)/velocity_step\(i).npy").asType(.float32)
    let dv = abs(v - gv).max().item(Float.self); let sv = abs(gv).max().item(Float.self)
    let relV = sv > 0 ? dv / sv : dv; worstVel = max(worstVel, relV)
    err(String(format: "  sigma %.4f: temb rel %.2e | velocity rel %.2e",
               sig, relT, relV))
}

// full denoise loop from the GS-equivalent captured noise -> final latent
final class W: @unchecked Sendable { var worst: Float = 0 }
let acc = W()
_ = pipe.denoise(img: img0, txt: vl, targetLen: targetLen, imgShapes: shapes,
                 scheduler: sched) { i, packed in
    guard i < 3, let g = try? npy("\(dir)/img_tokens_step\(i + 1).npy").asType(.float32) else { return }
    let d = abs(packed - g).max().item(Float.self); let s = abs(g).max().item(Float.self)
    acc.worst = max(acc.worst, s > 0 ? d / s : d)
}
err(String(format: "\n[e2e] worst temb rel %.2e | worst velocity rel %.2e | full-loop latent rel %.2e",
           worstTemb, worstVel, acc.worst))
// fp32 port vs bf16 oracle: a few % relative is expected (bf16 accumulation +
// the chaotic sigma sensitivity), well below anything structural.
let pass = worstVel < 8e-2 && acc.worst < 8e-2
err("[e2e] \(pass ? "PASS" : "FAIL")")
exit(pass ? 0 : 1)
