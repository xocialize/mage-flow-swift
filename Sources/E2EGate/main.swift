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
guard args.count >= 2 else { err("usage: E2EGate <transformerDir> <e2eDir>"); exit(2) }
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
let w = model.sanitize(weights: raw).mapValues { $0.asType(.float32) }
let keys = Set(model.parameters().flattened().map(\.0))
model.update(parameters: ModuleParameters.unflattened(w.filter { keys.contains($0.key) }))
eval(model)
err("[e2e] model loaded")

let sched = FlowMatchEulerScheduler(steps: 4, shift: 6.0)
let pipe = MageFlowPipeline(transformer: model)
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
