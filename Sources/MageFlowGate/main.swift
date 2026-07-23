// Parity gate: MageFlow DiT (Swift/MLX) vs the PyTorch oracle goldens.
//
//   MageFlowGate <transformerDir> <dit_goldens.safetensors>
//
// Drives the 12-block chain from the captured post-img_in / post-txt_in
// activations plus the captured rope, so a break localizes to a block rather
// than to the embedding or rope math. Also independently checks that the Swift
// rope reproduces the golden tables.
//
// CPU stream: Apple GPU fp32 accumulates ~8e-4/op, which both masks real op
// bugs and gets mistaken for them.

import Foundation
import MLX
import MLXNN
import MageFlow

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).flattened()
    let y = b.asType(.float32).flattened()
    let d = (x * y).sum().item(Float.self)
    let nx = sqrt((x * x).sum()).item(Float.self)
    let ny = sqrt((y * y).sum()).item(Float.self)
    return nx * ny == 0 ? 1 : d / (nx * ny)
}

struct Row { let name: String; let maxAbs: Float; let rel: Float; let cos: Float }

/// Boxed so the nonisolated `onBlock` callback can append: a top-level `var`
/// would be main-actor isolated under Swift 6 strict concurrency.
final class Rows: @unchecked Sendable { var items: [Row] = [] }
let rows = Rows()

/// Gate on RELATIVE error. This network's activations reach |x| ~ 1.2e5 by
/// block 11, so an absolute tolerance is meaningless — 0.35 absolute there is
/// 2.8e-6 relative, i.e. fp32 noise. Judging absolute would fail a correct port.
func check(_ name: String, _ a: MLXArray, _ g: MLXArray) {
    eval(a)
    let gf = g.asType(.float32)
    let d = abs(a.asType(.float32) - gf).max().item(Float.self)
    let scale = abs(gf).max().item(Float.self)
    rows.items.append(
        Row(name: name, maxAbs: d, rel: scale > 0 ? d / scale : d, cos: cosine(a, g)))
}

let args = Array(CommandLine.arguments.dropFirst())
// Weights-free NAX split-K GEMM probe at the exact Mage DiT FFN shapes.
// mlx-swift <=0.31.6 JIT-miscompiles steel_gemm_splitk_axpby_nax (mlx#3797,
// fixed by mlx#3810): M=896 clean / M>=1024 garbage. Run after any mlx-swift
// bump to decide whether MageFeedForward.downProjected can drop its row-chunk.
//   MageFlowGate --nax-probe
if args.first == "--nax-probe" {
    var lcg: UInt64 = 0x9E37_79B9_7F4A_7C15
    func rand(_ n: Int) -> [Float] {
        (0 ..< n).map { _ in
            lcg = lcg &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int64(bitPattern: lcg >> 11)) / Float(Int64.max >> 11)
        }
    }
    let (K, N) = (12288, 3072)   // Mage DiT FFN proj_out
    let b = MLXArray(rand(N * K), [N, K]).asType(.bfloat16)
    var ok = true
    for m in [512, 896, 1024, 1366, 2048] {
        let a = MLXArray(rand(m * K), [m, K]).asType(.bfloat16)
        let y = matmul(a, b.T)
        let yRef = matmul(a.asType(.float32), b.asType(.float32).T)
        eval(y, yRef)
        let mab = abs(y.asType(.float32) - yRef).max().item(Float.self)
        let c = cosine(y, yRef)
        let pass = c > 0.999 && mab.isFinite && mab < 100
        if !pass { ok = false }
        err(String(format: "  M=%d K=%d N=%d bf16: cos %.8f max_abs %.3e  %@",
                   m, K, N, c, mab, (pass ? "OK" : "BROKEN") as NSString))
    }
    err("[nax-probe] \(ok ? "PASS — kernel fixed, row-chunk removable" : "FAIL — keep the row-chunk")")
    exit(ok ? 0 : 1)
}

guard args.count >= 2 else {
    err("usage: MageFlowGate <transformerDir> <dit_goldens.safetensors>")
    exit(2)
}
let weightsDir = URL(fileURLWithPath: args[0])
let goldensURL = URL(fileURLWithPath: args[1])

Device.setDefault(device: Device(.cpu))

let g = try MLX.loadArrays(url: goldensURL)
err("[gate] loaded \(g.count) golden tensors")

// --- rope: independent check against the captured tables --------------------
// 512^2 edit pack: target 32x32 (frame 0) + one ref 32x32 (frame 1) = 2048 tokens.
let rope = MageFlowEmbedRope()
let (rc, rs) = rope(imgShapes: [(1, 32, 32), (1, 32, 32)])
check("rope cos", rc, g["rope_real"]!)
check("rope sin", rs, g["rope_imag"]!)

// --- model ------------------------------------------------------------------
let model = MageFlowTransformer()
let files = try FileManager.default.contentsOfDirectory(at: weightsDir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "safetensors" }
guard !files.isEmpty else { err("[gate] no safetensors in \(weightsDir.path)"); exit(1) }
var raw: [String: MLXArray] = [:]
for f in files { raw.merge(try MLX.loadArrays(url: f)) { a, _ in a } }
err("[gate] loaded \(raw.count) weight tensors")

var w = model.sanitize(weights: raw).mapValues { $0.asType(.float32) }
let moduleKeys = Set(model.parameters().flattened().map(\.0))
let missing = moduleKeys.subtracting(Set(w.keys)).sorted()
let extra = Set(w.keys).subtracting(moduleKeys).sorted()
if !missing.isEmpty {
    err("[gate] MISSING \(missing.count) module keys, e.g. \(missing.prefix(5))")
    exit(1)
}
if !extra.isEmpty { err("[gate] note: \(extra.count) unused checkpoint keys, e.g. \(extra.prefix(3))") }
w = w.filter { moduleKeys.contains($0.key) }
model.update(parameters: ModuleParameters.unflattened(w))
eval(model)
err("[gate] weights loaded, no missing keys")

// --- block chain from captured activations ----------------------------------
let hidden = g["img_in"]!.asType(.float32)
let encoder = g["txt_in"]!.asType(.float32)
let temb = g["temb"]!.asType(.float32)

let out = model.runBlocks(
    hidden: hidden, encoder: encoder, temb: temb, rope: (rc, rs)
) { i, txt, img in
    check("block \(i) txt", txt, g["block\(i)_txt"]!)
    check("block \(i) img", img, g["block\(i)_img"]!)
}
check("proj_out", out, g["proj_out"]!)

// --- report -----------------------------------------------------------------
err("")
err(String(format: "%-18@ %13@ %11@ %13@",
           "stage" as NSString, "max_abs" as NSString, "rel" as NSString, "cosine" as NSString))
err(String(repeating: "-", count: 60))
var worst: Float = 0
for r in rows.items {
    worst = max(worst, r.rel)
    let flag = r.rel < 1e-4 ? "" : (r.rel < 1e-3 ? "  <-- high" : "  <-- BUG")
    err(String(format: "%-18@ %13.4e %11.2e %13.8f%@",
               r.name as NSString, r.maxAbs, r.rel, r.cos, flag as NSString))
}
err(String(repeating: "-", count: 60))
// fp32 relative thresholds: a 12-block chain should stay well under 1e-4.
let pass = worst < 1e-4
err(String(format: "worst relative = %.3e  ->  %@", worst, (pass ? "PASS" : "FAIL") as NSString))
exit(pass ? 0 : 1)
