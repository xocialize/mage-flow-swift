// mage-flow-edit — instruction-based image editing on Apple Silicon (MLX-Swift).
//
//   edit: mage-flow-edit --repo <dir> --ref <image> --prompt "<instruction>" --out <out.png>
//   t2i : mage-flow-edit --repo <dir> --t2i --prompt "<prompt>" --out <out.png>
//   [--size 512] [--seed 42] [--steps N] [--cfg F] [--neg "<negative>"] [--renorm] [--no-filter]
//   Variant defaults: Base steps 30 / cfg 5.0 · RL steps 20 / cfg 5.0 · Turbo steps 4 / cfg 1.0
//
// <MageFlowEditRepoDir> is a downloaded microsoft/Mage-Flow-Edit* snapshot plus
// a folded_adaln.safetensors (from the port's dump_folded_adaln.py) at its root.

import Foundation
import MLX
import MageFlowEdit

func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(2) }

var repo: String?, ref: String?, prompt: String?, out = "edit.png"
var cfg = MageFlowEditConfig()
var filter = true
var t2iMode = false
var ditQuant: String?

var it = CommandLine.arguments.dropFirst().makeIterator()
while let arg = it.next() {
    switch arg {
    case "--repo": repo = it.next()
    case "--ref": ref = it.next()
    case "--prompt": prompt = it.next()
    case "--out": out = it.next() ?? out
    case "--size": cfg.size = Int(it.next() ?? "") ?? cfg.size
    case "--seed": cfg.seed = UInt64(it.next() ?? "") ?? cfg.seed
    case "--steps": cfg.steps = Int(it.next() ?? "") ?? cfg.steps
    case "--no-filter": filter = false
    case "--t2i": t2iMode = true
    case "--cfg": cfg.cfg = Float(it.next() ?? "") ?? cfg.cfg
    case "--neg": cfg.negPrompt = it.next() ?? cfg.negPrompt
    case "--renorm": cfg.renormalization = true
    case "--dit-quant": ditQuant = it.next()
    default: fail("unknown arg \(arg)")
    }
}
guard let repo, let prompt, t2iMode || ref != nil else {
    fail("usage: mage-flow-edit --repo <dir> [--t2i | --ref <image>] --prompt \"...\" --out out.png")
}
let root = URL(fileURLWithPath: repo)

let t0 = Date()
let pipe = try await MageFlowEditPipeline(
    textEncoderDir: root.appendingPathComponent("text_encoder"),
    transformerDir: root.appendingPathComponent("transformer"),
    vaeSafetensors: root.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"),
    foldedAdaLN: root.appendingPathComponent("folded_adaln.safetensors"),
    ditQuant: ditQuant.map { URL(fileURLWithPath: $0) },
    cfg: cfg)
FileHandle.standardError.write(Data("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s\n".utf8))

do {
    let t1 = Date()
    let img = t2iMode
        ? try pipe.t2i(prompt: prompt, screen: filter)
        : try pipe.edit(refImage: URL(fileURLWithPath: ref!), instruction: prompt, screen: filter)
    MageFlowEditPipeline.savePNG(img, to: URL(fileURLWithPath: out))
    FileHandle.standardError.write(
        Data("edited in \(String(format: "%.1f", Date().timeIntervalSince(t1)))s -> \(out)\n".utf8))
    FileHandle.standardError.write(
        Data(String(format: "peak GPU memory %.2f GB\n", Double(GPU.peakMemory) / 1e9).utf8))
} catch let e as MageFlowEditError {
    if case .refused(let v) = e {
        FileHandle.standardError.write(Data("REFUSED by content filter: \(v)\n".utf8))
        exit(3)
    }
    fail("\(e)")
}
