// mage-flow-edit — instruction-based image editing on Apple Silicon (MLX-Swift).
//
//   mage-flow-edit --repo <MageFlowEditRepoDir> --ref <image> --prompt "<instruction>"
//                  --out <out.png> [--size 512] [--seed 42] [--no-filter]
//
// <MageFlowEditRepoDir> is a downloaded microsoft/Mage-Flow-Edit* snapshot plus
// a folded_adaln.safetensors (from the port's dump_folded_adaln.py) at its root.

import Foundation
import MageFlowEdit

func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(2) }

var repo: String?, ref: String?, prompt: String?, out = "edit.png"
var cfg = MageFlowEditConfig()
var filter = true

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
    default: fail("unknown arg \(arg)")
    }
}
guard let repo, let ref, let prompt else {
    fail("usage: mage-flow-edit --repo <dir> --ref <image> --prompt \"...\" --out out.png")
}
let root = URL(fileURLWithPath: repo)

let t0 = Date()
let pipe = try await MageFlowEditPipeline(
    textEncoderDir: root.appendingPathComponent("text_encoder"),
    transformerDir: root.appendingPathComponent("transformer"),
    vaeSafetensors: root.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"),
    foldedAdaLN: root.appendingPathComponent("folded_adaln.safetensors"),
    cfg: cfg)
FileHandle.standardError.write(Data("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s\n".utf8))

do {
    let t1 = Date()
    let img = try pipe.edit(refImage: URL(fileURLWithPath: ref), instruction: prompt, screen: filter)
    MageFlowEditPipeline.savePNG(img, to: URL(fileURLWithPath: out))
    FileHandle.standardError.write(
        Data("edited in \(String(format: "%.1f", Date().timeIntervalSince(t1)))s -> \(out)\n".utf8))
} catch let e as MageFlowEditError {
    if case .refused(let v) = e {
        FileHandle.standardError.write(Data("REFUSED by content filter: \(v)\n".utf8))
        exit(3)
    }
    fail("\(e)")
}
