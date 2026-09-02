// mage-flow-edit — instruction-based image editing on Apple Silicon (MLX-Swift).
//
//   edit: mage-flow-edit --repo <dir> --ref <image> [--ref <image2> [--ref <image3>]] --prompt "<instruction>" --out <out.png>
//         (refs are Image 1, Image 2, Image 3 in the instruction; 1–3 = upstream's trained range)
//   t2i : mage-flow-edit --repo <dir> --t2i --prompt "<prompt>" --out <out.png>
//   [--size 512] [--seed 42] [--steps N] [--cfg F] [--neg "<negative>"] [--renorm] [--no-filter]
//   [--lora <adapter.safetensors> [--lora <second> …] [--lora-strength 1.0]]   runtime DiT-LoRA (AB-A-0050)
//   Variant defaults: Base steps 30 / cfg 5.0 · RL steps 20 / cfg 5.0 · Turbo steps 4 / cfg 1.0
//
// <MageFlowEditRepoDir> is a downloaded mage-flow-community/Mage-Flow-Edit* snapshot plus
// a folded_adaln.safetensors (from the port's dump_folded_adaln.py) at its root.

import Foundation
import MLX
import MageFlowEdit

func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(2) }

var repo: String?, refs: [String] = [], prompt: String?, out = "edit.png"
var cfg = MageFlowEditConfig()
var filter = true
var t2iMode = false
var ditQuant: String?
var loras: [String] = []            // --lora <path> (repeatable; rank-stacked)
var loraStrength: Float = 1.0       // --lora-strength s (applies to every --lora)

var it = CommandLine.arguments.dropFirst().makeIterator()
while let arg = it.next() {
    switch arg {
    case "--lora": if let p = it.next() { loras.append(p) }
    case "--lora-strength": loraStrength = Float(it.next() ?? "") ?? loraStrength
    case "--repo": repo = it.next()
    case "--ref": if let r = it.next() { refs.append(r) }
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
guard let repo, let prompt, t2iMode || !refs.isEmpty else {
    fail("usage: mage-flow-edit --repo <dir> [--t2i | --ref <image> [--ref <image2> [--ref <image3>]]] --prompt \"...\" --out out.png")
}
if refs.count > MageFlowEditConfig.maxReferences {
    fail("\(refs.count) --ref images; the trained range is 1…\(MageFlowEditConfig.maxReferences)")
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
if !loras.isEmpty {
    // Runtime DiT-LoRA (AB-A-0050): activation-path adapter on the resident DiT, every key must land.
    let s = try pipe.applyLoRA(loRAs: loras.map { (URL(fileURLWithPath: $0), loraStrength) })
    FileHandle.standardError.write(Data(
        "lora: \(loras.count) adapter(s) @ strength \(loraStrength) → \(s.total) targets (\(s.bf16Targets) bf16, \(s.quantizedTargets) quantized)\n".utf8))
}

do {
    let t1 = Date()
    let img = t2iMode
        ? try pipe.t2i(prompt: prompt, screen: filter)
        : try pipe.edit(refImages: refs.map { URL(fileURLWithPath: $0) }, instruction: prompt, screen: filter)
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
