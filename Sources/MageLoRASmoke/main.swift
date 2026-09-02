// mage-lora-smoke — STRUCTURAL smoke for the runtime DiT-LoRA applicator (AB-A-0050), the
// klein `lora-smoke` pattern. No weights needed: builds a config-dims MageFlowTransformer
// (random init), applies an adapter, and counts what landed.
//
//   mage-lora-smoke synth <outDir>          write two synthetic rank-32 adapters with the
//                                           trainer's exact key set (ai-toolkit comfy prefix,
//                                           144 modules / 288 tensors):
//                                             synthetic_rank32.safetensors  (random A and B)
//                                             zero_rank32.safetensors       (random A, B = 0 → INERT)
//                                           then applies synthetic_rank32 to a bf16 DiT AND to an
//                                           in-memory int8-quantized DiT (MageQuantConfig.int8 —
//                                           blocks 0…10 QuantizedLinear, block 11 bf16) and gates:
//                                             bf16: 144 targets (all LoRALinear)
//                                             int8: 132 QLoRALinear + 12 LoRALinear
//                                           plus an unknown-key adapter that MUST be rejected.
//   mage-lora-smoke apply <lora.safetensors> [bf16|int8]
//                                           apply a real checkpoint structurally, print the summary.
//
// The INERTNESS gate lives one level up (a gate that cannot fail measures nothing, AB-L-0026):
//   mage-flow-edit … --seed 42 --out a.png
//   mage-flow-edit … --seed 42 --lora zero_rank32.safetensors --out b.png     → a == b bit-exact
//   mage-flow-edit … --seed 42 --lora synthetic_rank32.safetensors --out c.png → c != a
// (README §Gates has the exact commands and the recorded results.)

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MageFlow
import MageFlowEdit

func die(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(2) }
let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else { die("usage: mage-lora-smoke synth <outDir> | apply <lora.safetensors> [bf16|int8]") }
Device.setDefault(device: Device(.cpu))

/// The trainer's per-block target list in SOURCE (diffusers/ai-toolkit) naming, with the port's
/// (in, out) dims so the synthetic factors have the right shapes.
let H = MageFlowConfig.hiddenSize
let ffInner = H * 4   // gelu-approximate FeedForward mult 4
let sourceTargets: [(rel: String, inDim: Int, outDim: Int)] = [
    ("attn.to_q", H, H), ("attn.to_k", H, H), ("attn.to_v", H, H),
    ("attn.add_q_proj", H, H), ("attn.add_k_proj", H, H), ("attn.add_v_proj", H, H),
    ("attn.to_out.0", H, H), ("attn.to_add_out", H, H),
    ("img_mlp.net.0.proj", H, ffInner), ("img_mlp.net.2", ffInner, H),
    ("txt_mlp.net.0.proj", H, ffInner), ("txt_mlp.net.2", ffInner, H),
]

func synth(rank: Int, zeroB: Bool, seed: UInt64) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    MLXRandom.seed(seed)
    for i in 0 ..< MageFlowConfig.depth {
        for t in sourceTargets {
            let base = "diffusion_model.transformer_blocks.\(i).\(t.rel)"
            out[base + ".lora_A.weight"] = (MLXRandom.normal([rank, t.inDim]) * 0.02).asType(.float16)
            out[base + ".lora_B.weight"] = zeroB
                ? MLXArray.zeros([t.outDim, rank], dtype: .float16)
                : (MLXRandom.normal([t.outDim, rank]) * 0.02).asType(.float16)
        }
    }
    return out
}

func freshDiT(int8: Bool) -> MageFlowTransformer {
    let dit = MageFlowTransformer()
    if int8 { quantize(model: dit, filter: MageQuantConfig.int8.spec) }
    return dit
}

func gate(_ ok: Bool, _ what: String) {
    print("[lora-smoke] \(ok ? "PASS" : "FAIL") \(what)")
    if !ok { exit(1) }
}

switch mode {
case "synth":
    guard args.count >= 2 else { die("synth needs <outDir>") }
    let dir = URL(fileURLWithPath: args[1])
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let synthURL = dir.appendingPathComponent("synthetic_rank32.safetensors")
    let zeroURL = dir.appendingPathComponent("zero_rank32.safetensors")
    let badURL = dir.appendingPathComponent("unknown_key.safetensors")
    try MLX.save(arrays: synth(rank: 32, zeroB: false, seed: 7), url: synthURL)
    try MLX.save(arrays: synth(rank: 32, zeroB: true, seed: 7), url: zeroURL)
    var bad = synth(rank: 4, zeroB: false, seed: 1)
    bad["diffusion_model.transformer_blocks.0.attn.to_qkv.lora_A.weight"] = MLXArray.zeros([4, H], dtype: .float16)
    bad["diffusion_model.transformer_blocks.0.attn.to_qkv.lora_B.weight"] = MLXArray.zeros([H, 4], dtype: .float16)
    try MLX.save(arrays: bad, url: badURL)
    print("[lora-smoke] wrote \(synthURL.lastPathComponent), \(zeroURL.lastPathComponent), \(badURL.lastPathComponent) (\(sourceTargets.count * MageFlowConfig.depth) modules / \(sourceTargets.count * MageFlowConfig.depth * 2) tensors each)")

    let expected = sourceTargets.count * MageFlowConfig.depth
    let s16 = try MageLoRA.apply(loRA: synthURL, to: freshDiT(int8: false))
    print("[lora-smoke] bf16 DiT: bf16Targets=\(s16.bf16Targets) quantizedTargets=\(s16.quantizedTargets)")
    gate(s16.total == expected && s16.quantizedTargets == 0, "bf16: \(expected) LoRALinear targets, zero unused keys")

    let s8 = try MageLoRA.apply(loRA: synthURL, to: freshDiT(int8: true))
    print("[lora-smoke] int8 DiT: bf16Targets=\(s8.bf16Targets) quantizedTargets=\(s8.quantizedTargets)")
    let keptBlocks = MageQuantConfig.int8.keepHiBlocks.count
    gate(s8.quantizedTargets == sourceTargets.count * (MageFlowConfig.depth - keptBlocks)
            && s8.bf16Targets == sourceTargets.count * keptBlocks,
         "int8: \(s8.quantizedTargets) QLoRALinear + \(s8.bf16Targets) LoRALinear (keepHiBlocks \(MageQuantConfig.int8.keepHiBlocks.sorted()))")

    let z = try MageLoRA.apply(loRA: zeroURL, to: freshDiT(int8: false))
    gate(z.total == expected, "zero-delta adapter lands on all \(expected) targets (inertness gate input)")

    do {
        _ = try MageLoRA.apply(loRA: badURL, to: freshDiT(int8: false))
        gate(false, "unknown key must be rejected")
    } catch MageLoRA.LoRAError.unknownKeys(_, let keys) {
        gate(keys == ["diffusion_model.transformer_blocks.0.attn.to_qkv"], "unknown key rejected loudly: \(keys)")
    }

case "apply":
    guard args.count >= 2 else { die("apply needs <lora.safetensors>") }
    let url = URL(fileURLWithPath: args[1])
    let int8 = args.count > 2 && args[2] == "int8"
    let raw = try MLX.loadArrays(url: url)
    print("[lora-smoke] \(url.lastPathComponent): \(raw.count) tensors")
    let s = try MageLoRA.apply(loRA: url, to: freshDiT(int8: int8))
    print("[lora-smoke] applied on \(int8 ? "int8" : "bf16") DiT: bf16Targets=\(s.bf16Targets) quantizedTargets=\(s.quantizedTargets) total=\(s.total)")
    let expected = MageLoRA.trainedTargetsPerBlock * MageFlowConfig.depth
    if s.total != expected {
        print("[lora-smoke] NOTE: \(s.total) targets ≠ the trainer's expected \(expected) (attn+MLP × \(MageFlowConfig.depth) blocks) — mod/io Linears present, or a partial adapter")
    }

default:
    die("unknown mode \(mode)")
}
