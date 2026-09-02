// Runtime DiT-LoRA (AB-A-0050) — the offline, weight-free half of the gate:
//   - key mapping: every trainer target (ai-toolkit comfy prefix, diffusers module names)
//     lands on the port's block-relative Linear, incl. the `net.0.proj → proj_in`,
//     `net.2 → proj_out`, `to_out.0`, `img_mod.1 → img_mod` renames and the accepted prefixes;
//   - anything else is nil (→ `unknownKeys`, a loud rejection — a silently-skipped key would
//     make the adapter a partial no-op, AB-L-0026);
//   - `loraPath` is a runtime override: never persisted, nil on decode.
// The structural apply (144 targets, int8 dispatch, unknown-key rejection) needs a
// config-dims DiT and runs in `mage-lora-smoke synth`; the inertness gate runs through the
// CLI (README §Gates).

import Foundation
import MLXToolKit
import XCTest

@testable import MLXMageFlow
@testable import MageFlowEdit

final class LoRATests: XCTestCase {

    func testTrainerTargetsMapOntoPortPaths() {
        let cases: [(String, String)] = [
            ("diffusion_model.transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_q"),
            ("diffusion_model.transformer_blocks.11.attn.add_v_proj", "transformer_blocks.11.attn.add_v_proj"),
            ("diffusion_model.transformer_blocks.3.attn.to_out.0", "transformer_blocks.3.attn.to_out.0"),
            ("diffusion_model.transformer_blocks.3.attn.to_add_out", "transformer_blocks.3.attn.to_add_out"),
            ("diffusion_model.transformer_blocks.5.img_mlp.net.0.proj", "transformer_blocks.5.img_mlp.proj_in"),
            ("diffusion_model.transformer_blocks.5.img_mlp.net.2", "transformer_blocks.5.img_mlp.proj_out"),
            ("diffusion_model.transformer_blocks.5.txt_mlp.net.0.proj", "transformer_blocks.5.txt_mlp.proj_in"),
            ("diffusion_model.transformer_blocks.5.txt_mlp.net.2", "transformer_blocks.5.txt_mlp.proj_out"),
            // accepted-if-present modulation Linears and alternate prefixes
            ("diffusion_model.transformer_blocks.2.img_mod.1", "transformer_blocks.2.img_mod"),
            ("transformer.transformer_blocks.2.txt_mod.1", "transformer_blocks.2.txt_mod"),
            ("base_model.model.diffusion_model.transformer_blocks.9.attn.to_k", "transformer_blocks.9.attn.to_k"),
            ("transformer_blocks.9.attn.to_v", "transformer_blocks.9.attn.to_v"),
        ]
        for (src, want) in cases {
            XCTAssertEqual(MageLoRA.expand(base: src), want, src)
        }
    }

    func testNonTargetsAreRejectedNotSkipped() {
        let bad = [
            "diffusion_model.transformer_blocks.0.attn.to_qkv",      // fused qkv is klein's dialect, not Mage's
            "diffusion_model.transformer_blocks.12.attn.to_q",       // block out of range (depth 12)
            "diffusion_model.single_transformer_blocks.0.attn.to_q", // Mage has no single-stream blocks
            "diffusion_model.img_in",                                // top-level io Linear
            "diffusion_model.transformer_blocks.0.img_norm1",        // not a Linear
            "text_encoder.layers.0.self_attn.q_proj",                // wrong component entirely
        ]
        for src in bad { XCTAssertNil(MageLoRA.expand(base: src), src) }
    }

    func testTrainedTargetCount() {
        // attn(8) + MLP(4) per block × 12 blocks = 144 modules — the trainer's expected result.
        let perBlock = MageLoRA.blockMap.values.filter { !$0.hasSuffix("_mod") }
        XCTAssertEqual(Set(perBlock).count, MageLoRA.trainedTargetsPerBlock)
    }

    func testLoraPathIsRuntimeOnly() throws {
        let cfg = MageFlowConfiguration(
            variant: .editBase, quant: .int8, loraPath: "/tmp/pose.safetensors", loraStrength: 0.8)
        let data = try JSONEncoder().encode(cfg)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("loraPath"), "loraPath must not persist")
        let back = try JSONDecoder().decode(MageFlowConfiguration.self, from: data)
        XCTAssertNil(back.loraPath)
        XCTAssertEqual(back.loraStrength, 1.0)
        XCTAssertEqual(back.variant, .editBase)
    }
}
