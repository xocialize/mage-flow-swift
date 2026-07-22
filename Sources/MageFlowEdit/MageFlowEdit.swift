// Standalone Mage-Flow-Edit pipeline: image + instruction -> edited image.
//
// Ties together the parity-locked numeric parts (MageFlow DiT, MageVAE, GS noise)
// with live Qwen3-VL conditioning and the mandatory content filter from
// qwen3vl-mlx-swift.
//
// Pipeline (mirrors upstream `generate_edits`, Turbo path):
//   1. content filter (AR classifier) — fail-closed, white refusal image
//   2. prompt template + edit body, tokenize, expand <|image_pad|> per grid
//   3. Qwen3-VL forward with the ref at 384px long-edge -> last_hidden_state,
//      slice off the first start_idx (64) tokens        [FLAT positionIds]
//   4. MageVAE-encode the ref at full target resolution -> ref latent
//   5. Gaussian-Shading watermarked target noise
//   6. pack [target, ref], 4-step denoise (target stepped, ref clean)
//   7. MageVAE-decode the target -> RGB

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import MLXNN
import MageFlow
import Qwen3VL
import Tokenizers
import UniformTypeIdentifiers

public struct MageFlowEditConfig {
    public var steps = 4
    public var shift: Float = 6.0
    public var size = 512               // target square side (floored to /16)
    public var vlCondLongEdge = 384     // VL sees a 384px long-edge ref (training match)
    public var gsKey: UInt64 = 20_260_720
    public var seed: UInt64 = 42
    public var startIdx = 64            // "mage-flow-edit" template: drop 64 tokens
    public init() {}
}

/// System + user template for "mage-flow-edit" (verbatim from models/utils.py).
public let mageFlowEditTemplate =
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture,"
    + " objects, background), then explain how the user's text instruction should alter or modify the image. "
    + "Generate a new image that meets the user's requirements while maintaining consistency with the original "
    + "input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

let visionPlaceholder = "<|vision_start|><|image_pad|><|vision_end|>"

public enum MageFlowEditError: Error, CustomStringConvertible {
    case refused(String)
    case load(String)
    public var description: String {
        switch self {
        case .refused(let s): return "content filter refused: \(s)"
        case .load(let s): return "load error: \(s)"
        }
    }
}

public final class MageFlowEditPipeline {
    let vl: Qwen3VL
    let tokenizer: Tokenizers.Tokenizer
    let imageProcessor: Qwen3VLImageProcessor
    let transformer: MageFlowTransformer
    let vae: VAEWeights
    let cfg: MageFlowEditConfig
    let ditDtype: DType

    public init(
        textEncoderDir: URL, transformerDir: URL, vaeSafetensors: URL, foldedAdaLN: URL,
        cfg: MageFlowEditConfig = MageFlowEditConfig()
    ) async throws {
        self.cfg = cfg
        // load everything on the CPU stream; heavy reads shouldn't ride a GPU buffer
        let loadedVL: Qwen3VL = try Device.withDefaultDevice(.cpu) {
            try Qwen3VLLoader.load(directory: textEncoderDir, dtype: .bfloat16)
        }
        self.vl = loadedVL
        self.tokenizer = try await AutoTokenizer.from(modelFolder: textEncoderDir)
        self.imageProcessor = Qwen3VLImageProcessor()

        let model = MageFlowTransformer()
        var raw: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: transformerDir, includingPropertiesForKeys: nil
        ).filter({ $0.pathExtension == "safetensors" }) {
            raw.merge(try MLX.loadArrays(url: f)) { x, _ in x }
        }
        let keys = Set(model.parameters().flattened().map(\.0))
        // The DiT runs in fp32 on the edit path. bf16 weights load fine, but the
        // MLX-Swift bf16 attention/GEMM produces grid garbage at >=512^2 on Apple
        // GPUs (same failure Boogu-Image hits -> useFP32DiT: true; upstream runs
        // bf16 on CUDA cleanly, so this is a Metal-kernel issue, not the math).
        // Opt back into bf16 with MAGEFLOW_BF16 once a surgical fix (fp32 SDPA
        // only, or per-block eval) lands.
        let ditDtype: DType = ProcessInfo.processInfo.environment["MAGEFLOW_BF16"] != nil ? .bfloat16 : .float32
        let w = model.sanitize(weights: raw).mapValues { $0.asType(ditDtype) }
        let missing = keys.subtracting(Set(w.keys))
        guard missing.isEmpty else { throw MageFlowEditError.load("DiT missing \(missing.count) keys") }
        model.update(parameters: ModuleParameters.unflattened(w.filter { keys.contains($0.key) }))
        eval(model)
        self.transformer = model
        self.ditDtype = ditDtype

        self.vae = try MageVAELoader.load(vae: vaeSafetensors, foldedAdaLN: foldedAdaLN)
    }

    // MARK: image helpers

    static func decodeRGB(_ url: URL) throws -> ([UInt8], Int, Int) {
        guard let d = try? Data(contentsOf: url),
              let src = CGImageSourceCreateWithData(d as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw MageFlowEditError.load("cannot decode \(url.path)")
        }
        let (w, h) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0 ..< w * h { for c in 0 ..< 3 { rgb[i * 3 + c] = rgba[i * 4 + c] } }
        return (rgb, w, h)
    }

    static func resize(_ rgb: [UInt8], _ w: Int, _ h: Int, _ ow: Int, _ oh: Int) -> [UInt8] {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0 ..< w * h { for c in 0 ..< 3 { rgba[i * 4 + c] = rgb[i * 3 + c] } }
        let cg = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        var out = [UInt8](repeating: 255, count: ow * oh * 4)
        let octx = CGContext(data: &out, width: ow, height: oh, bitsPerComponent: 8, bytesPerRow: ow * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        octx.interpolationQuality = .high
        octx.draw(cg, in: CGRect(x: 0, y: 0, width: ow, height: oh))
        var o = [UInt8](repeating: 0, count: ow * oh * 3)
        for i in 0 ..< ow * oh { for c in 0 ..< 3 { o[i * 3 + c] = out[i * 4 + c] } }
        return o
    }

    public static func savePNG(_ nhwc: MLXArray, to url: URL) {
        let x = clip(nhwc[0], min: -1, max: 1)
        let u = ((x + 1) * 127.5).asType(.uint8)
        eval(u)
        let (H, W) = (u.dim(0), u.dim(1))
        let rgb = u.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: H * W * 4)
        for i in 0 ..< H * W { for c in 0 ..< 3 { rgba[i * 4 + c] = rgb[i * 3 + c] } }
        let cg = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let dst = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, cg, nil)
        CGImageDestinationFinalize(dst)
    }

    // MARK: tokenization

    /// Tokenize the formatted prompt and expand each single `<|image_pad|>` into
    /// `grid.product / mergeSize^2` copies — the HF AutoProcessor contract.
    func buildInputIds(formatted: String, grid: THW) -> [Int32] {
        let merge = imageProcessor.mergeSize * imageProcessor.mergeSize
        let nImage = grid.product / merge
        let ids = tokenizer.encode(text: formatted)
        let pad = vl.config.imageTokenIndex
        var out: [Int32] = []
        out.reserveCapacity(ids.count + nImage)
        for id in ids {
            if id == pad {
                out.append(contentsOf: Array(repeating: Int32(pad), count: nImage))
            } else {
                out.append(Int32(id))
            }
        }
        return out
    }

    // MARK: the pipeline

    /// Returns the edited RGB as NHWC [1,H,W,3] in [-1,1]. `screen` runs the
    /// content filter; on refusal throws `.refused`.
    public func edit(refImage: URL, instruction: String, screen: Bool = true) throws -> MLXArray {
        let side = (cfg.size / 16) * 16
        let (rgb, iw, ih) = try Self.decodeRGB(refImage)

        // --- VL conditioning image at 384px long edge ---------------------
        // Must be PIL BICUBIC, and preprocess() then PIL-resizes AGAIN internally
        // (two-BICUBIC path, exactly as the oracle's _resize_long_edge + processor
        // smart_resize). CoreGraphics resampling here corrupts the resampling-
        // sensitive ViT features (cos 0.93 -> garbage edit).
        let longEdge = max(iw, ih)
        let (vw, vh): (Int, Int)
        if longEdge > cfg.vlCondLongEdge {
            let scale = Double(cfg.vlCondLongEdge) / Double(longEdge)
            vw = max(1, Int((Double(iw) * scale).rounded()))
            vh = max(1, Int((Double(ih) * scale).rounded()))
        } else {
            (vw, vh) = (iw, ih)
        }
        let vlRGB = PILResize.resize(rgb: rgb, width: iw, height: ih, outWidth: vw, outHeight: vh)
        let (pixelValues, grid) = imageProcessor.preprocess(rgb: vlRGB, width: vw, height: vh)

        // --- content filter (mandatory, fail-closed) ----------------------
        // Mirrors upstream screen_edit: real CONTENT_FILTER_EDIT_SYSTEM, the
        // exact user message, greedy generate, JSON verdict. The filter uses REAL
        // spatial M-RoPE (no flat override) — only the conditioning path is flat.
        if screen {
            let instr = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
            let userText = "There is 1 source image(s) above. Edit instruction: "
                + (instr.isEmpty ? "(no textual instruction)" : instr)
                + "\nClassify this edit request."
            let filterPrompt =
                "<|im_start|>system\n\(contentFilterEditSystem)<|im_end|>\n"
                + "<|im_start|>user\n\(visionPlaceholder)\(userText)<|im_end|>\n"
                + "<|im_start|>assistant\n"
            let fids = buildInputIds(formatted: filterPrompt, grid: grid)
            let verdictTokens = try vl.generate(
                inputIds: MLXArray(fids, [1, fids.count]),
                pixelValues: pixelValues, imageGridTHW: [grid], maxTokens: 192)
            let verdict = tokenizer.decode(tokens: verdictTokens.map { Int($0) })
                .replacingOccurrences(of: " ", with: "")
            if verdict.lowercased().contains("\"violates\":true") {
                throw MageFlowEditError.refused(verdict)
            }
        }

        // --- conditioning features ----------------------------------------
        let body = "Image 1: \(visionPlaceholder)\(instruction)"
        let formatted = mageFlowEditTemplate.replacingOccurrences(of: "{}", with: body)
        let ids = buildInputIds(formatted: formatted, grid: grid)
        let inputIds = MLXArray(ids, [1, ids.count])
        // Mage-Flow feeds FLAT positions — M-RoPE degenerates to 1-D.
        let flat = Qwen3VL.flatPositionIds(sequenceLength: ids.count)
        var feats = try vl.lastHiddenState(
            inputIds: inputIds, pixelValues: pixelValues, imageGridTHW: [grid], positionIds: flat)
        // slice off the first start_idx tokens (system preamble)
        feats = feats[0..., cfg.startIdx..., 0...].asType(.float32)
        eval(feats)

        // --- ref latent (full target resolution) --------------------------
        let refPx = Self.resize(rgb, iw, ih, side, side)
        var arr = [Float](repeating: 0, count: side * side * 3)
        for i in 0 ..< side * side * 3 { arr[i] = Float(refPx[i]) / 127.5 - 1 }
        let refImg = MLXArray(arr, [1, side, side, 3])
        let refLatent = vaeEncode(refImg, vae, samplePosterior: false)
        eval(refLatent)
        let (lh, lw) = (refLatent.dim(1), refLatent.dim(2))

        // --- Gaussian-Shading target noise --------------------------------
        let noise = gaussianShadingNoise(
            channels: 128, height: lh, width: lw, key: cfg.gsKey, seed: cfg.seed)
        let target = MLXArray(noise, [1, 128, lh, lw]).transposed(0, 2, 3, 1)

        // --- pack + denoise -----------------------------------------------
        let packed = concatenated([target.reshaped(1, lh * lw, 128),
                                   refLatent.reshaped(1, lh * lw, 128)], axis: 1).asType(ditDtype)
        let shapes = [(frame: 1, height: lh, width: lw), (frame: 1, height: lh, width: lw)]
        let sched = FlowMatchEulerScheduler(steps: cfg.steps, shift: cfg.shift)
        let pipe = MageFlowPipeline(transformer: transformer)
        let out = pipe.denoise(img: packed, txt: feats.asType(ditDtype),
                               targetLen: lh * lw, imgShapes: shapes, scheduler: sched)
        let targetLatent = out[0..., ..<(lh * lw), 0...].reshaped(1, lh, lw, 128).asType(.float32)
        eval(targetLatent)

        // --- decode -------------------------------------------------------
        let img = vaeDecode(targetLatent, vae)
        eval(img)
        return img
    }
}
