// Standalone Mage-Flow-Edit pipeline: image + instruction -> edited image.
//
// Ties together the parity-locked numeric parts (MageFlow DiT, MageVAE, GS noise)
// with live Qwen3-VL conditioning and the mandatory content filter from
// qwen3vl-mlx-swift.
//
// Pipeline (mirrors upstream `generate_edits`, Turbo path), for 1…3 references:
//   1. content filter (AR classifier) over ALL refs + instruction — fail-closed
//   2. prompt template + edit body `Image 1: <ph>Image 2: <ph>…{instruction}`,
//      tokenize, expand the k-th <|image_pad|> per the k-th ref's grid
//   3. Qwen3-VL forward with every ref at 384px long-edge -> last_hidden_state,
//      slice off the first start_idx (64) tokens        [FLAT positionIds]
//   4. MageVAE-encode every ref at the full target resolution -> ref latents
//      (output size comes from the request; upstream derives it from the FIRST ref)
//   5. Gaussian-Shading watermarked target noise
//   6. pack [target, ref_1 … ref_N], 4-step denoise (target stepped, refs clean);
//      the RoPE frame index j is the only thing telling ref_j from the target
//   7. MageVAE-decode the target -> RGB
//
// Multi-reference (AB-A-0047, 2026-09-02): upstream trains with up to 3 refs and
// the mechanism is exactly the single-ref one with an N loop — extra clean latent
// tokens in the same attention sequence, disambiguated by frame index + a per-image
// placeholder in the VL prompt. `pipeline.py:390-393` (placeholders), `:500-518`
// (all refs VAE-encoded at target size, shape_seq frame idx j per ref).
//
// Parity hooks (env, inert otherwise): MAGEFLOW_DUMP_FEATS=<path.npy> saves the
// conditioning features for a cosine check vs the oracle's txt_norm input;
// MAGEFLOW_DUMP_FILTER=<dir> saves the filter's ids + spatial M-RoPE positions for
// a check vs HF get_rope_index (both used by the 2026-09-02 multi-ref gate).

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

/// One reference image, tightly-packed RGB8, row-major — the engine-facing unit.
public struct MageRefImage: Sendable {
    public var rgb: [UInt8]
    public var width: Int
    public var height: Int
    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb; self.width = width; self.height = height
    }
}

public struct MageFlowEditConfig {
    /// Upstream's validated multi-reference range: trained with up to 3 refs
    /// (`generate_edits` docstring — "trained with up to 3, but more are accepted").
    /// The port stops at the trained range; beyond it is untested, not unsupported.
    public static let maxReferences = 3
    public var steps = 4
    public var shift: Float = 6.0
    public var size = 512               // target square side (floored to /16)
    public var vlCondLongEdge = 384     // VL sees a 384px long-edge ref (training match)
    public var gsKey: UInt64 = 20_260_720
    public var seed: UInt64 = 42
    public var startIdx = 64            // "mage-flow-edit" template: drop 64 tokens
    public var t2iStartIdx = 34         // "mage-flow" template: drop 34 tokens
    public var cfg: Float = 1.0         // Base 5.0 / RL 5.0 / Turbo 1.0
    public var negPrompt = " "          // upstream default: a single SPACE (truthy)
    public var renormalization = false
    public init() {}
}

/// System + user template for "mage-flow-edit" (verbatim from models/utils.py).
public let mageFlowEditTemplate =
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture,"
    + " objects, background), then explain how the user's text instruction should alter or modify the image. "
    + "Generate a new image that meets the user's requirements while maintaining consistency with the original "
    + "input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

let visionPlaceholder = "<|vision_start|><|image_pad|><|vision_end|>"

/// "mage-flow" T2I template (verbatim from models/utils.py), start_idx = 34.
/// NOTE: no space/newline between "background:" and <|im_end|> — the 34-token
/// slice is tied to this exact text.
public let mageFlowT2ITemplate =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
    + "text, spatial relationships of the objects and background:"
    + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

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
    /// The Qwen3-VL conditioner+filter (~8.3 GB bf16). Evictable: `dropConditioner()`
    /// releases it between requests (light tier); `conditioner()` reloads on demand.
    private var vlModel: Qwen3VL?
    let textEncoderDir: URL
    let tokenizer: Tokenizers.Tokenizer
    let imageProcessor: Qwen3VLImageProcessor
    let transformer: MageFlowTransformer
    let vae: VAEWeights
    /// Mutable so an engine wrapper can apply per-request overrides (steps/size/seed/cfg).
    public var cfg: MageFlowEditConfig
    let ditDtype: DType

    public init(
        textEncoderDir: URL, transformerDir: URL, vaeSafetensors: URL, foldedAdaLN: URL,
        ditQuant: URL? = nil,
        cfg: MageFlowEditConfig = MageFlowEditConfig(),
        deferConditioner: Bool = false
    ) async throws {
        self.cfg = cfg
        self.textEncoderDir = textEncoderDir
        // load everything on the CPU stream; heavy reads shouldn't ride a GPU buffer
        if deferConditioner {
            self.vlModel = nil   // loaded lazily by conditioner() on first encode
        } else {
            self.vlModel = try Device.withDefaultDevice(.cpu) {
                try Qwen3VLLoader.load(directory: textEncoderDir, dtype: .bfloat16)
            }
        }
        self.tokenizer = try await AutoTokenizer.from(modelFolder: textEncoderDir)
        self.imageProcessor = Qwen3VLImageProcessor()

        if let ditQuant {
            // pre-quantized DiT (int4/int8): no bf16 peak; unquantized layers stay bf16
            self.transformer = try MageQuant.loadQuantizedDiT(from: ditQuant)
            self.ditDtype = .bfloat16
        } else {
            let model = MageFlowTransformer()
            var raw: [String: MLXArray] = [:]
            for f in try FileManager.default.contentsOfDirectory(
                at: transformerDir, includingPropertiesForKeys: nil
            ).filter({ $0.pathExtension == "safetensors" }) {
                raw.merge(try MLX.loadArrays(url: f)) { x, _ in x }
            }
            let keys = Set(model.parameters().flattened().map(\.0))
            // bf16 by default (upstream dtype). The former "grid garbage at >=512^2"
            // was root-caused to the mlx-swift <=0.31.6 JIT-miscompiled NAX split-K
            // GEMM (ml-explore/mlx#3797, fixed by #3810) hitting the FFN proj_out at
            // >=1366 image tokens; MageFeedForward.downProjected now row-chunks
            // around it exactly. MAGEFLOW_FP32 remains for parity work.
            let ditDtype: DType = ProcessInfo.processInfo.environment["MAGEFLOW_FP32"] != nil ? .float32 : .bfloat16
            let w = model.sanitize(weights: raw).mapValues { $0.asType(ditDtype) }
            let missing = keys.subtracting(Set(w.keys))
            guard missing.isEmpty else { throw MageFlowEditError.load("DiT missing \(missing.count) keys") }
            model.update(parameters: ModuleParameters.unflattened(w.filter { keys.contains($0.key) }))
            eval(model)
            self.transformer = model
            self.ditDtype = ditDtype
        }

        self.vae = try MageVAELoader.load(vae: vaeSafetensors, foldedAdaLN: foldedAdaLN)
    }

    // MARK: conditioner lifecycle (light tier)

    /// The VL conditioner, loading it on demand if evicted/deferred.
    func conditioner() throws -> Qwen3VL {
        if let vlModel { return vlModel }
        let loaded: Qwen3VL = try Device.withDefaultDevice(.cpu) {
            try Qwen3VLLoader.load(directory: textEncoderDir, dtype: .bfloat16)
        }
        vlModel = loaded
        return loaded
    }

    /// Release the ~8.3 GB Qwen3-VL conditioner (it reloads on the next encode).
    /// Call after a request completes to keep the resident set to DiT + VAE.
    public func dropConditioner() {
        vlModel = nil
        MLX.Memory.clearCache()
    }

    // MARK: image helpers

    static func decodeRGB(_ url: URL) throws -> ([UInt8], Int, Int) {
        guard let d = try? Data(contentsOf: url) else {
            throw MageFlowEditError.load("cannot read \(url.path)")
        }
        return try decodeRGB(data: d)
    }

    /// Decode encoded image bytes (PNG/JPEG/...) → (RGB8, width, height).
    public static func decodeRGB(data d: Data) throws -> ([UInt8], Int, Int) {
        guard let src = CGImageSourceCreateWithData(d as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw MageFlowEditError.load("cannot decode image data")
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

    /// Encode NHWC [1,H,W,3] in [-1,1] → (PNG bytes, width, height) — the engine surface.
    public static func encodePNG(_ nhwc: MLXArray) throws -> (Data, Int, Int) {
        let x = clip(nhwc[0], min: -1, max: 1)
        let u = ((x + 1) * 127.5).asType(.uint8)
        eval(u)
        let (H, W) = (u.dim(0), u.dim(1))
        let rgb = u.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: H * W * 4)
        for i in 0 ..< H * W { for c in 0 ..< 3 { rgba[i * 4 + c] = rgb[i * 3 + c] } }
        guard let cg = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8,
            bytesPerRow: W * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        else { throw MageFlowEditError.load("PNG context") }
        let out = NSMutableData()
        guard let dst = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { throw MageFlowEditError.load("PNG encode") }
        CGImageDestinationAddImage(dst, cg, nil)
        guard CGImageDestinationFinalize(dst) else { throw MageFlowEditError.load("PNG finalize") }
        return (out as Data, W, H)
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

    /// Tokenize the formatted prompt and expand the k-th single `<|image_pad|>` into
    /// `grids[k].product / mergeSize^2` copies — the HF AutoProcessor contract, which
    /// consumes `image_grid_thw` rows in placeholder order.
    func buildInputIds(formatted: String, grids: [THW]) throws -> [Int32] {
        let merge = imageProcessor.mergeSize * imageProcessor.mergeSize
        let ids = tokenizer.encode(text: formatted)
        let pad = try conditioner().config.imageTokenIndex
        var out: [Int32] = []
        out.reserveCapacity(ids.count + grids.reduce(0) { $0 + $1.product / merge })
        var k = 0
        for id in ids {
            if id == pad {
                guard k < grids.count else {
                    throw MageFlowEditError.load("prompt has more <|image_pad|> than images (\(grids.count))")
                }
                out.append(contentsOf: Array(repeating: Int32(pad), count: grids[k].product / merge))
                k += 1
            } else {
                out.append(Int32(id))
            }
        }
        guard k == grids.count else {
            throw MageFlowEditError.load("prompt has \(k) <|image_pad|> for \(grids.count) images")
        }
        return out
    }

    func buildInputIds(formatted: String, grid: THW) throws -> [Int32] {
        try buildInputIds(formatted: formatted, grids: [grid])
    }

    /// The upstream `_edit_prompt_body`: `Image 1: <ph>Image 2: <ph>…{instruction}`.
    static func editPromptBody(instruction: String, refCount: Int) -> String {
        (1 ... max(refCount, 1)).map { "Image \($0): \(visionPlaceholder)" }.joined() + instruction
    }

    // MARK: text-only conditioning (T2I)

    /// Encode a T2I prompt: template, tokenize, text-only forward, drop 34 tokens.
    func encodeT2I(_ prompt: String) throws -> MLXArray {
        let formatted = mageFlowT2ITemplate.replacingOccurrences(of: "{}", with: prompt)
        let ids = tokenizer.encode(text: formatted).map { Int32($0) }
        var feats = try conditioner().lastHiddenState(inputIds: MLXArray(ids, [1, ids.count]))
        feats = feats[0..., cfg.t2iStartIdx..., 0...].asType(.float32)
        eval(feats)
        return feats
    }

    /// T2I content filter — upstream screen_text: CONTENT_FILTER_SYSTEM, greedy
    /// generate (<=160 new tokens), fail-closed.
    func screenT2I(_ prompt: String) throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let fp = "<|im_start|>system\n\(contentFilterT2ISystem)<|im_end|>\n"
            + "<|im_start|>user\nPrompt to classify:\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        let fids = tokenizer.encode(text: fp).map { Int32($0) }
        let toks = try conditioner().generate(inputIds: MLXArray(fids, [1, fids.count]), maxTokens: 160)
        let verdict = tokenizer.decode(tokens: toks.map { Int($0) })
            .replacingOccurrences(of: " ", with: "")
        if verdict.lowercased().contains("\"violates\":true") {
            throw MageFlowEditError.refused(verdict)
        }
    }

    /// Text-to-image (Mage-Flow / -Base / -Turbo). Returns NHWC [1,H,W,3] in [-1,1].
    public func t2i(prompt: String, screen: Bool = true,
                    shouldStop: (() -> Bool)? = nil) throws -> MLXArray {
        if screen {
            MageProgress.report(.screen)
            try screenT2I(prompt)
        }
        if shouldStop?() == true { throw CancellationError() }   // post-screen seam
        let side = (cfg.size / 16) * 16
        let (lh, lw) = (side / 16, side / 16)

        MageProgress.report(.encode)
        let feats = try encodeT2I(prompt)
        let useCFG = cfg.cfg > 1.0 && !cfg.negPrompt.isEmpty
        let negFeats = useCFG ? try encodeT2I(cfg.negPrompt) : nil

        let noise = gaussianShadingNoise(
            channels: 128, height: lh, width: lw, key: cfg.gsKey, seed: cfg.seed)
        let img0 = MLXArray(noise, [1, 128, lh, lw]).transposed(0, 2, 3, 1)
            .reshaped(1, lh * lw, 128).asType(ditDtype)

        let sched = FlowMatchEulerScheduler(steps: cfg.steps, shift: cfg.shift)
        let pipe = MageFlowPipeline(transformer: transformer)
        // whole sequence is the target; single shape entry, frame idx 0
        let out = pipe.denoise(
            img: img0, txt: feats.asType(ditDtype), targetLen: lh * lw,
            imgShapes: [(frame: 1, height: lh, width: lw)], scheduler: sched,
            negTxt: negFeats.map { $0.asType(ditDtype) }, cfg: cfg.cfg,
            renormalization: cfg.renormalization, shouldStop: shouldStop)
        if shouldStop?() == true { throw CancellationError() }   // pre-decode seam
        let latent = out.reshaped(1, lh, lw, 128).asType(.float32)
        eval(latent)
        MageProgress.report(.decode)
        let img = vaeDecode(latent, vae)
        eval(img)
        return img
    }

    // MARK: the pipeline

    /// Returns the edited RGB as NHWC [1,H,W,3] in [-1,1]. `screen` runs the
    /// content filter; on refusal throws `.refused`.
    public func edit(refImage: URL, instruction: String, screen: Bool = true,
                     shouldStop: (() -> Bool)? = nil) throws -> MLXArray {
        try edit(refImages: [refImage], instruction: instruction, screen: screen, shouldStop: shouldStop)
    }

    /// File entry for 1…`maxReferences` references, in prompt order (Image 1, Image 2, …).
    public func edit(refImages: [URL], instruction: String, screen: Bool = true,
                     shouldStop: (() -> Bool)? = nil) throws -> MLXArray {
        let refs = try refImages.map { url -> MageRefImage in
            let (rgb, w, h) = try Self.decodeRGB(url)
            return MageRefImage(rgb: rgb, width: w, height: h)
        }
        return try edit(refs: refs, instruction: instruction, screen: screen, shouldStop: shouldStop)
    }

    /// Bytes entry (engine surface): `refRGB` is tightly-packed RGB8, row-major.
    public func edit(refRGB rgb: [UInt8], width iw: Int, height ih: Int,
                     instruction: String, screen: Bool = true,
                     shouldStop: (() -> Bool)? = nil) throws -> MLXArray {
        try edit(refs: [MageRefImage(rgb: rgb, width: iw, height: ih)], instruction: instruction,
                 screen: screen, shouldStop: shouldStop)
    }

    /// The general entry: 1…`MageFlowEditConfig.maxReferences` references in prompt
    /// order. Ref 1 is the PRIMARY reference (upstream derives the output size from
    /// it; here the caller's `cfg.size` is authoritative and every ref is VAE-encoded
    /// at that size, exactly as upstream does once the size is fixed).
    public func edit(refs: [MageRefImage], instruction: String, screen: Bool = true,
                     shouldStop: (() -> Bool)? = nil) throws -> MLXArray {
        guard !refs.isEmpty else { throw MageFlowEditError.load("edit needs at least one reference image") }
        guard refs.count <= MageFlowEditConfig.maxReferences else {
            throw MageFlowEditError.load(
                "\(refs.count) reference images — the trained/validated range is 1…\(MageFlowEditConfig.maxReferences)")
        }
        let side = (cfg.size / 16) * 16
        let n = refs.count

        // --- VL conditioning images at 384px long edge --------------------
        // Must be PIL BICUBIC, and preprocess() then PIL-resizes AGAIN internally
        // (two-BICUBIC path, exactly as the oracle's _resize_long_edge + processor
        // smart_resize). CoreGraphics resampling here corrupts the resampling-
        // sensitive ViT features (cos 0.93 -> garbage edit).
        // Multi-image: the processor concatenates every image's patches along axis 0
        // and carries one grid row per image — the vision tower isolates images by
        // cu_seqlens and the LM scatters features into the placeholders in order.
        var condPixels: [MLXArray] = []
        var condGrids: [THW] = []
        for r in refs {
            let longEdge = max(r.width, r.height)
            let (vw, vh): (Int, Int)
            if longEdge > cfg.vlCondLongEdge {
                let scale = Double(cfg.vlCondLongEdge) / Double(longEdge)
                vw = max(1, Int((Double(r.width) * scale).rounded()))
                vh = max(1, Int((Double(r.height) * scale).rounded()))
            } else {
                (vw, vh) = (r.width, r.height)
            }
            let vlRGB = PILResize.resize(rgb: r.rgb, width: r.width, height: r.height,
                                         outWidth: vw, outHeight: vh)
            let (pv, grid) = imageProcessor.preprocess(rgb: vlRGB, width: vw, height: vh)
            condPixels.append(pv)
            condGrids.append(grid)
        }
        let pixelValues = condPixels.count == 1 ? condPixels[0] : concatenated(condPixels, axis: 0)

        // --- content filter (mandatory, fail-closed) ----------------------
        // Mirrors upstream screen_edit: real CONTENT_FILTER_EDIT_SYSTEM, the
        // exact user message, greedy generate, JSON verdict. The filter uses REAL
        // spatial M-RoPE (no flat override) — only the conditioning path is flat.
        // ⚠ The filter sees the ORIGINAL-resolution images (upstream pipeline.py
        // screens BEFORE the 384px _resize_long_edge, which is conditioning-only).
        // Reusing the 384px conditioning grid here changed the vision-token grid
        // enough to flip borderline verdicts (found live: a benign anime-style
        // edit refused with a self-contradictory rambling reason; upstream torch
        // passes the same input cleanly).
        // Multi-image: the chat template renders one placeholder per image, back to
        // back, then the text — "There are N source image(s) above." (screen_edit).
        if screen {
            MageProgress.report(.screen)
            var fPixels: [MLXArray] = []
            var fGrids: [THW] = []
            for r in refs {
                let (pv, g) = imageProcessor.preprocess(rgb: r.rgb, width: r.width, height: r.height)
                fPixels.append(pv); fGrids.append(g)
            }
            let filterPixels = fPixels.count == 1 ? fPixels[0] : concatenated(fPixels, axis: 0)
            let instr = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
            let userText = "There \(n == 1 ? "is" : "are") \(n) source image(s) above. Edit instruction: "
                + (instr.isEmpty ? "(no textual instruction)" : instr)
                + "\nClassify this edit request."
            let filterPrompt =
                "<|im_start|>system\n\(contentFilterEditSystem)<|im_end|>\n"
                + "<|im_start|>user\n\(String(repeating: visionPlaceholder, count: n))\(userText)<|im_end|>\n"
                + "<|im_start|>assistant\n"
            let fids = try buildInputIds(formatted: filterPrompt, grids: fGrids)
            // Parity hook: MAGEFLOW_DUMP_FILTER=<dir> saves the filter's input ids and the
            // spatial M-RoPE positions the LM will use, for a check against HF get_rope_index.
            if let dumpDir = ProcessInfo.processInfo.environment["MAGEFLOW_DUMP_FILTER"] {
                let d = URL(fileURLWithPath: dumpDir)
                try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
                let dbg = try conditioner().debugEdit(
                    inputIds: MLXArray(fids, [1, fids.count]), pixelValues: filterPixels, imageGridTHW: fGrids)
                try MLX.save(array: MLXArray(fids, [1, fids.count]), url: d.appendingPathComponent("ids.npy"))
                try MLX.save(array: dbg.positionIds, url: d.appendingPathComponent("pos.npy"))
                try MLX.save(array: dbg.merged.asType(.float32), url: d.appendingPathComponent("merged.npy"))
                let grids = fGrids.map { "\($0.t),\($0.h),\($0.w)" }.joined(separator: ";")
                try grids.write(to: d.appendingPathComponent("grids.txt"), atomically: true, encoding: .utf8)
            }
            let verdictTokens = try conditioner().generate(
                inputIds: MLXArray(fids, [1, fids.count]),
                pixelValues: filterPixels, imageGridTHW: fGrids, maxTokens: 192)
            let verdict = tokenizer.decode(tokens: verdictTokens.map { Int($0) })
                .replacingOccurrences(of: " ", with: "")
            if verdict.lowercased().contains("\"violates\":true") {
                throw MageFlowEditError.refused(verdict)
            }
        }
        if shouldStop?() == true { throw CancellationError() }   // post-screen seam

        // --- conditioning features ----------------------------------------
        // Mage-Flow feeds FLAT positions — M-RoPE degenerates to 1-D.
        func encodeEdit(_ instr: String) throws -> MLXArray {
            let body = Self.editPromptBody(instruction: instr, refCount: n)
            let formatted = mageFlowEditTemplate.replacingOccurrences(of: "{}", with: body)
            let ids = try buildInputIds(formatted: formatted, grids: condGrids)
            let flat = Qwen3VL.flatPositionIds(sequenceLength: ids.count)
            var f = try conditioner().lastHiddenState(
                inputIds: MLXArray(ids, [1, ids.count]), pixelValues: pixelValues,
                imageGridTHW: condGrids, positionIds: flat)
            // slice off the first start_idx tokens (system preamble)
            f = f[0..., cfg.startIdx..., 0...].asType(.float32)
            eval(f)
            return f
        }
        MageProgress.report(.encode)
        let feats = try encodeEdit(instruction)
        // Parity hook: MAGEFLOW_DUMP_FEATS=<path.npy> saves the conditioning features
        // (post start_idx slice) for a cosine check against the oracle's txt_norm input.
        if let dump = ProcessInfo.processInfo.environment["MAGEFLOW_DUMP_FEATS"] {
            try MLX.save(array: feats, url: URL(fileURLWithPath: dump))
        }
        // CFG (Base/RL): negative pass encodes the SAME ref images with the
        // negative instruction (upstream: edit_refs + edit_refs, pos + neg).
        let useCFG = cfg.cfg > 1.0 && !cfg.negPrompt.isEmpty
        let negFeats = useCFG ? try encodeEdit(cfg.negPrompt) : nil

        // --- ref latents (every ref at the full target resolution) --------
        var refLatents: [MLXArray] = []
        for r in refs {
            let refPx = Self.resize(r.rgb, r.width, r.height, side, side)
            var arr = [Float](repeating: 0, count: side * side * 3)
            for i in 0 ..< side * side * 3 { arr[i] = Float(refPx[i]) / 127.5 - 1 }
            let refImg = MLXArray(arr, [1, side, side, 3])
            let lat = vaeEncode(refImg, vae, samplePosterior: false)
            eval(lat)
            refLatents.append(lat)
        }
        let (lh, lw) = (refLatents[0].dim(1), refLatents[0].dim(2))

        // --- Gaussian-Shading target noise --------------------------------
        let noise = gaussianShadingNoise(
            channels: 128, height: lh, width: lw, key: cfg.gsKey, seed: cfg.seed)
        let target = MLXArray(noise, [1, 128, lh, lw]).transposed(0, 2, 3, 1)

        // --- pack + denoise -----------------------------------------------
        // [target, ref_1 … ref_N]; one shape entry per member — the enumeration
        // index IS the RoPE frame axis (target 0, ref_j = j).
        let packed = concatenated(
            [target.reshaped(1, lh * lw, 128)] + refLatents.map { $0.reshaped(1, lh * lw, 128) },
            axis: 1).asType(ditDtype)
        let shapes = Array(repeating: (frame: 1, height: lh, width: lw), count: n + 1)
        let sched = FlowMatchEulerScheduler(steps: cfg.steps, shift: cfg.shift)
        let pipe = MageFlowPipeline(transformer: transformer)
        let out = pipe.denoise(img: packed, txt: feats.asType(ditDtype),
                               targetLen: lh * lw, imgShapes: shapes, scheduler: sched,
                               negTxt: negFeats.map { $0.asType(ditDtype) }, cfg: cfg.cfg,
                               renormalization: cfg.renormalization, shouldStop: shouldStop)
        if shouldStop?() == true { throw CancellationError() }   // pre-decode seam
        let targetLatent = out[0..., ..<(lh * lw), 0...].reshaped(1, lh, lw, 128).asType(.float32)
        eval(targetLatent)

        // --- decode -------------------------------------------------------
        MageProgress.report(.decode)
        let img = vaeDecode(targetLatent, vae)
        eval(img)
        return img
    }
}
