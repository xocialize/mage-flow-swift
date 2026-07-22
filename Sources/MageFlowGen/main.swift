// Assembly validation: real ref image -> VAE encode -> GS-noise target -> pack ->
// DiT denoise (4-step Turbo) -> VAE decode -> PNG.
//
//   MageFlowGen <transformerDir> <vaeSafetensors> <foldedAdaLN> <vlFeatures.npy> <refImage> <out.png>
//
// Exercises the full generative numeric path assembled end-to-end with the
// parity-locked parts (VAE 1.08e-5, DiT 6.8e-6, GS noise bit-exact, denoise
// 2.8e-2). The ONLY seam not exercised here is live Qwen3-VL tokenization, which
// qwen3vl-mlx-swift owns and is load-tested at 4B separately; captured VL
// features stand in for it. A coherent dog->snowy-forest edit validates the whole.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MageFlow
import UniformTypeIdentifiers

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
let a = Array(CommandLine.arguments.dropFirst())
guard a.count >= 6 else {
    err("usage: MageFlowGen <transformerDir> <vae.st> <foldedAdaLN> <vl.npy> <ref> <out.png>")
    exit(2)
}
Device.setDefault(device: Device(.cpu))

func decodeRGB(_ url: URL) -> ([UInt8], Int, Int)? {
    guard let d = try? Data(contentsOf: url),
          let src = CGImageSourceCreateWithData(d as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let (w, h) = (cg.width, cg.height)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
        bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0 ..< w * h { for c in 0 ..< 3 { rgb[i * 3 + c] = rgba[i * 4 + c] } }
    return (rgb, w, h)
}

func resizeSquare(_ rgb: [UInt8], _ w: Int, _ h: Int, _ s: Int) -> [UInt8] {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0 ..< w * h { for c in 0 ..< 3 { rgba[i * 4 + c] = rgb[i * 3 + c] } }
    let ctx0 = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let cg = ctx0.makeImage()!
    var out = [UInt8](repeating: 255, count: s * s * 4)
    let octx = CGContext(data: &out, width: s, height: s, bitsPerComponent: 8, bytesPerRow: s * 4,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    octx.interpolationQuality = .high
    octx.draw(cg, in: CGRect(x: 0, y: 0, width: s, height: s))
    var o = [UInt8](repeating: 0, count: s * s * 3)
    for i in 0 ..< s * s { for c in 0 ..< 3 { o[i * 3 + c] = out[i * 4 + c] } }
    return o
}

func savePNG(_ nhwc: MLXArray, _ url: URL) {
    let x = clip(nhwc[0], min: -1, max: 1)
    let u = ((x + 1) * 127.5).asType(.uint8)
    eval(u)
    let (H, W) = (u.dim(0), u.dim(1))
    let rgb = u.asArray(UInt8.self)
    var rgba = [UInt8](repeating: 255, count: H * W * 4)
    for i in 0 ..< H * W { for c in 0 ..< 3 { rgba[i * 4 + c] = rgb[i * 3 + c] } }
    let ctx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let cg = ctx.makeImage()!
    let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, cg, nil)
    CGImageDestinationFinalize(dst)
}

guard let (rgb, iw, ih) = decodeRGB(URL(fileURLWithPath: a[4])) else { err("bad ref image"); exit(1) }
let side = 512
let px = resizeSquare(rgb, iw, ih, side)
var pf = [Float](repeating: 0, count: side * side * 3)
for i in 0 ..< side * side * 3 { pf[i] = Float(px[i]) / 127.5 - 1 }
let refImg = MLXArray(pf, [1, side, side, 3])

let w = try MageVAELoader.load(vae: URL(fileURLWithPath: a[1]), foldedAdaLN: URL(fileURLWithPath: a[2]))
let refLatent = vaeEncode(refImg, w, samplePosterior: false)
eval(refLatent)
let (lh, lw) = (refLatent.dim(1), refLatent.dim(2))
err("[gen] ref latent \(refLatent.shape)")

let noise = gaussianShadingNoise(channels: 128, height: lh, width: lw, key: 20_260_720, seed: 42)
let target = MLXArray(noise, [1, 128, lh, lw]).transposed(0, 2, 3, 1)

let packed = concatenated([target.reshaped(1, lh * lw, 128),
                           refLatent.reshaped(1, lh * lw, 128)], axis: 1)
let vl = try MLX.loadArray(url: URL(fileURLWithPath: a[3])).asType(.float32)
let shapes = [(frame: 1, height: lh, width: lw), (frame: 1, height: lh, width: lw)]

let model = MageFlowTransformer()
var raw: [String: MLXArray] = [:]
for f in try FileManager.default.contentsOfDirectory(
    at: URL(fileURLWithPath: a[0]), includingPropertiesForKeys: nil)
    .filter({ $0.pathExtension == "safetensors" }) {
    raw.merge(try MLX.loadArrays(url: f)) { x, _ in x }
}
let keys = Set(model.parameters().flattened().map(\.0))
model.update(parameters: ModuleParameters.unflattened(
    model.sanitize(weights: raw).mapValues { $0.asType(.float32) }.filter { keys.contains($0.key) }))
eval(model)
err("[gen] model loaded, denoising...")

let sched = FlowMatchEulerScheduler(steps: 4, shift: 6.0)
let pipe = MageFlowPipeline(transformer: model)
let out = pipe.denoise(img: packed, txt: vl, targetLen: lh * lw, imgShapes: shapes, scheduler: sched)
let targetLatent = out[0..., ..<(lh * lw), 0...].reshaped(1, lh, lw, 128)
eval(targetLatent)

let img = vaeDecode(targetLatent, w)
eval(img)
savePNG(img, URL(fileURLWithPath: a[5]))
err("[gen] wrote \(a[5])")
