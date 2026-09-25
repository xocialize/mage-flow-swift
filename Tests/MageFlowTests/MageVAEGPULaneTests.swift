// MageVAE on the GPU lane (conditioning-decoder route conv3d / raw Winograd) vs the CPU lane.
//
// The oracle golden (mage-flow-oracle/goldens/vae_goldens.safetensors) is 512², where the latent
// is 32² — below mlx's Winograd window (N·H·W ≥ 4096) — so it cannot see this. The window opens
// at 1024² output (latent 64²). Input: a real DIV2K photo, center-cropped to 1024², encoded on the
// CPU lane; the GPU decode of that latent (and its `codDecoder` conditioning) is compared with the
// CPU-lane decode. `testProbe` is weight-free.
//
// Run: swift test --filter MageVAEGPULaneTests/testProbe
//      MAGE_PARITY=1 MAGE_VAE=<vae/diffusion_pytorch_model.safetensors> \
//      MAGE_FOLD=<folded_adaln.safetensors> swift test -c release -Xswiftc -enable-testing \
//      --filter MageVAEGPULaneTests
// Override: MAGE_REAL_IMAGE.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXRandom
import XCTest

@testable import MageFlow

final class MageVAEGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let realImage = URL(
        fileURLWithPath: env["MAGE_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")

    static func stats(_ a: MLXArray, _ ref: MLXArray) -> (rel: Float, max: Float, text: String) {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        let mse = mean(d * d)
        eval(rel, mx, mse)
        let psnr = 10 * log10(4 / max(mse.item(Float.self), 1e-30))
        return (rel.item(Float.self), mx.item(Float.self),
                String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", rel.item(Float.self), mx.item(Float.self), psnr))
    }

    static func withRoute<T>(_ r: MageVAEConvRoute, _ f: () -> T) -> T {
        setenv("MAGEFLOW_VAE_CONV_ROUTE", r.rawValue, 1)
        defer { unsetenv("MAGEFLOW_VAE_CONV_ROUTE") }
        return f()
    }

    func testProbe() throws {
        for (c, o) in [(384, 384), (128, 384)] {
            MLXRandom.seed(UInt64(c + o))
            let x = MLXRandom.normal([1, 64, 64, c])
            let w = MLXRandom.normal([o, 3, 3, c]) * (1.0 / Float(c * 9).squareRoot())
            let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
                let r = conv2d(x, w, stride: 1, padding: 1)
                eval(r)
                return r
            }
            let raw = routedConv2d(x, w, stride: 1, padding: 1, groups: 1, route: .winograd)
            let routed = routedConv2d(x, w, stride: 1, padding: 1, groups: 1, route: .conv3d)
            eval(raw, routed)
            let r0 = Self.stats(raw, ref), r1 = Self.stats(routed, ref)
            print(String(format: "  %d→%d @64²: raw conv2d relL2 %.2e %@ | route relL2 %.2e",
                         c, o, r0.rel, r0.rel < 1e-5 ? "EXACT — route removable" : "lossy", r1.rel))
            XCTAssertLessThan(r1.rel, 1e-5, "conv3d route must be exact-class")
        }
    }

    func testRealPhoto1024() throws {
        try XCTSkipUnless(Self.env["MAGE_PARITY"] == "1", "set MAGE_PARITY=1 to run")
        let w = try MageVAELoader.load(
            vae: URL(fileURLWithPath: try XCTUnwrap(Self.env["MAGE_VAE"], "set MAGE_VAE")),
            foldedAdaLN: URL(fileURLWithPath: try XCTUnwrap(Self.env["MAGE_FOLD"], "set MAGE_FOLD")))
        guard let src = CGImageSourceCreateWithURL(Self.realImage as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "MageVAE", code: 1) }
        let side = 1024
        let (iw, ih) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: iw * ih * 4)
        let ctx = CGContext(
            data: &rgba, width: iw, height: ih, bitsPerComponent: 8, bytesPerRow: iw * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: iw, height: ih))
        let (x0, y0) = ((iw - side) / 2, (ih - side) / 2)
        var hwc = [Float](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            for x in 0..<side {
                let p = ((y0 + y) * iw + (x0 + x)) * 4
                for c in 0..<3 { hwc[(y * side + x) * 3 + c] = Float(rgba[p + c]) / 127.5 - 1 }
            }
        }
        let pixels = MLXArray(hwc, [1, side, side, 3])
        let (z, condRef, ref) = Device.withDefaultDevice(.cpu) { () -> (MLXArray, MLXArray, MLXArray) in
            let z = vaeEncode(pixels, w, samplePosterior: false)
            let c = codDecoder(z, w)
            let r = vaeDecode(z, w)
            eval(z, c, r)
            return (z, c, r)
        }
        Memory.clearCache()
        print("[real 1024² photo: \(Self.realImage.lastPathComponent), latent \(z.shape)]")
        for route in [MageVAEConvRoute.conv3d, .winograd] {
            let (cond, img, ms) = Self.withRoute(route) { () -> (MLXArray, MLXArray, Double) in
                let c = codDecoder(z, w)
                var img = vaeDecode(z, w)
                eval(c, img)
                let t0 = Date()
                for _ in 0..<3 { img = vaeDecode(z, w); eval(img) }
                return (c, img, Date().timeIntervalSince(t0) / 3 * 1000)
            }
            print(String(format: "  %@: cond %@ | decode %@ | %.0f ms", route.rawValue,
                         Self.stats(cond, condRef).text, Self.stats(img, ref).text, ms))
        }
    }
}
