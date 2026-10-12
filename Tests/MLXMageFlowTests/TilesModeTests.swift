// The `tiles` mode (G-5 small-Mac creative upscale): N independent edits in one request, one
// conditioner load. Offline tests validate the request before any model work (an unloaded package
// reports `notLoaded` once the guards pass) and the light-tier configuration. The live test
// (`MAGEFLOW_LIVE_STORE=<model store root>`, Mage-Flow-Edit-Turbo int8 materialized there) runs the
// real model: every band of a tiles response is byte-identical to a plain edit of that tile at
// seed + k, and the conditioner (and, on the lightest tier, the DiT) loads once per request
// instead of once per tile.

import CoreGraphics
import Foundation
import ImageIO
import MLXToolKit
import MageFlow
import MageFlowEdit
import UniformTypeIdentifiers
import XCTest

@testable import MLXMageFlow

final class TilesModeTests: XCTestCase {

    private func blank(_ n: Int) -> [Image] {
        (0 ..< n).map { _ in Image(format: .png, data: Data(), width: 1, height: 1) }
    }

    private func run(_ request: IEditRequest) async -> PackageError? {
        let package = MageFlowEditPackage(
            configuration: MageFlowConfiguration(variant: .editTurbo, quant: .int8))
        do {
            _ = try await package.run(request)
            return nil
        } catch let e as PackageError {
            return e
        } catch {
            XCTFail("unexpected error type \(type(of: error)): \(error)")
            return nil
        }
    }

    // MARK: - request validation (offline)

    func testDescriptorDeclaresTheTilesMode() {
        let surface = MageFlowEditPackage.manifest.surfaces.first
        XCTAssertEqual(surface?.supportedModes, [MageFlowModes.tiles])
        XCTAssertTrue(surface?.summary.contains("Mode `tiles`") ?? false)
    }

    func testTilesModeIsNotBoundByTheReferenceRange() async {
        // 5 tiles: past the 1…3 multi-reference range, which bounds ONE edit's references, not a batch.
        let e = await run(IEditRequest(images: blank(5), prompt: "probe", width: 512, height: 512,
                                       mode: MageFlowModes.tiles))
        XCTAssertEqual(e, .notLoaded)
        // the same five images without the mode are still a refused 5-reference edit
        guard case .unsupportedRequestFeature? = await run(IEditRequest(images: blank(5), prompt: "probe")) else {
            return XCTFail("5 references without the tiles mode must stay refused")
        }
    }

    func testTooManyTilesRejected() async {
        guard case .unsupportedRequestFeature(let why)? = await run(
            IEditRequest(images: blank(MageFlowModes.maxTiles + 1), prompt: "probe", mode: MageFlowModes.tiles))
        else { return XCTFail("more than maxTiles must be refused") }
        XCTAssertTrue(why.contains("at most \(MageFlowModes.maxTiles)"), why)
    }

    func testNonSquareTilesRejected() async {
        guard case .unsupportedRequestFeature(let why)? = await run(
            IEditRequest(images: blank(2), prompt: "probe", width: 512, height: 768, mode: MageFlowModes.tiles))
        else { return XCTFail("non-square tiles must be refused") }
        XCTAssertTrue(why.contains("non-square"), why)
    }

    func testBadInitSigmaRejectedBeforeModelWork() async {
        guard case .unsupportedRequestFeature? = await run(
            IEditRequest(images: blank(2), prompt: "probe", mode: MageFlowModes.tiles,
                         metaData: [MageFlowMetaKeys.initSigmaKey: .double(1.5)]))
        else { return XCTFail("initSigma 1.5 must be refused") }
    }

    // MARK: - light tiers (offline)

    func testEvictTransformerRoundTripsAndImpliesConditionerEviction() throws {
        let cfg = MageFlowConfiguration(variant: .editTurbo, quant: .int8, evictTransformer: true)
        let back = try JSONDecoder().decode(MageFlowConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertTrue(back.evictTransformer)
        XCTAssertFalse(back.evictConditioner)
        XCTAssertTrue(back.dropsConditioner)
        // older persisted configurations decode as the resident tier
        let old = try JSONDecoder().decode(
            MageFlowConfiguration.self, from: Data(#"{"variant":"Mage-Flow-Edit-Turbo","quant":"int8"}"#.utf8))
        XCTAssertFalse(old.evictTransformer)
        XCTAssertFalse(old.dropsConditioner)
    }

    func testLightTierFootprintsShrink() throws {
        let resident = MageFlowConfiguration(variant: .editTurbo, quant: .int8)
        let evict = MageFlowConfiguration(variant: .editTurbo, quant: .int8, evictConditioner: true)
        let lightest = MageFlowConfiguration(variant: .editTurbo, quant: .int8, evictTransformer: true)
        func peak(_ c: MageFlowConfiguration) -> UInt64 {
            let declared = MageFlowEditPackage.manifest.requirements.footprints.first { $0.quant == c.quant }!
            return (c.residentBytesHint ?? declared.residentBytes) + (c.peakActivationBytesHint ?? declared.peakActivationBytes)
        }
        XCTAssertLessThan(peak(lightest), peak(evict))
        XCTAssertLessThan(peak(evict), peak(resident))
        // the lightest tier is the 16 GB one: its declared peak sits under a 16 GB Mac's GPU budget (~12.7 GB)
        XCTAssertLessThan(peak(lightest), 12_000_000_000)
    }

    // MARK: - progress (offline)

    func testBatchStagesReachTheEnginePlane() {
        final class Box: @unchecked Sendable {
            let lock = NSLock()
            var items: [RunPhaseReport] = []
            func add(_ r: RunPhaseReport) { lock.lock(); items.append(r); lock.unlock() }
        }
        let box = Box()
        RunProgress.$sink.withValue({ box.add($0) }) {
            MageProgress.$sink.withValue(MageProgressBridge.forward) {
                MageProgress.report(.encode)
                MageProgress.$stage.withValue((2, 5)) {
                    MageProgress.report(.denoise, step: 3, totalSteps: 4)
                }
            }
        }
        XCTAssertEqual(box.items, [
            RunPhaseReport(phase: .encode),
            RunPhaseReport(phase: .denoise, step: 3, totalSteps: 4, stage: 2, totalStages: 5),
        ])
    }

    // MARK: - live (real weights)

    /// A deterministic, structured RGB test card (gradients, edges, a disc), as PNG.
    static func card(side: Int, variant: Int) -> Data {
        var rgba = [UInt8](repeating: 255, count: side * side * 4)
        let c = Double(side) / 2
        for y in 0 ..< side {
            for x in 0 ..< side {
                let i = (y * side + x) * 4
                let dx = Double(x) - c * (0.7 + 0.2 * Double(variant)), dy = Double(y) - c
                let disc = dx * dx + dy * dy < (c * 0.45) * (c * 0.45)
                rgba[i] = UInt8((x * 255) / side)
                rgba[i + 1] = disc ? 200 : UInt8((y * 255) / side)
                rgba[i + 2] = ((x / 32 + y / 32 + variant) % 2 == 0) ? 40 : 180
            }
        }
        let cg = CGContext(data: &rgba, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side * 4,
                           space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let out = NSMutableData()
        let dst = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, cg, nil)
        CGImageDestinationFinalize(dst)
        return out as Data
    }

    func testLiveTilesMatchPerTileEditsWithOneLoadPerRequest() async throws {
        guard let store = ProcessInfo.processInfo.environment["MAGEFLOW_LIVE_STORE"] else {
            throw XCTSkip("set MAGEFLOW_LIVE_STORE=<model store root> (Mage-Flow-Edit-Turbo int8) to run")
        }
        let side = 512, count = 3, seed: UInt64 = 1234
        let tiles = (0 ..< count).map { Image(format: .png, data: Self.card(side: side, variant: $0), width: side, height: side) }
        let instruction = "Upscale this image to high resolution: sharpen it and add fine, natural detail and texture."
        let meta: MetaData = [MageFlowMetaKeys.initSigmaKey: .double(0.9)]

        for tier in ["resident", "lightest"] {
            let cfg = MageFlowConfiguration(variant: .editTurbo, quant: .int8, evictTransformer: tier == "lightest",
                                            bypassContentFilter: true,
                                            modelsRootDirectory: URL(fileURLWithPath: store))
            let package = MageFlowEditPackage(configuration: cfg)
            try await package.load()

            // per tile: one plain edit per tile at seed + k
            let before = await package.modelLoads
            var bands: [[UInt8]] = []
            for (k, tile) in tiles.enumerated() {
                let r = try await package.run(IEditRequest(images: [tile], prompt: instruction, width: side, height: side,
                                                           seed: seed + UInt64(k), metaData: meta)) as! IEditResponse
                let (rgb, w, h) = try MageFlowEditPipeline.decodeRGB(data: r.image.data)
                XCTAssertEqual([w, h], [side, side])
                bands.append(rgb)
            }
            let perTile = await package.modelLoads

            // one request in the tiles mode
            let r = try await package.run(IEditRequest(images: tiles, prompt: instruction, width: side, height: side,
                                                       seed: seed, mode: MageFlowModes.tiles, metaData: meta)) as! IEditResponse
            let after = await package.modelLoads
            XCTAssertEqual([r.image.width, r.image.height], [side, side * count])
            let (strip, w, h) = try MageFlowEditPipeline.decodeRGB(data: r.image.data)
            XCTAssertEqual([w, h], [side, side * count])
            let band = side * side * 3
            for k in 0 ..< count {
                XCTAssertEqual(Array(strip[k * band ..< (k + 1) * band]), bands[k], "\(tier): band \(k) differs from its plain edit")
            }

            if tier == "resident" {
                // loaded once at load(); neither path reloads anything
                XCTAssertEqual(after.conditioner, 1, tier)
                XCTAssertEqual(after.transformer, 1, tier)
            } else {
                // lightest: per-tile reloads both models every tile; the tiles request loads each once
                XCTAssertEqual(perTile.conditioner - before.conditioner, count, tier)
                XCTAssertEqual(perTile.transformer - before.transformer, count, tier)
                XCTAssertEqual(after.conditioner - perTile.conditioner, 1, tier)
                XCTAssertEqual(after.transformer - perTile.transformer, 1, tier)
            }
            await package.unload()
        }
    }
}
