// Staged gate for the Gaussian-Shading RNG port.
//   GSGate <goldens/gs dir>
// Each generator is checked independently, so a mismatch localizes to SHA-256,
// PCG64/SeedSequence, MT19937, or ndtri rather than "the noise is wrong".

import Foundation
import MLX
import MageFlow

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
// Float64 crashes on the GPU (mlx-swift ops.cpp:456) — the goldens are fp64.
Device.setDefault(device: Device(.cpu))

let dir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."
func load(_ n: String) -> [Double] {
    let u = URL(fileURLWithPath: "\(dir)/\(n).npy")
    guard let a = try? MLX.loadArray(url: u) else { err("missing \(n).npy"); exit(2) }
    return a.asType(.float64).asArray(Double.self)
}

final class Flag: @unchecked Sendable { var ok = true }
let flag = Flag()
func stage(_ name: String, _ got: [Double], _ want: [Double], tol: Double) {
    guard got.count == want.count else {
        err("  \(name): COUNT \(got.count) vs \(want.count)"); flag.ok = false; return
    }
    var worst = 0.0, at = 0
    for i in 0 ..< got.count where abs(got[i] - want[i]) > worst {
        worst = abs(got[i] - want[i]); at = i
    }
    let pass = worst <= tol
    if !pass { flag.ok = false }
    err(String(format: "  %-26@ max_abs %.3e @%d  %@", name as NSString, worst, at,
               (pass ? "PASS" : "FAIL") as NSString))
    if !pass {
        err("      got \(got[at])  want \(want[at])   first5 got \(Array(got.prefix(5)))")
        err("                                        first5 want \(Array(want.prefix(5)))")
    }
}

err("[gs] staged RNG gate")
// 1. SHA-256 payload bits
stage("payload bits (SHA-256)", gsPayloadBits().map(Double.init), load("msg_bits"), tol: 0)

// 2. NumPy PCG64 + SeedSequence
let n = load("pad").count
var pcg = PCG64(seedSequence: NumPySeedSequence(entropy: NumPySeedSequence.entropyWords(20_260_720)))
let pad = numpyIntegersLemire32(&pcg, bound: 2, count: n)
let pos = numpyIntegersLemire32(&pcg, bound: 256, count: n)
stage("PCG64 pad", pad.map(Double.init), load("pad"), tol: 0)
stage("PCG64 pos", pos.map(Double.init), load("pos"), tol: 0)

// 3. torch CPU MT19937
var mt = TorchMT19937(seed: 42)
var u = [Double](repeating: 0, count: n)
for i in 0 ..< n { u[i] = mt.nextDouble() }
stage("MT19937 uniforms", u, load("u"), tol: 0)

// 4. ndtri + full assembly
let z = gaussianShadingNoise(channels: 128, height: 32, width: 32, key: 20_260_720, seed: 42)
stage("full GS noise", z.map(Double.init), load("final"), tol: 1e-5)

err("[gs] \(flag.ok ? "PASS" : "FAIL")")
exit(flag.ok ? 0 : 1)
