// Gaussian-Shading watermarked latent init.
//
// Mage-Flow does NOT start from randn. `get_noise()` is computed and then
// immediately overwritten by `mage_latent.encode_noise`, which embeds a
// SHA-256-derived payload into the initial noise. Upstream ships
// `invert_to_noise` as the matching detector and provides NO toggle to disable
// it, so a port that substitutes plain randn silently strips Microsoft's
// provenance mechanism and produces images their detector will not recognise.
//
// Reproducing it bit-exactly needs three generators that have no MLX (or
// Foundation) equivalent, each reimplemented here and each verified against a
// dumped golden stage (see dump_gs_stages.py / GSGate):
//
//   1. SHA-256              -> the 256-bit payload bit vector
//   2. NumPy PCG64          -> per-entry XOR pad + message-index map
//      (incl. NumPy SeedSequence, which turns the key into the PCG64 state)
//   3. torch CPU MT19937    -> the U(0,1) magnitudes
//
// then z = ndtri((half + u) / 2).
//
// Marginally the result is N(0,1), so output QUALITY does not depend on getting
// this exactly right — only detectability and bit-parity do. That makes it
// precisely the kind of thing that silently rots, hence the staged gate.

import Foundation

// MARK: - 1. payload bits (SHA-256)

/// `_payload_to_bits`: SHA-256("<payload>:<counter>") concatenated until n_bits,
/// unpacked **LSB-first within each byte** (`(byte >> k) & 1`).
public func gsPayloadBits(payload: String = "MageFlow", nBits: Int = 256) -> [UInt8] {
    var out: [UInt8] = []
    var counter = 0
    while out.count < nBits {
        let digest = sha256(Array("\(payload):\(counter)".utf8))
        for byte in digest {
            for k in 0 ..< 8 { out.append(UInt8((byte >> UInt8(k)) & 1)) }
        }
        counter += 1
    }
    return Array(out[0 ..< nBits])
}

/// Minimal SHA-256 (avoids a CryptoKit import for one call).
func sha256(_ message: [UInt8]) -> [UInt8] {
    let k: [UInt32] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]
    var h: [UInt32] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
    var msg = message
    let bitLen = UInt64(message.count) * 8
    msg.append(0x80)
    while msg.count % 64 != 56 { msg.append(0) }
    for i in (0 ..< 8).reversed() { msg.append(UInt8((bitLen >> UInt64(i * 8)) & 0xff)) }

    for chunk in stride(from: 0, to: msg.count, by: 64) {
        var w = [UInt32](repeating: 0, count: 64)
        for i in 0 ..< 16 {
            let o = chunk + i * 4
            w[i] = (UInt32(msg[o]) << 24) | (UInt32(msg[o + 1]) << 16)
                | (UInt32(msg[o + 2]) << 8) | UInt32(msg[o + 3])
        }
        for i in 16 ..< 64 {
            let s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3)
            let s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
        }
        var (a, b, c, d, e, f, g, hh) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7])
        for i in 0 ..< 64 {
            let S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)
            let ch = (e & f) ^ (~e & g)
            let t1 = hh &+ S1 &+ ch &+ k[i] &+ w[i]
            let S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)
            let maj = (a & b) ^ (a & c) ^ (b & c)
            let t2 = S0 &+ maj
            hh = g; g = f; f = e; e = d &+ t1
            d = c; c = b; b = a; a = t1 &+ t2
        }
        h[0] = h[0] &+ a; h[1] = h[1] &+ b; h[2] = h[2] &+ c; h[3] = h[3] &+ d
        h[4] = h[4] &+ e; h[5] = h[5] &+ f; h[6] = h[6] &+ g; h[7] = h[7] &+ hh
    }
    return h.flatMap { v in (0 ..< 4).reversed().map { UInt8((v >> UInt32($0 * 8)) & 0xff) } }
}

@inline(__always) func rotr32(_ x: UInt32, _ n: UInt32) -> UInt32 { (x >> n) | (x << (32 - n)) }
@inline(__always) func rotr64(_ x: UInt64, _ n: UInt64) -> UInt64 {
    n == 0 ? x : (x >> n) | (x << (64 - n))
}

// MARK: - 2. NumPy SeedSequence + PCG64

/// NumPy's SeedSequence: turns arbitrary entropy into a well-mixed state.
/// `np.random.default_rng(key)` runs the key through this before PCG64 sees it,
/// so seeding PCG64 with the raw key does NOT reproduce NumPy.
public struct NumPySeedSequence {
    static let INIT_A: UInt32 = 0x43b0_d7e5
    static let MULT_A: UInt32 = 0x931e_8875
    static let INIT_B: UInt32 = 0x8b51_f9dd
    static let MULT_B: UInt32 = 0x58f3_8ded
    static let MIX_L: UInt32 = 0xca01_f9dd
    static let MIX_R: UInt32 = 0x4973_f715
    static let XSHIFT: UInt32 = 16
    static let POOL_SIZE = 4

    let pool: [UInt32]

    public init(entropy: [UInt32]) {
        var hashConst = Self.INIT_A
        func hashmix(_ v: UInt32) -> UInt32 {
            var value = v
            value ^= hashConst
            hashConst = hashConst &* Self.MULT_A
            value = value &* hashConst
            value ^= value >> Self.XSHIFT
            return value
        }
        func mix(_ x: UInt32, _ y: UInt32) -> UInt32 {
            var r = (Self.MIX_L &* x) &- (Self.MIX_R &* y)
            r ^= r >> Self.XSHIFT
            return r
        }
        var p = [UInt32](repeating: 0, count: Self.POOL_SIZE)
        // entropy is zero-extended to POOL_SIZE
        for i in 0 ..< Self.POOL_SIZE {
            p[i] = hashmix(i < entropy.count ? entropy[i] : 0)
        }
        for iSrc in 0 ..< Self.POOL_SIZE {
            for iDst in 0 ..< Self.POOL_SIZE where iSrc != iDst {
                p[iDst] = mix(p[iDst], hashmix(p[iSrc]))
            }
        }
        // remaining entropy beyond the pool
        if entropy.count > Self.POOL_SIZE {
            for iSrc in Self.POOL_SIZE ..< entropy.count {
                for iDst in 0 ..< Self.POOL_SIZE {
                    p[iDst] = mix(p[iDst], hashmix(entropy[iSrc]))
                }
            }
        }
        self.pool = p
    }

    /// Convert a non-negative integer key into NumPy's entropy word array
    /// (little-endian 32-bit limbs, at least one word).
    public static func entropyWords(_ key: UInt64) -> [UInt32] {
        if key == 0 { return [0] }
        var v = key
        var out: [UInt32] = []
        while v > 0 { out.append(UInt32(truncatingIfNeeded: v)); v >>= 32 }
        return out
    }

    public func generateState(_ nWords: Int) -> [UInt32] {
        var hashConst = Self.INIT_B
        var out = [UInt32](repeating: 0, count: nWords)
        for i in 0 ..< nWords {
            var value = pool[i % Self.POOL_SIZE]
            value ^= hashConst
            hashConst = hashConst &* Self.MULT_B
            value = value &* hashConst
            value ^= value >> Self.XSHIFT
            out[i] = value
        }
        return out
    }
}

/// PCG64 (XSL-RR 128/64), as used by `np.random.default_rng`.
public struct PCG64 {
    static let MULT_HI: UInt64 = 0x2360_ED05_1FC6_5DA4
    static let MULT_LO: UInt64 = 0x4385_DF64_9FCC_F645

    var stateHi: UInt64, stateLo: UInt64
    var incHi: UInt64, incLo: UInt64

    public init(seedSequence ss: NumPySeedSequence) {
        // NumPy: `generate_state(4, uint64)` -> seed = val[0..1], inc = val[2..3],
        // and `PCG_128BIT_CONSTANT(high, low)` makes val[0] the HIGH word.
        // generate_state for uint64 produces 2n uint32 and views them as uint64,
        // i.e. little-endian pairs.
        let s = ss.generateState(8)
        func u64(_ lo: UInt32, _ hi: UInt32) -> UInt64 { UInt64(lo) | (UInt64(hi) << 32) }
        let v0 = u64(s[0], s[1]), v1 = u64(s[2], s[3])
        let v2 = u64(s[4], s[5]), v3 = u64(s[6], s[7])
        let (sh, sl) = (v0, v1)      // initstate: high = val[0], low = val[1]
        let (ih, il) = (v2, v3)      // initseq  : high = val[2], low = val[3]
        // increment = (inc << 1) | 1
        self.incHi = (ih << 1) | (il >> 63)
        self.incLo = (il << 1) | 1
        self.stateHi = 0; self.stateLo = 0
        _ = next()                       // state = state + inc, then step
        addInc(sh, sl)
        _ = next()
    }

    mutating func addInc(_ hi: UInt64, _ lo: UInt64) {
        let (nlo, carry) = stateLo.addingReportingOverflow(lo)
        stateLo = nlo
        stateHi = stateHi &+ hi &+ (carry ? 1 : 0)
    }

    mutating func stepState() {
        // state = state * MULT + inc   (128-bit)
        let (h1, l1) = mul128(stateHi, stateLo, Self.MULT_HI, Self.MULT_LO)
        stateHi = h1; stateLo = l1
        addInc(incHi, incLo)
    }

    public mutating func next() -> UInt64 {
        stepState()
        // XSL-RR: rotr64(hi ^ lo, hi >> 58)
        return rotr64(stateHi ^ stateLo, stateHi >> 58)
    }

    // NumPy buffers 32-bit draws as LOW half first, then HIGH half:
    //   next = next64(); return low32; (next call) return high32
    var hasU32 = false
    var bufU32: UInt32 = 0

    public mutating func next32() -> UInt32 {
        if hasU32 { hasU32 = false; return bufU32 }
        let n = next()
        bufU32 = UInt32(truncatingIfNeeded: n >> 32)
        hasU32 = true
        return UInt32(truncatingIfNeeded: n)
    }
}

/// 128x128 -> low 128 bits.
@inline(__always)
func mul128(_ aHi: UInt64, _ aLo: UInt64, _ bHi: UInt64, _ bLo: UInt64) -> (UInt64, UInt64) {
    let m = aLo.multipliedFullWidth(by: bLo)
    let hi = m.high &+ (aHi &* bLo) &+ (aLo &* bHi)
    return (hi, m.low)
}

/// `Generator.integers(0, bound)` for a small bound.
///
/// NOT `next_uint64() & mask` — that was the obvious guess and it is wrong.
/// NumPy routes small ranges through **32-bit** draws with **Lemire's**
/// multiply-shift (`random_buffered_bounded_lemire_uint32`), and PCG64's
/// `next_uint32` yields the LOW half of each uint64 first, then the HIGH half.
///
/// Verified against `PCG64.random_raw`: decomposing each raw uint64 low-then-high
/// and taking `(u32 * 2) >> 32` reproduces `integers(0, 2, n)` exactly, whereas
/// `raw & 1` does not.
///
/// For a power-of-two bound the Lemire rejection threshold is
/// `(2^32 - bound) % bound == 0`, so no rejection ever occurs — both bounds used
/// here (2 and 256) are powers of two.
public func numpyIntegersLemire32(_ rng: inout PCG64, bound: UInt32, count: Int) -> [UInt8] {
    precondition(bound > 0 && (bound & (bound - 1)) == 0, "no-rejection path needs a power of two")
    var out = [UInt8](repeating: 0, count: count)
    for i in 0 ..< count {
        out[i] = UInt8((UInt64(rng.next32()) &* UInt64(bound)) >> 32)
    }
    return out
}

// MARK: - 3. torch CPU MT19937

/// PyTorch's CPU generator. `torch.rand(dtype=float64)` draws two 32-bit words
/// and forms (x >> 11) * 2^-53.
public struct TorchMT19937 {
    var mt = [UInt32](repeating: 0, count: 624)
    var idx = 624
    var seeded = false

    public init(seed: UInt64) {
        // torch seeds MT19937 with the low 32 bits, Knuth's initializer.
        mt[0] = UInt32(truncatingIfNeeded: seed)
        for i in 1 ..< 624 {
            mt[i] = 1_812_433_253 &* (mt[i - 1] ^ (mt[i - 1] >> 30)) &+ UInt32(i)
        }
        idx = 624
    }

    mutating func generate() {
        for i in 0 ..< 624 {
            let y = (mt[i] & 0x8000_0000) | (mt[(i + 1) % 624] & 0x7fff_ffff)
            var n = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1 == 1 { n ^= 0x9908_b0df }
            mt[i] = n
        }
        idx = 0
    }

    public mutating func nextUInt32() -> UInt32 {
        if idx >= 624 { generate() }
        var y = mt[idx]; idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c_5680
        y ^= (y << 15) & 0xefc6_0000
        y ^= y >> 18
        return y
    }

    /// float64 uniform, torch's `uniform_real_distribution<double>`:
    ///   random64 = (r1 << 32) | r2;  x = (random64 & (2^53 - 1)) * 2^-53
    ///
    /// Note this takes the **LOW** 53 bits. NumPy instead uses the HIGH bits of
    /// two draws — taking the high bits here reproduces NumPy's famous
    /// seed-42 sequence [0.3745401, 0.9507143, ...] rather than torch's.
    public mutating func nextDouble() -> Double {
        let r1 = UInt64(nextUInt32())
        let r2 = UInt64(nextUInt32())
        let x = ((r1 << 32) | r2) & 0x001F_FFFF_FFFF_FFFF   // 2^53 - 1
        return Double(x) * (1.0 / 9_007_199_254_740_992.0)  // 2^-53
    }
}

// MARK: - 4. ndtri

/// Inverse normal CDF via erfinv: ndtri(p) = sqrt(2) * erfinv(2p - 1).
/// Acklam's rational approximation refined by one Halley step, which brings it
/// to double precision.
public func ndtri(_ p: Double) -> Double {
    let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
    let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
    let pLow = 0.02425, pHigh = 1 - pLow
    var x: Double
    if p < pLow {
        let q = (-2 * Foundation.log(p)).squareRoot()
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    } else if p <= pHigh {
        let q = p - 0.5, r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    } else {
        let q = (-2 * Foundation.log(1 - p)).squareRoot()
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    }
    // one Halley refinement against erfc
    let e = 0.5 * erfc(-x / 2.0.squareRoot()) - p
    let u = e * (2 * Double.pi).squareRoot() * Foundation.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return x
}

// MARK: - assembly

/// Reproduces `mage_latent.encode_noise((C,H,W), key:, seed:)`.
/// Returns row-major [C*H*W] to be reshaped to [1, C, H, W] (or NHWC).
public func gaussianShadingNoise(
    channels C: Int, height H: Int, width W: Int,
    key: UInt64 = 20_260_720, seed: UInt64 = 42, payload: String = "MageFlow"
) -> [Float] {
    let n = C * H * W
    let msg = gsPayloadBits(payload: payload)

    var pcg = PCG64(seedSequence: NumPySeedSequence(entropy: NumPySeedSequence.entropyWords(key)))
    let pad = numpyIntegersLemire32(&pcg, bound: 2, count: n)
    let pos = numpyIntegersLemire32(&pcg, bound: 256, count: n)

    var mt = TorchMT19937(seed: seed & 0x7FFF_FFFF)
    var out = [Float](repeating: 0, count: n)
    for i in 0 ..< n {
        let half = Double(msg[Int(pos[i])] ^ pad[i])
        let u = mt.nextDouble()
        let arg = min(max((half + u) / 2.0, 1e-6), 1.0 - 1e-6)
        out[i] = Float(ndtri(arg))
    }
    return out
}
