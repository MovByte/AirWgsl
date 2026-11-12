// Measure Metal's pack and unpack functions (MSL 6.15) and the integer functions at the edges MSL leaves undefined or
// implementation-defined (MSL Table 6.2), on this Mac's Metal device at the target's language version.
//
// MSL gives the conversion rules of the packed formats (8.7.1) but not where each component lies in the packed word;
// this records it. Usage: swift pack.swift OUTPUT

import Foundation
import Metal

struct K {
    let name: String
    let kind: String  // "f4", "h4", "f2", "h2", "f3", "h3" (pack: float input), "u" or "s16" (unpack: integer input)
    let outs: Int
    let body: String
}

var ks: [K] = []
func packF(_ fn: String, _ arg: String, _ t: String, ret: String = "uint") {
    ks.append(K(name: fn + "_" + t, kind: t, outs: 1, body: "o[i*4] = uint(\(fn)(\(arg)));"))
}
for fn in ["pack_float_to_unorm4x8", "pack_float_to_snorm4x8", "pack_float_to_srgb_unorm4x8", "pack_float_to_unorm10a2"] {
    packF(fn, "fv", "f4")
}
for fn in ["pack_half_to_unorm4x8", "pack_half_to_snorm4x8", "pack_half_to_unorm10a2"] {
    packF(fn, "hv", "h4")
}
for fn in ["pack_float_to_unorm2x16", "pack_float_to_snorm2x16"] {
    packF(fn, "fv.xy", "f2")
}
for fn in ["pack_half_to_unorm2x16", "pack_half_to_snorm2x16"] {
    packF(fn, "hv.xy", "h2")
}
packF("pack_float_to_unorm565", "fv.xyz", "f3")
packF("pack_half_to_unorm565", "hv.xyz", "h3")
func unpackU(_ fn: String, _ n: Int, half: Bool, arg: String = "u") {
    var body = "auto r = \(fn)(\(arg));"
    for k in 0..<n { body += " o[i*4+\(k)] = as_type<\(half ? "ushort" : "uint")>(r[\(k)]);" }
    ks.append(K(name: fn, kind: arg == "u" ? "u" : "s16", outs: n, body: body))
}
for (fn, n) in [("unpack_unorm4x8_to_float", 4), ("unpack_snorm4x8_to_float", 4), ("unpack_unorm4x8_srgb_to_float", 4),
                ("unpack_unorm2x16_to_float", 2), ("unpack_snorm2x16_to_float", 2), ("unpack_unorm10a2_to_float", 4)] {
    unpackU(fn, n, half: false)
}
for (fn, n) in [("unpack_unorm4x8_to_half", 4), ("unpack_snorm4x8_to_half", 4), ("unpack_unorm4x8_srgb_to_half", 4),
                ("unpack_unorm2x16_to_half", 2), ("unpack_snorm2x16_to_half", 2), ("unpack_unorm10a2_to_half", 4)] {
    unpackU(fn, n, half: true)
}
unpackU("unpack_unorm565_to_float", 3, half: false, arg: "s")
unpackU("unpack_unorm565_to_half", 3, half: true, arg: "s")
// Integer edges.
let ints: [(String, String)] = [
    ("rotate_i32", "o[i*4] = uint(rotate(int(u), int(v)));"),
    ("rotate_u16", "o[i*4] = uint(rotate(ushort(u), ushort(v)));"),
    ("rotate_i8", "o[i*4] = uint(uchar(rotate(char(u), char(v))));"),
    ("mul24_i32", "o[i*4] = uint(mul24(int(u), int(v)));"),
    ("mul24_u32", "o[i*4] = mul24(u, v);"),
    ("mad24_u32", "o[i*4] = mad24(u, v, 7u);"),
    ("extract_bits_u32", "o[i*4] = extract_bits(u, v & 63u, (v >> 8) & 63u);"),
    ("extract_bits_i32", "o[i*4] = uint(extract_bits(int(u), v & 63u, (v >> 8) & 63u));"),
    ("insert_bits_u32", "o[i*4] = insert_bits(u, 0xffffffffu, v & 63u, (v >> 8) & 63u);"),
    ("clz_u32", "o[i*4] = clz(u);"),
    ("ctz_u16", "o[i*4] = uint(ctz(ushort(u)));"),
    ("abs_i32", "o[i*4] = uint(abs(int(u)));"),
    ("absdiff_i32", "o[i*4] = absdiff(int(u), int(v));"),
    ("hadd_i32", "o[i*4] = uint(hadd(int(u), int(v)));"),
    ("rhadd_i32", "o[i*4] = uint(rhadd(int(u), int(v)));"),
    ("mulhi_i32", "o[i*4] = uint(mulhi(int(u), int(v)));"),
    ("madsat_i32", "o[i*4] = uint(madsat(int(u), int(v), int(u)));"),
]
for (n, b) in ints { ks.append(K(name: n, kind: "uu", outs: 1, body: b)) }

var src = "#include <metal_stdlib>\nusing namespace metal;\n"
for k in ks {
    src += "kernel void k_\(k.name)(device const float4* fa [[buffer(0)]], device const uint* ua [[buffer(1)]], "
    src += "device const uint* va [[buffer(2)]], device uint* o [[buffer(3)]], device const half4* ha [[buffer(4)]], "
    src += "uint i [[thread_position_in_grid]]) "
    // The half inputs come from memory: a conversion in the kernel may be folded into the pack by the compiler.
    src += "{ float4 fv = fa[i]; half4 hv = ha[i]; uint u = ua[i]; uint v = va[i]; ushort s = ushort(u); "
    src += "\(k.body) }\n"
}

let args = CommandLine.arguments
guard args.count == 2 else { fatalError("usage: pack.swift OUTPUT") }
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let opts = MTLCompileOptions()
opts.languageVersion = .version3_1
let lib = try device.makeLibrary(source: src, options: opts)
let queue = device.makeCommandQueue()!

func run(_ name: String, _ f: [Float], _ u: [UInt32], _ v: [UInt32], _ n: Int) -> [UInt32] {
    let pso = try! device.makeComputePipelineState(function: lib.makeFunction(name: "k_" + name)!)
    let bf = device.makeBuffer(bytes: f, length: max(16, f.count * 4), options: .storageModeShared)!
    let bu = device.makeBuffer(bytes: u, length: max(4, u.count * 4), options: .storageModeShared)!
    let bv = device.makeBuffer(bytes: v, length: max(4, v.count * 4), options: .storageModeShared)!
    // The same inputs rounded to half on the CPU (round to nearest even).
    let h = f.map { Float16($0) }
    let bh = device.makeBuffer(bytes: h, length: max(8, h.count * 2), options: .storageModeShared)!
    let o = device.makeBuffer(length: n * 16, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    e.setBuffer(bf, offset: 0, index: 0)
    e.setBuffer(bu, offset: 0, index: 1)
    e.setBuffer(bv, offset: 0, index: 2)
    e.setBuffer(o, offset: 0, index: 3)
    e.setBuffer(bh, offset: 0, index: 4)
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let p = o.contents().bindMemory(to: UInt32.self, capacity: n * 4)
    return Array(UnsafeBufferPointer(start: p, count: n * 4))
}

// Float inputs: each component from a list that covers the conversion rules' corner cases, in four rotations so every
// component position sees every value, plus a dense ramp over [-0.1, 1.1].
let comp: [Float] = [0, -0.0, 1, -1, 0.5, -0.5, 2, -2, 0.25, 0.75, 1.0 / 255, 0.5 / 255, 1.5 / 255, 254.5 / 255,
                     127.5 / 255, 0.0031308, 0.04045, 0.2, 0.9, Float.infinity, -Float.infinity, Float.nan, 1e-30, 0.1,
                     1.0 / 3, 2.0 / 3, 0.5 / 1023, 1022.5 / 1023, 0.5 / 31, 0.5 / 63, 32766.5 / 32767, 0.5 / 65535]
var fin: [Float] = []
for i in 0..<comp.count {
    for r in 0..<4 {
        var v = [Float](repeating: 0, count: 4)
        for k in 0..<4 { v[(k + r) % 4] = comp[(i + k * 7) % comp.count] }
        fin += v
    }
}
for k in 0..<512 {
    let x = Float(k) / 512 * 1.2 - 0.1
    fin += [x, 1 - x, x * 0.5, x * x]
}
let nf = fin.count / 4
var uin: [UInt32] = [0, 0xffffffff, 0xff, 0xff00, 0xff0000, 0xff000000, 0x7f, 0x80, 0x81, 0x7f7f7f7f, 0x80808080, 0x3ff,
                     0xffc00, 0x3ff00000, 0xc0000000, 0x40000000, 0xffff, 0xffff0000, 0x7fff, 0x8000, 0x8001, 0x1f, 0x7e0,
                     0xf800, 0x12345678, 0xdeadbeef]
for b in 0..<256 { uin.append(UInt32(b) | UInt32(b) << 8 | UInt32(255 - b) << 16 | UInt32(b ^ 0x5a) << 24) }
for b in 0..<65536 where b % 257 == 0 || b % 1021 == 3 { uin.append(UInt32(b) | UInt32(65535 - b) << 16) }
var vin: [UInt32] = []
var rng: UInt64 = 0x1234
for i in 0..<uin.count {
    rng = rng &* 6364136223846793005 &+ 1442695040888963407
    let r = UInt32(truncatingIfNeeded: rng >> 32)
    vin.append([0, 1, 7, 31, 32, 33, 63, 64, 0xffffffff, 0x80000000, 16, 24, 0x1010, 0x2008, 0x0820, r][i % 16])
}
var lines = ["# Metal pack, unpack and integer edges: \(device.name), macOS \(ProcessInfo.processInfo.operatingSystemVersionString), MSL 3.1",
             "# name inputs -> outputs (hex)"]
for k in ks {
    let isPack = k.kind.hasPrefix("f") || k.kind.hasPrefix("h")
    // The integer edges need only the first inputs, which cycle through the edge counts.
    let n = isPack ? nf : (k.kind == "uu" ? 512 : uin.count)
    let r = run(k.name, isPack ? fin : [0, 0, 0, 0], uin, vin, n)
    for i in 0..<n {
        let ins: String
        if isPack {
            ins = (0..<4).map { String(format: "%08x", fin[i * 4 + $0].bitPattern) }.joined(separator: " ")
        } else if k.kind == "uu" {
            ins = String(format: "%08x %08x", uin[i], vin[i])
        } else {
            ins = String(format: "%08x", uin[i])
        }
        let outs = (0..<k.outs).map { String(format: "%08x", r[i * 4 + $0]) }.joined(separator: " ")
        lines.append("\(k.name) \(ins) -> \(outs)")
    }
}
try (lines.joined(separator: "\n") + "\n").write(toFile: args[1], atomically: true, encoding: .utf8)
