// Measure what Metal returns for the special inputs of MSL's floating-point functions with fast math disabled.
//
// MSL requires NaN and infinity support when fast math is disabled (MSL 8.1) and states accuracy bounds, but not the
// value of each function at zeros, infinities, NaNs, poles and branch cuts. This program runs the functions on this
// Mac's Metal device, compiled from source with fast math disabled at the target's language version, over a set of
// boundary inputs, and writes one line per case: the function, its operands and its results as bit patterns.
//
// Usage: swift special.swift OUTPUT

import Foundation
import Metal

let specials: [UInt32] = [
    0x0000_0000, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x3f00_0000, 0xbf00_0000, 0x3fc0_0000, 0xbfc0_0000,
    0x4000_0000, 0xc000_0000, 0x4040_0000, 0xc040_0000, 0x3e80_0000, 0xbe80_0000, 0x3f40_0000, 0xbf40_0000,
    0x4b00_0000, 0xcb00_0000, 0x4b00_0001, 0xcb00_0001, 0x4b80_0000, 0xcb80_0000, 0x0080_0000, 0x8080_0000,
    0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000, 0x7fc0_1234, 0x4a80_0001,
    0x4040_0001, 0x3fff_ffff,
]

struct Fn {
    let name: String
    let arity: Int
    let outs: Int
    let body: String
}

func f1(_ n: String) -> Fn { Fn(name: n, arity: 1, outs: 1, body: "o[2*i] = as_type<uint>(\(n)(a[i]));") }
func f2(_ n: String) -> Fn { Fn(name: n, arity: 2, outs: 1, body: "o[2*i] = as_type<uint>(\(n)(a[i], b[i]));") }
func f3(_ n: String) -> Fn { Fn(name: n, arity: 3, outs: 1, body: "o[2*i] = as_type<uint>(\(n)(a[i], b[i], c[i]));") }

var fns: [Fn] = []
for n in ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "sinpi",
          "cospi", "tanpi", "exp", "exp2", "exp10", "log", "log2", "log10", "sqrt", "rsqrt", "fabs", "ceil", "floor",
          "trunc", "rint", "round", "fract", "saturate", "sign"] {
    fns.append(f1(n))
}
for n in ["atan2", "pow", "powr", "fmod", "fdim", "fmax", "fmin", "copysign", "nextafter", "step", "divide"] {
    fns.append(f2(n))
}
for n in ["fma", "fmax3", "fmin3", "fmedian3", "clamp", "mix", "smoothstep"] {
    fns.append(f3(n))
}
fns.append(Fn(name: "sincos", arity: 1, outs: 2,
              body: "float cv; float sv = sincos(a[i], cv); o[2*i] = as_type<uint>(sv); o[2*i+1] = as_type<uint>(cv);"))
fns.append(Fn(name: "frexp", arity: 1, outs: 2,
              body: "int e; float m = frexp(a[i], e); o[2*i] = as_type<uint>(m); o[2*i+1] = uint(e);"))
fns.append(Fn(name: "modf", arity: 1, outs: 2,
              body: "float ip; float fr = modf(a[i], ip); o[2*i] = as_type<uint>(fr); o[2*i+1] = as_type<uint>(ip);"))
fns.append(Fn(name: "ilogb", arity: 1, outs: 1, body: "o[2*i] = uint(ilogb(a[i]));"))
fns.append(Fn(name: "ldexp", arity: 2, outs: 1, body: "o[2*i] = as_type<uint>(ldexp(a[i], int(b[i])));"))
fns.append(Fn(name: "to_int", arity: 1, outs: 1, body: "o[2*i] = uint(int(a[i]));"))
fns.append(Fn(name: "to_uint", arity: 1, outs: 1, body: "o[2*i] = uint(a[i]);"))
fns.append(Fn(name: "to_short", arity: 1, outs: 1, body: "o[2*i] = uint(ushort(short(a[i])));"))
fns.append(Fn(name: "to_bool", arity: 1, outs: 1, body: "o[2*i] = bool(a[i]) ? 1u : 0u;"))
fns.append(Fn(name: "to_half", arity: 1, outs: 1, body: "o[2*i] = uint(as_type<ushort>(half(a[i])));"))
fns.append(Fn(name: "isnan", arity: 1, outs: 1, body: "o[2*i] = isnan(a[i]) ? 1u : 0u;"))

var src = "#include <metal_stdlib>\nusing namespace metal;\n"
for f in fns {
    src += "kernel void k_\(f.name)(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], "
    src += "device const float* c [[buffer(2)]], device uint* o [[buffer(3)]], uint i [[thread_position_in_grid]]) "
    src += "{ \(f.body) }\n"
}
// An integer converted to bool: whether the conversion tests for zero or keeps the low bit.
src += "kernel void k_int_to_bool(device const int* a [[buffer(0)]], device uint* o [[buffer(1)]], "
src += "uint i [[thread_position_in_grid]]) { o[i] = bool(a[i]) ? 1u : 0u; }\n"

let args = CommandLine.arguments
guard args.count == 2 else {
    FileHandle.standardError.write("usage: special.swift OUTPUT\n".data(using: .utf8)!)
    exit(2)
}
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let opts = MTLCompileOptions()
opts.languageVersion = .version3_1
opts.fastMathEnabled = false
let lib = try device.makeLibrary(source: src, options: opts)
let queue = device.makeCommandQueue()!

func run(_ kernel: String, _ inputs: [[UInt32]], _ words: Int) -> [UInt32] {
    let fn = lib.makeFunction(name: kernel)!
    let pso = try! device.makeComputePipelineState(function: fn)
    let n = inputs[0].count
    var bufs: [MTLBuffer] = inputs.map { v in
        device.makeBuffer(bytes: v, length: max(4, v.count * 4), options: .storageModeShared)!
    }
    let out = device.makeBuffer(length: max(4, n * words * 4), options: .storageModeShared)!
    bufs.append(out)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    for (k, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: k) }
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(64, n), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let p = out.contents().bindMemory(to: UInt32.self, capacity: n * words)
    return Array(UnsafeBufferPointer(start: p, count: n * words))
}

var lines: [String] = []
let os = ProcessInfo.processInfo.operatingSystemVersionString
lines.append("# Metal precise-math special values: \(device.name), macOS \(os), MSL 3.1, fastMathEnabled = false")
lines.append("# function operands -> results (hex bit patterns)")
for f in fns {
    var a: [UInt32] = []
    var b: [UInt32] = []
    var c: [UInt32] = []
    switch f.arity {
    case 1:
        a = specials
    case 2:
        for x in specials { for y in specials { a.append(x); b.append(y) } }
    default:
        let s3 = specials.filter { [0x0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x3f00_0000, 0x4000_0000,
                                   0x7f80_0000, 0xff80_0000, 0x7fc0_0000].contains($0) }
        for x in s3 { for y in s3 { for z in s3 { a.append(x); b.append(y); c.append(z) } } }
    }
    let n = a.count
    if b.isEmpty { b = Array(repeating: 0, count: n) }
    if c.isEmpty { c = Array(repeating: 0, count: n) }
    if f.name == "ldexp" {
        b = b.map { Float(bitPattern: $0).isFinite ? $0 : 0 }
    }
    let r = run("k_\(f.name)", [a, b, c], 2)
    for i in 0..<n {
        var ops = [a[i]]
        if f.arity > 1 { ops.append(b[i]) }
        if f.arity > 2 { ops.append(c[i]) }
        var res = [r[2 * i]]
        if f.outs > 1 { res.append(r[2 * i + 1]) }
        let hx = { (v: [UInt32]) in v.map { String(format: "%08x", $0) }.joined(separator: " ") }
        lines.append("\(f.name) \(hx(ops)) -> \(hx(res))")
    }
}
let ints: [UInt32] = [0, 1, 2, 3, 0xffff_ffff, 0x8000_0000, 0x100]
let rb = run("k_int_to_bool", [ints], 1)
for (k, v) in ints.enumerated() { lines.append(String(format: "int_to_bool %08x -> %08x", v, rb[k])) }
try (lines.joined(separator: "\n") + "\n").write(toFile: args[1], atomically: true, encoding: .utf8)
