// Runs compute kernels on this Mac's Metal device and writes back their output buffers, for the comparison of the
// IR interpreter with real Metal (tests/canon, examples/metal_diff.rs).
//
// Usage: swift run_kernels.swift MANIFEST
//
// Each manifest line is tab-separated:
//   id  source  language  entry  threads  buffers  output  constants
// where `source` is an MSL file compiled at run time with `language` ("3.1", "2.4", ...), `buffers` is a comma list of
// `index=path` (each file's bytes are bound at that buffer index; the output buffer's file gives its initial contents),
// `output` is `index=path` naming the buffer to write back to the path with an `.metal` suffix, and `constants` is a
// comma list of `index:mtldatatype:hexbytes` function constant values, or `-`. The grid is `threads` threads in one
// dimension, one thread per threadgroup. A failure to build or run a job is reported on stdout as `error <id> <text>`;
// each finished job prints `ok <id>`.

import Foundation
import Metal

func hexBytes(_ s: Substring) -> [UInt8] {
    var out: [UInt8] = []
    var i = s.startIndex
    while i < s.endIndex {
        let j = s.index(i, offsetBy: 2)
        out.append(UInt8(s[i..<j], radix: 16)!)
        i = j
    }
    return out
}

func languageVersion(_ s: String) -> MTLLanguageVersion? {
    switch s {
    case "2.0": return .version2_0
    case "2.1": return .version2_1
    case "2.2": return .version2_2
    case "2.3": return .version2_3
    case "2.4": return .version2_4
    case "3.0": return .version3_0
    case "3.1": return .version3_1
    default: return nil
    }
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var libraries: [String: MTLLibrary] = [:]
let manifest = try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
for line in manifest.split(separator: "\n") {
    let f = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
    let (id, source, language, entry) = (f[0], f[1], f[2], f[3])
    let threads = Int(f[4])!
    let key = source + "@" + language
    if libraries[key] == nil {
        let opts = MTLCompileOptions()
        guard let v = languageVersion(language) else { print("error \(id) language \(language)"); continue }
        opts.languageVersion = v
        do {
            libraries[key] = try device.makeLibrary(source: String(contentsOfFile: source, encoding: .utf8), options: opts)
        } catch {
            print("error \(id) compile \(error.localizedDescription.replacingOccurrences(of: "\n", with: " ").prefix(300))")
            continue
        }
    }
    let lib = libraries[key]!
    let cv = MTLFunctionConstantValues()
    if f[7] != "-" {
        for c in f[7].split(separator: ",") {
            let p = c.split(separator: ":")
            var bytes = hexBytes(p[2])
            cv.setConstantValue(&bytes, type: MTLDataType(rawValue: UInt(p[1])!)!, index: Int(p[0])!)
        }
    }
    let function: MTLFunction
    do {
        function = try lib.makeFunction(name: entry, constantValues: cv)
    } catch {
        print("error \(id) function \(error.localizedDescription)")
        continue
    }
    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(function: function)
    } catch {
        print("error \(id) pipeline \(error.localizedDescription)")
        continue
    }
    var bound: [Int: MTLBuffer] = [:]
    for b in f[5].split(separator: ",") {
        let p = b.split(separator: "=")
        let data = try! Data(contentsOf: URL(fileURLWithPath: String(p[1])))
        let buf = device.makeBuffer(length: max(data.count, 16), options: .storageModeShared)!
        data.withUnsafeBytes { raw in memcpy(buf.contents(), raw.baseAddress!, data.count) }
        bound[Int(p[0])!] = buf
    }
    let out = f[6].split(separator: "=")
    let outIndex = Int(out[0])!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    for (i, buf) in bound {
        enc.setBuffer(buf, offset: 0, index: i)
    }
    enc.dispatchThreadgroups(MTLSize(width: threads, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error {
        print("error \(id) run \(e.localizedDescription)")
        continue
    }
    let ob = bound[outIndex]!
    let data = Data(bytes: ob.contents(), count: ob.length)
    try! data.write(to: URL(fileURLWithPath: String(out[1]) + ".metal"))
    print("ok \(id)")
}
