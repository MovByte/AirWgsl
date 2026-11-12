// Runs one Metal compute function on this Mac's GPU and writes its buffers back: the reference side of the WGSL
// differential tests.
//
// Usage: metal_run SOURCE.metal FUNCTION TX,TY,TZ GX,GY,GZ TG_LENGTHS BUFFER...
//
// SOURCE is compiled with the device's MSL compiler (fast math off is not requested: the kernels of these tests are
// in the exact subset). The function is dispatched with dispatchThreadgroups: GX x GY x GZ threadgroups of
// TX x TY x TZ threads. TG_LENGTHS is a comma-separated list of threadgroup memory lengths by index ("-" for none).
// Each BUFFER is a file whose bytes are bound at the buffer index of its position; after the dispatch completes the
// buffer's contents are written to FILE.out.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
if args.count < 6 { fail("usage: metal_run SOURCE FUNCTION TX,TY,TZ GX,GY,GZ TG_LENGTHS BUFFER...") }
func triple(_ s: String) -> MTLSize {
    let v = s.split(separator: ",").map { Int($0)! }
    return MTLSize(width: v[0], height: v[1], depth: v[2])
}
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let source = try! String(contentsOfFile: args[1], encoding: .utf8)
let options = MTLCompileOptions()
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: options) } catch { fail("compile: \(error)") }
guard let function = library.makeFunction(name: args[2]) else { fail("no function \(args[2])") }
let pipeline: MTLComputePipelineState
do { pipeline = try device.makeComputePipelineState(function: function) } catch { fail("pipeline: \(error)") }
let threads = triple(args[3])
let groups = triple(args[4])
let tgLengths: [Int] = args[5] == "-" ? [] : args[5].split(separator: ",").map { Int($0)! }
var buffers: [MTLBuffer] = []
for path in args[6...] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    let buffer = data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max(data.count, 4), options: .storageModeShared)! }
    buffers.append(buffer)
}
let queue = device.makeCommandQueue()!
let cb = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
for (i, l) in tgLengths.enumerated() where l > 0 { enc.setThreadgroupMemoryLength(l, index: i) }
enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fail("command buffer: \(e)") }
for (i, path) in args[6...].enumerated() {
    let b = buffers[i]
    let n = (try! Data(contentsOf: URL(fileURLWithPath: path))).count
    let data = Data(bytes: b.contents(), count: n)
    try! data.write(to: URL(fileURLWithPath: path + ".out"))
}
