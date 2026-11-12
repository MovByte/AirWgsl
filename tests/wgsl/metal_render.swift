// Draws one triangle with a vertex and a fragment function on this Mac's GPU into a W x H RGBA32Uint target and
// writes the target and the fragment stage's buffers back: the reference side of the render differential tests.
//
// Usage: metal_render SOURCE.metal VERTEX FRAGMENT W H VERTEX_COUNT FRAGMENT_BUFFER...
//
// The draw is drawPrimitives(.triangle, vertexStart: 0, vertexCount: VERTEX_COUNT) with no vertex buffers. The
// target starts as zero; afterwards its bytes (row-major, 16 bytes a pixel) go to SOURCE.target and each fragment
// buffer FILE (bound at the fragment buffer index of its position) to FILE.out.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
if args.count < 7 { fail("usage: metal_render SOURCE VERTEX FRAGMENT W H VERTEX_COUNT BUFFER...") }
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let source = try! String(contentsOfFile: args[1], encoding: .utf8)
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: MTLCompileOptions()) } catch { fail("compile: \(error)") }
let w = Int(args[4])!, h = Int(args[5])!, count = Int(args[6])!
let desc = MTLRenderPipelineDescriptor()
desc.vertexFunction = library.makeFunction(name: args[2])
desc.fragmentFunction = library.makeFunction(name: args[3])
desc.colorAttachments[0].pixelFormat = .rgba32Uint
let pipeline: MTLRenderPipelineState
do { pipeline = try device.makeRenderPipelineState(descriptor: desc) } catch { fail("pipeline: \(error)") }
let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Uint, width: w, height: h, mipmapped: false)
td.usage = [.renderTarget, .shaderRead]
td.storageMode = .shared
let target = device.makeTexture(descriptor: td)!
var buffers: [MTLBuffer] = []
for path in args[7...] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    buffers.append(data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max(data.count, 4), options: .storageModeShared)! })
}
let rp = MTLRenderPassDescriptor()
rp.colorAttachments[0].texture = target
rp.colorAttachments[0].loadAction = .clear
rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
rp.colorAttachments[0].storeAction = .store
let queue = device.makeCommandQueue()!
let cb = queue.makeCommandBuffer()!
let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
enc.setRenderPipelineState(pipeline)
for (i, b) in buffers.enumerated() { enc.setFragmentBuffer(b, offset: 0, index: i) }
enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: count)
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fail("command buffer: \(e)") }
var bytes = [UInt8](repeating: 0, count: w * h * 16)
target.getBytes(&bytes, bytesPerRow: w * 16, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
try! Data(bytes).write(to: URL(fileURLWithPath: args[1] + ".target"))
for (i, path) in args[7...].enumerated() {
    let n = (try! Data(contentsOf: URL(fileURLWithPath: path))).count
    try! Data(bytes: buffers[i].contents(), count: n).write(to: URL(fileURLWithPath: path + ".out"))
}
