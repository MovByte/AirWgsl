// Fetches every element of a vertex buffer through a vertex descriptor on this Mac's GPU and writes the fetched
// attributes back: the reference side of the attribute fetch tests.
//
// Usage: metal_fetch SOURCE.metal VERTEX FRAGMENT FORMAT STRIDE OFFSET INPUT COUNT OUTPUT
//
// Attribute 0 has MTLVertexFormat FORMAT at OFFSET in buffer 1, whose layout has stride STRIDE and the per-vertex
// step function; the buffer holds INPUT's bytes. COUNT points are drawn into a 256-wide RGBA32Uint target (the vertex
// function places vertex v at pixel (v % 256, v / 256)); the first COUNT pixels (16 bytes each) go to OUTPUT. A
// pipeline Metal refuses is reported on standard output as "refused: ..." with exit status 0.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
if args.count != 10 { fail("usage: metal_fetch SOURCE VERTEX FRAGMENT FORMAT STRIDE OFFSET INPUT COUNT OUTPUT") }
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let source = try! String(contentsOfFile: args[1], encoding: .utf8)
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: MTLCompileOptions()) } catch { fail("compile: \(error)") }
let desc = MTLRenderPipelineDescriptor()
desc.vertexFunction = library.makeFunction(name: args[2])
desc.fragmentFunction = library.makeFunction(name: args[3])
desc.colorAttachments[0].pixelFormat = .rgba32Uint
let vd = MTLVertexDescriptor()
vd.attributes[0].format = MTLVertexFormat(rawValue: UInt(args[4])!)!
vd.attributes[0].offset = Int(args[6])!
vd.attributes[0].bufferIndex = 1
vd.layouts[1].stride = Int(args[5])!
vd.layouts[1].stepFunction = .perVertex
desc.vertexDescriptor = vd
let pipeline: MTLRenderPipelineState
do { pipeline = try device.makeRenderPipelineState(descriptor: desc) } catch {
    print("refused: \((error as NSError).localizedDescription)")
    exit(0)
}
let input = try! Data(contentsOf: URL(fileURLWithPath: args[7]))
let count = Int(args[8])!
let buffer = input.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: input.count, options: .storageModeShared)! }
let height = (count + 255) / 256
let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Uint, width: 256, height: height, mipmapped: false)
td.usage = [.renderTarget]
td.storageMode = .shared
let target = device.makeTexture(descriptor: td)!
let rp = MTLRenderPassDescriptor()
rp.colorAttachments[0].texture = target
rp.colorAttachments[0].loadAction = .clear
rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
rp.colorAttachments[0].storeAction = .store
let queue = device.makeCommandQueue()!
let cb = queue.makeCommandBuffer()!
let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
enc.setRenderPipelineState(pipeline)
enc.setVertexBuffer(buffer, offset: 0, index: 1)
enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: count)
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fail("command buffer: \(e)") }
var bytes = [UInt8](repeating: 0, count: 256 * height * 16)
target.getBytes(&bytes, bytesPerRow: 256 * 16, from: MTLRegionMake2D(0, 0, 256, height), mipmapLevel: 0)
try! Data(bytes[0..<(count * 16)]).write(to: URL(fileURLWithPath: args[9]))
