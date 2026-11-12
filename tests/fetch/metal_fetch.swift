// Runs a framebuffer-fetch scene on this Mac's GPU: the reference side of tests/fetch.
//
// Usage: metal_fetch SCENE.json
//
// The scene names the MSL source and its vertex and fragment functions, the color attachment (MTLPixelFormat, width,
// height, sample count), its initial contents (a file of texels, or a clear value for a multisampled attachment), the
// blending of attachment 0 (none, or one plus one), the primitive type and the draws: each draws vertexStart, count
// and instanceCount with the vertex buffer file at vertex index 0 and, when the scene has one, a fragment buffer
// (shared by the draws) at fragment index 0. The attachment's texels (per sample, sample-major within a pixel, as u32
// words for a multisampled 32-bit integer attachment) go to `out`, the fragment buffer to `out` + ".buffer". A
// pipeline Metal refuses is reported on standard output as "refused: ..." with exit status 0.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let scene = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
    as! [String: Any]
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let source = try! String(contentsOfFile: scene["source"] as! String, encoding: .utf8)
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: MTLCompileOptions()) } catch { fail("compile: \(error)") }
let format = MTLPixelFormat(rawValue: scene["format"] as! UInt)!
let w = scene["width"] as! Int, h = scene["height"] as! Int, samples = scene["samples"] as! Int
let bpp = scene["bpp"] as! Int
let d = MTLRenderPipelineDescriptor()
d.vertexFunction = library.makeFunction(name: scene["vertex"] as! String)
d.fragmentFunction = library.makeFunction(name: scene["fragment"] as! String)
d.colorAttachments[0].pixelFormat = format
d.rasterSampleCount = samples
if !(scene["write"] as! Bool) { d.colorAttachments[0].writeMask = [] }
if scene["blend"] as! Bool {
    d.colorAttachments[0].isBlendingEnabled = true
    d.colorAttachments[0].sourceRGBBlendFactor = .one
    d.colorAttachments[0].destinationRGBBlendFactor = .one
    d.colorAttachments[0].sourceAlphaBlendFactor = .one
    d.colorAttachments[0].destinationAlphaBlendFactor = .one
}
let pipeline: MTLRenderPipelineState
do { pipeline = try device.makeRenderPipelineState(descriptor: d) } catch {
    print("refused: \((error as NSError).localizedDescription)")
    exit(0)
}
let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
td.usage = [.renderTarget, .shaderRead]
if samples > 1 {
    td.textureType = .type2DMultisample
    td.sampleCount = samples
    td.storageMode = .private
} else {
    td.storageMode = .shared
}
let target = device.makeTexture(descriptor: td)!
let rp = MTLRenderPassDescriptor()
rp.colorAttachments[0].texture = target
rp.colorAttachments[0].storeAction = .store
if let initial = scene["initial"] as? String {
    let bytes = try! Data(contentsOf: URL(fileURLWithPath: initial))
    bytes.withUnsafeBytes {
        target.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: w * bpp)
    }
    rp.colorAttachments[0].loadAction = .load
} else {
    let c = scene["clear"] as! [Double]
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].clearColor = MTLClearColor(red: c[0], green: c[1], blue: c[2], alpha: c[3])
}
let fragmentBuffer: MTLBuffer? = (scene["fragmentBuffer"] as? String).map { path in
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! }
}
let primitive = MTLPrimitiveType(rawValue: scene["primitive"] as! UInt)!
let queue = device.makeCommandQueue()!
let cb = queue.makeCommandBuffer()!
let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
enc.setRenderPipelineState(pipeline)
if let b = fragmentBuffer { enc.setFragmentBuffer(b, offset: 0, index: 0) }
for draw in scene["draws"] as! [[String: Any]] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: draw["vertices"] as! String))
    let vb = data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! }
    enc.setVertexBuffer(vb, offset: 0, index: 0)
    enc.drawPrimitives(type: primitive, vertexStart: draw["start"] as! Int, vertexCount: draw["count"] as! Int,
                       instanceCount: draw["instances"] as! Int)
}
enc.endEncoding()
let out = scene["out"] as! String
var readback: MTLBuffer? = nil
if samples > 1 {
    let k = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void readms(texture2d_ms<uint> t [[texture(0)]], device uint *o [[buffer(0)]],
                       uint2 g [[thread_position_in_grid]]) {
        for (uint s = 0; s < t.get_num_samples(); s++) o[(g.y * t.get_width() + g.x) * t.get_num_samples() + s] = t.read(g, s).x;
    }
    """
    let kl = try! device.makeLibrary(source: k, options: MTLCompileOptions())
    let cp = try! device.makeComputePipelineState(function: kl.makeFunction(name: "readms")!)
    readback = device.makeBuffer(length: w * h * samples * 4, options: .storageModeShared)!
    let c = cb.makeComputeCommandEncoder()!
    c.setComputePipelineState(cp)
    c.setTexture(target, index: 0)
    c.setBuffer(readback, offset: 0, index: 0)
    c.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    c.endEncoding()
}
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fail("command buffer: \(e)") }
if let r = readback {
    try! Data(bytes: r.contents(), count: w * h * samples * 4).write(to: URL(fileURLWithPath: out))
} else {
    var bytes = [UInt8](repeating: 0, count: w * h * bpp)
    target.getBytes(&bytes, bytesPerRow: w * bpp, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    try! Data(bytes).write(to: URL(fileURLWithPath: out))
}
if let b = fragmentBuffer {
    try! Data(bytes: b.contents(), count: b.length).write(to: URL(fileURLWithPath: out + ".buffer"))
}
