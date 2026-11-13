// Draws tessellated patches with a post-tessellation vertex function on this Mac's GPU and writes the render target:
// the reference side of the full-pipeline tessellation tests.
//
// Usage: metal_tess SPEC OUTPUT
//
// SPEC is a text file of `key value...` lines: source (path of the MSL file), vertex, fragment (function names),
// quad (0 or 1), mode, max, winding (pipeline tessellation state), indexed (0 none, 1 UInt16, 2 UInt32 control-point
// indices: the draw is then drawIndexedPatches), points (control points per patch), patches, instances, base (base
// instance), width, height, factors (binary16 bit patterns, one MTLQuadTessellationFactorsHalf or
// MTLTriangleTessellationFactorsHalf per patch), cp (buffer 1: one float4 per control point, as binary32 bit
// patterns), tags (buffer 2: one uint per patch), cpindices (the control-point index buffer's elements). Attribute 0
// is a float4 of buffer 1 (MTLVertexStepFunctionPerPatchControlPoint), attribute 1 a uint of buffer 2
// (MTLVertexStepFunctionPerPatch); the factors step per patch. The target is R32Uint, cleared to 0.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
if args.count != 3 { fail("usage: metal_tess SPEC OUTPUT") }
var spec: [String: [String]] = [:]
for line in try! String(contentsOfFile: args[1], encoding: .utf8).split(separator: "\n") {
    let parts = line.split(separator: " ").map(String.init)
    if let k = parts.first { spec[k] = Array(parts.dropFirst()) }
}
func int(_ k: String) -> Int { Int(spec[k]![0])! }
func words(_ k: String) -> [UInt32] { (spec[k] ?? []).map { UInt32($0)! } }

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let source = try! String(contentsOfFile: spec["source"]![0], encoding: .utf8)
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: MTLCompileOptions()) } catch { fail("compile: \(error)") }
let desc = MTLRenderPipelineDescriptor()
desc.vertexFunction = library.makeFunction(name: spec["vertex"]![0])
desc.fragmentFunction = library.makeFunction(name: spec["fragment"]![0])
desc.colorAttachments[0].pixelFormat = .r32Uint
desc.tessellationPartitionMode = MTLTessellationPartitionMode(rawValue: UInt(int("mode")))!
desc.maxTessellationFactor = int("max")
desc.tessellationOutputWindingOrder = MTLWinding(rawValue: UInt(int("winding")))!
desc.tessellationFactorStepFunction = .perPatch
desc.tessellationControlPointIndexType = MTLTessellationControlPointIndexType(rawValue: UInt(int("indexed")))!
let vd = MTLVertexDescriptor()
vd.attributes[0].format = .float4
vd.attributes[0].offset = 0
vd.attributes[0].bufferIndex = 1
vd.layouts[1].stride = 16
vd.layouts[1].stepFunction = .perPatchControlPoint
vd.attributes[1].format = .uint
vd.attributes[1].offset = 0
vd.attributes[1].bufferIndex = 2
vd.layouts[2].stride = 4
vd.layouts[2].stepFunction = .perPatch
desc.vertexDescriptor = vd
let pipeline: MTLRenderPipelineState
do { pipeline = try device.makeRenderPipelineState(descriptor: desc) } catch { fail("pipeline: \(error)") }

func buffer(_ w: [UInt32]) -> MTLBuffer {
    let data = w.isEmpty ? [UInt32(0)] : w
    return data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}
func buffer16(_ w: [UInt32]) -> MTLBuffer {
    let data = (w.isEmpty ? [UInt32(0)] : w).map { UInt16($0) } + [0]
    return data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}
let factors = buffer16(words("factors"))
let cp = buffer(words("cp"))
let tags = buffer(words("tags"))
let indexed = int("indexed")
let cpIndices = indexed == 1 ? buffer16(words("cpindices")) : buffer(words("cpindices"))
let width = int("width"), height = int("height")
let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Uint, width: width, height: height, mipmapped: false)
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
enc.setVertexBuffer(cp, offset: 0, index: 1)
enc.setVertexBuffer(tags, offset: 0, index: 2)
enc.setTessellationFactorBuffer(factors, offset: 0, instanceStride: 0)
if indexed != 0 {
    enc.drawIndexedPatches(numberOfPatchControlPoints: int("points"), patchStart: 0, patchCount: int("patches"),
                           patchIndexBuffer: nil, patchIndexBufferOffset: 0, controlPointIndexBuffer: cpIndices,
                           controlPointIndexBufferOffset: 0, instanceCount: int("instances"), baseInstance: int("base"))
} else {
    enc.drawPatches(numberOfPatchControlPoints: int("points"), patchStart: 0, patchCount: int("patches"),
                    patchIndexBuffer: nil, patchIndexBufferOffset: 0, instanceCount: int("instances"),
                    baseInstance: int("base"))
}
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fail("command buffer: \(e)") }
var out = [UInt8](repeating: 0, count: width * height * 4)
target.getBytes(&out, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
FileManager.default.createFile(atPath: args[2], contents: Data(out))
