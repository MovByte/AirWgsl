// Runs one Metal compute function with textures and samplers on this Mac's GPU: the reference side of the texture
// differential tests.
//
// Usage: metal_tex SOURCE.metal SPEC
//
// SPEC holds one directive per line:
//   library FILE                          a metallib to load instead of compiling SOURCE
//   function NAME
//   threads TX TY TZ
//   groups GX GY GZ
//   buffer INDEX FILE                     bound at INDEX; written back to FILE.out after the dispatch
//   texture INDEX TYPE FORMAT W H D ARRAY LEVELS BPP WRITE SR SG SB SA FILE
//                                         MTLTextureType, MTLPixelFormat, size, array length, mipmap levels, bytes per
//                                         texel, 1 when the function writes it, the view swizzle (MTLTextureSwizzle);
//                                         FILE holds every level, each level's slices (faces) one after the other,
//                                         rows tightly packed
//   null INDEX                            no texture at INDEX (setTexture:nil)
//   sampler INDEX MIN MAG MIP S T R NORMALIZED COMPARE ANISOTROPY LODMIN LODMAX
//   readback INDEX FILE                   level 0 of the texture at INDEX, all slices, written to FILE

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
if args.count != 3 { fail("usage: metal_tex SOURCE SPEC") }
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let source = try! String(contentsOfFile: args[1], encoding: .utf8)
let spec = try! String(contentsOfFile: args[2], encoding: .utf8)
let queue = device.makeCommandQueue()!

var function = ""
var libraryPath: String? = nil
var threads = MTLSize(width: 1, height: 1, depth: 1)
var groups = MTLSize(width: 1, height: 1, depth: 1)
var buffers: [(Int, MTLBuffer, String, Int)] = []
var textures: [(Int, MTLTexture?)] = []
var storedTextures: [Int: (MTLTexture, Int, Int)] = [:]
var samplers: [(Int, MTLSamplerState)] = []
var readbacks: [(Int, String)] = []

func blit(_ body: (MTLBlitCommandEncoder) -> Void) {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeBlitCommandEncoder()!
    body(e)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let err = cb.error { fail("blit: \(err)") }
}

for line in spec.split(separator: "\n") {
    let f = line.split(separator: " ").map(String.init)
    switch f[0] {
    case "library": libraryPath = f[1]
    case "function": function = f[1]
    case "threads": threads = MTLSize(width: Int(f[1])!, height: Int(f[2])!, depth: Int(f[3])!)
    case "groups": groups = MTLSize(width: Int(f[1])!, height: Int(f[2])!, depth: Int(f[3])!)
    case "buffer":
        let data = try! Data(contentsOf: URL(fileURLWithPath: f[2]))
        let b = data.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: max(data.count, 16), options: .storageModeShared)!
        }
        buffers.append((Int(f[1])!, b, f[2], data.count))
    case "texture":
        let v = f[1...14].map { Int($0)! }
        let (index, type, format, w, h, d, array, levels, bpp, write) =
            (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])
        let desc = MTLTextureDescriptor()
        desc.textureType = MTLTextureType(rawValue: UInt(type))!
        desc.pixelFormat = MTLPixelFormat(rawValue: UInt(format))!
        desc.width = w
        desc.height = h
        desc.depth = d
        desc.arrayLength = array
        desc.mipmapLevelCount = levels
        desc.storageMode = .private
        var usage: MTLTextureUsage = [.shaderRead, .pixelFormatView]
        if write == 1 { usage.insert(.shaderWrite) }
        desc.usage = usage
        guard let t = device.makeTexture(descriptor: desc) else { fail("texture \(index)") }
        let data = try! Data(contentsOf: URL(fileURLWithPath: f[15]))
        let staging = data.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: max(data.count, 16), options: .storageModeShared)!
        }
        let slices = (type == 5 ? 6 : 1) * array
        var offset = 0
        blit { e in
            for l in 0..<levels {
                let lw = max(1, w >> l), lh = max(1, h >> l), ld = type == 7 ? max(1, d >> l) : 1
                let image = lw * lh * bpp
                for s in 0..<slices {
                    e.copy(from: staging, sourceOffset: offset, sourceBytesPerRow: lw * bpp,
                           sourceBytesPerImage: image, sourceSize: MTLSize(width: lw, height: lh, depth: ld),
                           to: t, destinationSlice: s, destinationLevel: l,
                           destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                    offset += image * ld
                }
            }
        }
        let sw = MTLTextureSwizzleChannels(red: MTLTextureSwizzle(rawValue: UInt8(v[10]))!,
                                           green: MTLTextureSwizzle(rawValue: UInt8(v[11]))!,
                                           blue: MTLTextureSwizzle(rawValue: UInt8(v[12]))!,
                                           alpha: MTLTextureSwizzle(rawValue: UInt8(v[13]))!)
        var bound: MTLTexture = t
        if v[10] != 2 || v[11] != 3 || v[12] != 4 || v[13] != 5 {
            bound = t.makeTextureView(pixelFormat: t.pixelFormat, textureType: t.textureType,
                                      levels: 0..<levels, slices: 0..<slices, swizzle: sw)!
        }
        textures.append((index, bound))
        storedTextures[index] = (t, bpp, slices)
    case "null": textures.append((Int(f[1])!, nil))
    case "sampler":
        let v = f[1...].map { Double($0)! }
        let d = MTLSamplerDescriptor()
        d.minFilter = MTLSamplerMinMagFilter(rawValue: UInt(v[1]))!
        d.magFilter = MTLSamplerMinMagFilter(rawValue: UInt(v[2]))!
        d.mipFilter = MTLSamplerMipFilter(rawValue: UInt(v[3]))!
        d.sAddressMode = MTLSamplerAddressMode(rawValue: UInt(v[4]))!
        d.tAddressMode = MTLSamplerAddressMode(rawValue: UInt(v[5]))!
        d.rAddressMode = MTLSamplerAddressMode(rawValue: UInt(v[6]))!
        d.normalizedCoordinates = v[7] != 0
        d.compareFunction = MTLCompareFunction(rawValue: UInt(v[8]))!
        d.maxAnisotropy = Int(v[9])
        d.lodMinClamp = Float(v[10])
        d.lodMaxClamp = Float(v[11])
        samplers.append((Int(v[0]), device.makeSamplerState(descriptor: d)!))
    case "readback": readbacks.append((Int(f[1])!, f[2]))
    default: fail("unknown directive \(f[0])")
    }
}

let library: MTLLibrary
do {
    if let p = libraryPath {
        library = try device.makeLibrary(URL: URL(fileURLWithPath: p))
    } else {
        library = try device.makeLibrary(source: source, options: MTLCompileOptions())
    }
} catch { fail("compile: \(error)") }
guard let fn = library.makeFunction(name: function) else { fail("no function \(function)") }
let pipeline: MTLComputePipelineState
do { pipeline = try device.makeComputePipelineState(function: fn) } catch { fail("pipeline: \(error)") }
let cb = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
for (i, b, _, _) in buffers { enc.setBuffer(b, offset: 0, index: i) }
for (i, t) in textures { enc.setTexture(t, index: i) }
for (i, s) in samplers { enc.setSamplerState(s, index: i) }
enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fail("command buffer: \(e)") }
for (_, b, path, n) in buffers {
    try! Data(bytes: b.contents(), count: n).write(to: URL(fileURLWithPath: path + ".out"))
}
for (index, path) in readbacks {
    let (t, bpp, slices) = storedTextures[index]!
    let image = t.width * t.height * bpp * (t.textureType == .type3D ? t.depth : 1)
    let out = device.makeBuffer(length: image * slices, options: .storageModeShared)!
    blit { e in
        for s in 0..<slices {
            e.copy(from: t, sourceSlice: s, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                   sourceSize: MTLSize(width: t.width, height: t.height, depth: t.textureType == .type3D ? t.depth : 1),
                   to: out, destinationOffset: s * image, destinationBytesPerRow: t.width * bpp,
                   destinationBytesPerImage: t.width * t.height * bpp)
        }
    }
    try! Data(bytes: out.contents(), count: image * slices).write(to: URL(fileURLWithPath: path))
}
