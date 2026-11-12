// Measures whether Metal accepts a specialization that gives two arguments of one binding kind the same location:
// a texture and a buffer whose index is a function constant, set to collide (k = 0) and not (k = 1).
//
// Usage: swift binding_locations.swift
//
// Apple M4, macOS 27.0.1: k = 0 fails newFunctionWithName:constantValues: with MTLLibraryErrorCompileFailure ("texture
// argument b has invalid location (<0,1>)", and the same for the buffer); k = 1 makes a function and a pipeline.

import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
print("# \(ProcessInfo.processInfo.operatingSystemVersionString), \(device.name)")
let src = """
#include <metal_stdlib>
using namespace metal;
constant int k [[function_constant(0)]];
kernel void tex(texture2d<float> a [[texture(0)]],
                texture2d<float, access::write> b [[texture(k)]],
                uint2 i [[thread_position_in_grid]]) { b.write(a.read(i), i); }
kernel void buf(device const float *a [[buffer(0)]],
                device float *b [[buffer(k)]],
                uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
"""
let lib = try! device.makeLibrary(source: src, options: MTLCompileOptions())
for name in ["tex", "buf"] {
    for v in [Int32(0), Int32(1)] {
        var x = v
        let cv = MTLFunctionConstantValues()
        cv.setConstantValue(&x, type: .int, index: 0)
        do {
            let f = try lib.makeFunction(name: name, constantValues: cv)
            do {
                let p = try device.makeComputePipelineState(function: f)
                _ = p
                print("\(name) k=\(v): function ok, pipeline ok")
            } catch {
                print("\(name) k=\(v): function ok, pipeline error: \((error as NSError).code) \(error.localizedDescription.prefix(200))")
            }
        } catch {
            print("\(name) k=\(v): function error \((error as NSError).domain) \((error as NSError).code): \(error.localizedDescription.prefix(300))")
        }
    }
}
