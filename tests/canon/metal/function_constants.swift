// Measures how Metal specializes function constants: which argument predicates keep their argument, what a missing
// value does, whether a value of another type is accepted, and what a bool byte other than 0 or 1 reads as.
//
// Usage: swift function_constants.swift OUTPUT LLVM_DIS
//
// The specialized code is read back through the private `bitcodeData` of the function that
// `newFunctionWithName:constantValues:error:` returns, disassembled with LLVM_DIS (llvm-dis of LLVM 21).
// OUTPUT gets one line per case; tests/canon/main.rs checks the passes against it.

import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let out = CommandLine.arguments[1]
let llvmDis = CommandLine.arguments[2]
var lines: [String] = ["# Metal \(ProcessInfo.processInfo.operatingSystemVersionString), \(device.name)"]

func library(_ src: String) -> MTLLibrary {
    try! device.makeLibrary(source: "#include <metal_stdlib>\nusing namespace metal;\n" + src, options: MTLCompileOptions())
}

func specializedText(_ f: MTLFunction) -> String {
    let d = (f as AnyObject).perform(NSSelectorFromString("bitcodeData"))!.takeUnretainedValue()
    let data: Data = (d as? Data) ?? Data(d as! DispatchData)
    let bc = NSTemporaryDirectory() + "fc-probe.bc"
    try! data.write(to: URL(fileURLWithPath: bc))
    let p = Process()
    p.executableURL = URL(fileURLWithPath: llvmDis)
    p.arguments = [bc, "-o", "-"]
    let pipe = Pipe()
    p.standardOutput = pipe
    try! p.run()
    let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!
    p.waitUntilExit()
    return text
}

func hex(_ b: [UInt8]) -> String { b.map { String(format: "%02x", $0) }.joined() }

// Predicates: an argument per predicate type; the kept arguments are those the specialized metadata still lists.
let predicates = library("""
constant int p_int [[function_constant(0)]];
constant bool2 p_bool2 [[function_constant(1)]];
constant float p_float [[function_constant(2)]];
constant uchar p_uchar [[function_constant(3)]];
kernel void k(device float *a_int [[buffer(5), function_constant(p_int)]],
              device float *a_bool2 [[buffer(6), function_constant(p_bool2)]],
              device float *a_float [[buffer(7), function_constant(p_float)]],
              device float *a_uchar [[buffer(8), function_constant(p_uchar)]],
              device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = 1; }
""")
let predicateCases: [(String, MTLDataType, Int, [[UInt8]])] = [
    ("int", .int, 0, [[2, 0, 0, 0], [0, 0, 0, 0], [0xff, 0xff, 0xff, 0xff], [1, 0, 0, 0]]),
    ("bool2", .bool2, 1, [[1, 0], [0, 0], [0, 1], [1, 1]]),
    ("float", .float, 2, [[0, 0, 0, 0x80], [0, 0, 0, 0], [0, 0, 0xc0, 0x7f], [0, 0, 0x80, 0x3f], [1, 0, 0, 0], [0, 0, 0x80, 0x7f]]),
    ("uchar", .uchar, 3, [[2], [0], [1], [255]]),
]
for (name, type, index, values) in predicateCases {
    for v in values {
        let cv = MTLFunctionConstantValues()
        var b = v
        cv.setConstantValue(&b, type: type, index: index)
        let f = try! predicates.makeFunction(name: "k", constantValues: cv)
        let kept = specializedText(f).contains("!\"a_\(name)\"")
        lines.append("predicate \(name) \(hex(v)) \(kept ? "kept" : "dropped")")
    }
    let f = try! predicates.makeFunction(name: "k", constantValues: MTLFunctionConstantValues())
    lines.append("predicate \(name) unset \(specializedText(f).contains("!\"a_\(name)\"") ? "kept" : "dropped")")
}

// A used constant without a value, and a defaulted one.
let missing = library("""
constant int m_used [[function_constant(0)]];
constant int m_maybe [[function_constant(1)]];
constant int m_default = is_function_constant_defined(m_maybe) ? m_maybe : 9;
kernel void used(device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = m_used; }
kernel void defaulted(device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = m_default; }
""")
for name in ["used", "defaulted"] {
    do {
        let f = try missing.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
        let body = specializedText(f).components(separatedBy: "\n").filter { $0.hasPrefix("  store") || $0.hasPrefix("  ret") }
        lines.append("missing \(name) ok \(body.map { $0.trimmingCharacters(in: .whitespaces).components(separatedBy: ",").first! }.joined(separator: "; "))")
    } catch {
        lines.append("missing \(name) error \((error as NSError).code)")
    }
}

// Values of another type.
let typed = library("""
constant int t_int [[function_constant(0)]];
constant uint t_uint [[function_constant(1)]];
constant bool t_bool [[function_constant(2)]];
constant bool2 t_bool2 [[function_constant(3)]];
kernel void ki(device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = uint(t_int); }
kernel void ku(device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = t_uint; }
kernel void kb(device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = t_bool ? 7u : 9u; }
kernel void kb2(device uint *r [[buffer(0)]], uint i [[thread_position_in_grid]]) { r[i] = uint(t_bool2.x) * 16u + uint(t_bool2.y); }
""")
let typedCases: [(String, String, Int, MTLDataType, [UInt8])] = [
    ("ki", "int", 0, .float, [0, 0, 0x80, 0x3f]),
    ("ki", "int", 0, .uint, [5, 0, 0, 0]),
    ("ki", "int", 0, .bool, [1]),
    ("ku", "uint", 1, .int, [5, 0, 0, 0]),
    ("kb", "bool", 2, .uchar, [1]),
    ("kb", "bool", 2, .bool, [1]),
    ("kb", "bool", 2, .bool, [2]),
    ("kb2", "bool2", 3, .bool2, [1, 0]),
    ("kb2", "bool2", 3, .bool2, [2, 3]),
]
for (fname, tname, index, type, v) in typedCases {
    let cv = MTLFunctionConstantValues()
    var b = v
    cv.setConstantValue(&b, type: type, index: index)
    do {
        let f = try typed.makeFunction(name: fname, constantValues: cv)
        let store = specializedText(f).components(separatedBy: "\n").first { $0.hasPrefix("  store") } ?? ""
        let value = store.trimmingCharacters(in: .whitespaces).components(separatedBy: ",").first ?? ""
        lines.append("typed \(tname) \(type.rawValue) \(hex(v)) ok \(value)")
    } catch {
        lines.append("typed \(tname) \(type.rawValue) \(hex(v)) error \((error as NSError).code)")
    }
}
try! (lines.joined(separator: "\n") + "\n").write(toFile: out, atomically: true, encoding: .utf8)
