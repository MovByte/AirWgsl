//! Atomics, barriers, argument buffers, function tables and indirect command buffers through the registered rules.
//!
//! The kernels run on real Metal (this Mac), on wgpu and on Dawn (`AW_DAWN_NODE`) through the WGSL differential
//! harness, with byte-identical buffers required. Kernels with contended atomics keep only results that do not depend
//! on the order the threads ran in: sums, differences, bitwise and, or and xor, minima and maxima, counts, and for
//! exchange the sum of the returned values when every thread of a word stores the same value.

#[allow(dead_code)]
#[path = "../wgsl/harness.rs"]
mod harness;

use airwgsl::api::*;
use airwgsl::wgsl::{self, hooks, tracked};
use harness::{air, bytes, case_dir, compile_kernel, differential};

const HEAD: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

fn report(name: &str, r: harness::Ran) {
    eprintln!("{}: Metal = wgpu{}", name, if r.dawn { " = Dawn" } else { " (Dawn not run: AW_DAWN_NODE unset)" });
}

const CONTEND: &str = r#"
#define R memory_order_relaxed
kernel void contend(device const uint* in [[buffer(0)]], device atomic_uint* u [[buffer(1)]],
                    device atomic_int* s [[buffer(2)]], device atomic_uint* x [[buffer(3)]],
                    device atomic_bool* flags [[buffer(4)]], device uint* out [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    uint v = in[i];
    uint k = i % 4u;
    atomic_fetch_add_explicit(&u[k], v, R);
    atomic_fetch_sub_explicit(&u[4u + k], v, R);
    atomic_fetch_and_explicit(&u[8u + k], v | 0x0f0f0f0fu, R);
    atomic_fetch_or_explicit(&u[12u + k], v & 0xf0f0f0f0u, R);
    atomic_fetch_xor_explicit(&u[16u + k], v, R);
    atomic_fetch_min_explicit(&u[20u + k], v, R);
    atomic_fetch_max_explicit(&u[24u + k], v, R);
    int sv = int(v);
    atomic_fetch_min_explicit(&s[k], sv, R);
    atomic_fetch_max_explicit(&s[4u + k], sv, R);
    atomic_fetch_add_explicit(&s[8u + k], sv >> 8, R);
    atomic_fetch_sub_explicit(&s[12u + k], sv >> 4, R);
    // An add through a compare-exchange loop.
    uint e = atomic_load_explicit(&u[28u + k], R);
    while (!atomic_compare_exchange_weak_explicit(&u[28u + k], &e, e + v, R, R)) {
    }
    // Every thread of a word exchanges the same value: the returned values sum to the initial value plus
    // (n - 1) times that value.
    uint old = atomic_exchange_explicit(&x[k], 0x01010101u * (k + 1u), R);
    atomic_fetch_add_explicit(&x[4u + k], old, R);
    // Exactly one thread of each flag sees it clear.
    bool was = atomic_exchange_explicit(&flags[k], true, R);
    if (!was) {
        atomic_fetch_add_explicit(&x[8u + k], 1u, R);
    }
    // No contention: a store and a load of the thread's own word.
    atomic_store_explicit(&x[12u + i], v ^ 0x5a5a5a5au, R);
    out[i] = atomic_load_explicit(&x[12u + i], R) + uint(atomic_load_explicit(&flags[k], R));
}
"#;

#[test]
fn device_atomics_under_contention() {
    let n = 256;
    let inputs = bytes(21, n * 4);
    let u = bytes(22, 32 * 4);
    let s = bytes(23, 16 * 4);
    let mut x = vec![0u8; (12 + n) * 4];
    x[..16].copy_from_slice(&bytes(24, 16));
    let flags = vec![0u8; 64];
    let out = vec![0xa5u8; n * 4];
    let r = differential(
        "contend",
        &format!("{}{}", HEAD, CONTEND),
        "contend",
        [64, 1, 1],
        [4, 1, 1],
        &[],
        &[inputs, u, s, x, flags, out],
    );
    report("contend", r);
}

const HISTOGRAM: &str = r#"
#define R memory_order_relaxed
kernel void histogram(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                      threadgroup atomic_uint* bins [[threadgroup(0)]], threadgroup atomic_int* sbins [[threadgroup(1)]],
                      uint lid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                      uint i [[thread_position_in_grid]]) {
    if (lid < 21u) {
        atomic_store_explicit(&bins[lid], 0u, R);
    }
    if (lid < 4u) {
        atomic_store_explicit(&sbins[lid], 0, R);
        atomic_store_explicit(&sbins[4u + lid], -1, R);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint v = in[i];
    atomic_fetch_add_explicit(&bins[v & 15u], 1u, R);
    atomic_fetch_max_explicit(&bins[16u + (v & 3u)], v >> 4, R);
    atomic_fetch_min_explicit(&sbins[(v >> 4) & 3u], int(v), R);
    atomic_fetch_and_explicit(&sbins[4u + ((v >> 6) & 3u)], int(v | 0x11111111u), R);
    // Only the first compare-exchange of the word finds it 0.
    uint e = 0u;
    bool first = atomic_compare_exchange_weak_explicit(&bins[20u], &e, 7u, R, R);
    atomic_fetch_add_explicit(&bins[20u], first ? 1u : 0x100u, R);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 21u) {
        out[gid * 29u + lid] = atomic_load_explicit(&bins[lid], R);
    }
    if (lid < 8u) {
        out[gid * 29u + 21u + lid] = uint(atomic_load_explicit(&sbins[lid], R));
    }
}
"#;

#[test]
fn threadgroup_atomics_between_barriers() {
    let n = 256;
    let inputs = bytes(31, n * 4);
    let out = vec![0xa5u8; 4 * 29 * 4];
    let r = differential(
        "histogram",
        &format!("{}{}", HEAD, HISTOGRAM),
        "histogram",
        [64, 1, 1],
        [4, 1, 1],
        &[84, 32],
        &[inputs, out],
    );
    report("histogram", r);
}

const REDUCE: &str = r#"
kernel void reduce(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                   threadgroup uint* t [[threadgroup(0)]], uint lid [[thread_index_in_threadgroup]],
                   uint gid [[threadgroup_position_in_grid]], uint i [[thread_position_in_grid]],
                   uint n [[threads_per_threadgroup]]) {
    t[lid] = in[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = n / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            t[lid] += t[lid + s] ^ s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[gid] = t[0];
    }
}

kernel void exchange(device uint* a [[buffer(0)]], device uint* b [[buffer(1)]], device uint* out [[buffer(2)]],
                     threadgroup uint* t [[threadgroup(0)]], uint lid [[thread_index_in_threadgroup]],
                     uint i [[thread_position_in_grid]]) {
    // Device memory written by a neighbor, ordered by a threadgroup barrier on device memory.
    a[i] = a[i] * 3u + lid;
    threadgroup_barrier(mem_flags::mem_device);
    out[3u * i] = a[i ^ 1u];
    // Both kinds of memory, ordered by one barrier with both flags.
    t[lid] = a[i] + 1u;
    b[i] = b[i] ^ lid;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    out[3u * i + 1u] = t[lid ^ 3u] + b[i ^ 2u];
    // A SIMD-group barrier in control flow uniform across the threadgroup; the neighbors are in the same SIMD-group.
    b[i] = b[i] + 7u;
    simdgroup_barrier(mem_flags::mem_device);
    threadgroup_barrier(mem_flags::mem_none);
    out[3u * i + 2u] = b[i ^ 1u];
}
"#;

#[test]
fn threadgroup_reduction_with_barriers() {
    let n = 256;
    let inputs = bytes(41, n * 4);
    let out = vec![0xa5u8; 4 * 4];
    let r =
        differential("reduce", &format!("{}{}", HEAD, REDUCE), "reduce", [64, 1, 1], [4, 1, 1], &[256], &[inputs, out]);
    report("reduce", r);
}

#[test]
fn barriers_order_device_and_threadgroup_memory() {
    let n = 256;
    let a = bytes(51, n * 4);
    let b = bytes(52, n * 4);
    let out = vec![0xa5u8; n * 3 * 4];
    let r = differential(
        "exchange",
        &format!("{}{}", HEAD, REDUCE),
        "exchange",
        [64, 1, 1],
        [4, 1, 1],
        &[256],
        &[a, b, out],
    );
    report("exchange", r);
}

const PARAMS: &str = r#"
struct Params {
    uint scale [[id(0)]];
    uint bias [[id(1)]];
    uint4 v [[id(2)]];
    ushort2 h [[id(3)]];
};
kernel void params(constant Params& p [[buffer(0)]], device uint* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    out[i] = i * p.scale + p.bias + p.v[i & 3u] + uint(p.h[i & 1u]);
}
"#;

#[test]
fn an_argument_buffer_of_constants_is_a_buffer() {
    let params = bytes(61, 48);
    let out = vec![0xa5u8; 256 * 4];
    let r =
        differential("params", &format!("{}{}", HEAD, PARAMS), "params", [64, 1, 1], [4, 1, 1], &[], &[params, out]);
    report("params", r);
}

// Refusals --------------------------------------------------------------------------------------------------------

fn refusal(name: &str, source: &str, function: &str, tg: &[u64]) -> Refusal {
    let dir = case_dir(name);
    let bc = air(&dir, &format!("{}{}", HEAD, source));
    match compile_kernel(&bc, function, [64, 1, 1], tg) {
        Ok(c) => panic!("{}: compiled:\n{}", name, c.wgsl[0].source),
        Err(r) => r,
    }
}

fn item(r: &Refusal) -> String {
    let (ns, id) = tracked::item_of(r).unwrap_or_else(|| panic!("untracked: {}", r));
    format!("{}:{}", ns.prefix(), id)
}

#[test]
fn a_barrier_in_non_uniform_control_flow_is_refused() {
    let src = r#"
kernel void k(device uint* o [[buffer(0)]], threadgroup uint* t [[threadgroup(0)]],
              uint lid [[thread_index_in_threadgroup]], uint i [[thread_position_in_grid]]) {
    t[lid] = o[i];
    if (lid < 5u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    o[i] = t[lid ^ 1u];
}
"#;
    let r = refusal("nonuniform", src, "k", &[256]);
    assert_eq!(item(&r), "compiler:barrier-uniformity", "{}", r);
    assert_eq!(r.kind, RefusalKind::NoExactLowering);
    assert!(r.reason.contains("local_invocation_index"), "{}", r);
    // Threads that returned early stop taking part in barriers on Apple GPUs; WGSL has no such form.
    let src = r#"
kernel void k(device uint* o [[buffer(0)]], threadgroup uint* t [[threadgroup(0)]], constant uint& n [[buffer(1)]],
              uint lid [[thread_index_in_threadgroup]], uint i [[thread_position_in_grid]]) {
    if (i >= n) {
        return;
    }
    t[lid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = t[lid ^ 1u];
}
"#;
    let r = refusal("early-return", src, "k", &[256]);
    assert_eq!(item(&r), "compiler:barrier-uniformity", "{}", r);
}

#[test]
fn a_barrier_in_a_fragment_function_is_refused() {
    let src = r#"
struct FIn { float4 pos [[position]]; };
fragment float4 fb(FIn in [[stage_in]], device uint* o [[buffer(0)]]) {
    o[0] = 1u;
    threadgroup_barrier(mem_flags::mem_device);
    return float4(float(o[1]));
}
"#;
    let dir = case_dir("fragment-barrier");
    let bc = air(&dir, &format!("{}{}", HEAD, src));
    let ir = airwgsl::parse_bitcode(&bc).unwrap();
    let module = airwgsl::air::AirModule::decode(ir).unwrap();
    let e = module.entry_points.iter().find(|e| e.name == "fb").unwrap().clone();
    let inp = wgsl::input_of(module, &e.name, e.stage.function_type()).unwrap();
    let r = wgsl::compile_entry(&inp, &HostCaps::default(), &hooks::registered(), [1, 1, 1], &[]).unwrap_err();
    assert_eq!(item(&r), "compiler:barrier-stage", "{}", r);
}

#[test]
fn indirect_command_encoding_is_refused_until_the_layout_is_defined() {
    let src = r#"
struct ICBContainer { command_buffer cmd [[id(0)]]; };
kernel void enc(device ICBContainer& c [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    render_command r(c.cmd, i);
    r.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}
"#;
    let r = refusal("icb", src, "enc", &[]);
    assert_eq!(item(&r), "compiler:icb-layout", "{}", r);
    assert_eq!(r.kind, RefusalKind::NotImplemented);
}

#[test]
fn function_tables_are_rejected_as_the_device_rejects_them() {
    let src = r#"
kernel void table(visible_function_table<int(int)> t [[buffer(1)]], device int* o [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
    o[i] = t[i & 1u](int(i));
}
"#;
    let r = refusal("table", src, "table", &[]);
    assert_eq!(item(&r), "family:function-tables-dynamic", "{}", r);
    assert_eq!(r.kind, RefusalKind::RejectedByDevice);
    assert!(r.reason.contains("supportsFunctionPointers"), "{}", r);
}

#[test]
fn argument_buffer_resources_are_refused_with_what_they_need() {
    let src = r#"
struct WithBuffer { device uint* b [[id(0)]]; uint n [[id(1)]]; };
kernel void withbuffer(constant WithBuffer& a [[buffer(0)]], device uint* out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = a.b[i % a.n];
}
"#;
    let r = refusal("argbuf-buffer", src, "withbuffer", &[]);
    assert_eq!(item(&r), "compiler:argument-buffer-buffer", "{}", r);
    let src = r#"
struct WithTexture { texture2d<float> t [[id(0)]]; };
kernel void withtexture(constant WithTexture& a [[buffer(0)]], device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[i] = a.t.read(uint2(i, 0)).x;
}
"#;
    let r = refusal("argbuf-texture", src, "withtexture", &[]);
    assert_eq!(item(&r), "compiler:argument-buffer-handle", "{}", r);
}
