//! The bitcode reader on the target's AIR (`xcrun metal` for `air64-apple-ios17.7`), which is written with typed
//! pointers: the pointee types the IR's opaque pointers drop stay in `Module::bitcode_types`.

use airwgsl::ir::{Module, Type, TypeId};
use std::path::PathBuf;
use std::process::Command;

fn air(name: &str, source: &str) -> Vec<u8> {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("reader-tests").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("k.metal");
    std::fs::write(&src, source).unwrap();
    let out = d.join("k.air");
    let o = Command::new("xcrun")
        .args(["-sdk", "iphoneos", "metal", "-std=metal3.1", "-target", "air64-apple-ios17.7", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("xcrun metal");
    assert!(o.status.success(), "metal: {}", String::from_utf8_lossy(&o.stderr));
    std::fs::read(&out).unwrap()
}

/// The IR type of the pointee of parameter `p` of function `f`, by the function's bitcode type.
fn param_pointee(m: &Module, f: usize, p: usize) -> TypeId {
    let fty = m.functions[f].bitcode_type.expect("function bitcode type");
    let fbt = &m.bitcode_types[fty as usize];
    // A function type's contained ids: the return type, then the parameters.
    let pbt = &m.bitcode_types[fbt.contained[p + 1] as usize];
    assert!(matches!(m.types.get(pbt.ty), Type::Ptr(_)), "parameter {} is not a pointer", p);
    m.bitcode_types[pbt.contained[0] as usize].ty
}

#[test]
fn typed_pointer_pointees_are_kept() {
    let bc = air(
        "pointees",
        "#include <metal_stdlib>\nusing namespace metal;\n\
         struct S { float4 a; uint b; };\n\
         kernel void k(device float4 *p [[buffer(0)]], const device S *s [[buffer(1)]],\n\
                       uint i [[thread_position_in_grid]]) { p[i] = s[i].a + float(s[i].b); }\n",
    );
    let m = airwgsl::parse_bitcode(&bc).expect("AIR");
    assert!(!m.bitcode_types.is_empty());
    // Every entry the reader kept is an IR type of the module.
    for bt in &m.bitcode_types {
        assert!((bt.ty.0 as usize) < m.types.len());
    }
    let f = m.function_by_name("k").expect("kernel k");
    let p0 = param_pointee(&m, f, 0);
    match m.types.get(p0) {
        Type::Vector { len: 4, elem, scalable: false } => assert!(matches!(m.types.get(*elem), Type::Float)),
        other => panic!("buffer 0's pointee: {:?}", other),
    }
    let p1 = param_pointee(&m, f, 1);
    match m.types.get(p1) {
        // `a` and `b`, then the padding Metal adds to the struct's 32-byte size.
        Type::Struct { body: Some(body), .. } => {
            assert!(matches!(m.types.get(body[0]), Type::Vector { len: 4, .. }));
            assert!(matches!(m.types.get(body[1]), Type::Int(32)));
        }
        other => panic!("buffer 1's pointee: {:?}", other),
    }
}
