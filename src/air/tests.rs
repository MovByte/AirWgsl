//! Tests of the AIR semantic layer on modules built in memory, including the mutation test of the closed grammar:
//! every key of a valid module, renamed to a name the grammar does not know, makes the module a refusal that names
//! it; so does an extra key in any record, a dropped required key and a repeated key.

use super::*;
use crate::api::RefusalKind;
use crate::ir::{ConstKind, Constant, GlobalCommon, GlobalId, Metadata, NamedMetadata, Type, Value};
use crate::ir::{DllStorage, Linkage, ThreadLocal, UnnamedAddr, Visibility};

/// Builds a module's metadata.
struct B {
    m: Module,
}

impl B {
    fn new() -> B {
        let mut m = Module::new();
        let void = m.types.void();
        let ty = m.types.intern(Type::Function { ret: void, params: Vec::new(), vararg: false });
        let attrs = crate::ir::AttrList::default();
        m.functions.push(crate::ir::Function {
            common: GlobalCommon {
                name: "k".into(),
                linkage: Linkage::External,
                visibility: Visibility::Default,
                dll_storage: DllStorage::Default,
                unnamed_addr: UnnamedAddr::None,
                thread_local: ThreadLocal::NotThreadLocal,
                dso_local: false,
                addrspace: 0,
                partition: String::new(),
            },
            ty,
            cc: 0,
            attrs,
            align: None,
            section: None,
            comdat: None,
            gc: None,
            prefix: None,
            prologue: None,
            personality: None,
            args: Vec::new(),
            blocks: Vec::new(),
            insts: Vec::new(),
            md: Vec::new(),
            bitcode_type: None,
        });
        B { m }
    }

    fn s(&mut self, s: &str) -> MdId {
        self.m.intern_md(Metadata::String(s.as_bytes().to_vec()))
    }

    fn i(&mut self, v: u64) -> MdId {
        let t = self.m.types.int(32);
        let c = self.m.intern_const(Constant { ty: t, kind: ConstKind::Int(vec![v]) });
        self.m.intern_md(Metadata::Value { ty: t, value: Value::Const(c) })
    }

    fn f(&mut self) -> MdId {
        let p = self.m.types.ptr(0);
        self.m.intern_md(Metadata::Value { ty: p, value: Value::Global(GlobalId::Function(0)) })
    }

    fn t(&mut self, ops: Vec<MdId>) -> MdId {
        self.m.intern_md(Metadata::Tuple { distinct: false, ops: ops.into_iter().map(Some).collect() })
    }

    /// A tuple of the given items: `"air.x"` strings become key strings, `#n` an `i32`, anything else a string.
    fn rec(&mut self, items: &[&str]) -> MdId {
        let ops = items
            .iter()
            .map(|x| match x.strip_prefix('#') {
                Some(n) => self.i(n.parse().unwrap()),
                None => self.s(x),
            })
            .collect();
        self.t(ops)
    }

    fn named(&mut self, name: &str, ops: Vec<MdId>) {
        self.m.named_metadata.push(NamedMetadata { name: name.into(), ops });
    }
}

const BUFFER: &[&str] = &[
    "#0",
    "air.buffer",
    "air.location_index",
    "#2",
    "#1",
    "air.read",
    "air.address_space",
    "#2",
    "air.arg_type_size",
    "#2",
    "air.arg_type_align_size",
    "#2",
    "air.arg_type_name",
    "ushort",
    "air.arg_name",
    "a",
];
const TEXTURE: &[&str] = &[
    "#1",
    "air.texture",
    "air.location_index",
    "#3",
    "#1",
    "air.sample",
    "air.arg_type_name",
    "texture2d<half, sample>",
    "air.arg_name",
    "t",
];
const SAMPLER: &[&str] =
    &["#2", "air.sampler", "air.location_index", "#1", "#1", "air.arg_type_name", "sampler", "air.arg_name", "s"];
const BUILTIN: &[&str] = &["#3", "air.thread_position_in_grid", "air.arg_type_name", "uint", "air.arg_name", "i"];
const RECORDS: &[&[&str]] = &[BUFFER, TEXTURE, SAMPLER, BUILTIN];

/// A kernel `k(constant ushort *a [[buffer(2)]], texture2d<half> t [[texture(3)]], sampler s [[sampler(1)]],
/// uint i [[thread_position_in_grid]])` with the module records the front end writes, built from `records`.
fn kernel(records: &[Vec<&str>]) -> Module {
    entry("air.kernel", &[], records)
}

/// A module whose function `k` is an entry point of `stage` (`air.vertex`...) with the given output and argument
/// records, and the module records the front end writes.
pub(crate) fn entry(stage: &str, outputs: &[Vec<&str>], records: &[Vec<&str>]) -> Module {
    let mut b = B::new();
    let outs: Vec<MdId> = outputs.iter().map(|r| b.rec(r)).collect();
    let args: Vec<MdId> = records.iter().map(|r| b.rec(r)).collect();
    let outputs = b.t(outs);
    let arguments = b.t(args);
    let f = b.f();
    let entry = b.t(vec![f, outputs, arguments]);
    b.named(stage, vec![entry]);
    let flag = b.rec(&["#7", "air.max_device_buffers", "#31"]);
    b.named("llvm.module.flags", vec![flag]);
    let v = b.rec(&["#2", "#6", "#0"]);
    b.named("air.version", vec![v]);
    let lv = b.rec(&["Metal", "#3", "#1", "#0"]);
    b.named("air.language_version", vec![lv]);
    let o1 = b.rec(&["air.compile.fast_math_enable"]);
    let o2 = b.rec(&["air.compile.denorms_disable"]);
    b.named("air.compile_options", vec![o1, o2]);
    b.m
}

/// The module [`kernel`] builds from the valid records.
pub(crate) fn valid() -> Module {
    kernel(&RECORDS.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
}

#[test]
fn decodes_a_kernel() {
    let a = AirModule::decode(valid()).unwrap();
    assert_eq!(a.air_version, Some((2, 6, 0)));
    assert_eq!(a.language_version.as_ref().map(|l| (l.major, l.minor)), Some((3, 1)));
    assert_eq!(a.compile_options.fast_math, Some(true));
    assert!(a.compile_options.denorms_disable);
    assert_eq!(a.flags.max_device_buffers, Some(31));
    let e = a.entry_point("k").unwrap();
    assert_eq!(e.stage, Stage::Kernel);
    assert_eq!(e.arguments.len(), 4);
    match &e.arguments[0].role {
        Role::Buffer(b) => {
            assert_eq!(b.location, Location { index: Slot::Value(2), count: Slot::Value(1) });
            assert_eq!(b.access, Access::Read);
            assert_eq!(b.address_space, Some(2));
        }
        r => panic!("{:?}", r),
    }
    assert!(matches!(e.arguments[1].role, Role::Texture { access: Access::Sample, .. }));
    assert!(matches!(e.arguments[2].role, Role::Sampler(_)));
    assert!(matches!(e.arguments[3].role, Role::Builtin { key: "air.thread_position_in_grid", .. }));
}

fn refused(m: Module) -> Refusal {
    match AirModule::decode(m) {
        Ok(_) => panic!("the mutated module decoded"),
        Err(r) => r,
    }
}

/// Every `air.*` string of the module, renamed to one the grammar does not know, is refused by that name.
#[test]
fn renamed_keys_are_refused() {
    let m = valid();
    let mut tried = 0;
    for (i, md) in m.metadata.iter().enumerate() {
        let Metadata::String(s) = md else { continue };
        if !s.starts_with(b"air.") {
            continue;
        }
        let mut x = m.clone();
        x.set_md(MdId(i as u32), Metadata::String(b"air.mutated".to_vec()));
        x.rebuild_md_unique();
        let r = refused(x);
        assert_eq!(r.kind, RefusalKind::NotImplemented);
        assert!(
            r.construct == "air.mutated" || r.reason.contains("air.mutated"),
            "renaming {} gave {}",
            String::from_utf8_lossy(s),
            r
        );
        tried += 1;
    }
    assert_eq!(tried, 15, "every distinct key string of the module was mutated");
    for n in 0..m.named_metadata.len() {
        let mut x = m.clone();
        x.named_metadata[n].name = "air.mutated".into();
        assert_eq!(refused(x).construct, "air.mutated");
    }
}

/// An extra key in any record, valid elsewhere or unknown, is refused.
#[test]
fn extra_keys_are_refused() {
    for (r, rec) in RECORDS.iter().enumerate() {
        for extra in ["air.unknown_key", "air.vertex_output", "air.flat", "air.depth"] {
            let mut records: Vec<Vec<&str>> = RECORDS.iter().map(|r| r.to_vec()).collect();
            records[r] = rec.iter().copied().chain([extra]).collect();
            let refusal = refused(kernel(&records));
            assert!(refusal.construct == extra || refusal.reason.contains(extra), "{}: {}", extra, refusal);
        }
    }
}

/// A record missing a key its role requires, or repeating one, is refused.
#[test]
fn missing_and_repeated_keys_are_refused() {
    for key in ["air.location_index", "air.read", "air.address_space"] {
        let i = BUFFER.iter().position(|k| *k == key).unwrap();
        let arity = match key {
            "air.location_index" => 2,
            "air.address_space" => 1,
            _ => 0,
        };
        let mut records: Vec<Vec<&str>> = RECORDS.iter().map(|r| r.to_vec()).collect();
        records[0].drain(i..i + 1 + arity);
        if key == "air.address_space" {
            // Two system modules omit the address space of a buffer; the record stays valid.
            assert!(AirModule::decode(kernel(&records)).is_ok());
            continue;
        }
        assert!(AirModule::decode(kernel(&records)).is_err(), "dropping {}", key);
    }
    let mut records: Vec<Vec<&str>> = RECORDS.iter().map(|r| r.to_vec()).collect();
    records[1].push("air.sample");
    assert!(refused(kernel(&records)).reason.contains("twice"));
    let mut records: Vec<Vec<&str>> = RECORDS.iter().map(|r| r.to_vec()).collect();
    records[0].push("air.write");
    assert!(AirModule::decode(kernel(&records)).is_err(), "two accesses");
}

/// An operand of the wrong kind after a key is refused.
#[test]
fn wrong_operands_are_refused() {
    let mut records: Vec<Vec<&str>> = RECORDS.iter().map(|r| r.to_vec()).collect();
    let i = BUFFER.iter().position(|k| *k == "air.arg_type_size").unwrap();
    records[0][i + 1] = "four";
    assert!(refused(kernel(&records)).reason.contains("i32"));
}
