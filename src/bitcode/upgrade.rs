//! The module-wide upgrades LLVM 21 applies after reading (`globalCleanup`, `materialize`, `materializeModule` in
//! `BitcodeReader.cpp`; `AutoUpgrade.cpp`).

use super::{Reader, Result};
use crate::ir::*;

pub(super) fn finish(r: &mut Reader) -> Result<()> {
    check_unimplemented_upgrades(r)?;
    super::intrinsics::upgrade_intrinsics(r)?;
    for f in 0..r.m.functions.len() {
        upgrade_function_attributes(r, f)?;
        remove_incompatible_call_attrs(r, f);
        strip_bad_branch_weights(r, f);
    }
    upgrade_global_ctors(r)?;
    verify_tbaa(r)?;
    linker_options(r);
    strip_debug_info(r)?;
    upgrade_module_flags(r);
    r.m.rebuild_md_unique();
    Ok(())
}

/// Upgrades LLVM applies that this reader does not implement; inputs that need them fail here.
fn check_unimplemented_upgrades(r: &Reader) -> Result<()> {
    if r.m.named_metadata.iter().any(|n| n.name == "nvvm.annotations") {
        return r.unsupported("nvvm.annotations (its upgrade to function attributes is not implemented)");
    }
    // `UpgradeARCRuntime`.
    if r.m.functions.iter().any(|f| f.common.name == "clang.arc.use") {
        return r.unsupported("clang.arc.use (its upgrade to llvm.objc.clang.arc.use is not implemented)");
    }
    let marker = "clang.arc.retainAutoreleasedReturnValueMarker";
    if r.m.named_metadata.iter().any(|n| n.name == marker) || module_flag(&r.m, marker).is_some() {
        return r.unsupported("the ARC retain/release marker (its runtime call upgrade is not implemented)");
    }
    Ok(())
}

/// The value operand of module flag `key`, if present.
fn module_flag(m: &Module, key: &str) -> Option<MdId> {
    let flags = m.named_metadata.iter().find(|n| n.name == "llvm.module.flags")?;
    for &op in &flags.ops {
        if let Metadata::Tuple { ops, .. } = m.md(op)
            && ops.len() >= 3
            && let Some(Some(k)) = ops.get(1)
            && let Metadata::String(s) = m.md(*k)
            && s.as_slice() == key.as_bytes()
        {
            return ops[2];
        }
    }
    None
}

/// `AttributeFuncs::typeIncompatible` (all kinds): the attributes a value of type `ty` cannot carry.
fn incompatible(m: &Module, ty: TypeId, set: &AttrSet) -> Vec<AttrKind> {
    let t = &m.types;
    let mut out = Vec::new();
    if !t.is_int(ty) {
        out.extend([AttrKind::AllocAlign, AttrKind::SExt, AttrKind::ZExt]);
    }
    if !t.is_int_or_int_vector(ty) {
        out.push(AttrKind::Range);
    } else if let Some(Attribute::Range(_, cr)) = set.get(AttrKind::Range)
        && cr.bits != t.scalar_bits(ty)
    {
        out.push(AttrKind::Range);
    }
    if !t.is_ptr(ty) {
        out.extend([
            AttrKind::NoAlias,
            AttrKind::NonNull,
            AttrKind::ReadNone,
            AttrKind::ReadOnly,
            AttrKind::Dereferenceable,
            AttrKind::DereferenceableOrNull,
            AttrKind::Writable,
            AttrKind::DeadOnUnwind,
            AttrKind::Initializes,
            AttrKind::Captures,
            AttrKind::DeadOnReturn,
            AttrKind::Nest,
            AttrKind::SwiftError,
            AttrKind::Preallocated,
            AttrKind::InAlloca,
            AttrKind::ByVal,
            AttrKind::StructRet,
            AttrKind::ByRef,
            AttrKind::ElementType,
            AttrKind::AllocatedPointer,
        ]);
    }
    if !t.is_ptr_or_ptr_vector(ty) {
        out.push(AttrKind::Alignment);
    }
    if !nofpclass_compatible(t, ty) {
        out.push(AttrKind::NoFPClass);
    }
    if t.is_void(ty) {
        out.push(AttrKind::NoUndef);
    }
    out
}

/// `AttributeFuncs::isNoFPClassCompatibleType`: floating point, vectors of it, and arrays of those.
fn nofpclass_compatible(t: &TypeTable, ty: TypeId) -> bool {
    let mut ty = ty;
    while let Type::Array { elem, .. } = t.get(ty) {
        ty = *elem;
    }
    t.is_fp_or_fp_vector(ty)
}

fn remove_kinds(m: &mut Module, id: AttrSetId, kinds: &[AttrKind]) -> AttrSetId {
    let set = m.attrs(id);
    if !kinds.iter().any(|k| set.has(*k)) {
        return id;
    }
    let mut s = set.clone();
    for k in kinds {
        s.remove(*k);
    }
    m.attr_set(s)
}

/// `UpgradeFunctionAttributes`.
fn upgrade_function_attributes(r: &mut Reader, f: usize) -> Result<()> {
    let m = &mut r.m;
    // Calls with strictfp in a function definition without it get nobuiltin instead.
    let fn_strict = m.attrs(m.functions[f].attrs.function).has(AttrKind::StrictFP);
    if !m.functions[f].is_declaration() && !fn_strict {
        for i in 0..m.functions[f].insts.len() {
            let call_attrs = match &m.functions[f].insts[i].op {
                Op::Call { attrs, callee, .. } => {
                    // Constrained FP intrinsics keep strictfp.
                    if let Value::Global(GlobalId::Function(c)) = callee
                        && m.functions[*c as usize].common.name.starts_with("llvm.experimental.constrained.")
                    {
                        continue;
                    }
                    attrs.clone()
                }
                _ => continue,
            };
            if m.attrs(call_attrs.function).has(AttrKind::StrictFP) {
                let mut s = m.attrs(call_attrs.function).clone();
                s.remove(AttrKind::StrictFP);
                s.add(Attribute::Enum(AttrKind::NoBuiltin));
                let id = m.attr_set(s);
                if let Op::Call { attrs, .. } = &mut m.functions[f].insts[i].op {
                    attrs.function = id;
                }
            }
        }
    }
    // Attributes incompatible with the return and parameter types.
    let fty = m.functions[f].ty;
    let (ret, params, _) = {
        let (a, b, c) = m.types.function(fty).unwrap();
        (a, b.to_vec(), c)
    };
    let mut list = m.functions[f].attrs.clone();
    let kinds = incompatible(m, ret, &m.attrs(list.ret).clone());
    list.ret = remove_kinds(m, list.ret, &kinds);
    for (i, p) in params.iter().enumerate() {
        if i < list.params.len() {
            let kinds = incompatible(m, *p, &m.attrs(list.params[i]).clone());
            list.params[i] = remove_kinds(m, list.params[i], &kinds);
        }
    }
    list.trim();
    // "implicit-section-name" sets the section.
    if let Some(v) = m.attrs(list.function).get_str(b"implicit-section-name").map(|v| v.to_vec()) {
        let s = match String::from_utf8(v) {
            Ok(s) => s,
            Err(_) => return r.unsupported("implicit-section-name that is not UTF-8"),
        };
        let mut set = m.attrs(list.function).clone();
        set.remove_str(b"implicit-section-name");
        list.function = m.attr_set(set);
        m.functions[f].section = Some(s);
    }
    if m.attrs(list.function).get_str(b"amdgpu-unsafe-fp-atomics").is_some() {
        return r.unsupported("amdgpu-unsafe-fp-atomics (its upgrade is not implemented)");
    }
    m.functions[f].attrs = list;
    Ok(())
}

/// Removes attributes incompatible with call argument and return types (`materialize`).
fn remove_incompatible_call_attrs(r: &mut Reader, f: usize) {
    let m = &mut r.m;
    for i in 0..m.functions[f].insts.len() {
        let (fn_ty, args, mut attrs) = match &m.functions[f].insts[i].op {
            Op::Call { fn_ty, args, attrs, .. } => (*fn_ty, args.clone(), attrs.clone()),
            _ => continue,
        };
        let ret = m.types.function(fn_ty).unwrap().0;
        let kinds = incompatible(m, ret, &m.attrs(attrs.ret).clone());
        attrs.ret = remove_kinds(m, attrs.ret, &kinds);
        for (k, a) in args.iter().enumerate() {
            if k < attrs.params.len() {
                let ty = m.value_type(f, *a);
                let kinds = incompatible(m, ty, &m.attrs(attrs.params[k]).clone());
                attrs.params[k] = remove_kinds(m, attrs.params[k], &kinds);
            }
        }
        attrs.trim();
        if let Op::Call { attrs: a, .. } = &mut m.functions[f].insts[i].op {
            *a = attrs;
        }
    }
}

/// Drops `!prof` branch weights whose operand count does not match the instruction (`materialize`).
fn strip_bad_branch_weights(r: &mut Reader, f: usize) {
    let prof = 2u32;
    let m = &mut r.m;
    for i in 0..m.functions[f].insts.len() {
        let md = match m.functions[f].insts[i].md.iter().find(|(k, _)| *k == prof) {
            Some((_, md)) => *md,
            None => continue,
        };
        let ops = match m.md(md) {
            Metadata::Tuple { ops, .. } => ops.clone(),
            _ => continue,
        };
        let name = match ops.first() {
            Some(Some(s)) => match m.md(*s) {
                Metadata::String(s) => s.clone(),
                _ => continue,
            },
            _ => continue,
        };
        if name != b"branch_weights" {
            continue;
        }
        let expected = match &m.functions[f].insts[i].op {
            Op::CondBr { .. } => 2,
            Op::Br { .. } => 1,
            Op::Switch { cases, .. } => cases.len() + 1,
            Op::Call { .. } => 1,
            Op::IndirectBr { dests, .. } => dests.len(),
            Op::Select { .. } => 2,
            _ => continue,
        };
        // `getBranchWeightOffset`: 2 when the second operand is the "expected" string.
        let offset = match ops.get(1) {
            Some(Some(s)) if matches!(m.md(*s), Metadata::String(x) if x.as_slice() == b"expected") => 2,
            _ => 1,
        };
        if ops.len() != offset + expected {
            m.functions[f].insts[i].md.retain(|(k, _)| *k != prof);
        }
    }
}

/// `UpgradeGlobalVariable`: two-field `llvm.global_ctors`/`llvm.global_dtors` entries gain a null third field.
fn upgrade_global_ctors(r: &mut Reader) -> Result<()> {
    for g in &r.m.globals {
        if g.common.name != "llvm.global_ctors" && g.common.name != "llvm.global_dtors" || g.initializer.is_none() {
            continue;
        }
        if let Type::Array { elem, .. } = r.m.types.get(g.value_type)
            && r.m.types.struct_elems(*elem).map(|e| e.len()) == Some(2)
        {
            return r.unsupported("two-field llvm.global_ctors (its upgrade is not implemented)");
        }
    }
    Ok(())
}

/// `materializeMetadata`: the "Linker Options" module flag becomes `!llvm.linker.options`.
fn linker_options(r: &mut Reader) {
    if r.m.named_metadata.iter().any(|n| n.name == "llvm.linker.options") {
        return;
    }
    if let Some(v) = module_flag(&r.m, "Linker Options")
        && let Metadata::Tuple { ops, .. } = r.m.md(v).clone()
    {
        let nodes: Vec<MdId> = ops.into_iter().flatten().collect();
        r.m.named_metadata.push(NamedMetadata { name: "llvm.linker.options".into(), ops: nodes });
    }
}

/// `UpgradeDebugInfo`: debug information without a "Debug Info Version" of 3 is stripped (`StripDebugInfo`).
fn strip_debug_info(r: &mut Reader) -> Result<()> {
    let version = module_flag(&r.m, "Debug Info Version").and_then(|md| match r.m.md(md) {
        Metadata::Value { value: Value::Const(c), .. } => match &r.m.constant(*c).kind {
            ConstKind::Int(w) => Some(w[0]),
            _ => None,
        },
        _ => None,
    });
    if version == Some(3) {
        // A current version keeps the debug information after LLVM verifies it; this reader has no debug
        // information support (debug records fail earlier), so there is nothing that could be kept.
        return Ok(());
    }
    r.m.named_metadata.retain(|n| !(n.name.starts_with("llvm.dbg.") || n.name == "llvm.gcov"));
    for g in r.m.globals.iter_mut() {
        g.md.retain(|(k, _)| *k != 0);
    }
    let heapallocsite = r.m.md_kinds.iter().position(|k| k == "heapallocsite").map(|p| p as u32);
    let assign_id = 38u32;
    for f in r.m.functions.iter_mut() {
        f.md.retain(|(k, _)| *k != 0);
        for inst in f.insts.iter_mut() {
            inst.dbg = None;
            inst.md.retain(|(k, _)| *k != assign_id && Some(*k) != heapallocsite);
        }
    }
    // Calls to debug intrinsics are dropped; none can be present without debug metadata, which fails earlier.
    let has_dbg_calls = r.m.functions.iter().any(|f| f.common.name.starts_with("llvm.dbg."));
    if has_dbg_calls {
        return r.unsupported("llvm.dbg intrinsics (debug information is not implemented)");
    }
    Ok(())
}

/// `UpgradeModuleFlags`.
fn upgrade_module_flags(r: &mut Reader) {
    let m = &mut r.m;
    let flags_idx = match m.named_metadata.iter().position(|n| n.name == "llvm.module.flags") {
        Some(i) => i,
        None => return,
    };
    let i32t = m.types.int(32);
    let i8t = m.types.int(8);
    let mut has_objc = false;
    let mut has_class_props = false;
    let mut swift: Option<(u64, u64, u64)> = None;
    let ops = m.named_metadata[flags_idx].ops.clone();
    let int_md = |m: &mut Module, ty: TypeId, v: u64| -> MdId {
        let c = m.intern_const(Constant { ty, kind: ConstKind::Int(vec![v]) });
        m.intern_md(Metadata::Value { ty, value: Value::Const(c) })
    };
    let const_int = |m: &Module, md: Option<MdId>| -> Option<(TypeId, u64)> {
        match md.map(|x| m.md(x)) {
            Some(Metadata::Value { ty, value: Value::Const(c) }) => match &m.constant(*c).kind {
                ConstKind::Int(w) => Some((*ty, w[0])),
                _ => None,
            },
            _ => None,
        }
    };
    for (i, op) in ops.iter().enumerate() {
        let node = match m.md(*op).clone() {
            Metadata::Tuple { ops, .. } if ops.len() == 3 => ops,
            _ => continue,
        };
        let id = match node[1].map(|x| m.md(x).clone()) {
            Some(Metadata::String(s)) => s,
            _ => continue,
        };
        let set_op = |m: &mut Module, new: Vec<Option<MdId>>| -> MdId {
            m.intern_md(Metadata::Tuple { distinct: false, ops: new })
        };
        let mut replacement = None;
        match id.as_slice() {
            b"Objective-C Image Info Version" => has_objc = true,
            b"Objective-C Class Properties" => has_class_props = true,
            b"PIC Level" => {
                if let Some((_, b)) = const_int(m, node[0]) {
                    // Error (1) or Max (7) become Min (8).
                    if b == 1 || b == 7 {
                        let b = int_md(m, i32t, 8);
                        replacement = Some(set_op(m, vec![Some(b), node[1], node[2]]));
                    }
                }
            }
            b"PIE Level" => {
                if let Some((_, 1)) = const_int(m, node[0]) {
                    let b = int_md(m, i32t, 7);
                    replacement = Some(set_op(m, vec![Some(b), node[1], node[2]]));
                }
            }
            b"Objective-C Image Info Section" => {
                if let Some(Metadata::String(v)) = node[2].map(|x| m.md(x).clone())
                    && v.contains(&b' ')
                {
                    let joined: Vec<u8> = v.split(|c| *c == b' ').flatten().copied().collect();
                    let s = m.intern_md(Metadata::String(joined));
                    replacement = Some(set_op(m, vec![node[0], node[1], Some(s)]));
                }
            }
            b"Objective-C Garbage Collection" => {
                if let Some((ty, val)) = const_int(m, node[2])
                    && ty != i8t
                {
                    if val & 0xff != val {
                        swift = Some(((val & 0xff00) >> 8, (val & 0xff00_0000) >> 24, (val & 0xff_0000) >> 16));
                    }
                    let b = int_md(m, i32t, 1);
                    let v = int_md(m, i8t, val & 0xff);
                    replacement = Some(set_op(m, vec![Some(b), node[1], Some(v)]));
                }
            }
            b"amdgpu_code_object_version" => {
                let s = m.intern_md(Metadata::String(b"amdhsa_code_object_version".to_vec()));
                replacement = Some(set_op(m, vec![node[0], Some(s), node[2]]));
            }
            _ => {
                if (id.as_slice() == b"branch-target-enforcement" || id.starts_with(b"sign-return-address"))
                    && let Some((_, 1)) = const_int(m, node[0])
                {
                    let b = int_md(m, i32t, 8);
                    replacement = Some(set_op(m, vec![Some(b), node[1], node[2]]));
                }
            }
        }
        if let Some(rep) = replacement {
            m.named_metadata[flags_idx].ops[i] = rep;
        }
    }
    let add_flag = |m: &mut Module, behavior: u64, key: &str, ty: TypeId, v: u64| {
        let b = int_md(m, i32t, behavior);
        let k = m.intern_md(Metadata::String(key.as_bytes().to_vec()));
        let val = int_md(m, ty, v);
        let node = m.intern_md(Metadata::Tuple { distinct: false, ops: vec![Some(b), Some(k), Some(val)] });
        m.named_metadata[flags_idx].ops.push(node);
    };
    if has_objc && !has_class_props {
        // Module::Override = 4.
        add_flag(m, 4, "Objective-C Class Properties", i32t, 0);
    }
    if let Some((abi, major, minor)) = swift {
        add_flag(m, 1, "Swift ABI Version", i32t, abi);
        add_flag(m, 1, "Swift Major Version", i8t, major);
        add_flag(m, 1, "Swift Minor Version", i8t, minor);
    }
}

/// `TBAAVerifier::visitTBAAMetadata`: when any `!tbaa` attachment is invalid, LLVM strips them all.
fn verify_tbaa(r: &mut Reader) -> Result<()> {
    let mut verifier = super::tbaa::Verifier::default();
    let mut valid = true;
    'outer: for f in 0..r.m.functions.len() {
        for i in 0..r.m.functions[f].insts.len() {
            let tag = match r.m.functions[f].insts[i].md.iter().find(|(k, _)| *k == MD_TBAA) {
                Some((_, t)) => *t,
                None => continue,
            };
            let ok = verifier.visit(&mut r.m, f, i, tag);
            if !ok {
                valid = false;
                break 'outer;
            }
        }
    }
    if !valid {
        for f in r.m.functions.iter_mut() {
            for inst in f.insts.iter_mut() {
                inst.md.retain(|(k, _)| *k != MD_TBAA);
            }
        }
    }
    Ok(())
}
