//! Scalar replacement of aggregates: a stack object split into smaller ones along the byte ranges its accesses use
//! (LLVM's SROA, by slices and partitions).
//!
//! An `alloca` in the entry block with a constant count is split when every use of it, and of the pointers derived
//! from it by casts and constant-index `getelementptr`s, is one of:
//!
//! - a non-volatile, non-atomic load or store (an unsplittable slice: its byte range),
//! - a non-volatile `llvm.memset`, `llvm.memcpy` or `llvm.memmove` of a constant length whose other pointer is not
//!   derived from the same object (a splittable slice),
//! - a lifetime marker (dropped: it only makes accesses undefined or contents undef, which the result refines),
//!
//! and every slice lies inside the object. Overlapping unsplittable slices form partitions; bytes that only
//! splittable slices touch form partitions of their own. Each partition becomes a new object: of the slices' type
//! when every slice in it covers it exactly with one type (so that [`super::mem2reg`] can promote it), of `[n x
//! i8]` otherwise. Accesses move to the new objects at the same offsets; memory transfers split into one per
//! partition they cross. No byte outside the slices is read or written by anything, so the program reads the same
//! bytes from the parts as from the whole.

use std::collections::HashSet;

use crate::api::Refusal;
use crate::ir::layout::DataLayout;
use crate::ir::{GepFlags, GlobalId, InstId, Module, Op, Type, TypeId, Value};

use super::util::{const_int, int_const, new_inst};

#[derive(Clone, Debug)]
enum Slice {
    /// A load or store: instruction, byte offset, size, accessed type, alignment.
    Access {
        inst: InstId,
        off: u64,
        size: u64,
        ty: TypeId,
        align: u64,
    },
    /// A memset, memcpy or memmove: instruction, byte offset of the object's side, length.
    Transfer {
        inst: InstId,
        off: u64,
        len: u64,
    },
    Marker(InstId),
}

/// Splits what it can in function `f`; returns whether anything changed.
#[allow(clippy::result_large_err)]
pub fn sroa(m: &mut Module, f: usize) -> Result<bool, Refusal> {
    let Ok(dl) = DataLayout::parse(&m.datalayout) else { return Ok(false) };
    let mut changed = false;
    let allocas: Vec<InstId> = match m.functions[f].blocks.first() {
        Some(b) => {
            b.insts.iter().copied().filter(|&i| matches!(m.functions[f].inst(i).op, Op::Alloca { .. })).collect()
        }
        None => return Ok(false),
    };
    let mut uses = super::util::use_map(&m.functions[f]);
    for a in allocas {
        if let Some(plan) = analyze(m, &dl, f, a, &uses) {
            rewrite(m, &dl, f, a, plan);
            changed = true;
            uses = super::util::use_map(&m.functions[f]);
        }
    }
    Ok(changed)
}

struct Plan {
    slices: Vec<Slice>,
    /// Byte ranges `[lo, hi)` of the new objects.
    parts: Vec<(u64, u64)>,
    align: u64,
    addrspace: u32,
}

fn analyze(m: &Module, dl: &DataLayout, f: usize, a: InstId, uses: &[Vec<InstId>]) -> Option<Plan> {
    let func = &m.functions[f];
    let users = |_: &crate::ir::Function, v: Value| -> Vec<InstId> {
        match v {
            Value::Inst(i) => {
                let mut u = uses[i.0 as usize].clone();
                u.dedup();
                u
            }
            _ => Vec::new(),
        }
    };
    let Op::Alloca { allocated, count, align, addrspace, inalloca: false, .. } = func.inst(a).op else { return None };
    let n = const_int(m, count)? as u64;
    let size = n.checked_mul(dl.alloc_size(&m.types, allocated)?)?;
    if size == 0 || size > 1 << 24 {
        return None;
    }
    // Pointers derived from the object, with their offsets.
    let mut derived: Vec<(InstId, u64)> = vec![(a, 0)];
    let mut k = 0;
    while k < derived.len() {
        let (p, off) = derived[k];
        k += 1;
        for u in users(func, Value::Inst(p)) {
            match &func.inst(u).op {
                Op::Gep { source, ptr, indices, .. } if *ptr == Value::Inst(p) => {
                    let o = gep_offset(m, dl, *source, indices)?;
                    let no = off as i128 + o;
                    if no < 0 || no > size as i128 {
                        return None;
                    }
                    if m.types.is_vector(func.inst(u).ty) {
                        return None;
                    }
                    derived.push((u, no as u64));
                }
                Op::Cast { op: crate::ir::CastOp::BitCast, .. } => {
                    if func.inst(u).ty != func.inst(a).ty {
                        return None;
                    }
                    derived.push((u, off));
                }
                Op::Cast { op: crate::ir::CastOp::AddrSpaceCast, .. } => return None,
                _ => {}
            }
        }
    }
    let set: HashSet<InstId> = derived.iter().map(|(p, _)| *p).collect();
    let mut slices = Vec::new();
    for &(p, off) in &derived {
        let pv = Value::Inst(p);
        for u in users(func, pv) {
            let inst = func.inst(u);
            match &inst.op {
                Op::Gep { ptr, indices, .. } if *ptr == pv && !indices.contains(&pv) => {}
                Op::Cast { op: crate::ir::CastOp::BitCast, .. } => {}
                Op::Load { volatile: false, atomic: None, align, .. } => {
                    let sz = dl.store_size(&m.types, inst.ty)?;
                    slices.push(Slice::Access { inst: u, off, size: sz, ty: inst.ty, align: *align });
                }
                Op::Store { ptr, value, volatile: false, atomic: None, align } if *ptr == pv && *value != pv => {
                    if matches!(value, Value::Inst(v) if set.contains(v)) {
                        return None;
                    }
                    let ty = value_type(m, f, *value)?;
                    let sz = dl.store_size(&m.types, ty)?;
                    slices.push(Slice::Access { inst: u, off, size: sz, ty, align: *align });
                }
                Op::Call { callee: Value::Global(GlobalId::Function(c)), args, .. } => {
                    let name = m.functions[*c as usize].common.name.as_str();
                    if name.starts_with("llvm.lifetime.") {
                        slices.push(Slice::Marker(u));
                        continue;
                    }
                    let transfer = name.starts_with("llvm.memcpy.") || name.starts_with("llvm.memmove.");
                    let set_ = name.starts_with("llvm.memset.");
                    if !(transfer || set_) || name.contains(".inline") {
                        return None;
                    }
                    if const_int(m, args[3]) != Some(0) {
                        return None;
                    }
                    let len = const_int(m, args[2])? as u64;
                    if transfer {
                        // One side in this object, the other outside it.
                        let ins = |v: Value| matches!(v, Value::Inst(i) if set.contains(&i));
                        if ins(args[0]) == ins(args[1]) {
                            return None;
                        }
                    } else if args[0] != pv || matches!(args[1], Value::Inst(i) if set.contains(&i)) {
                        return None;
                    }
                    slices.push(Slice::Transfer { inst: u, off, len });
                }
                _ => return None,
            }
        }
    }
    // Every slice inside the object.
    for s in &slices {
        let (o, l) = match s {
            Slice::Access { off, size, .. } => (*off, *size),
            Slice::Transfer { off, len, .. } => (*off, *len),
            Slice::Marker(_) => continue,
        };
        if o.checked_add(l)? > size {
            return None;
        }
    }
    // Partitions: overlapping unsplittable slices merge.
    let mut ranges: Vec<(u64, u64)> = slices
        .iter()
        .filter_map(|s| match s {
            Slice::Access { off, size, .. } if *size > 0 => Some((*off, off + size)),
            _ => None,
        })
        .collect();
    ranges.sort();
    let mut parts: Vec<(u64, u64)> = Vec::new();
    for (lo, hi) in ranges {
        match parts.last_mut() {
            Some(last) if lo < last.1 => last.1 = last.1.max(hi),
            _ => parts.push((lo, hi)),
        }
    }
    // Bytes only transfers touch.
    let mut covered = vec![false; size as usize];
    for s in &slices {
        if let Slice::Transfer { off, len, .. } = s {
            for b in *off..off + len {
                covered[b as usize] = true;
            }
        }
    }
    for &(lo, hi) in &parts {
        for b in lo..hi {
            covered[b as usize] = false;
        }
    }
    let mut b = 0;
    while b < size {
        if covered[b as usize] {
            let lo = b;
            while b < size && covered[b as usize] {
                b += 1;
            }
            parts.push((lo, b));
        } else {
            b += 1;
        }
    }
    parts.sort();
    if parts.len() == 1 && parts[0] == (0, size) {
        // Nothing to split: the object is accessed as one partition. It goes only if lifetime markers can.
        return None;
    }
    // Alignment: the new objects keep the old one's where their offsets allow, and every access stays aligned.
    for s in &slices {
        if let Slice::Access { off, align: al, .. } = s {
            let p = parts.iter().find(|(lo, hi)| off >= lo && off < hi).copied().unwrap_or((*off, *off));
            let base = part_align(align, p.0);
            if *al > base || (off - p.0) % *al.max(&1) != 0 {
                return None;
            }
        }
    }
    Some(Plan { slices, parts, align, addrspace })
}

/// Alignment of a new object starting at byte `lo` of one aligned to `align`.
fn part_align(align: u64, lo: u64) -> u64 {
    if lo == 0 { align.max(1) } else { align.max(1).min(1 << lo.trailing_zeros()) }
}

fn value_type(m: &Module, f: usize, v: Value) -> Option<TypeId> {
    let func = &m.functions[f];
    match v {
        Value::Inst(i) => Some(func.inst(i).ty),
        Value::Arg(a) => Some(func.args[a as usize].ty),
        Value::Const(c) => Some(m.constant(c).ty),
        Value::Global(g) => {
            m.types.iter().find(|(_, t)| matches!(t, Type::Ptr(a) if *a == m.global(g).addrspace)).map(|(i, _)| i)
        }
        _ => None,
    }
}

/// The byte offset of a `getelementptr` with constant indices (LangRef: sign-extended indices scaled by allocation
/// sizes, struct members at their layout offsets).
pub fn gep_offset(m: &Module, dl: &DataLayout, source: TypeId, indices: &[Value]) -> Option<i128> {
    let mut cur = source;
    let mut total: i128 = 0;
    for (k, &ix) in indices.iter().enumerate() {
        let raw = const_int(m, ix)?;
        let w = m.types.int_bits(m.constant(super::util::const_id(ix)?).ty)?;
        let v = if w >= 128 { raw as i128 } else { ((raw << (128 - w)) as i128) >> (128 - w) };
        if k == 0 {
            total += v * dl.alloc_size(&m.types, cur)? as i128;
            continue;
        }
        match m.types.get(cur) {
            Type::Struct { .. } | Type::LiteralStruct { .. } => {
                let l = dl.struct_layout(&m.types, cur)?;
                total += *l.offsets.get(v as usize)? as i128;
                cur = m.types.struct_elems(cur)?[v as usize];
            }
            Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                let e = *elem;
                total += v * dl.alloc_size(&m.types, e)? as i128;
                cur = e;
            }
            _ => return None,
        }
    }
    Some(total)
}

fn rewrite(m: &mut Module, dl: &DataLayout, f: usize, a: InstId, plan: Plan) {
    let i8t = m.types.int(8);
    let i64t = m.types.int(64);
    let pty = m.types.ptr(plan.addrspace);
    let i32t = m.types.int(32);
    let one = int_const(m, i32t, 1);
    // The new objects.
    let mut objects = Vec::new();
    for &(lo, hi) in &plan.parts {
        let mut ty = None;
        let mut exact = true;
        for s in &plan.slices {
            if let Slice::Access { off, size, ty: t, .. } = s
                && *off >= lo
                && *off < hi
            {
                if *off != lo || off + size != hi || ty.is_some_and(|x| x != *t) {
                    exact = false;
                }
                ty = Some(*t);
            }
        }
        let alloc_ty = match ty {
            Some(t) if exact && dl.alloc_size(&m.types, t) == Some(hi - lo) => t,
            _ => m.types.intern(Type::Array { len: hi - lo, elem: i8t }),
        };
        let align = part_align(plan.align, lo);
        let al = new_inst(
            &mut m.functions[f],
            pty,
            Op::Alloca {
                allocated: alloc_ty,
                count: one,
                align,
                addrspace: plan.addrspace,
                inalloca: false,
                swifterror: false,
            },
        );
        m.functions[f].blocks[0].insts.insert(0, al);
        objects.push((lo, hi, al));
    }
    let part_of = |o: u64| objects.iter().find(|(lo, hi, _)| o >= *lo && o < *hi).copied();
    // A pointer to byte `o` of the object, made before instruction `at`.
    let ptr_at = |m: &mut Module, at: InstId, o: u64, base: InstId, lo: u64, ty: TypeId| -> Value {
        if o == lo {
            return Value::Inst(base);
        }
        let idx = int_const(m, i64t, (o - lo) as u128);
        let g = new_inst(
            &mut m.functions[f],
            ty,
            Op::Gep { source: i8t, ptr: Value::Inst(base), indices: vec![idx], flags: GepFlags(GepFlags::INBOUNDS) },
        );
        insert_before(m, f, at, g);
        Value::Inst(g)
    };
    for s in &plan.slices {
        match s {
            Slice::Access { inst, off, .. } => {
                let (lo, _, base) = part_of(*off).expect("an access outside the partitions");
                let ty = ptr_type_of(m, f, *inst);
                let p = ptr_at(m, *inst, *off, base, lo, ty);
                match &mut m.functions[f].insts[inst.0 as usize].op {
                    Op::Load { ptr, .. } | Op::Store { ptr, .. } => *ptr = p,
                    _ => unreachable!(),
                }
            }
            Slice::Transfer { inst, off, len } => {
                let Op::Call { args, .. } = m.functions[f].inst(*inst).op.clone() else { unreachable!() };
                let callee_name = match &m.functions[f].inst(*inst).op {
                    Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } => {
                        m.functions[*c as usize].common.name.clone()
                    }
                    _ => unreachable!(),
                };
                let memset = callee_name.starts_with("llvm.memset.");
                // Which argument is the object's side.
                let side = if !memset && !derived_from(m, f, args[0], a) { 1 } else { 0 };
                let other = if memset { None } else { Some(args[1 - side]) };
                let len_ty = value_type(m, f, args[2]).unwrap_or(i64t);
                for &(lo, hi, base) in &objects {
                    let s0 = (*off).max(lo);
                    let s1 = (off + len).min(hi);
                    if s0 >= s1 {
                        continue;
                    }
                    let objp = ptr_at(m, *inst, s0, base, lo, ptr_type_of_value(m, f, args[side]));
                    let mut nargs = args.clone();
                    nargs[side] = objp;
                    if let Some(o) = other {
                        let oty = ptr_type_of_value(m, f, o);
                        let k = s0 - off;
                        nargs[1 - side] = if k == 0 {
                            o
                        } else {
                            let idx = int_const(m, i64t, k as u128);
                            let g = new_inst(
                                &mut m.functions[f],
                                oty,
                                Op::Gep { source: i8t, ptr: o, indices: vec![idx], flags: GepFlags(0) },
                            );
                            insert_before(m, f, *inst, g);
                            Value::Inst(g)
                        };
                    }
                    nargs[2] = int_const(m, len_ty, (s1 - s0) as u128);
                    let mut op = m.functions[f].inst(*inst).op.clone();
                    if let Op::Call { args, .. } = &mut op {
                        *args = nargs;
                    }
                    let void = m.functions[f].inst(*inst).ty;
                    let c = new_inst(&mut m.functions[f], void, op);
                    insert_before(m, f, *inst, c);
                }
                super::util::remove_inst(&mut m.functions[f], *inst);
            }
            Slice::Marker(i) => super::util::remove_inst(&mut m.functions[f], *i),
        }
    }
}

/// Whether pointer `v` is the object `a` or derived from it.
fn derived_from(m: &Module, f: usize, mut v: Value, a: InstId) -> bool {
    loop {
        match v {
            Value::Inst(i) if i == a => return true,
            Value::Inst(i) => match &m.functions[f].inst(i).op {
                Op::Gep { ptr, .. } => v = *ptr,
                Op::Cast { op: crate::ir::CastOp::BitCast | crate::ir::CastOp::AddrSpaceCast, value, .. } => v = *value,
                _ => return false,
            },
            _ => return false,
        }
    }
}

/// The pointer type of a load's or store's pointer operand.
fn ptr_type_of(m: &Module, f: usize, i: InstId) -> TypeId {
    match &m.functions[f].inst(i).op {
        Op::Load { ptr, .. } | Op::Store { ptr, .. } => ptr_type_of_value(m, f, *ptr),
        _ => unreachable!(),
    }
}

fn ptr_type_of_value(m: &Module, f: usize, v: Value) -> TypeId {
    value_type(m, f, v).expect("a pointer operand without a type")
}

fn insert_before(m: &mut Module, f: usize, at: InstId, new: InstId) {
    let func = &mut m.functions[f];
    for b in &mut func.blocks {
        if let Some(k) = b.insts.iter().position(|&x| x == at) {
            b.insts.insert(k, new);
            return;
        }
    }
}
