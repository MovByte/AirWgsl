//! LLVM intrinsic declarations: what LLVM 21's reader does to them (`UpgradeIntrinsicFunction` in `AutoUpgrade.cpp`;
//! `lookupIntrinsicID`, `getIntrinsicSignature`, `remangleIntrinsicFunction`, `getAttributes` in `Intrinsics.cpp`).
//!
//! A function whose name LLVM 21 recognizes as a target-independent intrinsic gets the intrinsic's attributes from
//! LLVM's tables when its type matches the intrinsic's signature, and is renamed when its name is not the one
//! LLVM 21 mangles from that signature (typed-pointer bitcode spells `p1i8` where LLVM 21 spells `p1`); the renamed
//! declaration is a new function at the end of the module, as LLVM creates it. The upgrades LLVM applies to specific
//! old intrinsics (changed signatures) are not implemented: inputs that need one fail with an error naming it, as do
//! target-specific intrinsics.

use super::llvm_intrinsics::{self as tbl, ArgAttr, FnAttr};
use super::{Reader, Result};
use crate::ir::*;

/// `findTargetSubtable` and `lookupLLVMIntrinsicByName`: the generic intrinsic `name` names, if any.
pub(crate) enum Lookup {
    NotIntrinsic,
    Generic(usize),
    /// A name in the sub-table of an LLVM target.
    Target(&'static str),
}

pub(crate) fn lookup(name: &str) -> Lookup {
    let rest = match name.strip_prefix("llvm.") {
        Some(r) => r,
        None => return Lookup::NotIntrinsic,
    };
    let target = rest.split('.').next().unwrap_or("");
    if let Some(t) = tbl::TARGETS.iter().find(|t| **t == target) {
        return Lookup::Target(t);
    }
    // Successive narrowing over the dotted components of the name, as LLVM's binary search does.
    let names: Vec<&str> = tbl::GENERIC.iter().map(|g| g.0).collect();
    let nb = name.as_bytes();
    let mut low = 0usize;
    let mut high = names.len();
    let mut last_low = low;
    let mut cmp_end = 4usize;
    // strncmp over [start, end) treating the end of a string as NUL.
    let byte = |s: &[u8], i: usize| -> u8 { if i < s.len() { s[i] } else { 0 } };
    let compare = |a: &[u8], b: &[u8], start: usize, end: usize| -> std::cmp::Ordering {
        for i in start..end {
            let (x, y) = (byte(a, i), byte(b, i));
            if x != y {
                return x.cmp(&y);
            }
            if x == 0 {
                break;
            }
        }
        std::cmp::Ordering::Equal
    };
    while cmp_end < nb.len() && high > low {
        let cmp_start = cmp_end;
        cmp_end = match name[cmp_start + 1..].find('.') {
            Some(p) => cmp_start + 1 + p,
            None => nb.len(),
        };
        last_low = low;
        let lo = low
            + names[low..high]
                .partition_point(|n| compare(n.as_bytes(), nb, cmp_start, cmp_end) == std::cmp::Ordering::Less);
        let hi = low
            + names[low..high]
                .partition_point(|n| compare(n.as_bytes(), nb, cmp_start, cmp_end) != std::cmp::Ordering::Greater);
        low = lo;
        high = hi;
    }
    if high > low {
        last_low = low;
    }
    if last_low >= names.len() {
        return Lookup::NotIntrinsic;
    }
    let found = names[last_low];
    let matches = name == found || (name.starts_with(found) && nb.get(found.len()) == Some(&b'.'));
    if !matches {
        return Lookup::NotIntrinsic;
    }
    // A non-overloaded intrinsic needs an exact match.
    if name.len() != found.len() && !tbl::GENERIC[last_low].1 {
        return Lookup::NotIntrinsic;
    }
    Lookup::Generic(last_low)
}

/// Whether LLVM 21 knows `name` as an intrinsic (`Function::getIntrinsicID` is not `not_intrinsic`).
pub fn is_known_intrinsic(name: &str) -> bool {
    !matches!(lookup(name), Lookup::NotIntrinsic)
}

/// An IIT type descriptor (`Intrinsic::IITDescriptor`).
#[derive(Clone, Copy, Debug)]
enum D {
    Void,
    VarArg,
    Mmx,
    Amx,
    Token,
    Metadata,
    Half,
    BFloat,
    Float,
    Double,
    Quad,
    PpcQuad,
    Integer(u32),
    Svcount,
    Vector(u32, bool),
    Pointer(u32),
    Struct(u32),
    Argument(u32),
    ExtendArgument(u32),
    TruncArgument(u32),
    OneNthEltsVecArgument(u32, u32),
    SameVecWidthArgument(u32),
    VecOfAnyPtrsToElt(u32, u32),
    VecElementArgument(u32),
    Subdivide2Argument(u32),
    Subdivide4Argument(u32),
    VecOfBitcastsToInt(u32),
}

/// `DecodeIITType`.
fn decode(next: &mut usize, infos: &[u8], last: u8, out: &mut Vec<D>) -> Option<()> {
    let scalable = last == 43;
    let info = *infos.get(*next)?;
    *next += 1;
    let arg = |next: &mut usize| -> u32 {
        if *next == infos.len() {
            0
        } else {
            *next += 1;
            infos[*next - 1] as u32
        }
    };
    let vec = |n: u32, next: &mut usize, out: &mut Vec<D>| -> Option<()> {
        out.push(D::Vector(n, scalable));
        decode(next, infos, info, out)
    };
    match info {
        0 => out.push(D::Void),
        29 => out.push(D::VarArg),
        17 => out.push(D::Mmx),
        51 => out.push(D::Amx),
        18 => out.push(D::Token),
        19 => out.push(D::Metadata),
        6 => out.push(D::Half),
        48 => out.push(D::BFloat),
        7 => out.push(D::Float),
        8 => out.push(D::Double),
        41 => out.push(D::Quad),
        52 => out.push(D::PpcQuad),
        1 => out.push(D::Integer(1)),
        57 => out.push(D::Integer(2)),
        58 => out.push(D::Integer(4)),
        59 => out.push(D::Svcount),
        2 => out.push(D::Integer(8)),
        3 => out.push(D::Integer(16)),
        4 => out.push(D::Integer(32)),
        5 => out.push(D::Integer(64)),
        35 => out.push(D::Integer(128)),
        28 => return vec(1, next, out),
        9 => return vec(2, next, out),
        53 => return vec(3, next, out),
        10 => return vec(4, next, out),
        60 => return vec(6, next, out),
        11 => return vec(8, next, out),
        61 => return vec(10, next, out),
        12 => return vec(16, next, out),
        13 => return vec(32, next, out),
        16 => return vec(64, next, out),
        47 => return vec(128, next, out),
        50 => return vec(256, next, out),
        36 => return vec(512, next, out),
        37 => return vec(1024, next, out),
        62 => return vec(2048, next, out),
        63 => return vec(4096, next, out),
        54 => out.push(D::Pointer(10)),
        55 => out.push(D::Pointer(20)),
        14 => out.push(D::Pointer(0)),
        27 => {
            let a = *infos.get(*next)? as u32;
            *next += 1;
            out.push(D::Pointer(a));
        }
        15 => out.push(D::Argument(arg(next))),
        25 => out.push(D::ExtendArgument(arg(next))),
        26 => out.push(D::TruncArgument(arg(next))),
        30 => {
            let argno = arg(next);
            let n = arg(next);
            out.push(D::OneNthEltsVecArgument(n, argno));
        }
        31 => out.push(D::SameVecWidthArgument(arg(next))),
        34 => {
            let argno = arg(next);
            let refno = arg(next);
            out.push(D::VecOfAnyPtrsToElt(argno, refno));
        }
        20 => out.push(D::Struct(0)),
        21..=24 | 38..=40 | 49 => {
            let n = match info {
                21 => 2,
                22 => 3,
                23 => 4,
                24 => 5,
                38 => 6,
                39 => 7,
                40 => 8,
                _ => 9,
            };
            out.push(D::Struct(n));
            for _ in 0..n {
                decode(next, infos, info, out)?;
            }
        }
        44 => out.push(D::Subdivide2Argument(arg(next))),
        45 => out.push(D::Subdivide4Argument(arg(next))),
        42 => out.push(D::VecElementArgument(arg(next))),
        43 => return decode(next, infos, info, out),
        46 => out.push(D::VecOfBitcastsToInt(arg(next))),
        _ => return None,
    }
    Some(())
}

/// `getIntrinsicInfoTableEntries`.
fn table_entries(index: usize) -> Option<Vec<D>> {
    let word = tbl::GENERIC[index].2;
    let mut out = Vec::new();
    let (entries, mut next): (Vec<u8>, usize) = if word >> 15 != 0 {
        (tbl::IIT_LONG.to_vec(), (word & 0x7fff) as usize)
    } else {
        let mut v = Vec::new();
        let mut w = word;
        loop {
            v.push((w & 0xf) as u8);
            w >>= 4;
            if w == 0 {
                break;
            }
        }
        (v, 0)
    };
    decode(&mut next, &entries, 0, &mut out)?;
    while next != entries.len() && entries[next] != 0 {
        decode(&mut next, &entries, 0, &mut out)?;
    }
    Some(out)
}

struct Matcher<'a> {
    types: &'a mut TypeTable,
    arg_tys: Vec<TypeId>,
    deferred: Vec<(TypeId, usize)>,
}

impl<'a> Matcher<'a> {
    /// `matchIntrinsicType`: true when `ty` does not match the descriptor at `infos[*pos]`.
    fn mismatch(&mut self, ty: TypeId, infos: &[D], pos: &mut usize, deferred_check: bool) -> bool {
        if *pos >= infos.len() {
            return true;
        }
        let start = *pos;
        let d = infos[*pos];
        *pos += 1;
        let t = self.types.get(ty).clone();
        match d {
            D::Void => !matches!(t, Type::Void),
            D::VarArg => true,
            D::Mmx => {
                !matches!(t, Type::Vector { len: 1, elem, scalable: false } if self.types.int_bits(elem) == Some(64))
            }
            D::Amx => !matches!(t, Type::X86Amx),
            D::Token => !matches!(t, Type::Token),
            D::Metadata => !matches!(t, Type::Metadata),
            D::Half => !matches!(t, Type::Half),
            D::BFloat => !matches!(t, Type::BFloat),
            D::Float => !matches!(t, Type::Float),
            D::Double => !matches!(t, Type::Double),
            D::Quad => !matches!(t, Type::Fp128),
            D::PpcQuad => !matches!(t, Type::PpcFp128),
            D::Integer(w) => t != Type::Int(w),
            D::Svcount => !matches!(t, Type::TargetExt { ref name, .. } if name == "aarch64.svcount"),
            D::Vector(n, scalable) => match t {
                Type::Vector { len, elem, scalable: s } if len == n && s == scalable => {
                    self.mismatch(elem, infos, pos, deferred_check)
                }
                _ => true,
            },
            D::Pointer(a) => t != Type::Ptr(a),
            D::Struct(n) => match t {
                Type::LiteralStruct { packed: false, elems } if elems.len() == n as usize => {
                    elems.iter().any(|e| self.mismatch(*e, infos, pos, deferred_check))
                }
                _ => true,
            },
            D::Argument(info) => {
                let num = (info >> 3) as usize;
                let kind = info & 7;
                if num < self.arg_tys.len() {
                    return ty != self.arg_tys[num];
                }
                if num > self.arg_tys.len() || kind == 7 {
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                self.arg_tys.push(ty);
                match kind {
                    0 => false,
                    1 => !self.types.is_int_or_int_vector(ty),
                    2 => !self.types.is_fp_or_fp_vector(ty),
                    3 => !self.types.is_vector(ty),
                    4 => !self.types.is_ptr(ty),
                    _ => true,
                }
            }
            D::ExtendArgument(info) | D::TruncArgument(info) => {
                let num = (info >> 3) as usize;
                if num >= self.arg_tys.len() {
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                let base = self.arg_tys[num];
                let extend = matches!(d, D::ExtendArgument(_));
                let new = match self.types.get(base).clone() {
                    Type::Vector { len, elem, scalable } => {
                        let e = match (self.types.get(elem).clone(), extend) {
                            (Type::Int(b), true) => self.types.int(2 * b),
                            (Type::Int(b), false) => self.types.int(b / 2),
                            (Type::Double, false) => self.types.intern(Type::Float),
                            (Type::Float, false) => self.types.intern(Type::Half),
                            _ => return true,
                        };
                        self.types.intern(Type::Vector { len, elem: e, scalable })
                    }
                    Type::Int(b) => {
                        if extend {
                            self.types.int(2 * b)
                        } else {
                            self.types.int(b / 2)
                        }
                    }
                    _ => return true,
                };
                ty != new
            }
            D::OneNthEltsVecArgument(n, refno) => {
                let r = refno as usize;
                if r >= self.arg_tys.len() {
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                match self.types.get(self.arg_tys[r]).clone() {
                    Type::Vector { len, elem, scalable } if n != 0 && len % n == 0 => {
                        let want = self.types.intern(Type::Vector { len: len / n, elem, scalable });
                        want != ty
                    }
                    _ => true,
                }
            }
            D::SameVecWidthArgument(info) => {
                let num = (info >> 3) as usize;
                if num >= self.arg_tys.len() {
                    *pos += 1;
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                let r = self.types.get(self.arg_tys[num]).clone();
                let elem_ty = match (r, t) {
                    (Type::Vector { len: rl, scalable: rs, .. }, Type::Vector { len, elem, scalable }) => {
                        if rl != len || rs != scalable {
                            return true;
                        }
                        elem
                    }
                    (Type::Vector { .. }, _) | (_, Type::Vector { .. }) => return true,
                    _ => ty,
                };
                self.mismatch(elem_ty, infos, pos, deferred_check)
            }
            D::VecOfAnyPtrsToElt(argno, refno) => {
                let r = refno as usize;
                if r >= self.arg_tys.len() {
                    if deferred_check {
                        return true;
                    }
                    self.arg_tys.push(ty);
                    self.deferred.push((ty, start));
                    return false;
                }
                if !deferred_check {
                    // LLVM asserts this of its own tables ("Table consistency error").
                    debug_assert_eq!(argno as usize, self.arg_tys.len());
                    self.arg_tys.push(ty);
                }
                match (self.types.get(self.arg_tys[r]).clone(), t) {
                    (Type::Vector { len: rl, scalable: rs, .. }, Type::Vector { len, elem, scalable })
                        if rl == len && rs == scalable =>
                    {
                        !self.types.is_ptr(elem)
                    }
                    _ => true,
                }
            }
            D::VecElementArgument(info) => {
                let num = (info >> 3) as usize;
                if num >= self.arg_tys.len() {
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                match self.types.get(self.arg_tys[num]).clone() {
                    Type::Vector { elem, .. } => ty != elem,
                    _ => true,
                }
            }
            D::Subdivide2Argument(info) | D::Subdivide4Argument(info) => {
                let num = (info >> 3) as usize;
                if num >= self.arg_tys.len() {
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                let subdivs = if matches!(d, D::Subdivide2Argument(_)) { 1 } else { 2 };
                let mut v = self.arg_tys[num];
                for _ in 0..subdivs {
                    v = match self.types.get(v).clone() {
                        Type::Vector { len, elem, scalable } => {
                            let e = match self.types.get(elem).clone() {
                                Type::Int(b) => self.types.int(b / 2),
                                Type::Double => self.types.intern(Type::Float),
                                Type::Float => self.types.intern(Type::Half),
                                _ => return true,
                            };
                            self.types.intern(Type::Vector { len: len * 2, elem: e, scalable })
                        }
                        _ => return true,
                    };
                }
                ty != v
            }
            D::VecOfBitcastsToInt(info) => {
                let num = (info >> 3) as usize;
                if num >= self.arg_tys.len() {
                    if deferred_check {
                        return true;
                    }
                    self.deferred.push((ty, start));
                    return false;
                }
                match (self.types.get(self.arg_tys[num]).clone(), t) {
                    (Type::Vector { len: rl, elem: re, scalable: rs }, Type::Vector { .. }) => {
                        let bits = self.types.scalar_bits(re);
                        if bits == 0 {
                            return true;
                        }
                        let ie = self.types.int(bits);
                        let want = self.types.intern(Type::Vector { len: rl, elem: ie, scalable: rs });
                        want != ty
                    }
                    _ => true,
                }
            }
        }
    }
}

/// `Intrinsic::getIntrinsicSignature`: the overload types when the function type matches.
fn signature(types: &mut TypeTable, index: usize, fty: TypeId) -> Option<Vec<TypeId>> {
    let infos = table_entries(index)?;
    let (ret, params, vararg) = {
        let (r, p, v) = types.function(fty)?;
        (r, p.to_vec(), v)
    };
    let mut m = Matcher { types, arg_tys: Vec::new(), deferred: Vec::new() };
    let mut pos = 0usize;
    if m.mismatch(ret, &infos, &mut pos, false) {
        return None;
    }
    for p in &params {
        if m.mismatch(*p, &infos, &mut pos, false) {
            return None;
        }
    }
    let mut i = 0;
    while i < m.deferred.len() {
        let (t, at) = m.deferred[i];
        let mut p = at;
        if m.mismatch(t, &infos, &mut p, true) {
            return None;
        }
        i += 1;
    }
    // `matchIntrinsicVarArg`.
    let rest = &infos[pos..];
    let vararg_mismatch = if rest.is_empty() {
        vararg
    } else if rest.len() != 1 {
        true
    } else {
        match rest[0] {
            D::VarArg => !vararg,
            _ => true,
        }
    };
    if vararg_mismatch {
        return None;
    }
    Some(m.arg_tys)
}

/// `getMangledTypeStr`.
fn mangle(types: &TypeTable, t: TypeId, unnamed: &mut bool) -> String {
    match types.get(t) {
        Type::Ptr(a) => format!("p{}", a),
        Type::Array { len, elem } => format!("a{}{}", len, mangle(types, *elem, unnamed)),
        Type::Struct { name, .. } => {
            if name.is_empty() {
                *unnamed = true;
                "s_s".into()
            } else {
                format!("s_{}s", name)
            }
        }
        Type::LiteralStruct { elems, .. } => {
            let mut s = String::from("sl_");
            for e in elems {
                s.push_str(&mangle(types, *e, unnamed));
            }
            s.push('s');
            s
        }
        Type::Function { ret, params, vararg } => {
            let mut s = format!("f_{}", mangle(types, *ret, unnamed));
            for p in params {
                s.push_str(&mangle(types, *p, unnamed));
            }
            if *vararg {
                s.push_str("vararg");
            }
            s.push('f');
            s
        }
        Type::Vector { len, elem, scalable } => {
            format!("{}v{}{}", if *scalable { "nx" } else { "" }, len, mangle(types, *elem, unnamed))
        }
        Type::TargetExt { name, types: ts, ints } => {
            let mut s = format!("t{}", name);
            for x in ts {
                s.push('_');
                s.push_str(&mangle(types, *x, unnamed));
            }
            for i in ints {
                s.push_str(&format!("_{}", i));
            }
            s.push('t');
            s
        }
        Type::Void => "isVoid".into(),
        Type::Metadata => "Metadata".into(),
        Type::Half => "f16".into(),
        Type::BFloat => "bf16".into(),
        Type::Float => "f32".into(),
        Type::Double => "f64".into(),
        Type::X86Fp80 => "f80".into(),
        Type::Fp128 => "f128".into(),
        Type::PpcFp128 => "ppcf128".into(),
        Type::X86Amx => "x86amx".into(),
        Type::Int(b) => format!("i{}", b),
        Type::Label | Type::Token => String::new(),
    }
}

/// `Intrinsic::getAttributes`: the intrinsic's attribute list for function type `fty`.
fn attributes(m: &mut Module, index: usize, fty: TypeId) -> AttrList {
    let (_, _, _, fn_set, list) = tbl::GENERIC[index];
    let (entries, has_fn) = tbl::LISTS[list as usize];
    let (ret, params) = {
        let (r, p, _) = m.types.function(fty).unwrap();
        (r, p.to_vec())
    };
    let mut out = AttrList::default();
    if has_fn {
        let mut s = AttrSet::new();
        for a in tbl::FN_SETS[fn_set as usize] {
            match a {
                FnAttr::E(k) => s.add(Attribute::Enum(*k)),
                FnAttr::Memory(v) => s.add(Attribute::Int(AttrKind::Memory, *v)),
            }
        }
        out.function = m.attr_set(s);
    }
    for (idx, set) in entries {
        let ty = if *idx == 0 { ret } else { params[*idx as usize - 1] };
        let bits = m.types.scalar_bits(ty);
        let mut s = AttrSet::new();
        for a in tbl::ARG_SETS[*set as usize] {
            match a {
                ArgAttr::E(k) => s.add(Attribute::Enum(*k)),
                ArgAttr::CapturesNone => s.add(Attribute::Int(AttrKind::Captures, 0)),
                ArgAttr::Align(v) => s.add(Attribute::Int(AttrKind::Alignment, *v)),
                ArgAttr::Range(lo, hi) => {
                    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
                    let ext = |v: i64| -> Vec<u64> {
                        let n = bits.div_ceil(64) as usize;
                        let mut w = vec![if v < 0 { u64::MAX } else { 0 }; n.max(1)];
                        w[0] = v as u64;
                        let last = w.len() - 1;
                        if bits % 64 != 0 {
                            w[last] &= mask;
                        }
                        w
                    };
                    s.add(Attribute::Range(AttrKind::Range, ConstantRange { bits, lower: ext(*lo), upper: ext(*hi) }));
                }
            }
        }
        let id = m.attr_set(s);
        if *idx == 0 {
            out.ret = id;
        } else {
            let p = *idx as usize - 1;
            if out.params.len() <= p {
                out.params.resize(p + 1, AttrSetId::EMPTY);
            }
            out.params[p] = id;
        }
    }
    out.trim();
    out
}

/// The special upgrades of `upgradeIntrinsicFunction1` that this reader does not implement, by the conditions LLVM
/// tests; `Some(description)` when `name` (without `llvm.`) with `nargs` parameters needs one.
fn special_upgrade(name: &str, nargs: usize) -> Option<&'static str> {
    let sw = |p: &str| name.starts_with(p);
    if nargs == 1 && (sw("ctlz.") || sw("cttz.")) {
        return Some("one-argument ctlz/cttz");
    }
    if nargs == 2 && name == "coro.end" {
        return Some("two-argument coro.end");
    }
    if sw("dbg.") {
        return Some("debug intrinsics");
    }
    if let Some(rest) = name.strip_prefix("experimental.vector.") {
        let r = |p: &str| rest.starts_with(p);
        if !r("extract.last.active.")
            && (r("extract.")
                || r("insert.")
                || r("splice.")
                || r("reverse.")
                || r("interleave2.")
                || r("deinterleave2."))
        {
            return Some("experimental.vector intrinsics");
        }
        if r("reduce.") {
            return Some("experimental.vector.reduce intrinsics");
        }
    }
    if sw("experimental.stepvector.") {
        return Some("experimental.stepvector");
    }
    if sw("flt.rounds") {
        return Some("flt.rounds");
    }
    if sw("invariant.group.barrier") {
        return Some("invariant.group.barrier");
    }
    if (sw("memcpy.") || sw("memmove.") || sw("memset.")) && nargs == 5 {
        return Some("memory intrinsics with an alignment argument");
    }
    if sw("objectsize.") && (nargs == 2 || nargs == 3) {
        return Some("two- and three-argument objectsize");
    }
    if sw("ptr.annotation.") && nargs == 4 {
        return Some("four-argument ptr.annotation");
    }
    if name == "stackprotectorcheck" {
        return Some("stackprotectorcheck");
    }
    if name == "thread.pointer" {
        return Some("thread.pointer");
    }
    if name == "var.annotation" && nargs == 4 {
        return Some("four-argument var.annotation");
    }
    None
}

/// Applies `UpgradeIntrinsicFunction` to every function, and redirects calls to remangled declarations.
pub(super) fn upgrade_intrinsics(r: &mut Reader) -> Result<()> {
    let n = r.m.functions.len();
    let mut replaced: Vec<(usize, usize)> = Vec::new();
    for f in 0..n {
        let name = r.m.functions[f].common.name.clone();
        let index = match lookup(&name) {
            Lookup::NotIntrinsic => continue,
            Lookup::Target(t) => {
                return r.unsupported(format!("target-specific LLVM intrinsic {} (target {})", name, t));
            }
            Lookup::Generic(i) => i,
        };
        let fty = r.m.functions[f].ty;
        let nargs = r.m.types.function(fty).unwrap().1.len();
        if let Some(what) = special_upgrade(&name[5..], nargs) {
            return r.unsupported(format!("{}: the upgrade of {} is not implemented", name, what));
        }
        let ret = r.m.types.function(fty).unwrap().0;
        if matches!(r.m.types.get(ret), Type::Struct { .. } | Type::LiteralStruct { packed: true, .. }) {
            if let Some(Some(D::Struct(_))) = table_entries(index).map(|t| t.first().copied()) {
                return r.unsupported(format!("{}: the struct return upgrade is not implemented", name));
            }
        }
        let sig = match signature(&mut r.m.types, index, fty) {
            Some(s) => s,
            None => continue,
        };
        // `remangleIntrinsicFunction`.
        let mut unnamed = false;
        let mut wanted = tbl::GENERIC[index].0.to_string();
        for t in &sig {
            wanted.push('.');
            wanted.push_str(&mangle(&r.m.types, *t, &mut unnamed));
        }
        if unnamed {
            return r.unsupported(format!("{}: intrinsics overloaded on unnamed types", name));
        }
        if wanted == name {
            r.m.functions[f].attrs = attributes(&mut r.m, index, fty);
            continue;
        }
        let existing = r.m.functions.iter().position(|g| g.common.name == wanted);
        let target = match existing {
            Some(e) if r.m.functions[e].ty == fty => e,
            Some(_) => {
                return r.unsupported(format!("{}: a different function already has its LLVM 21 name {}", name, wanted));
            }
            None => {
                if r.m.globals.iter().any(|g| g.common.name == wanted)
                    || r.m.aliases.iter().any(|g| g.common.name == wanted)
                {
                    return r.unsupported(format!("{}: a global already has its LLVM 21 name {}", name, wanted));
                }
                let old = &r.m.functions[f];
                let program_as = r.layout.as_ref().map(|l| l.program_addrspace).unwrap_or(0);
                let new = Function {
                    common: GlobalCommon {
                        name: wanted.clone(),
                        linkage: Linkage::External,
                        visibility: Visibility::Default,
                        dll_storage: DllStorage::Default,
                        unnamed_addr: UnnamedAddr::None,
                        thread_local: ThreadLocal::NotThreadLocal,
                        dso_local: false,
                        addrspace: program_as,
                        partition: String::new(),
                    },
                    ty: fty,
                    cc: old.cc,
                    attrs: AttrList::default(),
                    align: None,
                    section: None,
                    comdat: None,
                    gc: None,
                    prefix: None,
                    prologue: None,
                    personality: None,
                    args: old.args.iter().map(|a| Argument { ty: a.ty, name: None }).collect(),
                    blocks: Vec::new(),
                    insts: Vec::new(),
                    md: Vec::new(),
                    bitcode_type: old.bitcode_type,
                };
                r.m.functions.push(new);
                r.m.functions.len() - 1
            }
        };
        if !r.m.functions[f].is_declaration() {
            return r.unsupported(format!("{}: a definition with an intrinsic name", name));
        }
        r.m.functions[target].attrs = attributes(&mut r.m, index, fty);
        replaced.push((f, target));
    }
    if replaced.is_empty() {
        return Ok(());
    }
    // Calls and other uses move to the new declarations; the old ones are erased.
    let map: std::collections::HashMap<u32, u32> = replaced.iter().map(|&(a, b)| (a as u32, b as u32)).collect();
    redirect_uses(&mut r.m, &map)?;
    let mut keep: Vec<bool> = vec![true; r.m.functions.len()];
    for &(a, _) in &replaced {
        keep[a] = false;
    }
    let mut renumber = vec![u32::MAX; r.m.functions.len()];
    let mut next = 0u32;
    for (i, k) in keep.iter().enumerate() {
        if *k {
            renumber[i] = next;
            next += 1;
        }
    }
    let old = std::mem::take(&mut r.m.functions);
    r.m.functions = old.into_iter().enumerate().filter(|(i, _)| keep[*i]).map(|(_, f)| f).collect();
    renumber_functions(&mut r.m, &renumber);
    Ok(())
}

/// Replaces references to function `from` with `to` for every pair in `map`.
fn redirect_uses(m: &mut Module, map: &std::collections::HashMap<u32, u32>) -> Result<()> {
    let fix = |v: &mut Value| {
        if let Value::Global(GlobalId::Function(f)) = v {
            if let Some(t) = map.get(f) {
                *f = *t;
            }
        }
    };
    for f in m.functions.iter_mut() {
        for inst in f.insts.iter_mut() {
            super::function::visit_values_mut(&mut inst.op, &mut |v| fix(v));
        }
        for v in [&mut f.prefix, &mut f.prologue, &mut f.personality].into_iter().flatten() {
            fix(v);
        }
    }
    let used_in_constants = m.consts.iter().any(|c| const_refs_any(c, map));
    let used_in_md = m.metadata.iter().any(
        |md| matches!(md, Metadata::Value { value: Value::Global(GlobalId::Function(f)), .. } if map.contains_key(f)),
    );
    let used_in_globals = m
        .globals
        .iter()
        .any(|g| matches!(g.initializer, Some(Value::Global(GlobalId::Function(f))) if map.contains_key(&f)))
        || m.aliases.iter().any(|a| matches!(a.aliasee, Value::Global(GlobalId::Function(f)) if map.contains_key(&f)));
    if used_in_constants || used_in_md || used_in_globals {
        return Err(super::unsupported(None, "a renamed intrinsic used outside call instructions"));
    }
    Ok(())
}

fn const_refs_any(c: &Constant, map: &std::collections::HashMap<u32, u32>) -> bool {
    let refs = |v: &Value| matches!(v, Value::Global(GlobalId::Function(f)) if map.contains_key(f));
    match &c.kind {
        ConstKind::Struct(e) | ConstKind::Array(e) | ConstKind::Vector(e) => e.iter().any(refs),
        ConstKind::Expr(ConstExpr::Cast { value, .. }) => refs(value),
        ConstKind::Expr(ConstExpr::Gep { ptr, indices, .. }) => refs(ptr) || indices.iter().any(refs),
        ConstKind::Expr(ConstExpr::Binary { lhs, rhs, .. }) => refs(lhs) || refs(rhs),
        _ => false,
    }
}

/// Renumbers function references after functions were removed (`renumber[old] = new`).
fn renumber_functions(m: &mut Module, renumber: &[u32]) {
    let fix = |v: &mut Value| {
        if let Value::Global(GlobalId::Function(f)) = v {
            *f = renumber[*f as usize];
        }
    };
    for f in m.functions.iter_mut() {
        for inst in f.insts.iter_mut() {
            super::function::visit_values_mut(&mut inst.op, &mut |v| fix(v));
        }
        for v in [&mut f.prefix, &mut f.prologue, &mut f.personality].into_iter().flatten() {
            fix(v);
        }
    }
    for g in m.globals.iter_mut() {
        if let Some(v) = &mut g.initializer {
            fix(v);
        }
    }
    for a in m.aliases.iter_mut() {
        fix(&mut a.aliasee);
    }
    for a in m.ifuncs.iter_mut() {
        fix(&mut a.resolver);
    }
    let consts = std::mem::take(&mut m.consts);
    let mut new_consts = Vec::with_capacity(consts.len());
    for mut c in consts {
        match &mut c.kind {
            ConstKind::Struct(e) | ConstKind::Array(e) | ConstKind::Vector(e) => e.iter_mut().for_each(fix),
            ConstKind::Expr(ConstExpr::Cast { value, .. }) => fix(value),
            ConstKind::Expr(ConstExpr::Gep { ptr, indices, .. }) => {
                fix(ptr);
                indices.iter_mut().for_each(fix);
            }
            ConstKind::Expr(ConstExpr::Binary { lhs, rhs, .. }) => {
                fix(lhs);
                fix(rhs);
            }
            ConstKind::Expr(ConstExpr::ExtractElement { vec, idx }) => {
                fix(vec);
                fix(idx);
            }
            ConstKind::Expr(ConstExpr::InsertElement { vec, elt, idx }) => {
                fix(vec);
                fix(elt);
                fix(idx);
            }
            ConstKind::Expr(ConstExpr::ShuffleVector { a, b, .. }) => {
                fix(a);
                fix(b);
            }
            _ => {}
        }
        new_consts.push(c);
    }
    m.replace_consts(new_consts);
    let mds = std::mem::take(&mut m.metadata);
    let mut new_md = Vec::with_capacity(mds.len());
    for mut md in mds {
        match &mut md {
            Metadata::Value { value, .. } => fix(value),
            Metadata::Local { function, .. } => *function = renumber[*function as usize],
            _ => {}
        }
        new_md.push(md);
    }
    m.metadata = new_md;
}
