//! The byte-addressed memory model (plan 5.4).
//!
//! AIR is pointer-based LLVM IR; WGSL has typed variables and no pointer arithmetic or reinterpretation. Every
//! guest-visible memory region is one WGSL array of `u32` words (of `atomic<u32>` where it must be): each bound
//! buffer, each threadgroup variable and threadgroup buffer, each alloca that survives canonicalization, and each
//! module-scope variable. Nothing about the guest's types survives into WGSL declarations: layouts come from the
//! module's data layout string.
//!
//! A pointer value is a region and a byte offset. The region is static where the analysis proves a single one (the
//! value's WGSL carrier is then the offset, a `u32`); otherwise the carrier is `vec2<u32>(offset, region tag)` and each
//! access is a selection over the candidate regions. A pointer whose regions cannot be bounded (an `inttoptr` of an
//! integer that does not come from a `ptrtoint`) is refused.
//!
//! Addresses: the address of byte `o` of the region with tag `t` is the 64-bit value `t << 32 | o` (tag 0 is the null
//! pointer's). Regions are disjoint and each starts at a multiple of 2^32, so pointer comparison, `ptrtoint` and
//! `inttoptr` behave as on a flat address space in which every region is allocated at a multiple of 2^32; buffer
//! regions keep their low address bits, because the executor binds each buffer from an aligned base and passes the
//! residual offset (plan 5.4), which is the region's base offset here. The offset arithmetic is modulo 2^32, which is
//! exact for every in-bounds address: WebGPU bindings are smaller than 4 GiB.
//!
//! Loads and stores of any type are word accesses with shifts and masks ([`access`]). A sub-word or misaligned store
//! into a region other invocations can access concurrently (device memory in fragment and compute functions,
//! threadgroup memory) is an `atomicAnd` followed by an `atomicOr`, and such a region is declared atomic so that every
//! access to it is atomic; neighbors written concurrently are never disturbed.

// The refusal is the compiler's only failure value (api::compile) and is returned by value throughout.
#![allow(clippy::result_large_err)]

pub mod access;

use crate::api::Refusal;
use crate::ir::layout::DataLayout;
use crate::ir::{CastOp, ConstExpr, ConstKind, GlobalId, InstId, Module, Op, Type, Value};
use crate::wgsl::tracked;
use std::collections::{BTreeSet, HashMap, HashSet};

pub type RegionId = usize;

/// The null pointer's region: no storage; tag 0.
pub const NULL: RegionId = 0;

/// What a region is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Null,
    /// A buffer argument: IR argument index, Metal buffer index, address space (1 device, 2 constant).
    Buffer {
        arg: u32,
        index: u64,
        space: u32,
    },
    /// A `[[threadgroup(index)]]` buffer argument.
    ThreadgroupArg {
        arg: u32,
        index: u64,
    },
    /// A module-scope variable in address space 3.
    ThreadgroupGlobal {
        global: u32,
    },
    /// An alloca of the entry block.
    Alloca {
        inst: InstId,
    },
    /// A module-scope variable in address space 0 (thread-private).
    PrivateGlobal {
        global: u32,
    },
    /// A module-scope variable in address space 2 (program-scope constant data).
    ConstantGlobal {
        global: u32,
    },
    /// A texture or sampler argument: a handle, not memory. Rules of the resource lowering resolve it.
    Handle {
        arg: u32,
    },
}

/// One region.
#[derive(Clone, Debug)]
pub struct Region {
    pub kind: RegionKind,
    /// The WGSL variable.
    pub name: String,
    /// Size in words when the variable has a fixed size.
    pub words: Option<u64>,
    /// Other invocations may access it concurrently.
    pub shared: bool,
    /// Declared `array<atomic<u32>>`; every access is atomic.
    pub atomic: bool,
    pub written: bool,
    pub read: bool,
    /// The high word of its addresses.
    pub tag: u32,
    /// The WGSL expression of the offset of the region's first byte, as pointers into it count offsets.
    pub base: String,
    /// The initial bytes of a module-scope variable with an initializer.
    pub init: Option<Vec<u8>>,
}

/// How a region is declared and bound; filled in by the stage interface for argument regions.
#[derive(Clone, Debug)]
pub struct ArgRegion {
    pub kind: RegionKind,
    pub name: String,
    pub base: String,
    pub words: Option<u64>,
}

/// The regions of a function and the provenance of its pointer values.
pub struct Memory {
    pub regions: Vec<Region>,
    /// Region sets of pointer values and of integers that carry a pointer's provenance (`ptrtoint` and arithmetic on
    /// it). Function-local values only; constants and globals are computed on demand.
    prov: HashMap<Value, BTreeSet<RegionId>>,
    /// Pointer values (and provenance integers converted back) whose regions cannot be bounded.
    unbounded: HashSet<Value>,
    arg_region: HashMap<u32, RegionId>,
    global_region: HashMap<u32, RegionId>,
    alloca_region: HashMap<InstId, RegionId>,
}

fn int_operand(m: &Module, v: Value) -> Option<u64> {
    if let Value::Const(c) = v
        && let ConstKind::Int(w) = &m.constant(c).kind
    {
        return Some(w[0]);
    }
    None
}

impl Memory {
    /// Analyze function `fi`: create the regions of its arguments (`args`, by IR argument index), the module-scope
    /// variables it uses and its allocas, and compute the provenance of every pointer value.
    pub fn analyze(
        m: &Module,
        fi: usize,
        dl: &DataLayout,
        args: &HashMap<u32, ArgRegion>,
        shared_device: bool,
    ) -> Result<Memory, Refusal> {
        let f = &m.functions[fi];
        let mut mem = Memory {
            regions: vec![Region {
                kind: RegionKind::Null,
                name: String::new(),
                words: Some(0),
                shared: false,
                atomic: false,
                written: false,
                read: false,
                tag: 0,
                base: "0u".into(),
                init: None,
            }],
            prov: HashMap::new(),
            unbounded: HashSet::new(),
            arg_region: HashMap::new(),
            global_region: HashMap::new(),
            alloca_region: HashMap::new(),
        };
        let loc = |i: InstId| crate::api::Location {
            function: f.common.name.clone(),
            instruction: Some(crate::ir::print::print_instruction(m, fi, i)),
        };
        let mut arg_ids: Vec<&u32> = args.keys().collect();
        arg_ids.sort();
        for a in arg_ids {
            let r = &args[a];
            let shared = match r.kind {
                RegionKind::Buffer { space: 1, .. } => shared_device,
                RegionKind::ThreadgroupArg { .. } => true,
                _ => false,
            };
            let id = mem.push(r.kind.clone(), r.name.clone(), r.words, shared, r.base.clone(), None);
            mem.arg_region.insert(*a, id);
        }
        // Module-scope variables the function names, directly or inside constant expressions.
        let mut globals = BTreeSet::new();
        for inst in &f.insts {
            for v in inst.op.operands() {
                collect_globals(m, v, &mut globals, 0);
            }
        }
        for g in globals {
            let gv = &m.globals[g as usize];
            let size = dl.alloc_size(&m.types, gv.value_type).unwrap_or(0);
            let words = size.div_ceil(4).max(1);
            let (kind, shared, prefix) = match gv.common.addrspace {
                3 => (RegionKind::ThreadgroupGlobal { global: g }, true, "aw_tg"),
                0 => (RegionKind::PrivateGlobal { global: g }, false, "aw_pv"),
                2 => (RegionKind::ConstantGlobal { global: g }, false, "aw_cn"),
                a => {
                    return Err(tracked::refuse(
                        "unbounded-pointer-provenance",
                        format!("@{}", gv.common.name),
                        None,
                        format!("a module-scope variable in address space {}", a),
                    ));
                }
            };
            let init = match (gv.common.addrspace, gv.initializer) {
                (3, _) | (_, None) => None,
                (_, Some(init)) => {
                    let mut bytes = vec![0u8; size as usize];
                    const_bytes(m, dl, init, &mut bytes, 0)?;
                    Some(bytes)
                }
            };
            let id = mem.regions.len();
            let id = mem.push(kind, format!("{}{}", prefix, id), Some(words), shared, "0u".into(), init);
            mem.global_region.insert(g, id);
        }
        // Allocas: entry-block, constant count.
        for (bi, b) in f.blocks.iter().enumerate() {
            for &i in &b.insts {
                if let Op::Alloca { allocated, count, .. } = &f.inst(i).op {
                    let n = int_operand(m, *count);
                    if bi != 0 || n.is_none() {
                        return Err(tracked::refuse(
                            "dynamic-alloca",
                            "alloca",
                            Some(loc(i)),
                            "an alloca outside the entry block or with a variable count",
                        ));
                    }
                    let size = dl.alloc_size(&m.types, *allocated).unwrap_or(0) * n.unwrap();
                    let id = mem.regions.len();
                    let id = mem.push(
                        RegionKind::Alloca { inst: i },
                        format!("aw_fn{}", id),
                        Some(size.div_ceil(4).max(1)),
                        false,
                        "0u".into(),
                        None,
                    );
                    mem.alloca_region.insert(i, id);
                }
            }
        }
        mem.propagate(m, fi)?;
        Ok(mem)
    }

    fn push(
        &mut self,
        kind: RegionKind,
        name: String,
        words: Option<u64>,
        shared: bool,
        base: String,
        init: Option<Vec<u8>>,
    ) -> RegionId {
        let id = self.regions.len();
        self.regions.push(Region {
            kind,
            name,
            words,
            shared,
            atomic: false,
            written: false,
            read: false,
            tag: id as u32,
            base,
            init,
        });
        id
    }

    /// The region of an argument, when it has one.
    pub fn arg_region(&self, arg: u32) -> Option<RegionId> {
        self.arg_region.get(&arg).copied()
    }

    /// Provenance of every function-local pointer value and of the integers derived from `ptrtoint`, to a fixed
    /// point (phis make it cyclic).
    fn propagate(&mut self, m: &Module, fi: usize) -> Result<(), Refusal> {
        let f = &m.functions[fi];
        for (a, &r) in &self.arg_region {
            self.prov.insert(Value::Arg(*a), BTreeSet::from([r]));
        }
        loop {
            let mut changed = false;
            for (idx, inst) in f.insts.iter().enumerate() {
                let id = InstId(idx as u32);
                let me = Value::Inst(id);
                let is_ptr = m.types.is_ptr(inst.ty);
                let is_int = m.types.is_int(inst.ty);
                if !is_ptr && !is_int {
                    continue;
                }
                let mut set: Option<BTreeSet<RegionId>> = None;
                let mut unbounded = false;
                {
                    let mut add = |s: Option<BTreeSet<RegionId>>, ub: bool| {
                        unbounded |= ub;
                        if let Some(s) = s {
                            set.get_or_insert_with(BTreeSet::new).extend(s);
                        }
                    };
                    match &inst.op {
                        Op::Alloca { .. } if is_ptr => add(Some(BTreeSet::from([self.alloca_region[&id]])), false),
                        Op::Gep { ptr, .. } => add(self.lookup(m, *ptr), self.is_unbounded(m, *ptr)),
                        Op::Cast { op: CastOp::BitCast | CastOp::AddrSpaceCast, value, .. } if is_ptr => {
                            add(self.lookup(m, *value), self.is_unbounded(m, *value))
                        }
                        Op::Cast { op: CastOp::IntToPtr, value, .. } => {
                            let s = self.lookup(m, *value);
                            let ub = s.is_none() || self.is_unbounded(m, *value);
                            add(s, ub);
                        }
                        Op::Cast { op: CastOp::PtrToInt, value, .. } => add(self.lookup(m, *value), false),
                        Op::Binary { op, lhs, rhs, .. } if is_int => {
                            use crate::ir::BinOp::*;
                            let (l, r) = (self.lookup(m, *lhs), self.lookup(m, *rhs));
                            match (op, &l, &r) {
                                (Add | Or | Xor | And, Some(_), None) | (Sub, Some(_), None) => add(l, false),
                                (Add | Or | Xor | And, None, Some(_)) => add(r, false),
                                _ => {}
                            }
                        }
                        Op::Select { t, f: fv, .. } => {
                            add(self.lookup(m, *t), self.is_unbounded(m, *t));
                            add(self.lookup(m, *fv), self.is_unbounded(m, *fv));
                        }
                        Op::Phi { incoming, .. } => {
                            for (v, _) in incoming {
                                add(self.lookup(m, *v), self.is_unbounded(m, *v));
                            }
                        }
                        Op::Freeze { value } => add(self.lookup(m, *value), self.is_unbounded(m, *value)),
                        _ if is_ptr => unbounded = true,
                        _ => {}
                    }
                }
                if is_ptr && set.is_none() && !unbounded {
                    // A pointer from operands with no regions at all (a phi of undef only): the null region.
                    set = Some(BTreeSet::from([NULL]));
                }
                if unbounded && is_ptr && self.unbounded.insert(me) {
                    changed = true;
                }
                if let Some(s) = set {
                    let e = self.prov.entry(me).or_default();
                    let before = e.len();
                    e.extend(s);
                    changed |= e.len() != before;
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }

    /// The region set of a value: pointers and provenance-carrying integers; `None` for an integer without one.
    fn lookup(&self, m: &Module, v: Value) -> Option<BTreeSet<RegionId>> {
        match v {
            Value::Arg(_) | Value::Inst(_) => self.prov.get(&v).cloned(),
            Value::Global(GlobalId::Var(g)) => self.global_region.get(&g).map(|&r| BTreeSet::from([r])),
            Value::Const(c) => {
                let k = &m.constant(c);
                match &k.kind {
                    ConstKind::NullPtr | ConstKind::Undef | ConstKind::Poison | ConstKind::Zero
                        if m.types.is_ptr(k.ty) =>
                    {
                        Some(BTreeSet::from([NULL]))
                    }
                    ConstKind::Expr(ConstExpr::Gep { ptr, .. }) => self.lookup(m, *ptr),
                    ConstKind::Expr(ConstExpr::Cast {
                        op: CastOp::BitCast | CastOp::AddrSpaceCast | CastOp::PtrToInt,
                        value,
                    }) => self.lookup(m, *value),
                    ConstKind::Expr(ConstExpr::Cast { op: CastOp::IntToPtr, value }) => self.lookup(m, *value),
                    ConstKind::Expr(ConstExpr::Binary { lhs, rhs, .. }) => {
                        match (self.lookup(m, *lhs), self.lookup(m, *rhs)) {
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn is_unbounded(&self, m: &Module, v: Value) -> bool {
        match v {
            Value::Const(c) => {
                let k = m.constant(c);
                m.types.is_ptr(k.ty) && self.lookup(m, v).is_none()
            }
            Value::Global(GlobalId::Var(_)) => false,
            Value::Global(_) => true,
            _ => self.unbounded.contains(&v),
        }
    }

    /// The regions a pointer value may point into, or a refusal when they cannot be bounded.
    pub fn regions(&self, m: &Module, v: Value) -> Result<Vec<RegionId>, Refusal> {
        if self.is_unbounded(m, v) {
            return Err(tracked::refuse(
                "unbounded-pointer-provenance",
                "inttoptr",
                None,
                "a pointer whose provenance cannot be bounded to the regions the shader can access",
            ));
        }
        match self.lookup(m, v) {
            Some(s) => Ok(s.into_iter().collect()),
            None => {
                Err(tracked::refuse("unbounded-pointer-provenance", "ptr", None, "a pointer without a known region"))
            }
        }
    }

    /// The region set of an integer value that carries a pointer's provenance, if it does.
    pub fn int_regions(&self, m: &Module, v: Value) -> Option<Vec<RegionId>> {
        self.lookup(m, v).map(|s| s.into_iter().collect())
    }

    pub fn global_region(&self, g: u32) -> Option<RegionId> {
        self.global_region.get(&g).copied()
    }

    pub fn alloca_region(&self, i: InstId) -> Option<RegionId> {
        self.alloca_region.get(&i).copied()
    }
}

fn collect_globals(m: &Module, v: Value, out: &mut BTreeSet<u32>, depth: u32) {
    if depth > 64 {
        return;
    }
    match v {
        Value::Global(GlobalId::Var(g)) => {
            out.insert(g);
        }
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Expr(e) => {
                let ops: Vec<Value> = match e {
                    ConstExpr::Cast { value, .. } => vec![*value],
                    ConstExpr::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
                    ConstExpr::Gep { ptr, indices, .. } => {
                        std::iter::once(*ptr).chain(indices.iter().copied()).collect()
                    }
                    ConstExpr::ExtractElement { vec, idx } => vec![*vec, *idx],
                    ConstExpr::InsertElement { vec, elt, idx } => vec![*vec, *elt, *idx],
                    ConstExpr::ShuffleVector { a, b, .. } => vec![*a, *b],
                };
                for o in ops {
                    collect_globals(m, o, out, depth + 1);
                }
            }
            ConstKind::Struct(vs) | ConstKind::Array(vs) | ConstKind::Vector(vs) => {
                for o in vs {
                    collect_globals(m, *o, out, depth + 1);
                }
            }
            _ => {}
        },
        _ => {}
    }
}

/// Write the bytes of constant `v` (of its type, as the data layout places it) at `at`.
pub fn const_bytes(m: &Module, dl: &DataLayout, v: Value, out: &mut [u8], at: usize) -> Result<(), Refusal> {
    let unsupported = |what: &str| {
        tracked::refuse("pointer-in-memory", what.to_string(), None, format!("an initializer holding {}", what))
    };
    let Value::Const(c) = v else {
        return Err(unsupported("a global's address"));
    };
    let k = m.constant(c);
    let ty = k.ty;
    let store = dl.store_size(&m.types, ty).unwrap_or(0) as usize;
    let put = |out: &mut [u8], at: usize, bits: u128, n: usize| {
        for i in 0..n {
            if at + i < out.len() {
                out[at + i] = (bits >> (8 * i)) as u8;
            }
        }
    };
    match &k.kind {
        ConstKind::Zero | ConstKind::Undef | ConstKind::Poison | ConstKind::NullPtr => {
            if m.types.is_ptr(ty) && !matches!(k.kind, ConstKind::Undef | ConstKind::Poison) {
                // A null pointer is all zero bits (address 0, tag 0).
            }
            for i in 0..store {
                if at + i < out.len() {
                    out[at + i] = 0;
                }
            }
        }
        ConstKind::Int(words) => {
            let bits = m.types.int_bits(ty).unwrap_or(0);
            if bits == 1 {
                put(out, at, (words[0] & 1) as u128, 1);
            } else {
                let mut v: u128 = 0;
                for (i, w) in words.iter().enumerate().take(2) {
                    v |= (*w as u128) << (64 * i);
                }
                put(out, at, v, store);
            }
        }
        ConstKind::Float(b) => put(out, at, *b, store),
        ConstKind::DataArray(es) | ConstKind::DataVector(es) => {
            let elem = match m.types.get(ty) {
                Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
                _ => return Err(unsupported("a malformed data constant")),
            };
            let stride = if matches!(m.types.get(ty), Type::Vector { .. }) {
                dl.store_size(&m.types, elem).unwrap_or(0) as usize
            } else {
                dl.alloc_size(&m.types, elem).unwrap_or(0) as usize
            };
            let es_size = dl.store_size(&m.types, elem).unwrap_or(0) as usize;
            for (i, e) in es.iter().enumerate() {
                put(out, at + i * stride, *e as u128, es_size);
            }
        }
        ConstKind::Array(vs) | ConstKind::Vector(vs) => {
            let elem = match m.types.get(ty) {
                Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
                _ => return Err(unsupported("a malformed aggregate constant")),
            };
            if matches!(m.types.get(ty), Type::Vector { .. }) && m.types.int_bits(elem) == Some(1) {
                return Err(tracked::refuse("vector-shape", "<N x i1>", None, "a vector of i1 in memory"));
            }
            let stride = if matches!(m.types.get(ty), Type::Vector { .. }) {
                dl.store_size(&m.types, elem).unwrap_or(0) as usize
            } else {
                dl.alloc_size(&m.types, elem).unwrap_or(0) as usize
            };
            for (i, e) in vs.iter().enumerate() {
                const_bytes(m, dl, *e, out, at + i * stride)?;
            }
        }
        ConstKind::Struct(vs) => {
            let l = dl.struct_layout(&m.types, ty).ok_or_else(|| unsupported("an unsized struct"))?;
            for (i, e) in vs.iter().enumerate() {
                const_bytes(m, dl, *e, out, at + l.offsets[i] as usize)?;
            }
        }
        _ => return Err(unsupported("a pointer or constant expression")),
    }
    Ok(())
}
