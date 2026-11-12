//! The function-body emitter: the structured control flow of [`crate::cfg`] with each instruction lowered over the
//! value carriers of [`super::types`] and the memory model of [`crate::mem`].
//!
//! Every instruction result is bound once, in program order: a `let` when all its uses are inside the WGSL scope its
//! block's code opens (the same block, or a block the dominator tree places inline after it), otherwise a
//! function-scope `var` declared at the top and assigned at the definition. A phi is a `var` assigned on each incoming
//! edge, with all of an edge's incoming values read before any phi is assigned (the copies are parallel).
//!
//! Integer operations follow LLVM's semantics on the carriers: narrow results are re-masked, signed operations
//! sign-extend, and an operation whose result LLVM leaves poison (a shift by the width or more) or undefined (division
//! by zero) yields zero when its operands are constants, because WGSL rejects such constant expressions. Operations on
//! constants alone are folded here for the same reason.

use super::hooks::{CallSite, Effect, RuleCx, Rules};
use super::io::Interface;
use super::tracked;
use super::types::{Shape, Types, num_ty};
use crate::api::{HostRequirements, Location, MetalStage, Refusal, RefusalKind};
use crate::cfg::{Analysis, Behavior, Node, Structured};
use crate::ir::layout::DataLayout;
use crate::ir::{
    BinOp, CastOp, ConstExpr, ConstKind, FloatPredicate, Function, GlobalId, InstId, IntPredicate, Module, Op, Type,
    TypeId, Value,
};
use crate::mem::access::{Access, Code, Ptr, store_is_partial};
use crate::mem::{Memory, NULL, RegionId, RegionKind};
use crate::num::{self, prelude::Prelude};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

type R<T> = Result<T, Refusal>;

/// Static knowledge of a pointer value's offset.
#[derive(Clone, Copy, Debug, Default)]
struct PInfo {
    konst: Option<u64>,
    mod4: Option<u32>,
}

/// The emitter of one function.
pub struct FnEmit<'a> {
    pub m: &'a Module,
    pub fi: usize,
    f: &'a Function,
    dl: &'a DataLayout,
    pub types: Types,
    pub mem: Memory,
    a: &'a Analysis,
    rules: &'a Rules,
    target: num::Target<'a>,
    iface: &'a Interface,
    stage: MetalStage,
    block_of: Vec<usize>,
    hoisted: HashSet<InstId>,
    pinfo: HashMap<Value, PInfo>,
    lines: Vec<String>,
    indent: usize,
    next: u32,
    /// Function-scope declarations (hoisted values, phis, the jump flags, alloca arrays).
    pub decls: Vec<String>,
    pub prelude: Prelude,
    /// Library items of `lib.wgsl`.
    pub lib: BTreeSet<&'static str>,
    pub requirements: HostRequirements,
    pub enables: BTreeSet<String>,
    /// Module-scope items from rules, by key.
    pub module_items: BTreeMap<String, String>,
    /// Derivatives are lowered (the module disables the derivative uniformity diagnostic, see [`super`]).
    pub derivatives: bool,
    /// The jump targets whose flag the body sets or tests: each has its own `bool` variable, so that WGSL's
    /// uniformity analysis, which follows each variable's value through the control flow (WGSL 15.2.5), sees a jump
    /// taken in uniform control flow as uniform even where another jump of the function is not.
    flags: BTreeSet<u32>,
    /// The entry point's argument records.
    pub records: Vec<crate::air::Argument>,
    /// Bindings rules asked for: (name, WGSL type, resource, source).
    pub resources: Vec<(String, String, crate::api::BindingResource, crate::api::BindingSource)>,
    /// What the pipeline description binds to the stage.
    pub bindings: Option<crate::api::StageBindings>,
}

fn fmf(f: crate::ir::FastMath) -> num::FastMath {
    use crate::ir::FastMath as F;
    num::FastMath {
        nnan: f.0 & F::NNAN != 0,
        ninf: f.0 & F::NINF != 0,
        nsz: f.0 & F::NSZ != 0,
        arcp: f.0 & F::ARCP != 0,
        contract: f.0 & F::CONTRACT != 0,
        afn: f.0 & F::AFN != 0,
        reassoc: f.0 & F::REASSOC != 0,
    }
}

fn fpred(p: FloatPredicate) -> num::FPred {
    use FloatPredicate as P;
    use num::FPred as N;
    match p {
        P::False => N::False,
        P::Oeq => N::Oeq,
        P::Ogt => N::Ogt,
        P::Oge => N::Oge,
        P::Olt => N::Olt,
        P::Ole => N::Ole,
        P::One => N::One,
        P::Ord => N::Ord,
        P::Uno => N::Uno,
        P::Ueq => N::Ueq,
        P::Ugt => N::Ugt,
        P::Uge => N::Uge,
        P::Ult => N::Ult,
        P::Ule => N::Ule,
        P::Une => N::Une,
        P::True => N::True,
    }
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

fn sext(v: u64, bits: u32) -> i64 {
    if bits >= 64 { v as i64 } else { ((v << (64 - bits)) as i64) >> (64 - bits) }
}

/// LLVM's integer binary operation on constants of `bits` bits; `None` when the result is poison or undefined.
fn fold(op: BinOp, bits: u32, a: u64, b: u64) -> Option<u64> {
    let m = mask(bits);
    let (a, b) = (a & m, b & m);
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    Some(
        (match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::UDiv => a.checked_div(b)?,
            BinOp::URem => a.checked_rem(b)?,
            BinOp::SDiv => {
                if b == 0 || (sa == sext(1u64 << (bits - 1), bits) && sb == -1) {
                    return None;
                }
                (sa / sb) as u64
            }
            BinOp::SRem => {
                if b == 0 || (sa == sext(1u64 << (bits - 1), bits) && sb == -1) {
                    return None;
                }
                (sa % sb) as u64
            }
            BinOp::Shl => {
                if b >= bits as u64 {
                    return None;
                }
                a << b
            }
            BinOp::LShr => {
                if b >= bits as u64 {
                    return None;
                }
                a >> b
            }
            BinOp::AShr => {
                if b >= bits as u64 {
                    return None;
                }
                (sa >> b) as u64
            }
            BinOp::And => a & b,
            BinOp::Or => a | b,
            BinOp::Xor => a ^ b,
            _ => return None,
        }) & m,
    )
}

/// The literal of an integer constant in its carrier.
fn int_lit(bits: u32, v: u64) -> String {
    let v = v & mask(bits);
    match bits {
        1 => (if v != 0 { "true" } else { "false" }).into(),
        64 => format!("vec2<u32>({}u, {}u)", v as u32, (v >> 32) as u32),
        _ => format!("{}u", v),
    }
}

impl<'a> FnEmit<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: &'a Module,
        fi: usize,
        dl: &'a DataLayout,
        mem: Memory,
        a: &'a Analysis,
        rules: &'a Rules,
        target: num::Target<'a>,
        iface: &'a Interface,
        stage: MetalStage,
    ) -> FnEmit<'a> {
        let f = &m.functions[fi];
        let mut block_of = vec![usize::MAX; f.insts.len()];
        for (bi, b) in f.blocks.iter().enumerate() {
            for &i in &b.insts {
                block_of[i.0 as usize] = bi;
            }
        }
        FnEmit {
            m,
            fi,
            f,
            dl,
            types: Types::new(),
            mem,
            a,
            rules,
            target,
            iface,
            stage,
            block_of,
            hoisted: HashSet::new(),
            pinfo: HashMap::new(),
            lines: Vec::new(),
            indent: 1,
            next: 0,
            decls: Vec::new(),
            prelude: Prelude::new(),
            lib: BTreeSet::new(),
            requirements: HostRequirements::default(),
            enables: BTreeSet::new(),
            module_items: BTreeMap::new(),
            derivatives: false,
            flags: BTreeSet::new(),
            records: Vec::new(),
            resources: Vec::new(),
            bindings: None,
        }
    }

    fn loc(&self, i: Option<InstId>) -> Option<Location> {
        Some(Location {
            function: self.f.common.name.clone(),
            instruction: i.map(|i| crate::ir::print::print_instruction(self.m, self.fi, i)),
        })
    }

    fn at(&self, i: InstId, mut r: Refusal) -> Refusal {
        if r.location.is_none() {
            r.location = self.loc(Some(i));
        }
        r
    }

    fn need(&mut self, item: &str) -> R<()> {
        self.prelude.require(item).map_err(|e| {
            tracked::refuse(
                "unregistered-lowering",
                item.to_string(),
                None,
                format!("numerics library defect: {}", e.0),
            )
        })
    }

    // Analysis before emission ---------------------------------------------------------------------------------------

    /// Whether a use in block `u` sees a `let` bound in block `d`'s code.
    fn in_scope(&self, d: usize, u: usize) -> bool {
        if d == u {
            return true;
        }
        if !self.a.reachable(u) || !self.a.dominates(d, u) {
            return false;
        }
        let mut c = u;
        loop {
            let p = self.a.idom[c].unwrap();
            if p == d {
                return !self.a.merge[c];
            }
            c = p;
        }
    }

    /// Decide which values are function-scope variables, and mark the memory each access touches.
    pub fn prepare(&mut self) -> R<()> {
        let f = self.f;
        for (bi, b) in f.blocks.iter().enumerate() {
            if !self.a.reachable(bi) {
                continue;
            }
            for &i in &b.insts {
                let inst = f.inst(i);
                let uses: Vec<(Value, usize)> = match &inst.op {
                    Op::Phi { incoming, .. } => {
                        self.hoisted.insert(i);
                        incoming.iter().map(|(v, p)| (*v, p.0 as usize)).collect()
                    }
                    op => op.operands().into_iter().map(|v| (v, bi)).collect(),
                };
                for (v, ub) in uses {
                    if let Value::Inst(d) = v {
                        let db = self.block_of[d.0 as usize];
                        if !self.in_scope(db, ub) {
                            self.hoisted.insert(d);
                        }
                    }
                }
            }
        }
        // Memory effects: reads, writes, and the stores that make a shared region atomic.
        let mut partial: HashSet<RegionId> = HashSet::new();
        let mut atomic: HashSet<RegionId> = HashSet::new();
        let mut written: HashSet<RegionId> = HashSet::new();
        for (bi, b) in f.blocks.iter().enumerate() {
            if !self.a.reachable(bi) {
                continue;
            }
            for &i in &b.insts {
                let inst = f.inst(i);
                let mut effects: Vec<(Value, Effect)> = Vec::new();
                match &inst.op {
                    Op::Load { ptr, .. } => effects.push((*ptr, Effect::Read)),
                    Op::Store { value, ptr, align, .. } => {
                        let t = self.value_type(*value);
                        let bytes = self.dl.store_size(&self.m.types, t).unwrap_or(0);
                        if store_is_partial(&self.m.types, self.dl, t, if *align >= 4 { Some(0) } else { None }) {
                            effects.push((*ptr, Effect::Write { bytes, align: 1 }));
                        } else {
                            effects.push((*ptr, Effect::Write { bytes, align: *align }));
                        }
                    }
                    Op::Call { callee, args, .. } => {
                        let name = self.callee_name(*callee).unwrap_or_default();
                        let site = CallSite { name: &name, inst: i, args, ret: inst.ty };
                        if let Some(rule) = self.rules.find(&name) {
                            for (k, e) in rule.effects(&site) {
                                if let Some(v) = args.get(k) {
                                    effects.push((*v, e));
                                }
                            }
                        } else if name.starts_with("llvm.memcpy") || name.starts_with("llvm.memmove") {
                            let len = self.const_int(args[2]);
                            effects.push((
                                args[0],
                                Effect::Write {
                                    bytes: len.unwrap_or(1),
                                    align: if len.is_some() { self.param_align(inst, 0) } else { 1 },
                                },
                            ));
                            effects.push((args[1], Effect::Read));
                        } else if name.starts_with("llvm.memset") {
                            let len = self.const_int(args[2]);
                            effects.push((
                                args[0],
                                Effect::Write {
                                    bytes: len.unwrap_or(1),
                                    align: if len.is_some() { self.param_align(inst, 0) } else { 1 },
                                },
                            ));
                        } else if name.starts_with("air.") && is_numeric_family(tracked::family_of(&name)) {
                            // Numerics calls store through a pointer operand (sincos); other calls nobody lowers are
                            // refused when they are emitted.
                            for v in args.iter() {
                                if !matches!(v, Value::Metadata(_)) && self.m.types.is_ptr(self.value_type(*v)) {
                                    let t = inst.ty;
                                    let bytes = self.dl.store_size(&self.m.types, t).unwrap_or(4);
                                    let al = self.dl.abi_align(&self.m.types, t).unwrap_or(1);
                                    effects.push((*v, Effect::Write { bytes, align: al }));
                                }
                            }
                        }
                    }
                    Op::AtomicRmw { ptr, .. } | Op::CmpXchg { ptr, .. } => effects.push((*ptr, Effect::Atomic)),
                    _ => {}
                }
                for (v, e) in effects {
                    let regions = self.mem.regions(self.m, v).map_err(|r| self.at(i, r))?;
                    for r in regions {
                        if r == NULL {
                            continue;
                        }
                        match e {
                            Effect::Read => self.mem.regions[r].read = true,
                            Effect::Write { bytes, align } => {
                                written.insert(r);
                                if align < 4 || bytes % 4 != 0 {
                                    partial.insert(r);
                                }
                            }
                            Effect::Atomic => {
                                written.insert(r);
                                atomic.insert(r);
                            }
                        }
                    }
                }
            }
        }
        for r in 0..self.mem.regions.len() {
            let reg = &mut self.mem.regions[r];
            reg.written |= written.contains(&r);
            reg.atomic |= atomic.contains(&r) || (reg.shared && partial.contains(&r));
            if self.stage == MetalStage::Vertex
                && reg.written
                && matches!(reg.kind, RegionKind::Buffer { space: 1, .. })
            {
                return Err(tracked::refuse_row(
                    "vertex-stage-writes",
                    RefusalKind::NoExactLowering,
                    format!("buffer {}", reg.name),
                    self.loc(None),
                    "a vertex function writes a device buffer",
                    "WebGPU 8.1.1 bind group layout creation",
                ));
            }
            if matches!(reg.kind, RegionKind::ConstantGlobal { .. } | RegionKind::Buffer { space: 2, .. })
                && reg.written
            {
                // Constant address space memory is read-only in MSL; a store is malformed input.
                return Err(tracked::refuse(
                    "constant-store",
                    reg.name.clone(),
                    self.loc(None),
                    "a store into constant address space memory",
                ));
            }
        }
        // Slots sharing one WGSL variable (alias groups) share its atomicity.
        let mut by_name: HashMap<String, bool> = HashMap::new();
        for r in &self.mem.regions {
            if matches!(r.kind, RegionKind::Buffer { .. }) {
                *by_name.entry(r.name.clone()).or_default() |= r.atomic;
            }
        }
        for r in self.mem.regions.iter_mut() {
            if matches!(r.kind, RegionKind::Buffer { .. }) && by_name[&r.name] {
                r.atomic = true;
            }
        }
        if self.mem.regions.iter().any(|r| r.atomic && !r.shared && !matches!(r.kind, RegionKind::Buffer { .. })) {
            // Atomic access to private memory is lowered as plain access (no other invocation sees it).
            for r in self.mem.regions.iter_mut() {
                if !r.shared && !matches!(r.kind, RegionKind::Buffer { .. }) {
                    r.atomic = false;
                }
            }
        }
        Ok(())
    }

    fn param_align(&self, inst: &crate::ir::Instruction, k: usize) -> u64 {
        if let Op::Call { attrs, .. } = &inst.op
            && let Some(crate::ir::Attribute::Int(_, a)) =
                self.m.attrs(attrs.param(k)).get(crate::ir::AttrKind::Alignment)
        {
            return *a;
        }
        1
    }

    // Values -----------------------------------------------------------------------------------------------------------

    fn value_type(&self, v: Value) -> TypeId {
        match v {
            Value::Arg(i) => self.f.args[i as usize].ty,
            Value::Inst(i) => self.f.inst(i).ty,
            Value::Const(c) => self.m.constant(c).ty,
            Value::Global(g) => {
                let a = self.m.global(g).addrspace;
                self.ptr_type(a)
            }
            Value::InlineAsm(_) => self.ptr_type(0),
            Value::Metadata(_) => self.f.args.first().map(|a| a.ty).unwrap_or(TypeId(0)),
        }
    }

    /// The pointer type of an address space (present in the table: a global of it is used).
    fn ptr_type(&self, a: u32) -> TypeId {
        self.m.types.iter().find(|(_, t)| matches!(t, Type::Ptr(x) if *x == a)).map(|(id, _)| id).unwrap_or(TypeId(0))
    }

    fn const_int(&self, v: Value) -> Option<u64> {
        if let Value::Const(c) = v
            && let ConstKind::Int(w) = &self.m.constant(c).kind
        {
            return Some(w[0]);
        }
        None
    }

    fn callee_name(&self, callee: Value) -> Option<String> {
        match callee {
            Value::Global(GlobalId::Function(fi)) => Some(self.m.functions[fi as usize].common.name.clone()),
            _ => None,
        }
    }

    /// The regions of a pointer value and its carrier: the offset (`u32`) for one region, `vec2<u32>(offset, tag)`
    /// for several.
    fn ptr_regions(&self, v: Value) -> R<Vec<RegionId>> {
        self.mem.regions(self.m, v)
    }

    /// A constant pointer: its region and offset.
    fn const_ptr(&self, v: Value) -> Option<(RegionId, u64)> {
        match v {
            Value::Global(GlobalId::Var(g)) => Some((self.mem.global_region(g)?, 0)),
            Value::Const(c) => match &self.m.constant(c).kind {
                ConstKind::NullPtr | ConstKind::Zero | ConstKind::Undef | ConstKind::Poison => Some((NULL, 0)),
                ConstKind::Expr(ConstExpr::Cast { op: CastOp::BitCast | CastOp::AddrSpaceCast, value }) => {
                    self.const_ptr(*value)
                }
                ConstKind::Expr(ConstExpr::Gep { source, ptr, indices, .. }) => {
                    let (r, base) = self.const_ptr(*ptr)?;
                    let mut off = base;
                    let mut t = *source;
                    for (k, idx) in indices.iter().enumerate() {
                        let iv = sext(self.const_int(*idx)?, self.m.types.int_bits(self.value_type(*idx))?);
                        if k == 0 {
                            off = off.wrapping_add((iv * self.dl.alloc_size(&self.m.types, t)? as i64) as u64);
                            continue;
                        }
                        match self.m.types.get(t) {
                            Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                                let l = self.dl.struct_layout(&self.m.types, t)?;
                                off = off.wrapping_add(l.offsets[iv as usize]);
                                t = elems[iv as usize];
                            }
                            Type::Array { elem, .. } => {
                                off = off.wrapping_add((iv * self.dl.alloc_size(&self.m.types, *elem)? as i64) as u64);
                                t = *elem;
                            }
                            Type::Vector { elem, .. } => {
                                off = off
                                    .wrapping_add((iv * (self.dl.size_bits(&self.m.types, *elem)? / 8) as i64) as u64);
                                t = *elem;
                            }
                            _ => return None,
                        }
                    }
                    Some((r, off & 0xffff_ffff))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The WGSL expression of an operand in its carrier.
    pub fn operand(&mut self, v: Value) -> R<String> {
        match v {
            Value::Inst(i) => Ok(format!("v{}", i.0)),
            Value::Arg(a) => Ok(format!("aw_a{}", a)),
            Value::Global(GlobalId::Var(_)) => Ok("0u".into()),
            Value::Const(c) => self.constant(c),
            _ => Err(tracked::refuse("llvm-instruction", format!("{:?}", v), None, "an operand without a WGSL value")),
        }
    }

    fn constant(&mut self, c: crate::ir::ConstId) -> R<String> {
        let k = self.m.constant(c).clone();
        let tt = &self.m.types;
        if tt.is_ptr(k.ty) {
            let (r, off) = self.const_ptr(Value::Const(c)).ok_or_else(|| {
                tracked::refuse(
                    "unbounded-pointer-provenance",
                    "constant pointer",
                    None,
                    "a constant pointer outside every region",
                )
            })?;
            let _ = r;
            return Ok(format!("{}u", off as u32));
        }
        Ok(match &k.kind {
            ConstKind::Int(w) => int_lit(tt.int_bits(k.ty).unwrap_or(32), w[0]),
            ConstKind::Float(b) => format!("{}u", *b as u32),
            ConstKind::Zero | ConstKind::Undef | ConstKind::Poison => self.types.zero(tt, k.ty)?,
            ConstKind::DataVector(es) | ConstKind::DataArray(es) => {
                let elem = match tt.get(k.ty) {
                    Type::Vector { elem, .. } | Type::Array { elem, .. } => *elem,
                    _ => unreachable!(),
                };
                let bits = tt.int_bits(elem);
                let parts: Vec<String> = es
                    .iter()
                    .map(|e| match bits {
                        Some(b) => int_lit(b, *e),
                        None => format!("{}u", *e as u32),
                    })
                    .collect();
                format!("{}({})", self.types.carrier(tt, k.ty)?, parts.join(", "))
            }
            ConstKind::Vector(vs) | ConstKind::Array(vs) | ConstKind::Struct(vs) => {
                let mut parts = Vec::new();
                for v in vs {
                    parts.push(self.operand(*v)?);
                }
                format!("{}({})", self.types.carrier(tt, k.ty)?, parts.join(", "))
            }
            ConstKind::Expr(ConstExpr::Cast { op: CastOp::PtrToInt, value }) => {
                let (r, off) = self.const_ptr(*value).ok_or_else(|| {
                    tracked::refuse(
                        "unbounded-pointer-provenance",
                        "ptrtoint",
                        None,
                        "a constant pointer outside every region",
                    )
                })?;
                let addr = ((self.mem.regions[r].tag as u64) << 32) | off;
                int_lit(tt.int_bits(k.ty).unwrap_or(64), addr)
            }
            _ => {
                return Err(tracked::refuse(
                    "llvm-instruction",
                    "constant expression",
                    None,
                    format!("a constant expression {:?}", k.kind),
                ));
            }
        })
    }

    /// The carrier expression of a pointer value as a pointer into `target` regions.
    fn ptr_as(&mut self, v: Value, target: &[RegionId]) -> R<String> {
        let own = self.ptr_regions(v)?;
        let e = self.operand(v)?;
        if target.len() == 1 || own.len() > 1 {
            return Ok(e);
        }
        Ok(format!("vec2<u32>({}, {}u)", e, self.mem.regions[own[0]].tag))
    }

    /// A pointer operand for access.
    pub fn ptr(&mut self, v: Value) -> R<Ptr> {
        let regions = self.ptr_regions(v)?;
        let e = self.operand(v)?;
        if regions.len() > 1 {
            return Ok(Ptr {
                regions,
                off: format!("{}.x", e),
                konst: None,
                tag: Some(format!("{}.y", e)),
                mod4: None,
            });
        }
        let info = match v {
            Value::Const(_) | Value::Global(_) => {
                let (_, off) = self.const_ptr(v).unwrap_or((NULL, 0));
                PInfo { konst: Some(off), mod4: Some((off % 4) as u32) }
            }
            Value::Arg(_) => PInfo::default(),
            _ => self.pinfo.get(&v).copied().unwrap_or_default(),
        };
        let off = match info.konst {
            Some(k) => format!("{}u", k as u32),
            None => e,
        };
        Ok(Ptr { regions, off, konst: info.konst, tag: None, mod4: info.mod4 })
    }

    // Output -----------------------------------------------------------------------------------------------------------

    fn push(&mut self, s: String) {
        self.lines.push(format!("{}{}", "    ".repeat(self.indent), s));
    }

    fn tmp(&mut self) -> String {
        self.next += 1;
        format!("t{}", self.next)
    }

    fn carrier_of_inst(&mut self, i: InstId) -> R<String> {
        let t = self.f.inst(i).ty;
        if self.m.types.is_ptr(t) {
            Ok(if self.ptr_regions(Value::Inst(i))?.len() > 1 { "vec2<u32>".to_string() } else { "u32".to_string() })
        } else {
            self.types.carrier(&self.m.types, t)
        }
    }

    /// Declare the function-scope variables of phis (assigned on edges, so declared before any code).
    pub fn declare_phis(&mut self) -> R<()> {
        for (bi, b) in self.f.blocks.iter().enumerate() {
            if !self.a.reachable(bi) {
                continue;
            }
            for &i in &b.insts {
                if matches!(self.f.inst(i).op, Op::Phi { .. }) {
                    let c = self.carrier_of_inst(i).map_err(|r| self.at(i, r))?;
                    self.decls.push(format!("var v{}: {};", i.0, c));
                }
            }
        }
        Ok(())
    }

    /// Bind the result of instruction `i`.
    fn def(&mut self, i: InstId, expr: String) -> R<()> {
        let carrier = self.carrier_of_inst(i)?;
        if self.hoisted.contains(&i) {
            self.decls.push(format!("var v{}: {};", i.0, carrier));
            self.push(format!("v{} = {};", i.0, expr));
        } else {
            self.push(format!("let v{}: {} = {};", i.0, carrier, expr));
        }
        Ok(())
    }

    fn access(&mut self) -> (Access<'_>, &mut Vec<String>, &mut usize, &mut u32) {
        (
            Access { tt: &self.m.types, dl: self.dl, types: &mut self.types, mem: &self.mem },
            &mut self.lines,
            &mut self.indent,
            &mut self.next,
        )
    }

    // Structured statements --------------------------------------------------------------------------------------------

    /// Emit a structured statement list.
    pub fn nodes(&mut self, list: &[Node]) -> R<()> {
        for n in list {
            self.node(n)?;
        }
        Ok(())
    }

    fn open(&mut self, s: String) {
        self.push(s);
        self.indent += 1;
    }

    fn close(&mut self) {
        self.indent -= 1;
        self.push("}".into());
    }

    fn node(&mut self, n: &Node) -> R<()> {
        match n {
            Node::Code(b) => {
                let b = *b;
                let insts = self.f.blocks[b].insts.clone();
                for &i in &insts[..insts.len().saturating_sub(1)] {
                    self.inst(i).map_err(|r| self.at(i, r))?;
                }
            }
            Node::Block { body, .. } => {
                self.open("switch 0i {".into());
                self.open("default {".into());
                self.nodes(body)?;
                self.close();
                self.close();
            }
            Node::Loop { body, .. } => {
                self.open("loop {".into());
                self.nodes(body)?;
                self.close();
            }
            Node::If { block, then_, else_ } => {
                let term = *self.f.blocks[*block].insts.last().unwrap();
                let Op::CondBr { cond, .. } = &self.f.inst(term).op else { unreachable!() };
                let c = self.operand(*cond)?;
                self.open(format!("if {} {{", c));
                self.nodes(then_)?;
                self.indent -= 1;
                self.push("} else {".into());
                self.indent += 1;
                self.nodes(else_)?;
                self.close();
            }
            Node::Switch { block, arms, default } => {
                let term = *self.f.blocks[*block].insts.last().unwrap();
                let Op::Switch { cond, cases, .. } = self.f.inst(term).op.clone() else { unreachable!() };
                let ct = self.value_type(cond);
                let bits = self.m.types.int_bits(ct).unwrap_or(32);
                let sel = self.tmp();
                let c = self.operand(cond)?;
                self.push(format!("let {} = {};", sel, c));
                if arms.is_empty() {
                    return self.nodes(default);
                }
                for (k, (ks, body)) in arms.iter().enumerate() {
                    let tests: Vec<String> = ks
                        .iter()
                        .map(|&ci| {
                            let v = match &self.m.constant(cases[ci].0).kind {
                                ConstKind::Int(w) => w[0],
                                _ => 0,
                            };
                            if bits == 64 {
                                format!("all({} == {})", sel, int_lit(64, v))
                            } else {
                                format!("{} == {}", sel, int_lit(bits, v))
                            }
                        })
                        .collect();
                    let head = format!("if {} {{", tests.join(" || "));
                    if k == 0 {
                        self.open(head);
                    } else {
                        self.indent -= 1;
                        self.push(format!("}} else {}", head));
                        self.indent += 1;
                    }
                    self.nodes(body)?;
                }
                self.indent -= 1;
                self.push("} else {".into());
                self.indent += 1;
                self.nodes(default)?;
                self.close();
            }
            Node::Edge { from, to } => self.edge(*from, *to)?,
            Node::Return(b) => {
                let term = *self.f.blocks[*b].insts.last().unwrap();
                let Op::Ret { value } = self.f.inst(term).op.clone() else { unreachable!() };
                let s = self.ret(value)?;
                self.push(s);
            }
            Node::Unreachable(_) => {
                // Reaching `unreachable` is undefined behavior: leave the function.
                let s = match &self.iface.ret_type {
                    Some(t) => format!("return {}();", t),
                    None => "return;".to_string(),
                };
                self.push(s);
            }
            Node::Break => self.push("break;".into()),
            Node::Continue => self.push("continue;".into()),
            Node::SetFlag(k) => {
                self.flags.insert(*k);
                self.push(format!("aw_flow{} = true;", k));
            }
            Node::Dispatch(arms) => {
                for (k, (flag, body)) in arms.iter().enumerate() {
                    self.flags.insert(*flag);
                    let head = format!("if aw_flow{} {{", flag);
                    if k == 0 {
                        self.open(head);
                    } else {
                        self.indent -= 1;
                        self.push(format!("}} else {}", head));
                        self.indent += 1;
                    }
                    self.push(format!("aw_flow{} = false;", flag));
                    self.nodes(body)?;
                }
                self.close();
            }
            Node::Goto(_) => unreachable!("unresolved jump"),
        }
        Ok(())
    }

    /// `return` with the stage's outputs built from the IR return value.
    pub fn ret(&mut self, value: Option<Value>) -> R<String> {
        let Some(t) = self.iface.ret_type.clone() else {
            return Ok("return;".into());
        };
        let v = match value {
            Some(v) => self.operand(v)?,
            None => return Ok(format!("return {}();", t)),
        };
        let r = self.tmp();
        self.push(format!("let {} = {};", r, v));
        let fields: Vec<String> = self.iface.ret_fields.iter().flatten().map(|c| c.replace("{}", &r)).collect();
        Ok(format!("return {}({});", t, fields.join(", ")))
    }

    /// The phi copies of edge `from -> to`, in parallel.
    fn edge(&mut self, from: usize, to: usize) -> R<()> {
        let mut copies = Vec::new();
        for &i in &self.f.blocks[to].insts {
            let Op::Phi { incoming, .. } = &self.f.inst(i).op else {
                break;
            };
            let v = incoming.iter().find(|(_, b)| b.0 as usize == from).map(|(v, _)| *v).ok_or_else(|| {
                tracked::refuse(
                    "llvm-instruction",
                    "phi",
                    self.loc(Some(i)),
                    "a phi without an incoming value for an edge",
                )
            })?;
            copies.push((i, v));
        }
        let mut tmps = Vec::new();
        for (i, v) in &copies {
            let e = if self.m.types.is_ptr(self.f.inst(*i).ty) {
                let target = self.ptr_regions(Value::Inst(*i))?;
                self.ptr_as(*v, &target)?
            } else {
                self.operand(*v)?
            };
            let t = self.tmp();
            self.push(format!("let {} = {};", t, e));
            tmps.push(t);
        }
        for ((i, _), t) in copies.iter().zip(tmps) {
            self.push(format!("v{} = {};", i.0, t));
        }
        Ok(())
    }

    // Instructions -----------------------------------------------------------------------------------------------------

    fn lanes(&self, t: TypeId) -> (u32, TypeId, bool) {
        match self.m.types.get(t) {
            Type::Vector { len, elem, .. } => (*len, *elem, true),
            _ => (1, t, false),
        }
    }

    fn lane(e: &str, vector: bool, i: u32) -> String {
        if vector { format!("{}[{}]", e, i) } else { e.to_string() }
    }

    /// Build a value of type `t` from per-lane expressions.
    fn build(&mut self, t: TypeId, lanes: Vec<String>) -> R<String> {
        if lanes.len() == 1 && !self.m.types.is_vector(t) {
            return Ok(lanes.into_iter().next().unwrap());
        }
        Ok(format!("{}({})", self.types.carrier(&self.m.types, t)?, lanes.join(", ")))
    }

    fn inst(&mut self, i: InstId) -> R<()> {
        let inst = self.f.inst(i).clone();
        let t = inst.ty;
        match &inst.op {
            Op::Phi { .. } => Ok(()),
            Op::Binary { op, lhs, rhs, flags: _, fmf: fm } => {
                if op.is_float() {
                    let ni = match op {
                        BinOp::FAdd => num::Inst::FAdd,
                        BinOp::FSub => num::Inst::FSub,
                        BinOp::FMul => num::Inst::FMul,
                        BinOp::FDiv => num::Inst::FDiv,
                        _ => num::Inst::FRem,
                    };
                    return self.numeric(i, num::Op::Inst(ni), &[*lhs, *rhs], Some(t), *fm, op.name());
                }
                let e = self.int_binary(*op, *lhs, *rhs, t)?;
                self.def(i, e)
            }
            Op::FNeg { value, fmf: fm } => {
                self.numeric(i, num::Op::Inst(num::Inst::FNeg), &[*value], Some(t), *fm, "fneg")
            }
            Op::FCmp { pred, lhs, rhs, fmf: fm } => {
                self.numeric(i, num::Op::Inst(num::Inst::FCmp(fpred(*pred))), &[*lhs, *rhs], Some(t), *fm, "fcmp")
            }
            Op::ICmp { pred, lhs, rhs, .. } => {
                let e = self.icmp(i, *pred, *lhs, *rhs)?;
                self.def(i, e)
            }
            Op::Cast { op, value, fmf: fm, .. } => self.cast(i, *op, *value, t, *fm),
            Op::Select { cond, t: tv, f: fv, .. } => {
                let e = self.select(i, *cond, *tv, *fv, t)?;
                self.def(i, e)
            }
            Op::Freeze { value } => {
                let e = if self.m.types.is_ptr(t) {
                    let target = self.ptr_regions(Value::Inst(i))?;
                    self.ptr_as(*value, &target)?
                } else {
                    self.operand(*value)?
                };
                if let Some(p) = self.pinfo.get(value).copied() {
                    self.pinfo.insert(Value::Inst(i), p);
                }
                self.def(i, e)
            }
            Op::Alloca { .. } => {
                self.pinfo.insert(Value::Inst(i), PInfo { konst: Some(0), mod4: Some(0) });
                self.def(i, "0u".into())
            }
            Op::Gep { source, ptr, indices, .. } => self.gep(i, *source, *ptr, indices),
            Op::Load { ptr, align, atomic, .. } => {
                if atomic.is_some() {
                    return Err(tracked::refuse("llvm-instruction", "load atomic", None, "an LLVM atomic load"));
                }
                if self.m.types.is_ptr(t) {
                    return Err(tracked::refuse("pointer-in-memory", "load ptr", None, "a pointer loaded from memory"));
                }
                let p = self.ptr(*ptr)?;
                let e = self.with_code(|a, c| a.load(c, &p, t, *align))?;
                self.def(i, e)
            }
            Op::Store { value, ptr, align, atomic, .. } => {
                if atomic.is_some() {
                    return Err(tracked::refuse("llvm-instruction", "store atomic", None, "an LLVM atomic store"));
                }
                let vt = self.value_type(*value);
                if self.m.types.is_ptr(vt) {
                    return Err(tracked::refuse("pointer-in-memory", "store ptr", None, "a pointer stored to memory"));
                }
                let p = self.ptr(*ptr)?;
                let v = self.operand(*value)?;
                self.with_code(|a, c| a.store(c, &p, vt, *align, &v))
            }
            Op::ExtractElement { vec, idx } => {
                let (n, _, _) = self.lanes(self.value_type(*vec));
                let v = self.operand(*vec)?;
                let e = match self.const_int(*idx) {
                    Some(k) if k >= n as u64 => self.types.zero(&self.m.types, t)?,
                    Some(k) => format!("{}[{}]", v, k),
                    None => format!("{}[{}]", v, self.index_u32(*idx)?),
                };
                self.def(i, e)
            }
            Op::InsertElement { vec, elt, idx } => {
                let (n, _, _) = self.lanes(t);
                let v = self.operand(*vec)?;
                let x = self.operand(*elt)?;
                let e = match self.const_int(*idx) {
                    Some(k) if k >= n as u64 => self.types.zero(&self.m.types, t)?,
                    Some(k) => {
                        let tmp = self.tmp();
                        self.push(format!("var {} = {};", tmp, v));
                        self.push(format!("{}[{}] = {};", tmp, k, x));
                        tmp
                    }
                    None => {
                        let ix = self.index_u32(*idx)?;
                        let tmp = self.tmp();
                        self.push(format!("var {} = {};", tmp, v));
                        self.push(format!("{}[{}] = {};", tmp, ix, x));
                        tmp
                    }
                };
                self.def(i, e)
            }
            Op::ShuffleVector { a, b, mask } => {
                let (n, _, _) = self.lanes(self.value_type(*a));
                let (_, el, _) = self.lanes(t);
                let ea = self.operand(*a)?;
                let eb = self.operand(*b)?;
                let zero = self.types.zero(&self.m.types, el)?;
                let lanes: Vec<String> = mask
                    .iter()
                    .map(|&k| {
                        if k < 0 {
                            zero.clone()
                        } else if (k as u32) < n {
                            format!("{}[{}]", ea, k)
                        } else {
                            format!("{}[{}]", eb, k as u32 - n)
                        }
                    })
                    .collect();
                let e = self.build(t, lanes)?;
                self.def(i, e)
            }
            Op::ExtractValue { agg, indices } => {
                let path = self.agg_path(self.value_type(*agg), indices)?;
                let e = format!("{}{}", self.operand(*agg)?, path);
                self.def(i, e)
            }
            Op::InsertValue { agg, value, indices } => {
                let path = self.agg_path(t, indices)?;
                let a = self.operand(*agg)?;
                let v = self.operand(*value)?;
                let tmp = self.tmp();
                self.push(format!("var {} = {};", tmp, a));
                self.push(format!("{}{} = {};", tmp, path, v));
                self.def(i, tmp)
            }
            Op::Call { callee, args, fmf: fm, .. } => self.call(i, *callee, args, *fm),
            Op::AtomicRmw { .. } | Op::CmpXchg { .. } | Op::Fence { .. } => Err(tracked::refuse(
                "llvm-instruction",
                op_name(&inst.op),
                None,
                "LLVM atomic instructions and fences (AIR uses air.atomic.* and air.*barrier)",
            )),
            Op::VaArg { .. } | Op::IndirectBr { .. } => {
                Err(tracked::refuse("llvm-instruction", op_name(&inst.op), None, "not produced for MSL"))
            }
            Op::Ret { .. } | Op::Br { .. } | Op::CondBr { .. } | Op::Switch { .. } | Op::Unreachable => Ok(()),
        }
    }

    fn agg_path(&self, mut t: TypeId, indices: &[u32]) -> R<String> {
        let mut p = String::new();
        for &k in indices {
            match self.m.types.get(t) {
                Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                    p.push_str(&format!(".m{}", k));
                    t = elems[k as usize];
                }
                Type::Array { elem, .. } => {
                    p.push_str(&format!("[{}]", k));
                    t = *elem;
                }
                _ => {
                    return Err(tracked::refuse(
                        "llvm-instruction",
                        "extractvalue",
                        None,
                        "a malformed aggregate index",
                    ));
                }
            }
        }
        Ok(p)
    }

    /// An integer operand as a u32 index (sign-extended as GEP indices are, then taken modulo 2^32).
    fn index_u32(&mut self, v: Value) -> R<String> {
        let t = self.value_type(v);
        let bits = self
            .m
            .types
            .int_bits(t)
            .ok_or_else(|| tracked::refuse("pointer-aggregate", "vector index", None, "a vector of indices"))?;
        if let Some(k) = self.const_int(v) {
            return Ok(format!("{}u", sext(k, bits) as u64 as u32));
        }
        let e = self.operand(v)?;
        Ok(match bits {
            64 => format!("{}.x", e),
            32 => e,
            1 => format!("select(0u, 0xffffffffu, {})", e),
            b => {
                self.need("nm_sext")?;
                format!("bitcast<u32>(nm_sext({}, {}u))", e, b)
            }
        })
    }

    fn with_code<T>(&mut self, f: impl FnOnce(&mut Access<'_>, &mut dyn Code) -> R<T>) -> R<T> {
        let (mut a, lines, indent, next) = self.access();
        let mut sink = Sink { lines, indent, next };
        f(&mut a, &mut sink)
    }

    fn gep(&mut self, i: InstId, source: TypeId, ptr: Value, indices: &[Value]) -> R<()> {
        let regions = self.ptr_regions(Value::Inst(i))?;
        let base = self.ptr(ptr)?;
        let mut konst = base.konst;
        let mut mod4 = base.mod4.or(base.konst.map(|k| (k % 4) as u32));
        let mut c: u64 = 0;
        let mut terms: Vec<String> = Vec::new();
        let mut t = source;
        for (k, idx) in indices.iter().enumerate() {
            let (stride, next_t, is_struct) = if k == 0 {
                (self.dl.alloc_size(&self.m.types, t).unwrap_or(0), t, false)
            } else {
                match self.m.types.get(t).clone() {
                    Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                        let l = self.dl.struct_layout(&self.m.types, t).unwrap();
                        let f = self.const_int(*idx).unwrap_or(0) as usize;
                        c = c.wrapping_add(l.offsets[f]);
                        t = elems[f];
                        (0, t, true)
                    }
                    Type::Array { elem, .. } => (self.dl.alloc_size(&self.m.types, elem).unwrap_or(0), elem, false),
                    Type::Vector { elem, .. } => (self.dl.size_bits(&self.m.types, elem).unwrap_or(0) / 8, elem, false),
                    _ => return Err(tracked::refuse("llvm-instruction", "getelementptr", None, "a malformed GEP")),
                }
            };
            if is_struct {
                continue;
            }
            t = next_t;
            if let Some(v) = self.const_int(*idx) {
                let bits = self.m.types.int_bits(self.value_type(*idx)).unwrap_or(64);
                c = c.wrapping_add((sext(v, bits) as u64).wrapping_mul(stride));
            } else {
                let e = self.index_u32(*idx)?;
                terms.push(if stride == 1 { e } else { format!("{} * {}u", e, stride as u32) });
                konst = None;
                if stride % 4 != 0 {
                    mod4 = None;
                }
            }
        }
        let c32 = c as u32;
        konst = konst.map(|k| (k.wrapping_add(c)) & 0xffff_ffff);
        mod4 = mod4.map(|m| ((m as u64 + c) % 4) as u32);
        let e = if let Some(k) = konst {
            format!("{}u", k as u32)
        } else {
            let mut parts = vec![base.off.clone()];
            parts.extend(terms);
            if c32 != 0 {
                parts.push(format!("{}u", c32));
            }
            format!("({})", parts.join(" + "))
        };
        let e =
            if regions.len() > 1 { format!("vec2<u32>({}, {})", e, base.tag.clone().unwrap_or_default()) } else { e };
        self.pinfo.insert(Value::Inst(i), PInfo { konst, mod4 });
        self.def(i, e)
    }

    // Integer operations -------------------------------------------------------------------------------------------

    fn int_binary(&mut self, op: BinOp, lhs: Value, rhs: Value, t: TypeId) -> R<String> {
        let (n, el, vector) = self.lanes(t);
        let bits =
            self.m.types.int_bits(el).ok_or_else(|| tracked::refuse("integer-width", "?", None, "not an integer"))?;
        if !matches!(bits, 1 | 8 | 16 | 32 | 64) {
            return Err(tracked::refuse("integer-width", format!("i{}", bits), None, "an integer operation"));
        }
        let ea = self.operand(lhs)?;
        let eb = self.operand(rhs)?;
        let mut lanes = Vec::new();
        for k in 0..n {
            // Constant lanes are folded, and a constant shift amount or divisor is checked, per lane: WGSL rejects
            // the constant expressions LLVM leaves poison or undefined.
            let (ca, cb) = (self.const_lane(lhs, k), self.const_lane(rhs, k));
            if let (Some(a), Some(b)) = (ca, cb) {
                lanes.push(int_lit(bits, fold(op, bits, a, b).unwrap_or(0)));
                continue;
            }
            let a = Self::lane(&ea, vector, k);
            let b = Self::lane(&eb, vector, k);
            lanes.push(self.int_lane(op, bits, &a, &b, cb)?);
        }
        self.build(t, lanes)
    }

    /// Lane `k` of an integer constant (scalar or vector); undef and poison lanes are 0.
    fn const_lane(&self, v: Value, k: u32) -> Option<u64> {
        let Value::Const(c) = v else {
            return None;
        };
        match &self.m.constant(c).kind {
            ConstKind::Int(w) => Some(w[0]),
            ConstKind::Zero | ConstKind::Undef | ConstKind::Poison => Some(0),
            ConstKind::DataVector(es) => es.get(k as usize).copied(),
            ConstKind::Vector(vs) => vs.get(k as usize).and_then(|e| self.const_lane(*e, 0)),
            _ => None,
        }
    }

    fn int_lane(&mut self, op: BinOp, bits: u32, a: &str, b: &str, cb: Option<u64>) -> R<String> {
        use BinOp::*;
        // A constant shift amount at or past the width, or a constant zero divisor: poison or undefined (and a
        // WGSL shader-creation error as a constant expression).
        if let Some(k) = cb {
            let k = k & mask(bits);
            if matches!(op, Shl | LShr | AShr) && k >= bits as u64 {
                return Ok(int_lit(bits, 0));
            }
            if matches!(op, UDiv | SDiv | URem | SRem) && k == 0 {
                return Ok(int_lit(bits, 0));
            }
        }
        if bits == 1 {
            return Ok(match op {
                And | Mul => format!("({} & {})", a, b),
                Or => format!("({} | {})", a, b),
                Xor | Add | Sub => format!("({} != {})", a, b),
                // i1 division and remainder: the divisor must be 1 (true); shifts by 1 or more are poison.
                UDiv | SDiv | Shl | LShr | AShr => a.to_string(),
                URem | SRem => "false".into(),
                _ => unreachable!(),
            });
        }
        if bits == 64 {
            let call = |s: &mut Self, f: &str, args: String| -> R<String> {
                s.need(f)?;
                Ok(format!("{}({})", f, args))
            };
            return match op {
                Add => call(self, "nm_add64", format!("{}, {}", a, b)),
                Sub => call(self, "nm_sub64", format!("{}, {}", a, b)),
                Mul => call(self, "nm_i64_mul", format!("{}, {}", a, b)),
                And => Ok(format!("({} & {})", a, b)),
                Or => Ok(format!("({} | {})", a, b)),
                Xor => Ok(format!("({} ^ {})", a, b)),
                Shl => call(self, "nm_shl64", format!("{}, {}.x & 63u", a, b)),
                LShr => call(self, "nm_shr64", format!("{}, {}.x & 63u", a, b)),
                AShr => {
                    self.lib.insert("aw_sar64");
                    Ok(format!("aw_sar64({}, {}.x & 63u)", a, b))
                }
                UDiv | URem => {
                    self.lib.insert("aw_udivrem64");
                    let sw = if op == UDiv { "xy" } else { "zw" };
                    Ok(format!("aw_udivrem64({}, {}).{}", a, b, sw))
                }
                SDiv | SRem => {
                    self.lib.insert("aw_sdivrem64");
                    let sw = if op == SDiv { "xy" } else { "zw" };
                    Ok(format!("aw_sdivrem64({}, {}).{}", a, b, sw))
                }
                _ => unreachable!(),
            };
        }
        let m = mask(bits);
        let wrap = |e: String| if bits == 32 { e } else { format!("({} & {}u)", e, m) };
        let sx = |s: &mut Self, x: &str| -> R<String> {
            if bits == 32 {
                Ok(format!("bitcast<i32>({})", x))
            } else {
                s.need("nm_sext")?;
                Ok(format!("nm_sext({}, {}u)", x, bits))
            }
        };
        Ok(match op {
            Add => wrap(format!("({} + {})", a, b)),
            Sub => wrap(format!("({} - {})", a, b)),
            Mul => wrap(format!("({} * {})", a, b)),
            UDiv => format!("({} / {})", a, b),
            URem => format!("({} % {})", a, b),
            SDiv => {
                let (x, y) = (sx(self, a)?, sx(self, b)?);
                wrap(format!("bitcast<u32>({} / {})", x, y))
            }
            SRem => {
                let (x, y) = (sx(self, a)?, sx(self, b)?);
                wrap(format!("bitcast<u32>({} % {})", x, y))
            }
            Shl => wrap(format!("({} << {})", a, b)),
            LShr => format!("({} >> {})", a, b),
            AShr => {
                let x = sx(self, a)?;
                wrap(format!("bitcast<u32>({} >> {})", x, b))
            }
            And => format!("({} & {})", a, b),
            Or => format!("({} | {})", a, b),
            Xor => format!("({} ^ {})", a, b),
            _ => unreachable!(),
        })
    }

    fn icmp(&mut self, inst: InstId, pred: IntPredicate, lhs: Value, rhs: Value) -> R<String> {
        let lt = self.value_type(lhs);
        if self.m.types.is_ptr(lt) {
            return self.ptr_cmp(pred, lhs, rhs);
        }
        let (n, el, vector) = self.lanes(lt);
        let bits = self.m.types.int_bits(el).unwrap_or(32);
        if !vector && let (Some(a), Some(b)) = (self.const_int(lhs), self.const_int(rhs)) {
            let (ua, ub) = (a & mask(bits), b & mask(bits));
            let (sa, sb) = (sext(ua, bits), sext(ub, bits));
            use IntPredicate::*;
            let r = match pred {
                Eq => ua == ub,
                Ne => ua != ub,
                Ugt => ua > ub,
                Uge => ua >= ub,
                Ult => ua < ub,
                Ule => ua <= ub,
                Sgt => sa > sb,
                Sge => sa >= sb,
                Slt => sa < sb,
                Sle => sa <= sb,
            };
            return Ok(r.to_string());
        }
        let ea = self.operand(lhs)?;
        let eb = self.operand(rhs)?;
        let mut lanes = Vec::new();
        for k in 0..n {
            let a = Self::lane(&ea, vector, k);
            let b = Self::lane(&eb, vector, k);
            lanes.push(self.icmp_lane(pred, bits, &a, &b)?);
        }
        if vector {
            // vecN<bool> for up to 4 lanes, an array beyond (the carrier of <N x i1>).
            let t = self.f.inst(inst).ty;
            self.build(t, lanes)
        } else {
            Ok(lanes.remove(0))
        }
    }

    fn icmp_lane(&mut self, pred: IntPredicate, bits: u32, a: &str, b: &str) -> R<String> {
        use IntPredicate::*;
        if bits == 1 {
            // i1: true is 1 unsigned and -1 signed.
            let (ua, ub) = (format!("select(0u, 1u, {})", a), format!("select(0u, 1u, {})", b));
            return Ok(match pred {
                Eq => format!("({} == {})", a, b),
                Ne => format!("({} != {})", a, b),
                Ugt | Slt => format!("({} > {})", ua, ub),
                Uge | Sle => format!("({} >= {})", ua, ub),
                Ult | Sgt => format!("({} < {})", ua, ub),
                Ule | Sge => format!("({} <= {})", ua, ub),
            });
        }
        if bits == 64 {
            let lt = |s: &mut Self, x: &str, y: &str, signed: bool| -> R<String> {
                let f = if signed { "nm_i64_slt" } else { "nm_lt64" };
                s.need(f)?;
                Ok(format!("{}({}, {})", f, x, y))
            };
            return Ok(match pred {
                Eq => format!("all({} == {})", a, b),
                Ne => format!("any({} != {})", a, b),
                Ult => lt(self, a, b, false)?,
                Ugt => lt(self, b, a, false)?,
                Ule => format!("!{}", lt(self, b, a, false)?),
                Uge => format!("!{}", lt(self, a, b, false)?),
                Slt => lt(self, a, b, true)?,
                Sgt => lt(self, b, a, true)?,
                Sle => format!("!{}", lt(self, b, a, true)?),
                Sge => format!("!{}", lt(self, a, b, true)?),
            });
        }
        let sx = |s: &mut Self, x: &str| -> R<String> {
            if bits == 32 {
                Ok(format!("bitcast<i32>({})", x))
            } else {
                s.need("nm_sext")?;
                Ok(format!("nm_sext({}, {}u)", x, bits))
            }
        };
        let (x, y, op) = match pred {
            Eq => (a.to_string(), b.to_string(), "=="),
            Ne => (a.to_string(), b.to_string(), "!="),
            Ugt => (a.to_string(), b.to_string(), ">"),
            Uge => (a.to_string(), b.to_string(), ">="),
            Ult => (a.to_string(), b.to_string(), "<"),
            Ule => (a.to_string(), b.to_string(), "<="),
            Sgt => (sx(self, a)?, sx(self, b)?, ">"),
            Sge => (sx(self, a)?, sx(self, b)?, ">="),
            Slt => (sx(self, a)?, sx(self, b)?, "<"),
            Sle => (sx(self, a)?, sx(self, b)?, "<="),
        };
        Ok(format!("({} {} {})", x, op, y))
    }

    /// Pointer comparison: offsets within one region, (tag, offset) addresses otherwise.
    fn ptr_cmp(&mut self, pred: IntPredicate, lhs: Value, rhs: Value) -> R<String> {
        let (ra, rb) = (self.ptr_regions(lhs)?, self.ptr_regions(rhs)?);
        if ra.len() == 1 && ra == rb {
            let (a, b) = (self.operand(lhs)?, self.operand(rhs)?);
            let p = match pred {
                IntPredicate::Slt => IntPredicate::Ult,
                IntPredicate::Sle => IntPredicate::Ule,
                IntPredicate::Sgt => IntPredicate::Ugt,
                IntPredicate::Sge => IntPredicate::Uge,
                p => p,
            };
            return self.icmp_lane(p, 32, &a, &b);
        }
        let mut all: Vec<RegionId> = ra.iter().chain(rb.iter()).copied().collect();
        all.sort();
        all.dedup();
        let (a, b) = (self.ptr_as(lhs, &all)?, self.ptr_as(rhs, &all)?);
        let a = if ra.len() == 1 && all.len() == 1 { format!("vec2<u32>({}, 0u)", a) } else { a };
        let b = if rb.len() == 1 && all.len() == 1 { format!("vec2<u32>({}, 0u)", b) } else { b };
        self.icmp_lane(pred, 64, &a, &b)
    }

    fn select(&mut self, i: InstId, cond: Value, tv: Value, fv: Value, t: TypeId) -> R<String> {
        let c = self.operand(cond)?;
        if self.m.types.is_ptr(t) {
            let target = self.ptr_regions(Value::Inst(i))?;
            let (a, b) = (self.ptr_as(tv, &target)?, self.ptr_as(fv, &target)?);
            return Ok(format!("select({}, {}, {})", b, a, c));
        }
        let (a, b) = (self.operand(tv)?, self.operand(fv)?);
        let shape = self.types.shape(&self.m.types, t)?;
        let cond_vector = self.m.types.is_vector(self.value_type(cond));
        match shape {
            Shape::Bool | Shape::Word { .. } | Shape::Wide | Shape::Vector { .. } => {
                Ok(format!("select({}, {}, {})", b, a, c))
            }
            Shape::Array { len, is_vector: true, .. } if cond_vector => {
                let lanes: Vec<String> =
                    (0..len).map(|k| format!("select({}[{}], {}[{}], {}[{}])", b, k, a, k, c, k)).collect();
                self.build(t, lanes)
            }
            _ => {
                let ct = self.types.carrier(&self.m.types, t)?;
                let tmp = self.tmp();
                self.push(format!("var {}: {} = {};", tmp, ct, b));
                self.push(format!("if {} {{ {} = {}; }}", c, tmp, a));
                Ok(tmp)
            }
        }
    }

    fn cast(&mut self, i: InstId, op: CastOp, value: Value, t: TypeId, fm: crate::ir::FastMath) -> R<()> {
        let st = self.value_type(value);
        let ni = match op {
            CastOp::FPToSI => Some(num::Inst::FPToSI),
            CastOp::FPToUI => Some(num::Inst::FPToUI),
            CastOp::SIToFP => Some(num::Inst::SIToFP),
            CastOp::UIToFP => Some(num::Inst::UIToFP),
            CastOp::FPTrunc => Some(num::Inst::FPTrunc),
            CastOp::FPExt => Some(num::Inst::FPExt),
            _ => None,
        };
        if let Some(ni) = ni {
            return self.numeric(i, num::Op::Inst(ni), &[value], Some(t), fm, op.name());
        }
        let e = match op {
            CastOp::BitCast | CastOp::AddrSpaceCast if self.m.types.is_ptr(t) => {
                if let Some(p) = self.pinfo.get(&value).copied() {
                    self.pinfo.insert(Value::Inst(i), p);
                }
                let target = self.ptr_regions(Value::Inst(i))?;
                self.ptr_as(value, &target)?
            }
            CastOp::BitCast => self.bitcast(value, st, t)?,
            CastOp::PtrToInt => {
                let regions = self.ptr_regions(value)?;
                let e = self.operand(value)?;
                let addr = if regions.len() > 1 {
                    e
                } else {
                    format!("vec2<u32>({}, {}u)", e, self.mem.regions[regions[0]].tag)
                };
                let bits = self.m.types.int_bits(t).unwrap_or(64);
                match bits {
                    64 => addr,
                    1 => format!("(({}.x & 1u) != 0u)", addr),
                    b => format!("({}.x & {}u)", addr, mask(b)),
                }
            }
            CastOp::IntToPtr => {
                let regions = self.ptr_regions(Value::Inst(i))?;
                let e = self.operand(value)?;
                let bits = self.m.types.int_bits(st).unwrap_or(64);
                match (bits, regions.len() > 1) {
                    (64, false) => format!("{}.x", e),
                    (64, true) => e,
                    (_, false) => e,
                    (_, true) => {
                        return Err(tracked::refuse(
                            "unbounded-pointer-provenance",
                            "inttoptr",
                            None,
                            "a narrow integer converted to a pointer into several regions",
                        ));
                    }
                }
            }
            CastOp::Trunc | CastOp::ZExt | CastOp::SExt => {
                let (n, sel, vector) = self.lanes(st);
                let (_, del, _) = self.lanes(t);
                let sb = self.m.types.int_bits(sel).unwrap_or(32);
                let db = self.m.types.int_bits(del).unwrap_or(32);
                if !matches!(sb, 1 | 8 | 16 | 32 | 64) || !matches!(db, 1 | 8 | 16 | 32 | 64) {
                    return Err(tracked::refuse("integer-width", format!("i{} to i{}", sb, db), None, "a cast"));
                }
                if !vector && let Some(k) = self.const_int(value) {
                    let k = k & mask(sb);
                    let r = match op {
                        CastOp::SExt => sext(k, sb) as u64,
                        _ => k,
                    };
                    int_lit(db, r & mask(db))
                } else {
                    let e = self.operand(value)?;
                    let mut lanes = Vec::new();
                    for k in 0..n {
                        let x = Self::lane(&e, vector, k);
                        lanes.push(self.int_cast(op, sb, db, &x)?);
                    }
                    self.build(t, lanes)?
                }
            }
            _ => unreachable!(),
        };
        self.def(i, e)
    }

    fn int_cast(&mut self, op: CastOp, sb: u32, db: u32, x: &str) -> R<String> {
        Ok(match op {
            CastOp::Trunc => match (sb, db) {
                (64, 1) => format!("(({}.x & 1u) != 0u)", x),
                (_, 1) => format!("(({} & 1u) != 0u)", x),
                (64, 32) => format!("{}.x", x),
                (64, d) => format!("({}.x & {}u)", x, mask(d)),
                (_, d) => format!("({} & {}u)", x, mask(d)),
            },
            CastOp::ZExt => match (sb, db) {
                (1, 64) => format!("vec2<u32>(select(0u, 1u, {}), 0u)", x),
                (1, _) => format!("select(0u, 1u, {})", x),
                (_, 64) => format!("vec2<u32>({}, 0u)", x),
                _ => x.to_string(),
            },
            CastOp::SExt => match (sb, db) {
                (1, 64) => format!("select(vec2<u32>(0u, 0u), vec2<u32>(0xffffffffu, 0xffffffffu), {})", x),
                (1, d) => format!("select(0u, {}u, {})", mask(d), x),
                (32, 64) => {
                    self.need("nm_sext64")?;
                    format!("nm_sext64({})", x)
                }
                (s, 64) => {
                    self.need("nm_sext")?;
                    self.need("nm_sext64")?;
                    format!("nm_sext64(bitcast<u32>(nm_sext({}, {}u)))", x, s)
                }
                (s, d) => {
                    self.need("nm_sext")?;
                    format!("(bitcast<u32>(nm_sext({}, {}u)) & {}u)", x, s, mask(d))
                }
            },
            _ => unreachable!(),
        })
    }

    /// A bitcast between non-pointer types of one size, through the value's bits (element 0 in the low bits, as on
    /// a little-endian target).
    fn bitcast(&mut self, value: Value, st: TypeId, t: TypeId) -> R<String> {
        let e = self.operand(value)?;
        let ss = self.types.shape(&self.m.types, st)?;
        let ts = self.types.shape(&self.m.types, t)?;
        if ss == ts
            || self.types.carrier(&self.m.types, st)? == self.types.carrier(&self.m.types, t)?
                && !matches!(ss, Shape::Vector { .. })
        {
            return Ok(e);
        }
        // Source lanes to 32-bit words.
        let (sn, sel, sv) = self.lanes(st);
        let (tn, tel, tv) = self.lanes(t);
        let sbits = self.dl.size_bits(&self.m.types, sel).unwrap_or(32) as u32;
        let tbits = self.dl.size_bits(&self.m.types, tel).unwrap_or(32) as u32;
        let total = sn * sbits;
        if total != tn * tbits || total == 0 {
            return Err(tracked::refuse(
                "llvm-instruction",
                "bitcast",
                None,
                "a bitcast between types of different sizes",
            ));
        }
        let words = total.div_ceil(32) as usize;
        let mut parts: Vec<Vec<String>> = vec![Vec::new(); words];
        let src = self.tmp();
        self.push(format!("let {} = {};", src, e));
        for k in 0..sn {
            let x = Self::lane(&src, sv, k);
            let pos = k * sbits;
            match sbits {
                64 => {
                    parts[(pos / 32) as usize].push(format!("{}.x", x));
                    parts[(pos / 32) as usize + 1].push(format!("{}.y", x));
                }
                1 => parts[(pos / 32) as usize].push(format!("(select(0u, 1u, {}) << {}u)", x, pos % 32)),
                _ => parts[(pos / 32) as usize].push(if pos.is_multiple_of(32) {
                    x
                } else {
                    format!("({} << {}u)", x, pos % 32)
                }),
            }
        }
        let w: Vec<String> = parts
            .into_iter()
            .map(|p| if p.is_empty() { "0u".into() } else { format!("({})", p.join(" | ")) })
            .collect();
        let mut lanes = Vec::new();
        for k in 0..tn {
            let pos = k * tbits;
            let wi = (pos / 32) as usize;
            lanes.push(match tbits {
                64 => format!("vec2<u32>({}, {})", w[wi], w[wi + 1]),
                32 => format!("({})", w[wi]),
                1 => format!("((({}) >> {}u) & 1u) != 0u", w[wi], pos % 32),
                b => format!("((({}) >> {}u) & {}u)", w[wi], pos % 32, mask(b)),
            });
        }
        if !tv {
            return Ok(lanes.remove(0));
        }
        self.build(t, lanes)
    }

    // Calls ----------------------------------------------------------------------------------------------------------

    fn numeric(
        &mut self,
        i: InstId,
        op: num::Op,
        args: &[Value],
        ret: Option<TypeId>,
        fm: crate::ir::FastMath,
        construct: &str,
    ) -> R<()> {
        let mut exprs = Vec::new();
        let mut tys = Vec::new();
        let mut consts = Vec::new();
        for &a in args {
            let at = self.value_type(a);
            if self.m.types.is_ptr(at) {
                // A pointer operand (sincos's cosine output): the lowering returns the value to store through it.
                exprs.push("0u".to_string());
                tys.push(num::Ty::scalar(num::Scalar::I32));
                consts.push(None);
                continue;
            }
            self.types.shape(&self.m.types, at)?;
            let ty = num_ty(&self.m.types, at).ok_or_else(|| {
                tracked::refuse(
                    "unregistered-lowering",
                    construct.to_string(),
                    None,
                    "an operand type the numerics lowering does not take",
                )
            })?;
            exprs.push(self.operand(a)?);
            tys.push(ty);
            consts.push(match a {
                Value::Const(c) => match &self.m.constant(c).kind {
                    ConstKind::Float(b) => Some(*b as u64),
                    ConstKind::Int(w) => Some(w[0]),
                    _ => None,
                },
                _ => None,
            });
        }
        let operands: Vec<num::Operand> = exprs
            .iter()
            .zip(&tys)
            .zip(&consts)
            .map(|((e, t), c)| num::Operand { expr: e, ty: *t, constant: *c })
            .collect();
        if let Some(t) = ret {
            self.types.shape(&self.m.types, t)?;
        }
        let rty = match ret {
            Some(t) if !self.m.types.is_void(t) => Some(num_ty(&self.m.types, t).ok_or_else(|| {
                tracked::refuse(
                    "unregistered-lowering",
                    construct.to_string(),
                    None,
                    "a result type the numerics lowering does not take",
                )
            })?),
            _ => None,
        };
        let prefix = format!("n{}", i.0);
        let l = num::lower(op, &operands, rty, fmf(fm), self.target, &prefix).map_err(|r| num_refusal(r, construct))?;
        for s in &l.stmts {
            self.push(s.clone());
        }
        self.prelude.extend(&l.prelude);
        self.requirements.features.extend(l.requirements.features.iter().cloned());
        self.requirements.wgsl_language_features.extend(l.requirements.wgsl_language_features.iter().cloned());
        self.enables.extend(l.enables.iter().cloned());
        if let num::Op::Call(name) = op
            && (name.starts_with("air.dfdx") || name.starts_with("air.dfdy") || name.starts_with("air.fwidth"))
        {
            self.derivatives = true;
        }
        if let Some(v) = l.value {
            self.def(i, v)?;
        }
        for s in l.stores {
            let p = self.ptr(args[s.operand])?;
            let t = ret.unwrap();
            let al = self.dl.abi_align(&self.m.types, t).unwrap_or(1);
            self.with_code(|a, c| a.store(c, &p, t, al, &s.value))?;
        }
        Ok(())
    }

    fn call(&mut self, i: InstId, callee: Value, args: &[Value], fm: crate::ir::FastMath) -> R<()> {
        let Some(name) = self.callee_name(callee) else {
            return Err(tracked::refuse("llvm-instruction", "call", None, "an indirect call"));
        };
        if let Value::Global(GlobalId::Function(fx)) = callee
            && !self.m.functions[fx as usize].is_declaration()
        {
            return Err(tracked::refuse(
                "non-canonical-input",
                name,
                None,
                "a call of a defined function (inlining is canonicalization's)",
            ));
        }
        let t = self.f.inst(i).ty;
        if let Some(rule) = self.rules.find(&name) {
            let site = CallSite { name: &name, inst: i, args, ret: t };
            let v = rule.lower(self, &site)?;
            if let Some(v) = v {
                self.def(i, v)?;
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("llvm.") {
            let head = rest.split('.').next().unwrap_or("");
            match head {
                "lifetime" | "assume" | "experimental" | "dbg" | "donothing" | "invariant" | "sideeffect" => {
                    // No effect on the program's values (lifetime markers only make memory undefined).
                    return Ok(());
                }
                "memcpy" | "memmove" | "memset" => return self.mem_intrinsic(i, head, args),
                _ => {}
            }
            return self.numeric(i, num::Op::Call(&name), args, Some(t), fm, &name).map_err(|r| {
                if r.reason.contains("not a function of the numerics families") {
                    tracked::refuse("llvm-intrinsic", name.clone(), None, "an llvm intrinsic without a lowering")
                } else {
                    r
                }
            });
        }
        if name.starts_with("air.") {
            let family = tracked::family_of(&name);
            let numeric = is_numeric_family(family);
            if numeric {
                return self.numeric(i, num::Op::Call(&name), args, Some(t), fm, &name).map_err(|r| {
                    if r.reason.contains("not a function of the numerics families") {
                        tracked::refuse_family(family.unwrap(), name.clone(), None)
                    } else {
                        r
                    }
                });
            }
            return Err(match family {
                Some(f) => tracked::refuse_family(f, name, None),
                None => tracked::refuse(
                    "unregistered-lowering",
                    name,
                    None,
                    "an air function the target's tables do not name",
                ),
            });
        }
        if name.starts_with("___metal_") {
            return Err(tracked::refuse("runtime-library-function", name, None, "a Metal runtime-library math helper"));
        }
        Err(tracked::refuse("foreign-callee", name, None, "a call of a function that is not linked in"))
    }

    fn mem_intrinsic(&mut self, i: InstId, head: &str, args: &[Value]) -> R<()> {
        let inst = self.f.inst(i).clone();
        let dst = self.ptr(args[0])?;
        let dal = self.param_align(&inst, 0);
        let len = self.const_int(args[2]);
        if head == "memset" {
            let v = self.operand(args[1])?;
            return match len {
                Some(n) => self.with_code(|a, c| a.fill(c, &dst, &v, n, dal)),
                None => {
                    let l = self.len_u32(args[2])?;
                    self.with_code(|a, c| a.bytes_dynamic(c, &dst, None, &v, &l, false))
                }
            };
        }
        let src = self.ptr(args[1])?;
        let sal = self.param_align(&inst, 1);
        let overlap = head == "memmove";
        match len {
            Some(n) => {
                let (dst, src) = (dst.with_align(dal), src.with_align(sal));
                self.with_code(|a, c| a.copy(c, &dst, &src, n, overlap))
            }
            None => {
                let l = self.len_u32(args[2])?;
                self.with_code(|a, c| a.bytes_dynamic(c, &dst, Some(&src), "0u", &l, overlap))
            }
        }
    }

    fn len_u32(&mut self, v: Value) -> R<String> {
        let e = self.operand(v)?;
        Ok(if self.m.types.int_bits(self.value_type(v)) == Some(64) { format!("{}.x", e) } else { e })
    }

    /// The emitted body and whether its end is reachable for WGSL's behavior analysis.
    pub fn body(&mut self, s: &Structured) -> R<(Vec<String>, Behavior)> {
        self.declare_phis()?;
        self.nodes(&s.body)?;
        for k in std::mem::take(&mut self.flags) {
            self.decls.push(format!("var aw_flow{}: bool = false;", k));
        }
        Ok((std::mem::take(&mut self.lines), Structured::behavior(&s.body)))
    }
}

/// The intrinsics.toml families the numerics lowering ([`crate::num::lower`]) owns.
fn is_numeric_family(f: Option<&str>) -> bool {
    matches!(
        f,
        Some("math-float" | "math-geometric" | "integer" | "relational" | "conversion" | "pack-unpack" | "derivative")
    )
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::AtomicRmw { .. } => "atomicrmw",
        Op::CmpXchg { .. } => "cmpxchg",
        Op::Fence { .. } => "fence",
        Op::VaArg { .. } => "va_arg",
        Op::IndirectBr { .. } => "indirectbr",
        _ => "instruction",
    }
}

/// Name the numerics refusal's tracked item.
fn num_refusal(mut r: Refusal, construct: &str) -> Refusal {
    if tracked::item_of(&r).is_some() {
        return r;
    }
    if let Some(rest) = r.reason.strip_prefix("numerics.toml row ")
        && let Some(id) = rest.split_whitespace().next()
    {
        r.reason = format!("[numerics:{}] {}", id, r.reason);
        return r;
    }
    if r.reason.contains("explicit rounding mode or saturation") {
        r.reason = format!("[family:conversion-explicit-rounding] {}", r.reason);
        return r;
    }
    if let Some(f) = tracked::family_of(construct) {
        r.reason = format!("[family:{}] {}", f, r.reason);
        return r;
    }
    let mut t = tracked::refuse(
        "unregistered-lowering",
        if r.construct.is_empty() { construct.to_string() } else { r.construct.clone() },
        r.location.take(),
        r.reason.clone(),
    );
    t.kind = r.kind;
    t
}

struct Sink<'b> {
    lines: &'b mut Vec<String>,
    indent: &'b mut usize,
    next: &'b mut u32,
}

impl Code for Sink<'_> {
    fn fresh(&mut self) -> String {
        *self.next += 1;
        format!("t{}", self.next)
    }

    fn line(&mut self, s: String) {
        self.lines.push(format!("{}{}", "    ".repeat(*self.indent), s));
    }

    fn open(&mut self, head: String) {
        self.line(head);
        *self.indent += 1;
    }

    fn close(&mut self) {
        *self.indent -= 1;
        self.line("}".into());
    }
}

impl RuleCx for FnEmit<'_> {
    fn module(&self) -> &Module {
        self.m
    }

    fn function(&self) -> usize {
        self.fi
    }

    fn operand(&mut self, v: Value) -> Result<String, Refusal> {
        FnEmit::operand(self, v)
    }

    fn pointer(&mut self, v: Value) -> Result<Ptr, Refusal> {
        self.ptr(v)
    }

    fn region_name(&self, r: usize) -> String {
        self.mem.regions[r].name.clone()
    }

    fn region(&self, r: usize) -> &crate::mem::Region {
        &self.mem.regions[r]
    }

    fn handle_arg(&self, v: Value) -> Option<u32> {
        let r = self.mem.regions(self.m, v).ok()?;
        match r.as_slice() {
            [r] => match self.mem.regions[*r].kind {
                RegionKind::Handle { arg } => Some(arg),
                _ => None,
            },
            _ => None,
        }
    }

    fn carrier(&mut self, t: TypeId) -> Result<String, Refusal> {
        self.types.carrier(&self.m.types, t)
    }

    fn fresh(&mut self) -> String {
        self.tmp()
    }

    fn line(&mut self, s: String) {
        self.push(s);
    }

    fn module_item(&mut self, key: &str, text: String) {
        self.module_items.entry(key.to_string()).or_insert(text);
    }

    fn resource(
        &mut self,
        resource: crate::api::BindingResource,
        source: crate::api::BindingSource,
        wgsl_type: &str,
    ) -> Result<String, Refusal> {
        if matches!(source, crate::api::BindingSource::Uniforms(_) | crate::api::BindingSource::MetalBuffer { .. }) {
            return Err(tracked::refuse(
                "unregistered-lowering",
                "binding",
                None,
                "buffers and the uniform block are bound by the memory model and the layout",
            ));
        }
        let name = crate::layout::Layout::resource_name(self.stage, &source);
        if !self.resources.iter().any(|r| r.0 == name) {
            self.resources.push((name.clone(), wgsl_type.to_string(), resource, source));
        }
        Ok(name)
    }

    fn arg_record(&self, arg: u32) -> Option<crate::air::Argument> {
        self.records.iter().find(|a| a.position == arg).cloned()
    }

    fn require_feature(&mut self, feature: &str) {
        self.requirements.features.insert(feature.to_string());
    }

    fn enable(&mut self, extension: &str) {
        self.enables.insert(extension.to_string());
    }

    fn stage(&self) -> MetalStage {
        self.stage
    }

    fn bindings(&self) -> Option<&crate::api::StageBindings> {
        self.bindings.as_ref()
    }

    fn host(&self) -> &crate::api::HostCaps {
        self.target.host
    }

    fn prelude(&mut self, item: &'static str) -> Result<(), Refusal> {
        self.need(item)
    }

    fn derivatives(&mut self) {
        self.derivatives = true;
    }
}
