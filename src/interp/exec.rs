//! Execution: frames, control flow and the instructions.

use std::collections::{HashMap, HashSet};

use super::memory::{Byte, Layout, Memory, RegionId, RegionKind};
use super::ops::{self, Prim, Ub};
use super::value::{Scalar, Val, mask};
use super::{Error, Event, Options, Outcome, float};
use crate::ir::layout::DataLayout;
use crate::ir::{
    AttrKind, Attribute, BinOp, BlockId, CastOp, ConstId, FastMath, Function, GepFlags, GlobalId, InstId, Module, Op,
    RmwOp, Type, TypeId, Value,
};

pub(super) type R<T> = Result<T, Error>;

/// Runs function `function` of `m` with `args`, against `memory`.
pub fn run(m: &Module, function: usize, args: &[Val], memory: &mut Memory, opts: &Options) -> R<Outcome> {
    let dl = DataLayout::parse(&m.datalayout)
        .map_err(|e| Error::Unsupported { what: format!("data layout: {}", e), location: String::new() })?;
    let mut it = Interp {
        m,
        dl,
        mem: memory,
        opts,
        steps: 0,
        depth: 0,
        events: Vec::new(),
        uninterpreted: 0,
        choice_path: false,
        consts: HashMap::new(),
        lifetime_allocas: HashMap::new(),
        globals_creating: HashSet::new(),
    };
    let f = &m.functions[function];
    if f.is_declaration() {
        return Err(Error::Unsupported {
            what: format!("running the declaration {}", f.common.name),
            location: f.common.name.clone(),
        });
    }
    let ret = it.call_function(function, args.to_vec())?;
    Ok(Outcome {
        ret,
        events: it.events,
        uninterpreted: it.uninterpreted,
        choice_path: it.choice_path,
        steps: it.steps,
    })
}

/// The region of global variable `g` in `memory`, created and initialized if it does not exist yet.
pub fn global_region(m: &Module, g: u32, memory: &mut Memory) -> R<RegionId> {
    let dl = DataLayout::parse(&m.datalayout)
        .map_err(|e| Error::Unsupported { what: format!("data layout: {}", e), location: String::new() })?;
    let opts = Options::default();
    let mut it = Interp {
        m,
        dl,
        mem: memory,
        opts: &opts,
        steps: 0,
        depth: 0,
        events: Vec::new(),
        uninterpreted: 0,
        choice_path: false,
        consts: HashMap::new(),
        lifetime_allocas: HashMap::new(),
        globals_creating: HashSet::new(),
    };
    it.global_region(g)
}

/// Evaluates instruction `inst` of function `f` alone, against `memory`: for instructions whose operands are all
/// module-level values (constants and globals), which the canonicalization folds. Arguments and other instructions
/// as operands are unsupported here.
pub fn eval_instruction(m: &Module, f: usize, inst: InstId, memory: &mut Memory, opts: &Options) -> R<Option<Val>> {
    let dl = DataLayout::parse(&m.datalayout)
        .map_err(|e| Error::Unsupported { what: format!("data layout: {}", e), location: String::new() })?;
    let mut it = Interp {
        m,
        dl,
        mem: memory,
        opts,
        steps: 0,
        depth: 0,
        events: Vec::new(),
        uninterpreted: 0,
        choice_path: false,
        consts: HashMap::new(),
        lifetime_allocas: HashMap::new(),
        globals_creating: HashSet::new(),
    };
    let func = &m.functions[f];
    let mut fr =
        Frame { f, args: Vec::new(), vals: vec![None; func.insts.len()], allocas: Vec::new(), cur: Some(inst) };
    if func.inst(inst).op.operands().iter().any(|v| matches!(v, Value::Arg(_) | Value::Inst(_))) {
        return Err(it.unsupported(&fr, "an instruction with local operands"));
    }
    let r = it.instruction(&mut fr, inst);
    for a in &fr.allocas {
        it.mem.region_mut(*a).live = false;
    }
    r
}

pub(super) struct Interp<'a> {
    pub m: &'a Module,
    pub dl: DataLayout,
    pub mem: &'a mut Memory,
    pub opts: &'a Options,
    pub steps: u64,
    pub depth: usize,
    pub events: Vec<Event>,
    pub uninterpreted: u64,
    pub choice_path: bool,
    pub consts: HashMap<ConstId, Val>,
    /// Per function: the allocas some `llvm.lifetime.start` names (they start dead, LangRef `llvm.lifetime.start`).
    lifetime_allocas: HashMap<usize, HashSet<InstId>>,
    pub globals_creating: HashSet<u32>,
}

/// One activation.
pub(super) struct Frame {
    pub f: usize,
    pub args: Vec<Val>,
    pub vals: Vec<Option<Val>>,
    pub allocas: Vec<RegionId>,
    pub cur: Option<InstId>,
}

/// A type as the interpreter needs it: an id from the table, or the pointer type of a global (which the table may
/// lack).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TyRef {
    Id(TypeId),
    Ptr(u32),
}

impl<'a> Interp<'a> {
    pub fn layout(&self) -> Layout<'_> {
        Layout { dl: &self.dl, types: &self.m.types }
    }

    pub fn loc(&self, fr: &Frame) -> String {
        if fr.f == usize::MAX {
            return String::new();
        }
        let f = &self.m.functions[fr.f];
        match fr.cur {
            Some(i) => format!("{}: {}", f.common.name, crate::ir::print::print_instruction(self.m, fr.f, i)),
            None => f.common.name.clone(),
        }
    }

    pub fn ub(&self, fr: &Frame, reason: impl Into<String>) -> Error {
        Error::Ub { reason: reason.into(), location: self.loc(fr) }
    }

    pub fn unsupported(&self, fr: &Frame, what: impl Into<String>) -> Error {
        Error::Unsupported { what: what.into(), location: self.loc(fr) }
    }

    pub fn ub_of(&self, fr: &Frame, u: Ub) -> Error {
        self.ub(fr, u.0)
    }

    // Types -----------------------------------------------------------------------------------------------------------

    pub fn type_of(&self, fr: &Frame, v: Value) -> TyRef {
        match v {
            Value::Arg(i) => TyRef::Id(self.m.functions[fr.f].args[i as usize].ty),
            Value::Inst(i) => TyRef::Id(self.m.functions[fr.f].insts[i.0 as usize].ty),
            Value::Const(c) => TyRef::Id(self.m.constant(c).ty),
            Value::Global(g) => TyRef::Ptr(self.m.global(g).addrspace),
            Value::Metadata(_) | Value::InlineAsm(_) => TyRef::Ptr(0),
        }
    }

    /// The kind of a scalar type, or of a vector's elements.
    pub fn prim(&self, t: TyRef) -> Option<Prim> {
        match t {
            TyRef::Ptr(a) => Some(Prim::Ptr(a, self.dl.pointer_spec(a).bits)),
            TyRef::Id(t) => {
                let s = self.m.types.scalar(t);
                match self.m.types.get(s) {
                    Type::Int(w) => Some(Prim::Int(*w)),
                    Type::Ptr(a) => Some(Prim::Ptr(*a, self.dl.pointer_spec(*a).bits)),
                    _ => self.m.types.float_kind(s).and_then(float::Format::of).map(Prim::Float),
                }
            }
        }
    }

    pub fn prim_of(&self, fr: &Frame, t: TyRef) -> R<Prim> {
        let p = self.prim(t).ok_or_else(|| self.unsupported(fr, format!("values of type {:?}", t)))?;
        if p.width() > 128 {
            return Err(self.unsupported(fr, format!("{}-bit integers", p.width())));
        }
        Ok(p)
    }

    pub fn is_vector(&self, t: TyRef) -> bool {
        matches!(t, TyRef::Id(t) if self.m.types.is_vector(t))
    }

    // Operands --------------------------------------------------------------------------------------------------------

    pub fn eval(&mut self, fr: &Frame, v: Value) -> R<Val> {
        match v {
            Value::Arg(i) => Ok(fr.args[i as usize].clone()),
            Value::Inst(i) => match &fr.vals[i.0 as usize] {
                Some(v) => Ok(v.clone()),
                None => Err(self.unsupported(fr, format!("a use of %{} before its definition executed", i.0))),
            },
            Value::Const(c) => self.eval_const(c).map_err(|e| relocate(e, &self.loc(fr))),
            Value::Global(g) => self.global_ptr(g).map(Val::S).map_err(|e| relocate(e, &self.loc(fr))),
            Value::Metadata(md) => Ok(Val::Md(md)),
            Value::InlineAsm(_) => Err(self.unsupported(fr, "inline assembly")),
        }
    }

    fn scalar(&mut self, fr: &Frame, v: Value) -> R<Scalar> {
        match self.eval(fr, v)? {
            Val::S(s) => Ok(s),
            other => Err(self.unsupported(fr, format!("a scalar operand that is {:?}", other))),
        }
    }

    /// A pointer to global value `g` (creating a global variable's region on first use).
    pub fn global_ptr(&mut self, g: GlobalId) -> R<Scalar> {
        match g {
            GlobalId::Var(i) => {
                let r = self.global_region(i)?;
                Ok(self.mem.pointer(r, 0))
            }
            GlobalId::Function(i) => {
                let name = self.m.functions[i as usize].common.name.clone();
                let r = self.mem.function(i, &name);
                Ok(self.mem.pointer(r, 0))
            }
            GlobalId::Alias(i) => {
                let a = self.m.aliases[i as usize].aliasee;
                match a {
                    Value::Const(c) => match self.eval_const(c)? {
                        Val::S(s) => Ok(s),
                        _ => Err(Error::Unsupported { what: "an alias of a vector".into(), location: String::new() }),
                    },
                    Value::Global(g2) => self.global_ptr(g2),
                    _ => Err(Error::Unsupported { what: "an alias of a local value".into(), location: String::new() }),
                }
            }
            GlobalId::IFunc(_) => Err(Error::Unsupported { what: "an ifunc".into(), location: String::new() }),
        }
    }

    /// The region of global variable `i`, created and initialized on first use.
    pub fn global_region(&mut self, i: u32) -> R<RegionId> {
        if let Some(r) = self.mem.global(i) {
            return Ok(r);
        }
        let g = &self.m.globals[i as usize];
        let size = self.layout().alloc_size(g.value_type).ok_or_else(|| Error::Unsupported {
            what: format!("global @{} of an unsized type", g.common.name),
            location: String::new(),
        })?;
        // The region exists before its initializer is evaluated, so initializers that point at each other resolve.
        let writable = !g.constant;
        let r = self.mem.add(
            RegionKind::Global(i),
            g.common.addrspace,
            vec![Byte::UNDEF; size as usize],
            true,
            format!("global @{}", g.common.name),
        );
        self.mem.set_global(i, r);
        if let Some(init) = g.initializer {
            if !self.globals_creating.insert(i) {
                return Ok(r);
            }
            let v = match init {
                Value::Const(c) => self.eval_const(c)?,
                Value::Global(g2) => Val::S(self.global_ptr(g2)?),
                _ => {
                    return Err(Error::Unsupported {
                        what: format!("initializer of @{}", g.common.name),
                        location: String::new(),
                    });
                }
            };
            let bytes = self.layout().encode(g.value_type, &v).map_err(|e| Error::Unsupported {
                what: format!("initializer of @{}: {}", g.common.name, e),
                location: String::new(),
            })?;
            self.mem.region_mut(r).bytes[..bytes.len()].copy_from_slice(&bytes);
            self.globals_creating.remove(&i);
        }
        self.mem.region_mut(r).writable = writable;
        Ok(r)
    }

    // Functions -------------------------------------------------------------------------------------------------------

    pub fn call_function(&mut self, fi: usize, args: Vec<Val>) -> R<Option<Val>> {
        let f: &Function = &self.m.functions[fi];
        if self.depth >= self.opts.depth_limit {
            return Err(Error::Limit(format!("call depth {} reached calling {}", self.depth, f.common.name)));
        }
        if args.len() != f.args.len() {
            return Err(Error::Unsupported {
                what: format!("a call of {} with {} arguments", f.common.name, args.len()),
                location: f.common.name.clone(),
            });
        }
        self.depth += 1;
        let mut fr = Frame { f: fi, args, vals: vec![None; f.insts.len()], allocas: Vec::new(), cur: None };
        let r = self.body(&mut fr);
        for a in &fr.allocas {
            self.mem.region_mut(*a).live = false;
        }
        self.depth -= 1;
        r
    }

    fn body(&mut self, fr: &mut Frame) -> R<Option<Val>> {
        let m = self.m;
        let f = &m.functions[fr.f];
        let mut block = 0usize;
        let mut pred: Option<usize> = None;
        loop {
            let b = &f.blocks[block];
            // The phis of a block read their operands together, on entry from the predecessor.
            let mut k = 0;
            let mut phis = Vec::new();
            while k < b.insts.len() {
                let id = b.insts[k];
                let Op::Phi { incoming, .. } = &f.inst(id).op else { break };
                fr.cur = Some(id);
                let p = pred.ok_or_else(|| self.unsupported(fr, "a phi in the entry block"))?;
                let v = incoming
                    .iter()
                    .find(|(_, bb)| bb.0 as usize == p)
                    .map(|(v, _)| *v)
                    .ok_or_else(|| self.unsupported(fr, "a phi without an operand for its predecessor"))?;
                phis.push((id, self.eval(fr, v)?));
                k += 1;
            }
            for (id, v) in phis {
                fr.vals[id.0 as usize] = Some(v);
            }
            let mut next = None;
            for &id in &b.insts[k..] {
                self.steps += 1;
                if self.steps > self.opts.step_limit {
                    return Err(Error::Limit(format!("{} instructions executed", self.opts.step_limit)));
                }
                fr.cur = Some(id);
                let inst = f.inst(id);
                match &inst.op {
                    Op::Ret { value } => {
                        let v = match value {
                            Some(v) => Some(self.eval(fr, *v)?),
                            None => None,
                        };
                        if let Some(v) = &v {
                            let ra = m.attrs(f.attrs.ret);
                            if ra.has(AttrKind::NoUndef) && !well_defined(v) {
                                return Err(
                                    self.ub(fr, "returning a value that is not well defined from a noundef return")
                                );
                            }
                        }
                        return Ok(v);
                    }
                    Op::Br { dest } => {
                        next = Some(*dest);
                    }
                    Op::CondBr { cond, t, f: fb } => {
                        let c = self.scalar(fr, *cond)?;
                        if c.poison || c.undef != 0 {
                            return Err(self.ub(fr, "branch on an undef or poison condition"));
                        }
                        if c.choice != 0 {
                            self.choice_path = true;
                        }
                        next = Some(if c.bits & 1 != 0 { *t } else { *fb });
                    }
                    Op::Switch { cond, default, cases } => {
                        let c = self.scalar(fr, *cond)?;
                        if c.poison || c.undef != 0 {
                            return Err(self.ub(fr, "switch on an undef or poison value"));
                        }
                        if c.choice != 0 {
                            self.choice_path = true;
                        }
                        let mut dest = *default;
                        for (cv, bb) in cases {
                            if let Val::S(s) = self.eval_const(*cv)?
                                && s.bits == c.bits
                            {
                                dest = *bb;
                                break;
                            }
                        }
                        next = Some(dest);
                    }
                    Op::IndirectBr { .. } => return Err(self.unsupported(fr, "indirectbr")),
                    Op::Unreachable => return Err(self.ub(fr, "unreachable executed")),
                    _ => {
                        let v = self.instruction(fr, id)?;
                        fr.vals[id.0 as usize] = v;
                    }
                }
                if next.is_some() {
                    break;
                }
            }
            match next {
                Some(BlockId(n)) => {
                    pred = Some(block);
                    block = n as usize;
                }
                None => return Err(self.unsupported(fr, "a block without a terminator")),
            }
        }
    }

    // Instructions ----------------------------------------------------------------------------------------------------

    fn instruction(&mut self, fr: &mut Frame, id: InstId) -> R<Option<Val>> {
        let m = self.m;
        let inst = m.functions[fr.f].inst(id);
        let ty = TyRef::Id(inst.ty);
        Ok(Some(match &inst.op {
            Op::FNeg { value, fmf } => {
                let p = self.prim_of(fr, ty)?;
                let Prim::Float(fmt) = p else { return Err(self.unsupported(fr, "fneg of a non-float")) };
                let v = self.eval(fr, *value)?;
                map_lanes(&v, |a| ops::fneg(*fmf, fmt, a))
            }
            Op::Binary { op, lhs, rhs, flags, fmf } => {
                let p = self.prim_of(fr, ty)?;
                let a = self.eval(fr, *lhs)?;
                let b = self.eval(fr, *rhs)?;
                let mut err = None;
                let r = zip_lanes(&a, &b, |x, y| {
                    let r = match p {
                        Prim::Float(fmt) => Ok(ops::float_binop(*op, *fmf, fmt, x, y)),
                        _ => ops::int_binop(*op, *flags, p.width(), x, y),
                    };
                    r.unwrap_or_else(|u| {
                        err.get_or_insert(u);
                        Scalar::POISON
                    })
                });
                if let Some(u) = err {
                    return Err(self.ub_of(fr, u));
                }
                r
            }
            Op::Cast { op, value, flags, fmf } => {
                let st = self.type_of(fr, *value);
                let v = self.eval(fr, *value)?;
                self.cast_value(fr, *op, *flags, *fmf, st, inst.ty, v)?
            }
            Op::ICmp { pred, lhs, rhs, samesign } => {
                let p = self.prim_of(fr, self.type_of(fr, *lhs))?;
                let a = self.eval(fr, *lhs)?;
                let b = self.eval(fr, *rhs)?;
                zip_lanes(&a, &b, |x, y| ops::icmp(*pred, *samesign, p.width(), x, y))
            }
            Op::FCmp { pred, lhs, rhs, fmf } => {
                let Prim::Float(fmt) = self.prim_of(fr, self.type_of(fr, *lhs))? else {
                    return Err(self.unsupported(fr, "fcmp of a non-float"));
                };
                let a = self.eval(fr, *lhs)?;
                let b = self.eval(fr, *rhs)?;
                zip_lanes(&a, &b, |x, y| ops::fcmp(*pred, *fmf, fmt, x, y))
            }
            Op::Select { cond, t, f, fmf } => {
                let c = self.eval(fr, *cond)?;
                let tv = self.eval(fr, *t)?;
                let fv = self.eval(fr, *f)?;
                let r = select(&c, &tv, &fv);
                self.select_fmf(inst.ty, *fmf, r)
            }
            Op::Phi { .. } => return Err(self.unsupported(fr, "a phi after the start of its block")),
            Op::Alloca { allocated, count, addrspace, .. } => {
                let n = self.scalar(fr, *count)?;
                if !n.is_defined() {
                    return Err(self.ub(fr, "alloca of an undef or poison count"));
                }
                let cw = self.prim_of(fr, self.type_of(fr, *count))?.width();
                let n = n.bits & mask(cw);
                let es = self
                    .layout()
                    .alloc_size(*allocated)
                    .ok_or_else(|| self.unsupported(fr, "alloca of an unsized type"))?;
                let size = (n as u64)
                    .checked_mul(es)
                    .filter(|s| *s < (1 << 32))
                    .ok_or_else(|| self.unsupported(fr, format!("alloca of {} elements of {} bytes", n, es)))?;
                let label = format!("alloca %{} in {}", id.0, m.functions[fr.f].common.name);
                let r = self.mem.add(RegionKind::Alloca, *addrspace, vec![Byte::UNDEF; size as usize], true, label);
                if self.starts_dead(fr.f, id) {
                    self.mem.region_mut(r).live = false;
                }
                fr.allocas.push(r);
                Val::S(self.mem.pointer(r, 0))
            }
            Op::Load { ptr, align, .. } => {
                let p = self.scalar(fr, *ptr)?;
                let v = self.load(fr, inst.ty, &p, *align)?;
                self.load_metadata(fr, id, v)?
            }
            Op::Store { value, ptr, align, .. } => {
                let vt = self.type_of(fr, *value);
                let v = self.eval(fr, *value)?;
                let p = self.scalar(fr, *ptr)?;
                let vt = match vt {
                    TyRef::Id(t) => t,
                    TyRef::Ptr(a) => self.ptr_type(fr, a)?,
                };
                self.store(fr, vt, &v, &p, *align)?;
                return Ok(None);
            }
            Op::Gep { source, ptr, indices, flags } => {
                let base = self.eval(fr, *ptr)?;
                let mut idx = Vec::new();
                for i in indices {
                    let w = self.prim_of(fr, self.type_of(fr, *i))?.width();
                    idx.push((self.eval(fr, *i)?, w));
                }
                self.gep(fr, *source, &base, &idx, *flags)?
            }
            Op::ExtractElement { vec, idx } => {
                let v = self.eval(fr, *vec)?;
                let i = self.scalar(fr, *idx)?;
                let lanes = v.lanes();
                Val::S(extract_lane(&lanes, &i))
            }
            Op::InsertElement { vec, elt, idx } => {
                let v = self.eval(fr, *vec)?;
                let e = self.scalar(fr, *elt)?;
                let i = self.scalar(fr, *idx)?;
                Val::V(insert_lane(&v.lanes(), &e, &i))
            }
            Op::ShuffleVector { a, b, mask: sel } => {
                let x = self.eval(fr, *a)?.lanes();
                let y = self.eval(fr, *b)?.lanes();
                let all: Vec<Scalar> = x.iter().chain(y.iter()).copied().collect();
                Val::V(
                    sel.iter()
                        .map(|&k| if k < 0 || k as usize >= all.len() { Scalar::POISON } else { all[k as usize] })
                        .collect(),
                )
            }
            Op::ExtractValue { agg, indices } => {
                let mut v = self.eval(fr, *agg)?;
                for &i in indices {
                    v = match v {
                        Val::A(mut items) if (i as usize) < items.len() => items.swap_remove(i as usize),
                        _ => return Err(self.unsupported(fr, "extractvalue out of its aggregate")),
                    };
                }
                v
            }
            Op::InsertValue { agg, value, indices } => {
                let mut a = self.eval(fr, *agg)?;
                let v = self.eval(fr, *value)?;
                if !insert_at(&mut a, indices, v) {
                    return Err(self.unsupported(fr, "insertvalue out of its aggregate"));
                }
                a
            }
            Op::Call { .. } => return self.call(fr, id),
            Op::Freeze { value } => {
                let v = self.eval(fr, *value)?;
                let t = TyRef::Id(inst.ty);
                self.freeze(fr, t, &v)?
            }
            Op::AtomicRmw { op, ptr, value, align, .. } => {
                let p = self.scalar(fr, *ptr)?;
                let v = self.eval(fr, *value)?;
                let old = self.load(fr, inst.ty, &p, *align)?;
                let prim = self.prim_of(fr, ty)?;
                let new = self.rmw(fr, *op, prim, &old, &v)?;
                self.store(fr, inst.ty, &new, &p, *align)?;
                old
            }
            Op::CmpXchg { ptr, cmp, new, align, weak, .. } => {
                let p = self.scalar(fr, *ptr)?;
                let c = self.scalar(fr, *cmp)?;
                let n = self.eval(fr, *new)?;
                let vt = match self.type_of(fr, *new) {
                    TyRef::Id(t) => t,
                    TyRef::Ptr(a) => self.ptr_type(fr, a)?,
                };
                let old = self.load(fr, vt, &p, *align)?;
                let o = old.scalar().unwrap_or(Scalar::POISON);
                if !o.is_defined() || !c.is_defined() {
                    return Err(self.unsupported(fr, "cmpxchg comparing undef or poison"));
                }
                let eq = o.bits == c.bits;
                if eq {
                    self.store(fr, vt, &n, &p, *align)?;
                }
                if *weak && eq {
                    // A weak cmpxchg may fail spuriously; the interpreter chooses success.
                    self.choice_path = true;
                }
                Val::A(vec![old, Val::S(Scalar::bool(eq))])
            }
            Op::Fence { .. } => return Ok(None),
            Op::VaArg { .. } => return Err(self.unsupported(fr, "va_arg")),
            _ => return Err(self.unsupported(fr, "a terminator in the middle of a block")),
        }))
    }

    fn ptr_type(&self, fr: &Frame, addrspace: u32) -> R<TypeId> {
        self.m
            .types
            .iter()
            .find(|(_, t)| matches!(t, Type::Ptr(a) if *a == addrspace))
            .map(|(id, _)| id)
            .ok_or_else(|| self.unsupported(fr, format!("no type ptr addrspace({}) in the module", addrspace)))
    }

    fn starts_dead(&mut self, f: usize, alloca: InstId) -> bool {
        let set = self.lifetime_allocas.entry(f).or_insert_with(|| lifetime_started(self.m, f));
        set.contains(&alloca)
    }

    /// `select`'s fast-math flags: `nnan`/`ninf` make a NaN or infinite result poison (floating-point results only).
    fn select_fmf(&self, ty: TypeId, fmf: FastMath, v: Val) -> Val {
        if fmf.0 & (FastMath::NNAN | FastMath::NINF) == 0 {
            return v;
        }
        let Some(Prim::Float(fmt)) = self.prim(TyRef::Id(ty)) else { return v };
        map_lanes(&v, |s| {
            if s.poison || s.undef != 0 {
                return *s;
            }
            let b = s.bits as u64;
            if (fmf.0 & FastMath::NNAN != 0 && fmt.is_nan(b)) || (fmf.0 & FastMath::NINF != 0 && fmt.is_inf(b)) {
                Scalar::POISON
            } else {
                *s
            }
        })
    }

    fn freeze(&mut self, fr: &Frame, t: TyRef, v: &Val) -> R<Val> {
        Ok(match v {
            Val::A(items) => {
                let TyRef::Id(t) = t else { return Err(self.unsupported(fr, "freeze of a global")) };
                let elems: Vec<TypeId> = match self.m.types.get(t) {
                    Type::Array { elem, len } => vec![*elem; *len as usize],
                    _ => self.m.types.struct_elems(t).map(|e| e.to_vec()).unwrap_or_default(),
                };
                let mut out = Vec::new();
                for (k, it) in items.iter().enumerate() {
                    out.push(self.freeze(fr, TyRef::Id(elems[k]), it)?);
                }
                Val::A(out)
            }
            Val::S(_) | Val::V(_) => {
                let w = self.prim_of(fr, t)?.width();
                map_lanes(v, |s| freeze_scalar(s, w))
            }
            other => other.clone(),
        })
    }

    pub fn rmw(&self, fr: &Frame, op: RmwOp, p: Prim, old: &Val, v: &Val) -> R<Val> {
        let (Some(a), Some(b)) = (old.scalar(), v.scalar()) else {
            return Err(self.unsupported(fr, "atomicrmw on a vector"));
        };
        let w = p.width();
        let int = |o: BinOp| ops::int_binop(o, 0, w, &a, &b).map_err(|u| self.ub_of(fr, u));
        let cmp = |pr: crate::ir::IntPredicate| ops::icmp(pr, false, w, &a, &b);
        let pick = |c: Scalar, x: &Scalar, y: &Scalar| select(&Val::S(c), &Val::S(*x), &Val::S(*y));
        use crate::ir::IntPredicate as IP;
        Ok(match (op, p) {
            (RmwOp::Xchg, _) => Val::S(b),
            (RmwOp::Add, _) => Val::S(int(BinOp::Add)?),
            (RmwOp::Sub, _) => Val::S(int(BinOp::Sub)?),
            (RmwOp::And, _) => Val::S(int(BinOp::And)?),
            (RmwOp::Or, _) => Val::S(int(BinOp::Or)?),
            (RmwOp::Xor, _) => Val::S(int(BinOp::Xor)?),
            (RmwOp::Nand, _) => {
                let x = int(BinOp::And)?;
                Val::S(ops::int_binop(BinOp::Xor, 0, w, &x, &Scalar::int(mask(w))).map_err(|u| self.ub_of(fr, u))?)
            }
            (RmwOp::Max, _) => pick(cmp(IP::Sgt), &a, &b),
            (RmwOp::Min, _) => pick(cmp(IP::Slt), &a, &b),
            (RmwOp::UMax, _) => pick(cmp(IP::Ugt), &a, &b),
            (RmwOp::UMin, _) => pick(cmp(IP::Ult), &a, &b),
            (RmwOp::FAdd, Prim::Float(f)) => Val::S(ops::float_binop(BinOp::FAdd, FastMath(0), f, &a, &b)),
            (RmwOp::FSub, Prim::Float(f)) => Val::S(ops::float_binop(BinOp::FSub, FastMath(0), f, &a, &b)),
            (RmwOp::FMax | RmwOp::FMin | RmwOp::FMaximum | RmwOp::FMinimum, Prim::Float(f)) => {
                let name = match op {
                    RmwOp::FMax => "maxnum",
                    RmwOp::FMin => "minnum",
                    RmwOp::FMaximum => "maximum",
                    _ => "minimum",
                };
                Val::S(super::intrinsics::minmax(name, f, FastMath(0), &a, &b))
            }
            (RmwOp::UIncWrap, _) => {
                // old >= v ? 0 : old + 1
                let ge = cmp(IP::Uge);
                let inc = ops::int_binop(BinOp::Add, 0, w, &a, &Scalar::int(1)).map_err(|u| self.ub_of(fr, u))?;
                pick(ge, &Scalar::int(0), &inc)
            }
            (RmwOp::UDecWrap, _) => {
                // (old == 0 || old > v) ? v : old - 1
                let z = ops::icmp(IP::Eq, false, w, &a, &Scalar::int(0));
                let gt = cmp(IP::Ugt);
                let either = ops::int_binop(BinOp::Or, 0, 1, &z, &gt).map_err(|u| self.ub_of(fr, u))?;
                let dec = ops::int_binop(BinOp::Sub, 0, w, &a, &Scalar::int(1)).map_err(|u| self.ub_of(fr, u))?;
                pick(either, &b, &dec)
            }
            (RmwOp::USubCond, _) => {
                // old >= v ? old - v : old
                let ge = cmp(IP::Uge);
                let d = int(BinOp::Sub)?;
                pick(ge, &d, &a)
            }
            (RmwOp::USubSat, _) => {
                let ge = cmp(IP::Uge);
                let d = int(BinOp::Sub)?;
                pick(ge, &d, &Scalar::int(0))
            }
            _ => return Err(self.unsupported(fr, format!("atomicrmw {}", op.name()))),
        })
    }

    // Memory ----------------------------------------------------------------------------------------------------------

    pub fn load(&mut self, fr: &Frame, ty: TypeId, p: &Scalar, align: u64) -> R<Val> {
        let size = self.layout().store_size(ty).ok_or_else(|| self.unsupported(fr, "a load of an unsized type"))?;
        self.check_access(fr, p, align)?;
        let bytes = self.mem.read(p, size).map_err(|e| self.ub(fr, e))?;
        self.layout().decode(ty, &bytes).map_err(|e| self.unsupported(fr, e))
    }

    pub fn store(&mut self, fr: &Frame, ty: TypeId, v: &Val, p: &Scalar, align: u64) -> R<()> {
        self.check_access(fr, p, align)?;
        let bytes = self.layout().encode(ty, v).map_err(|e| self.unsupported(fr, e))?;
        self.mem.write(p, &bytes).map_err(|e| self.ub(fr, e))
    }

    fn check_access(&mut self, fr: &Frame, p: &Scalar, align: u64) -> R<()> {
        if p.poison || p.undef != 0 {
            return Err(self.ub(fr, "memory access through a pointer that is not well defined"));
        }
        if p.choice != 0 {
            self.choice_path = true;
        }
        if align > 1 && !(p.bits as u64).is_multiple_of(align) {
            return Err(self.ub(fr, format!("access at {:#x} declared aligned to {}", p.bits as u64, align)));
        }
        Ok(())
    }

    /// The value-changing metadata of a load: `!range`, `!nonnull` and `!align` make a violating value poison;
    /// `!noundef` makes an undef or poison value undefined behavior.
    fn load_metadata(&mut self, fr: &Frame, id: InstId, v: Val) -> R<Val> {
        let inst = self.m.functions[fr.f].inst(id);
        let mut v = v;
        for &(kind, md) in &inst.md {
            match self.m.md_kinds.get(kind as usize).map(String::as_str) {
                Some("range") => {
                    let ranges = super::consts::md_ranges(self.m, md);
                    let w = self.prim_of(fr, TyRef::Id(inst.ty))?.width();
                    v = map_lanes(&v, |s| if in_ranges(s, &ranges, w) { *s } else { Scalar::POISON });
                }
                Some("nonnull") => {
                    v = map_lanes(&v, |s| if s.is_defined() && s.bits == 0 { Scalar::POISON } else { *s });
                }
                Some("align") => {
                    let a = super::consts::md_int(self.m, md).unwrap_or(1) as u128;
                    v = map_lanes(&v, |s| if s.is_defined() && a > 1 && s.bits % a != 0 { Scalar::POISON } else { *s });
                }
                Some("noundef") if !well_defined(&v) => {
                    return Err(self.ub(fr, "a !noundef load of a value that is not well defined"));
                }
                _ => {}
            }
        }
        Ok(v)
    }

    /// `getelementptr` (LangRef: indices sign-extended or truncated to the index width, scaled by allocation sizes,
    /// added with wrapping; `nusw`/`nuw`/`inbounds` violations give poison).
    pub fn gep(&mut self, fr: &Frame, source: TypeId, base: &Val, idx: &[(Val, u32)], flags: GepFlags) -> R<Val> {
        let lanes = std::iter::once(base)
            .chain(idx.iter().map(|(v, _)| v))
            .filter_map(|v| if let Val::V(l) = v { Some(l.len()) } else { None })
            .max();
        let pick = |v: &Val, k: usize| -> Scalar {
            match v {
                Val::S(s) => *s,
                Val::V(l) => l[k],
                _ => Scalar::POISON,
            }
        };
        match lanes {
            None => {
                let b = pick(base, 0);
                let ix: Vec<(Scalar, u32)> = idx.iter().map(|(v, w)| (pick(v, 0), *w)).collect();
                Ok(Val::S(self.gep_scalar(fr, source, &b, &ix, flags)?))
            }
            Some(n) => {
                let mut out = Vec::new();
                for k in 0..n {
                    let b = pick(base, k);
                    let ix: Vec<(Scalar, u32)> = idx.iter().map(|(v, w)| (pick(v, k), *w)).collect();
                    out.push(self.gep_scalar(fr, source, &b, &ix, flags)?);
                }
                Ok(Val::V(out))
            }
        }
    }

    fn gep_scalar(
        &mut self,
        fr: &Frame,
        source: TypeId,
        base: &Scalar,
        idx: &[(Scalar, u32)],
        flags: GepFlags,
    ) -> R<Scalar> {
        if base.poison || idx.iter().any(|(s, _)| s.poison) {
            return Ok(Scalar::POISON);
        }
        let iw = self.dl.pointer_spec(0).index_bits;
        let nusw = flags.0 & (GepFlags::NUSW | GepFlags::INBOUNDS) != 0;
        let nuw = flags.0 & GepFlags::NUW != 0;
        let inbounds = flags.inbounds();
        let any_undef = base.undef != 0 || idx.iter().any(|(s, _)| s.undef != 0);
        let any_choice = base.choice != 0 || idx.iter().any(|(s, _)| s.choice != 0);
        let all_zero = idx.iter().all(|(s, _)| s.undef == 0 && s.bits == 0);
        if any_undef {
            if (inbounds || nusw || nuw) && !all_zero {
                return Ok(Scalar::POISON);
            }
            let mut r = *base;
            r.undef = mask(64);
            r.bits = 0;
            return Ok(r);
        }
        // The object's bounds, for inbounds.
        let bounds = if inbounds && !all_zero {
            let id = if base.prov != 0 { Some(base.prov) } else { self.mem.region_at(base.bits as u64) };
            match id {
                Some(id) => {
                    let r = self.mem.region(id);
                    Some((r.base, r.base + r.bytes.len() as u64))
                }
                None => return Ok(Scalar::POISON),
            }
        } else {
            None
        };
        let mut addr = base.bits as u64;
        let mut total: i64 = 0;
        let mut cur = source;
        let mut poison = false;
        let in_bounds = |a: u64| bounds.is_none_or(|(lo, hi)| a >= lo && a <= hi);
        if !in_bounds(addr) {
            poison = true;
        }
        for (k, (s, w)) in idx.iter().enumerate() {
            let raw = s.bits & mask(*w);
            let signed = sext_to_i128(raw, *w);
            // Truncation to the index width.
            if *w > iw && ((nusw && signed != sext_to_i128(raw & mask(iw), iw)) || (nuw && raw >> iw != 0)) {
                poison = true;
            }
            let ix = signed as i64;
            let off: i64 = if k == 0 {
                let es =
                    self.layout().alloc_size(cur).ok_or_else(|| self.unsupported(fr, "gep over an unsized type"))?;
                let p = (ix as i128) * (es as i128);
                if (nusw && (p < i64::MIN as i128 || p > i64::MAX as i128))
                    || (nuw && raw * (es as u128) > u64::MAX as u128)
                {
                    poison = true;
                }
                p as i64
            } else {
                match self.m.types.get(cur).clone() {
                    Type::Struct { .. } | Type::LiteralStruct { .. } => {
                        let l = self
                            .dl
                            .struct_layout(&self.m.types, cur)
                            .ok_or_else(|| self.unsupported(fr, "gep into an opaque struct"))?;
                        let i = raw as usize;
                        let o = *l.offsets.get(i).ok_or_else(|| self.unsupported(fr, "gep past a struct's members"))?;
                        cur = self.m.types.struct_elems(cur).unwrap()[i];
                        o as i64
                    }
                    Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                        cur = elem;
                        let es = self
                            .layout()
                            .alloc_size(elem)
                            .ok_or_else(|| self.unsupported(fr, "gep over an unsized type"))?;
                        let p = (ix as i128) * (es as i128);
                        if (nusw && (p < i64::MIN as i128 || p > i64::MAX as i128))
                            || (nuw && raw * (es as u128) > u64::MAX as u128)
                        {
                            poison = true;
                        }
                        p as i64
                    }
                    t => return Err(self.unsupported(fr, format!("gep index into {:?}", t))),
                }
            };
            let (nt, o1) = total.overflowing_add(off);
            if nusw && o1 {
                poison = true;
            }
            if nuw && (total as u64).checked_add(off as u64).is_none() {
                poison = true;
            }
            total = nt;
            let (na, o2) = addr.overflowing_add(off as u64);
            if nuw && o2 {
                poison = true;
            }
            if nusw && ((off >= 0 && o2) || (off < 0 && !o2)) {
                poison = true;
            }
            addr = na;
            if !in_bounds(addr) {
                poison = true;
            }
        }
        if poison {
            return Ok(Scalar::POISON);
        }
        let mut r = Scalar::ptr(base.prov, addr);
        if any_choice {
            r.choice = mask(64);
        }
        Ok(r)
    }

    // Calls -------------------------------------------------------------------------------------------------------------

    fn call(&mut self, fr: &mut Frame, id: InstId) -> R<Option<Val>> {
        let m = self.m;
        let inst = m.functions[fr.f].inst(id);
        let Op::Call { callee, fn_ty, args, attrs, .. } = &inst.op else { unreachable!() };
        let target = match callee {
            Value::Global(GlobalId::Function(i)) => *i as usize,
            Value::InlineAsm(_) => return Err(self.unsupported(fr, "a call of inline assembly")),
            other => {
                let p = self.scalar(fr, *other)?;
                if !p.is_defined() {
                    return Err(self.ub(fr, "a call through a pointer that is not well defined"));
                }
                let id = if p.prov != 0 { Some(p.prov) } else { self.mem.region_at(p.bits as u64) };
                match id.map(|r| self.mem.region(r).kind.clone()) {
                    Some(RegionKind::Function(i)) if p.bits as u64 == self.mem.region(id.unwrap()).base => i as usize,
                    _ => return Err(self.ub(fr, "a call through a pointer that is not a function")),
                }
            }
        };
        let callee_fn = &m.functions[target];
        if callee_fn.ty != *fn_ty {
            return Err(
                self.unsupported(fr, format!("a call of {} with a different function type", callee_fn.common.name))
            );
        }
        let mut vals = Vec::with_capacity(args.len());
        let (_, params, _) = m.types.function(*fn_ty).unwrap();
        for (k, a) in args.iter().enumerate() {
            let mut v = self.eval(fr, *a)?;
            let cs = m.attrs(attrs.param(k));
            let fs = m.attrs(callee_fn.attrs.param(k));
            if (cs.has(AttrKind::NoUndef) || fs.has(AttrKind::NoUndef)) && !well_defined(&v) {
                return Err(self.ub(fr, format!("argument {} is not well defined but the parameter is noundef", k)));
            }
            for set in [cs, fs] {
                v = self.param_attrs(fr, set, params.get(k).copied(), v)?;
            }
            if let Some(Attribute::Type(AttrKind::ByVal, Some(t))) = cs.get(AttrKind::ByVal).or(fs.get(AttrKind::ByVal))
            {
                // byval: the callee gets a copy of the pointee.
                let p = v.scalar().unwrap_or(Scalar::POISON);
                let size = self.layout().alloc_size(*t).unwrap_or(0);
                let bytes = self.mem.read(&p, size).map_err(|e| self.ub(fr, e))?;
                let label = format!("byval copy of argument {} at %{}", k, id.0);
                let r = self.mem.add(RegionKind::Alloca, 0, bytes, true, label);
                fr.allocas.push(r);
                v = Val::S(self.mem.pointer(r, 0));
            }
            vals.push(v);
        }
        let ret = if callee_fn.is_declaration() {
            self.intrinsic(fr, id, target, vals)?
        } else {
            self.call_function(target, vals)?
        };
        // Call-site return attributes and metadata.
        let ret = match ret {
            Some(v) => {
                let ra = m.attrs(attrs.ret);
                let fa = m.attrs(callee_fn.attrs.ret);
                if (ra.has(AttrKind::NoUndef) || fa.has(AttrKind::NoUndef)) && !well_defined(&v) {
                    return Err(self.ub(fr, "a noundef return value that is not well defined"));
                }
                let mut v = v;
                for set in [ra, fa] {
                    v = self.param_attrs(fr, set, Some(inst.ty), v)?;
                }
                Some(self.load_metadata(fr, id, v)?)
            }
            None => None,
        };
        Ok(ret)
    }

    /// Attributes that turn a violating argument or return value into poison: `nonnull`, `range`, `align`.
    fn param_attrs(&self, fr: &Frame, set: &crate::ir::AttrSet, ty: Option<TypeId>, v: Val) -> R<Val> {
        let mut v = v;
        for a in set.iter() {
            match a {
                Attribute::Enum(AttrKind::NonNull) => {
                    v = map_lanes(&v, |s| if s.is_defined() && s.bits == 0 { Scalar::POISON } else { *s });
                }
                Attribute::Int(AttrKind::Alignment, al) => {
                    let al = *al as u128;
                    v = map_lanes(
                        &v,
                        |s| if s.is_defined() && al > 1 && s.bits % al != 0 { Scalar::POISON } else { *s },
                    );
                }
                Attribute::Range(AttrKind::Range, r) => {
                    let w = match ty {
                        Some(t) => self.prim_of(fr, TyRef::Id(t))?.width(),
                        None => r.bits,
                    };
                    let lo = words(&r.lower);
                    let hi = words(&r.upper);
                    v = map_lanes(&v, |s| if in_ranges(s, &[(lo, hi)], w) { *s } else { Scalar::POISON });
                }
                _ => {}
            }
        }
        Ok(v)
    }

    pub fn event(&mut self, callee: &str, args: Vec<Val>) {
        self.events.push(Event { callee: callee.to_string(), args });
    }
}

fn words(w: &[u64]) -> u128 {
    let mut v = 0u128;
    for (i, x) in w.iter().enumerate().take(2) {
        v |= (*x as u128) << (64 * i);
    }
    v
}

/// Re-attaches a location to an error raised while evaluating a constant or a global.
fn relocate(e: Error, loc: &str) -> Error {
    match e {
        Error::Ub { reason, location } if location.is_empty() => Error::Ub { reason, location: loc.to_string() },
        Error::Unsupported { what, location } if location.is_empty() => {
            Error::Unsupported { what, location: loc.to_string() }
        }
        other => other,
    }
}

/// The allocas of function `f` some `llvm.lifetime.start` names, through no-op pointer casts.
fn lifetime_started(m: &Module, f: usize) -> HashSet<InstId> {
    let func = &m.functions[f];
    let mut out = HashSet::new();
    for b in &func.blocks {
        for &id in &b.insts {
            let Op::Call { callee: Value::Global(GlobalId::Function(c)), args, .. } = &func.inst(id).op else {
                continue;
            };
            if !m.functions[*c as usize].common.name.starts_with("llvm.lifetime.start") {
                continue;
            }
            let mut v = args.last().copied();
            while let Some(Value::Inst(i)) = v {
                match &func.inst(i).op {
                    Op::Alloca { .. } => {
                        out.insert(i);
                        break;
                    }
                    Op::Cast { op: CastOp::BitCast | CastOp::AddrSpaceCast, value, .. } => v = Some(*value),
                    Op::Gep { ptr, indices, .. }
                        if indices.iter().all(|x| matches!(x, Value::Const(c) if is_zero_const(m, *c))) =>
                    {
                        v = Some(*ptr)
                    }
                    _ => break,
                }
            }
        }
    }
    out
}

fn is_zero_const(m: &Module, c: ConstId) -> bool {
    match &m.constant(c).kind {
        crate::ir::ConstKind::Int(w) => w.iter().all(|x| *x == 0),
        crate::ir::ConstKind::Zero => true,
        _ => false,
    }
}

fn sext_to_i128(v: u128, w: u32) -> i128 {
    if w >= 128 {
        return v as i128;
    }
    let sh = 128 - w;
    ((v << sh) as i128) >> sh
}

/// Whether `s` lies in the union of half-open wrapping ranges `[lo, hi)` (undef bits: whether all values do).
fn in_ranges(s: &Scalar, ranges: &[(u128, u128)], w: u32) -> bool {
    if s.poison {
        return true;
    }
    let m = mask(w);
    let inside = |v: u128| {
        ranges.iter().any(|&(lo, hi)| {
            let (lo, hi) = (lo & m, hi & m);
            if lo == hi {
                // LangRef: a pair a,b must have a != b; the full set is how ConstantRange spells max,max.
                true
            } else if lo < hi {
                v >= lo && v < hi
            } else {
                v >= lo || v < hi
            }
        })
    };
    if s.undef == 0 {
        return inside(s.bits & m);
    }
    // With undef bits the value is in range only if every value of them is. Up to 2^12 values are checked; with
    // more undef bits the answer is "not in range", which makes the result poison, the outcome that admits all.
    let und = s.undef & m;
    if und.count_ones() > 12 {
        return false;
    }
    let base = s.bits & !s.undef & m;
    let bits: Vec<u32> = (0..w).filter(|b| und >> b & 1 != 0).collect();
    (0u32..(1 << bits.len())).all(|k| {
        let mut v = base;
        for (j, b) in bits.iter().enumerate() {
            if k >> j & 1 != 0 {
                v |= 1 << b;
            }
        }
        inside(v)
    })
}

/// LangRef "Well-Defined Values": no undef bit and not poison, element-wise.
pub(super) fn well_defined(v: &Val) -> bool {
    match v {
        Val::S(s) => s.is_defined(),
        Val::V(l) => l.iter().all(Scalar::is_defined),
        Val::A(a) => a.iter().all(well_defined),
        Val::Md(_) | Val::Token => true,
    }
}

pub(super) fn map_lanes(v: &Val, f: impl Fn(&Scalar) -> Scalar) -> Val {
    match v {
        Val::S(s) => Val::S(f(s)),
        Val::V(l) => Val::V(l.iter().map(f).collect()),
        other => other.clone(),
    }
}

pub(super) fn zip_lanes(a: &Val, b: &Val, mut f: impl FnMut(&Scalar, &Scalar) -> Scalar) -> Val {
    match (a, b) {
        (Val::S(x), Val::S(y)) => Val::S(f(x, y)),
        (Val::V(x), Val::V(y)) => Val::V(x.iter().zip(y.iter()).map(|(p, q)| f(p, q)).collect()),
        _ => Val::S(Scalar::POISON),
    }
}

/// `freeze` of one element: undef bits and poison become choice bits set to zero.
fn freeze_scalar(s: &Scalar, w: u32) -> Scalar {
    if s.poison {
        return Scalar { bits: 0, undef: 0, choice: mask(w), prov: 0, poison: false };
    }
    Scalar { bits: s.bits & !s.undef, undef: 0, choice: s.choice | s.undef, prov: s.prov, poison: false }
}

/// `select` (LangRef: a poison condition gives poison; an undef condition may pick either operand, so the result's
/// bits are defined only where both operands agree).
pub(super) fn select(c: &Val, t: &Val, f: &Val) -> Val {
    match c {
        Val::S(cs) => {
            if cs.poison {
                return poison_like(t);
            }
            if cs.undef & 1 != 0 {
                return merge(t, f);
            }
            let mut r = if cs.bits & 1 != 0 { t.clone() } else { f.clone() };
            if cs.choice & 1 != 0 && t != f {
                r = choose_marked(&r, t, f);
            }
            r
        }
        Val::V(cl) => {
            let (tl, fl) = (t.lanes(), f.lanes());
            Val::V(
                cl.iter()
                    .enumerate()
                    .map(|(k, c)| match select(&Val::S(*c), &Val::S(tl[k]), &Val::S(fl[k])) {
                        Val::S(s) => s,
                        _ => Scalar::POISON,
                    })
                    .collect(),
            )
        }
        _ => poison_like(t),
    }
}

/// `r` (one of `t`, `f`) with every bit where `t` and `f` differ marked as a choice.
fn choose_marked(r: &Val, t: &Val, f: &Val) -> Val {
    match (r, t, f) {
        (Val::S(r), Val::S(a), Val::S(b)) => {
            let mut x = *r;
            if !x.poison {
                x.choice |= (a.bits ^ b.bits) | a.undef | b.undef | a.choice | b.choice;
                x.choice &= !x.undef;
                if a.poison || b.poison || a.prov != b.prov {
                    x.choice = u128::MAX;
                }
            }
            Val::S(x)
        }
        (Val::V(r), Val::V(a), Val::V(b)) => Val::V(
            r.iter()
                .zip(a.iter().zip(b.iter()))
                .map(|(r, (a, b))| match choose_marked(&Val::S(*r), &Val::S(*a), &Val::S(*b)) {
                    Val::S(s) => s,
                    _ => Scalar::POISON,
                })
                .collect(),
        ),
        (Val::A(r), Val::A(a), Val::A(b)) => {
            Val::A(r.iter().zip(a.iter().zip(b.iter())).map(|(r, (a, b))| choose_marked(r, a, b)).collect())
        }
        (r, _, _) => r.clone(),
    }
}

/// The set {a, b} as one value: bits where both are defined and equal stay, the rest are undef; poison if either is.
fn merge(a: &Val, b: &Val) -> Val {
    match (a, b) {
        (Val::S(x), Val::S(y)) => {
            if x.poison || y.poison {
                return Val::S(Scalar::POISON);
            }
            let und = x.undef | y.undef | (x.bits ^ y.bits);
            Val::S(Scalar {
                bits: x.bits & !und,
                undef: und,
                choice: (x.choice | y.choice) & !und,
                prov: if x.prov == y.prov { x.prov } else { 0 },
                poison: false,
            })
        }
        (Val::V(x), Val::V(y)) => Val::V(
            x.iter()
                .zip(y.iter())
                .map(|(p, q)| match merge(&Val::S(*p), &Val::S(*q)) {
                    Val::S(s) => s,
                    _ => Scalar::POISON,
                })
                .collect(),
        ),
        (Val::A(x), Val::A(y)) => Val::A(x.iter().zip(y.iter()).map(|(p, q)| merge(p, q)).collect()),
        (x, _) => x.clone(),
    }
}

fn poison_like(v: &Val) -> Val {
    match v {
        Val::S(_) => Val::S(Scalar::POISON),
        Val::V(l) => Val::V(vec![Scalar::POISON; l.len()]),
        Val::A(a) => Val::A(a.iter().map(poison_like).collect()),
        other => other.clone(),
    }
}

/// `extractelement` (LangRef: an index past the end gives poison; an undef index may select any lane it can name).
fn extract_lane(lanes: &[Scalar], i: &Scalar) -> Scalar {
    if i.poison {
        return Scalar::POISON;
    }
    let n = lanes.len() as u128;
    if i.undef == 0 {
        return if i.bits < n { lanes[i.bits as usize] } else { Scalar::POISON };
    }
    if (i.bits | i.undef) >= n {
        return Scalar::POISON;
    }
    let mut acc: Option<Val> = None;
    for k in 0..n {
        if (i.bits ^ k) & !i.undef == 0 {
            let v = Val::S(lanes[k as usize]);
            acc = Some(match acc {
                None => v,
                Some(a) => merge(&a, &v),
            });
        }
    }
    acc.and_then(|v| v.scalar()).unwrap_or(Scalar::POISON)
}

/// `insertelement` (LangRef: an index past the end gives a poison vector).
fn insert_lane(lanes: &[Scalar], e: &Scalar, i: &Scalar) -> Vec<Scalar> {
    let n = lanes.len() as u128;
    if i.poison || (i.bits | i.undef) >= n {
        return vec![Scalar::POISON; lanes.len()];
    }
    let mut out = lanes.to_vec();
    if i.undef == 0 {
        out[i.bits as usize] = *e;
        return out;
    }
    for k in 0..n {
        if (i.bits ^ k) & !i.undef == 0
            && let Val::S(s) = merge(&Val::S(lanes[k as usize]), &Val::S(*e))
        {
            out[k as usize] = s;
        }
    }
    out
}

fn insert_at(a: &mut Val, indices: &[u32], v: Val) -> bool {
    match indices.split_first() {
        None => {
            *a = v;
            true
        }
        Some((&i, rest)) => match a {
            Val::A(items) if (i as usize) < items.len() => insert_at(&mut items[i as usize], rest, v),
            _ => false,
        },
    }
}
