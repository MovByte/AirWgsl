//! Inlining every call of a defined function into the entry point.
//!
//! A call `%r = call @g(args)` in block B is replaced by a copy of `@g`'s body: B ends with a branch to the copy of
//! `@g`'s entry, the instructions after the call move to a new block that every copied `ret` branches to, and `%r`
//! becomes the returned value (a phi when `@g` returns from several blocks, poison when it never returns). Arguments
//! are substituted (a `byval` argument gets a fresh copy of its pointee, as the callee would). Allocas with constant
//! counts in `@g`'s entry block move to the caller's entry block: the callee's frame dies when it returns, so
//! sharing one object across calls is unobservable except through accesses after the return, which are undefined.
//!
//! The copy drops what only restricts the program further: parameter and return attributes of the call, `!dbg`,
//! and `!alias.scope`/`!noalias` (whose scopes would otherwise be shared between copies of one callee).
//!
//! Each copied instruction remembers the functions it was inlined through; a call of one of them, or of the entry
//! point, is a call cycle: refused, since MSL forbids recursion and WGSL has no way to express it.

use std::collections::{HashMap, HashSet};

use crate::api::{Refusal, RefusalKind};
use crate::ir::{AttrKind, Attribute, Block, BlockId, GlobalId, InstId, Module, Op, Value};

use super::util::{new_inst, operands_mut, remove_inst, replace_uses, successors_mut, terminator};
use super::{refusal, refusal_at};

/// Instructions in the entry point above which inlining refuses (a guard against exponential growth).
const MAX_INSTS: usize = 4_000_000;

/// Inlines every call of a defined function in function `f`, transitively.
#[allow(clippy::result_large_err)]
pub fn inline_all(m: &mut Module, f: usize) -> Result<(), Refusal> {
    // The functions each instruction was inlined through.
    let mut chains: HashMap<InstId, std::rc::Rc<HashSet<usize>>> = HashMap::new();
    let empty = std::rc::Rc::new(HashSet::new());
    loop {
        let Some((call, callee)) = next_call(m, f) else { return Ok(()) };
        let chain = chains.get(&call).cloned().unwrap_or_else(|| empty.clone());
        if callee == f || chain.contains(&callee) {
            return Err(refusal_at(
                m,
                f,
                call,
                RefusalKind::NoExactLowering,
                &format!("call of @{}", m.functions[callee].common.name),
                "a call cycle (recursion), which MSL forbids and WGSL cannot express",
            ));
        }
        if m.functions[f].insts.len() > MAX_INSTS {
            return Err(refusal(
                RefusalKind::NotImplemented,
                "inlining",
                &m.functions[f].common.name,
                &format!("the inlined function exceeds {} instructions", MAX_INSTS),
            ));
        }
        let mut inner: HashSet<usize> = (*chain).clone();
        inner.insert(callee);
        let inner = std::rc::Rc::new(inner);
        let first_new = m.functions[f].insts.len() as u32;
        inline_call(m, f, call, callee)?;
        for i in first_new..m.functions[f].insts.len() as u32 {
            chains.insert(InstId(i), inner.clone());
        }
    }
}

/// The first call of a defined function in `f`.
fn next_call(m: &Module, f: usize) -> Option<(InstId, usize)> {
    let func = &m.functions[f];
    for b in &func.blocks {
        for &i in &b.insts {
            if let Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } = &func.inst(i).op
                && !m.functions[*c as usize].is_declaration()
            {
                return Some((i, *c as usize));
            }
        }
    }
    None
}

#[allow(clippy::result_large_err)]
fn inline_call(m: &mut Module, f: usize, call: InstId, callee: usize) -> Result<(), Refusal> {
    let g = m.functions[callee].clone();
    let (_, _, vararg) = m.types.function(g.ty).map(|(r, p, v)| (r, p.to_vec(), v)).unwrap();
    if vararg {
        return Err(refusal_at(
            m,
            f,
            call,
            RefusalKind::NotImplemented,
            "variadic call",
            "a call of a variadic function",
        ));
    }
    let Op::Call { args, fn_ty, attrs, .. } = m.functions[f].inst(call).op.clone() else { unreachable!() };
    if fn_ty != g.ty {
        return Err(refusal_at(
            m,
            f,
            call,
            RefusalKind::RejectedByDevice,
            "call",
            "the call's function type differs from the callee's",
        ));
    }
    let ret_ty = m.functions[f].inst(call).ty;
    let blk = m.functions[f].blocks.iter().position(|b| b.insts.contains(&call)).unwrap();
    let pos = m.functions[f].blocks[blk].insts.iter().position(|&x| x == call).unwrap();

    // Argument values; byval arguments get a copy.
    let mut arg_vals: Vec<Value> = args.clone();
    let mut prelude: Vec<InstId> = Vec::new();
    let i64t = m.types.int(64);
    let i1t = m.types.int(1);
    let void = m.types.void();
    for (k, a) in args.iter().enumerate() {
        let byval =
            [m.attrs(attrs.param(k)), m.attrs(g.attrs.param(k))].iter().find_map(|s| match s.get(AttrKind::ByVal) {
                Some(Attribute::Type(_, Some(t))) => Some(*t),
                _ => None,
            });
        if let Some(t) = byval {
            let dl = crate::ir::layout::DataLayout::parse(&m.datalayout).unwrap_or_default();
            let size = dl.alloc_size(&m.types, t).unwrap_or(0);
            let align = dl.abi_align(&m.types, t).unwrap_or(1);
            let as_ = m.types.pointer_addrspace(g.args[k].ty).unwrap_or(0);
            let one = super::util::int_const(m, i64t, 1);
            let pty = g.args[k].ty;
            let al = new_inst(
                &mut m.functions[f],
                pty,
                Op::Alloca { allocated: t, count: one, align, addrspace: as_, inalloca: false, swifterror: false },
            );
            let memcpy = memcpy_decl(m, as_)?;
            let len = super::util::int_const(m, i64t, size as u128);
            let no = super::util::int_const(m, i1t, 0);
            let fnty = m.functions[memcpy].ty;
            let cp = new_inst(
                &mut m.functions[f],
                void,
                Op::Call {
                    callee: Value::Global(GlobalId::Function(memcpy as u32)),
                    fn_ty: fnty,
                    args: vec![Value::Inst(al), *a, len, no],
                    attrs: Default::default(),
                    cc: 0,
                    tail: Default::default(),
                    fmf: Default::default(),
                    bundles: Vec::new(),
                },
            );
            prelude.push(al);
            prelude.push(cp);
            arg_vals[k] = Value::Inst(al);
        }
    }

    let func = &mut m.functions[f];
    let boff = func.blocks.len() as u32;
    let ioff = func.insts.len() as u32;
    // The continuation block takes the instructions after the call.
    let cont = BlockId(boff + g.blocks.len() as u32);
    let tail: Vec<InstId> = func.blocks[blk].insts.split_off(pos + 1);
    func.blocks[blk].insts.pop();
    // Phis in the old block's successors now come from the continuation.
    if let Some(&t) = tail.last() {
        let succ: Vec<usize> = func.inst(t).op.successors().into_iter().map(|b| b.0 as usize).collect();
        for s in succ {
            let ids = func.blocks[s].insts.clone();
            for i in ids {
                if let Op::Phi { incoming, .. } = &mut func.insts[i.0 as usize].op {
                    for (_, p) in incoming.iter_mut() {
                        if p.0 as usize == blk {
                            *p = cont;
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }
    // Copies of the callee's instructions.
    let map_val = |v: Value| -> Value {
        match v {
            Value::Arg(i) => arg_vals[i as usize],
            Value::Inst(i) => Value::Inst(InstId(i.0 + ioff)),
            other => other,
        }
    };
    let alias_kinds: Vec<u32> = ["alias.scope", "noalias"]
        .iter()
        .filter_map(|k| m.md_kinds.iter().position(|x| x == k).map(|p| p as u32))
        .collect();
    let func = &mut m.functions[f];
    for inst in &g.insts {
        let mut c = inst.clone();
        for o in operands_mut(&mut c.op) {
            *o = map_val(*o);
        }
        for s in successors_mut(&mut c.op) {
            *s = BlockId(s.0 + boff);
        }
        if let Op::Phi { incoming, .. } = &mut c.op {
            for (_, b) in incoming.iter_mut() {
                *b = BlockId(b.0 + boff);
            }
        }
        c.name = None;
        c.dbg = None;
        c.md.retain(|(k, _)| !alias_kinds.contains(k));
        func.insts.push(c);
    }
    let mut returns: Vec<(Option<Value>, BlockId)> = Vec::new();
    for (k, b) in g.blocks.iter().enumerate() {
        let insts: Vec<InstId> = b.insts.iter().map(|i| InstId(i.0 + ioff)).collect();
        func.blocks.push(Block { name: None, insts, use_order: None });
        let nb = boff as usize + k;
        if let Some(t) = terminator(func, nb)
            && let Op::Ret { value } = func.inst(t).op.clone()
        {
            func.insts[t.0 as usize].op = Op::Br { dest: cont };
            returns.push((value, BlockId(nb as u32)));
        }
    }
    func.blocks.push(Block { name: None, insts: tail, use_order: None });
    // The call's block runs the byval copies and enters the callee.
    func.blocks[blk].insts.extend(prelude);
    let br = new_inst(func, void, Op::Br { dest: BlockId(boff) });
    func.blocks[blk].insts.push(br);
    // The returned value.
    if !m.types.is_void(ret_ty) {
        let rv = match returns.len() {
            0 => super::util::kind_const(m, ret_ty, crate::ir::ConstKind::Poison),
            1 => returns[0].0.unwrap(),
            _ => {
                let incoming = returns.iter().map(|(v, b)| (v.unwrap(), *b)).collect();
                let func = &mut m.functions[f];
                let phi = new_inst(func, ret_ty, Op::Phi { incoming, fmf: Default::default() });
                func.blocks[cont.0 as usize].insts.insert(0, phi);
                Value::Inst(phi)
            }
        };
        replace_uses(&mut m.functions[f], Value::Inst(call), rv);
    }
    // Static allocas of the callee's entry block go to the caller's entry block.
    let func = &mut m.functions[f];
    let entry_new = boff as usize;
    let statics: Vec<InstId> = func.blocks[entry_new]
        .insts
        .iter()
        .copied()
        .filter(|&i| matches!(&func.inst(i).op, Op::Alloca { count: Value::Const(_), .. }))
        .collect();
    for a in statics.iter().rev() {
        remove_inst(func, *a);
        func.blocks[0].insts.insert(0, *a);
    }
    Ok(())
}

/// `llvm.memcpy.p<as>.p<as>.i64`, declared if the module lacks it.
#[allow(clippy::result_large_err)]
fn memcpy_decl(m: &mut Module, addrspace: u32) -> Result<usize, Refusal> {
    let name = format!("llvm.memcpy.p{}.p{}.i64", addrspace, addrspace);
    if let Some(i) = m.function_by_name(&name) {
        return Ok(i);
    }
    Err(refusal(
        RefusalKind::NotImplemented,
        "byval argument",
        &name,
        "a byval argument needs llvm.memcpy, which the module does not declare",
    ))
}
