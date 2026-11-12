//! 32-bit atomic functions on device and threadgroup memory (MSL 6.16, plan 5.8).
//!
//! AIR names them `air.atomic.<space>.<op>[.<s|u>].i32`, space `global` (device, address space 1) or `local`
//! (threadgroup, address space 3), with the operands (measured from the target's front end, `-std=metal3.1` for
//! `air64-apple-ios17`):
//!
//! | op | operands |
//! | --- | --- |
//! | `load` | object, order, scope, volatile |
//! | `store`, `xchg`, `add`, `sub`, `and`, `or`, `xor`, `min`, `max` | object, value, order, scope, volatile |
//! | `cmpxchg.weak` | object, expected (a thread pointer), desired, success order, failure order, scope, volatile |
//!
//! The order is `0` for `memory_order_relaxed`, the only order MSL before 3.2 has and the only one the A10X accepts
//! (D1: `supportsMemoryOrderAtomics` = false); the scope is `2` for device and `1` for threadgroup atomics. Any other
//! encoding is refused. `atomic_bool` is a 32-bit atomic in AIR (D2), so it needs nothing of its own; float and 64-bit
//! atomics are outside the A10X's family and are refused with their families (`atomic-float`, `atomic-int64`).
//!
//! The object's region is declared `array<atomic<u32>>` through the [`Effect::Atomic`] the rule states, so every access
//! to it is atomic (plan 5.8). WGSL's atomics are relaxed with the same memory scopes (storage: queue family,
//! workgroup: workgroup; WGSL 14.5.3), and their operations on `u32` are MSL's on the bits of `int` and `uint` except
//! signed minimum and maximum, which are a compare-exchange loop: an exact read-modify-write, the loop retrying until
//! no other invocation changed the word between the read and the exchange.
//!
//! Compare-exchange: the front end derives the success of `atomic_compare_exchange_weak_explicit` from the returned
//! value (`icmp eq expected, result`) and reloads `*expected` after the call, so `air.atomic.*.cmpxchg.weak` returns
//! the value the object held and stores it into `*expected`, and its exchange happened exactly when that value equals
//! the expected one. WGSL's `atomicCompareExchangeWeak` may fail spuriously (`exchanged` false although the values
//! are equal), so it is retried until it either exchanges or reports a different value. `*expected` is thread memory,
//! which no other invocation sees: storing the value unconditionally is the same as storing it on failure.

use crate::api::{Location, Refusal};
use crate::ir::{ConstKind, Value};
use crate::mem::{NULL, Region};
use crate::wgsl::hooks::{CallRule, CallSite, Effect, RuleCx};
use crate::wgsl::tracked;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Load,
    Store,
    Xchg,
    CmpXchg,
    Add,
    Sub,
    And,
    Or,
    Xor,
    Min { signed: bool },
    Max { signed: bool },
}

/// A parsed atomic function name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Name {
    /// `local`: threadgroup memory.
    pub local: bool,
    pub op: Op,
}

/// The atomic function `name` is, when it is a 32-bit integer one.
pub fn parse(name: &str) -> Option<Name> {
    let rest = name.strip_prefix("air.atomic.")?;
    let (space, rest) = rest.split_once('.')?;
    let local = match space {
        "global" => false,
        "local" => true,
        _ => return None,
    };
    let op = rest.strip_suffix(".i32")?;
    let (base, sign) = match op.rsplit_once('.') {
        Some((b, "s")) => (b, Some(true)),
        Some((b, "u")) => (b, Some(false)),
        _ => (op, None),
    };
    let op = match (base, sign) {
        ("load", None) => Op::Load,
        ("store", None) => Op::Store,
        ("xchg", None) => Op::Xchg,
        ("cmpxchg.weak", None) => Op::CmpXchg,
        ("add", _) => Op::Add,
        ("sub", _) => Op::Sub,
        ("and", _) => Op::And,
        ("or", _) => Op::Or,
        ("xor", _) => Op::Xor,
        ("min", Some(s)) => Op::Min { signed: s },
        ("max", Some(s)) => Op::Max { signed: s },
        _ => return None,
    };
    Some(Name { local, op })
}

/// The operand positions of an op: (value, expected, orders, scope, count).
fn layout(op: Op) -> (Option<usize>, Option<usize>, &'static [usize], usize, usize) {
    match op {
        Op::Load => (None, None, &[1], 2, 4),
        Op::CmpXchg => (Some(2), Some(1), &[3, 4], 5, 7),
        _ => (Some(1), None, &[2], 3, 5),
    }
}

/// The lowering of the 32-bit atomic functions.
pub struct Atomics;

fn int_const(cx: &dyn RuleCx, v: Value) -> Option<u64> {
    match v {
        Value::Const(c) => match &cx.module().constant(c).kind {
            ConstKind::Int(w) => Some(w[0]),
            _ => None,
        },
        _ => None,
    }
}

fn loc(cx: &dyn RuleCx, call: &CallSite<'_>) -> Option<Location> {
    let fi = cx.function();
    Some(Location {
        function: cx.module().functions[fi].common.name.clone(),
        instruction: Some(crate::ir::print::print_instruction(cx.module(), fi, call.inst)),
    })
}

/// A word of a region: an atomic access where the region is declared atomic, a plain one otherwise (thread memory,
/// which no other invocation accesses).
fn read_word(r: &Region, w: &str) -> String {
    if r.atomic { format!("atomicLoad(&{}[{}])", r.name, w) } else { format!("{}[{}]", r.name, w) }
}

fn write_word(r: &Region, w: &str, v: &str) -> String {
    if r.atomic { format!("atomicStore(&{}[{}], {});", r.name, w, v) } else { format!("{}[{}] = {};", r.name, w, v) }
}

/// The statements that perform `op` on word `w` of region `r` and leave the old value in `res` (a declared `var`),
/// with `v` the operand and `e` the expected value of a compare-exchange.
fn op_lines(cx: &mut dyn RuleCx, r: &Region, w: &str, op: Op, v: &str, e: &str, res: &str) -> Vec<String> {
    let at = format!("&{}[{}]", r.name, w);
    let mut out = Vec::new();
    let combine = |old: &str| -> String {
        match op {
            Op::Add => format!("({} + {})", old, v),
            Op::Sub => format!("({} - {})", old, v),
            Op::And => format!("({} & {})", old, v),
            Op::Or => format!("({} | {})", old, v),
            Op::Xor => format!("({} ^ {})", old, v),
            Op::Min { signed: false } => format!("min({}, {})", old, v),
            Op::Max { signed: false } => format!("max({}, {})", old, v),
            Op::Min { signed: true } => format!("bitcast<u32>(min(bitcast<i32>({}), bitcast<i32>({})))", old, v),
            Op::Max { signed: true } => format!("bitcast<u32>(max(bitcast<i32>({}), bitcast<i32>({})))", old, v),
            Op::Xchg => v.to_string(),
            Op::CmpXchg => format!("select({}, {}, {} == {})", old, v, old, e),
            Op::Load | Op::Store => old.to_string(),
        }
    };
    if !r.atomic {
        // Thread memory: a plain read-modify-write.
        match op {
            Op::Load => out.push(format!("{} = {}[{}];", res, r.name, w)),
            Op::Store => out.push(format!("{}[{}] = {};", r.name, w, v)),
            _ => {
                out.push(format!("{} = {}[{}];", res, r.name, w));
                out.push(format!("{}[{}] = {};", r.name, w, combine(res)));
            }
        }
        return out;
    }
    let direct = |f: &str| format!("{} = {}({}, {});", res, f, at, v);
    match op {
        Op::Load => out.push(format!("{} = atomicLoad({});", res, at)),
        Op::Store => out.push(format!("atomicStore({}, {});", at, v)),
        Op::Xchg => out.push(direct("atomicExchange")),
        Op::Add => out.push(direct("atomicAdd")),
        Op::Sub => out.push(direct("atomicSub")),
        Op::And => out.push(direct("atomicAnd")),
        Op::Or => out.push(direct("atomicOr")),
        Op::Xor => out.push(direct("atomicXor")),
        Op::Min { signed: false } => out.push(direct("atomicMin")),
        Op::Max { signed: false } => out.push(direct("atomicMax")),
        Op::Min { signed: true } | Op::Max { signed: true } => {
            // A read-modify-write through compare-exchange: retried until the word did not change between the read
            // and the exchange.
            let x = cx.fresh();
            out.push(format!("{} = atomicLoad({});", res, at));
            out.push("loop {".into());
            out.push(format!("    let {} = atomicCompareExchangeWeak({}, {}, {});", x, at, res, combine(res)));
            out.push(format!("    if {}.exchanged {{", x));
            out.push("        break;".into());
            out.push("    }".into());
            out.push(format!("    {} = {}.old_value;", res, x));
            out.push("}".into());
        }
        Op::CmpXchg => {
            // A strong compare-exchange: a spurious failure (not exchanged, the same value) is retried.
            let x = cx.fresh();
            out.push("loop {".into());
            out.push(format!("    let {} = atomicCompareExchangeWeak({}, {}, {});", x, at, e, v));
            out.push(format!("    if {}.exchanged || {}.old_value != {} {{", x, x, e));
            out.push(format!("        {} = {}.old_value;", res, x));
            out.push("        break;".into());
            out.push("    }".into());
            out.push("}".into());
        }
    }
    out
}

/// Run `body` for each candidate region of a pointer: directly for one, under a test of the tag for several. A
/// null candidate is skipped (an access through a null pointer is undefined behavior).
fn per_region(
    cx: &mut dyn RuleCx,
    p: &crate::mem::access::Ptr,
    mut body: impl FnMut(&mut dyn RuleCx, &Region, &str) -> Vec<String>,
) {
    let w = match p.konst {
        Some(k) => format!("{}u", (k >> 2) as u32),
        None => format!("({} >> 2u)", p.off),
    };
    let several = p.regions.len() > 1;
    for &r in &p.regions {
        if r == NULL {
            continue;
        }
        let reg = cx.region(r).clone();
        let lines = body(cx, &reg, &w);
        if several {
            let tag = p.tag.as_ref().expect("a pointer into several regions has a tag");
            cx.line(format!("if {} == {}u {{", tag, reg.tag));
            for l in lines {
                cx.line(format!("    {}", l));
            }
            cx.line("}".into());
        } else {
            for l in lines {
                cx.line(l);
            }
        }
    }
}

impl CallRule for Atomics {
    fn handles(&self, name: &str) -> bool {
        parse(name).is_some()
    }

    fn effects(&self, call: &CallSite<'_>) -> Vec<(usize, Effect)> {
        let Some(n) = parse(call.name) else { return Vec::new() };
        let mut out = vec![(0, Effect::Atomic)];
        if n.op == Op::CmpXchg {
            out.push((1, Effect::Read));
            out.push((1, Effect::Write { bytes: 4, align: 4 }));
        }
        out
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal> {
        let n = parse(call.name).expect("handles() accepted the name");
        let (value, expected, orders, scope, count) = layout(n.op);
        let refuse = |cx: &dyn RuleCx, detail: String| {
            tracked::refuse("sync-operand", call.name.to_string(), loc(cx, call), detail)
        };
        if call.args.len() != count {
            return Err(tracked::refuse(
                "entry-metadata",
                call.name.to_string(),
                loc(cx, call),
                format!("{} operands where the target's front end passes {}", call.args.len(), count),
            ));
        }
        for &o in orders {
            match int_const(cx, call.args[o]) {
                Some(0) => {}
                Some(k) => {
                    return Err(refuse(
                        cx,
                        format!(
                            "memory order {}: only memory_order_relaxed (0) is accepted by the A10X \
                             (supportsMemoryOrderAtomics = false)",
                            k
                        ),
                    ));
                }
                None => return Err(refuse(cx, "a memory order that is not a constant".into())),
            }
        }
        let want_scope = if n.local { 1 } else { 2 };
        match int_const(cx, call.args[scope]) {
            Some(s) if s == want_scope => {}
            other => {
                return Err(refuse(
                    cx,
                    format!(
                        "scope {:?}: the target's front end passes {} for {} atomics",
                        other,
                        want_scope,
                        if n.local { "threadgroup" } else { "device" }
                    ),
                ));
            }
        }
        if int_const(cx, call.args[count - 1]).is_none() {
            return Err(refuse(cx, "a volatile flag that is not a constant".into()));
        }
        let p = cx.pointer(call.args[0])?;
        let v = match value {
            Some(k) => {
                let e = cx.operand(call.args[k])?;
                let t = cx.fresh();
                cx.line(format!("let {} = {};", t, e));
                t
            }
            None => String::new(),
        };
        // The expected value of a compare-exchange, read from its thread memory.
        let e = match expected {
            Some(k) => {
                let ep = cx.pointer(call.args[k])?;
                let t = cx.fresh();
                cx.line(format!("var {}: u32;", t));
                per_region(cx, &ep, |_, r, w| vec![format!("{} = {};", t, read_word(r, w))]);
                t
            }
            None => String::new(),
        };
        // The old value; a store has none.
        let res = if n.op == Op::Store { String::new() } else { cx.fresh() };
        if n.op != Op::Store {
            cx.line(format!("var {}: u32 = 0u;", res));
        }
        let op = n.op;
        per_region(cx, &p, |cx, r, w| op_lines(cx, r, w, op, &v, &e, &res));
        if let Some(k) = expected {
            let ep = cx.pointer(call.args[k])?;
            per_region(cx, &ep, |_, r, w| vec![write_word(r, w, &res)]);
        }
        Ok(if n.op == Op::Store { None } else { Some(res) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse_as_the_target_spells_them() {
        assert_eq!(parse("air.atomic.global.add.s.i32"), Some(Name { local: false, op: Op::Add }));
        assert_eq!(parse("air.atomic.local.min.s.i32"), Some(Name { local: true, op: Op::Min { signed: true } }));
        assert_eq!(parse("air.atomic.global.max.u.i32"), Some(Name { local: false, op: Op::Max { signed: false } }));
        assert_eq!(parse("air.atomic.global.cmpxchg.weak.i32"), Some(Name { local: false, op: Op::CmpXchg }));
        assert_eq!(parse("air.atomic.local.xchg.i32"), Some(Name { local: true, op: Op::Xchg }));
        assert_eq!(parse("air.atomic.global.load.i32"), Some(Name { local: false, op: Op::Load }));
        // Other widths and types belong to the device-unsupported families.
        assert_eq!(parse("air.atomic.global.add.f32"), None);
        assert_eq!(parse("air.atomic.global.max.u.i64"), None);
        assert_eq!(parse("air.atomic.global.min.i32"), None);
        assert_eq!(parse("air.atomic.fence"), None);
        for n in ["air.atomic.global.add.s.i32", "air.atomic.local.cmpxchg.weak.i32", "air.atomic.global.store.i32"] {
            assert_eq!(tracked::family_of(n), Some("atomic-int32"), "{}", n);
        }
    }
}
