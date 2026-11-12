//! Function constant specialization.
//!
//! A module's function constants are listed in `!air.function_constants`: `!{ptr addrspace(2) @g, !"type", !"name",
//! i32 index}`, where `@g` is an `externally_initialized` constant in section `air.fc_initializer` whose initializer
//! is `undef`. Specialization gives `@g` the value as its initializer and replaces `air.is_function_constant_defined(@g)`
//! by whether it has one. A constant without a value keeps its `undef` initializer and is not defined.
//!
//! What Metal does, measured on this Mac's Metal (macOS 27.0.1, Apple M4) through `newFunctionWithName:
//! constantValues:error:` and the specialized bitcode the resulting function holds (the probes are in
//! `tests/canon/metal/function_constants.swift`, their output in `tests/canon/data/function-constants.txt`):
//!
//! - A value whose `MTLDataType` differs from the constant's type is an error (`MTLLibraryErrorCompileFailure`:
//!   "constant fc_i (0) is of type MTLDataTypeInt but value found has type MTLDataTypeUInt"), with no conversion
//!   between types of the same size. This is [`RefusalKind::RejectedByDevice`].
//! - A constant the function uses without a value is not an error: the specialized code computes with `undef` (a
//!   store of it is deleted), which is what the unchanged initializer gives here.
//! - A `bool` constant set to a byte other than 0 or 1 is read as `true` by a scalar `bool` and by its lowest bit in
//!   a `bool2` (the front end's `!range` and `and 1` on the loads, as Metal's optimizer happened to resolve them);
//!   with `!range [0, 2)` such a load is poison in LLVM's semantics. Such values are outside the API's `bool` and are
//!   refused rather than resolved differently from Metal.
//! - An argument with `[[function_constant(p)]]` is kept exactly when `p`'s value has a nonzero bit: an `int` 2 or
//!   -1, a `bool2` (false, true), a `float` -0.0, NaN or the smallest subnormal keep it; zero, `+0.0` and an unset
//!   predicate drop it. The front end stores `air.normalize_function_constant_predicate(p)` into a
//!   `__metal_implicit_fc_pred_N` global that the argument's metadata names; the interpreter defines that function
//!   as the identity, which with [`predicate_present`] reproduces every measured case.

use std::collections::HashSet;

use crate::api::Refusal;
use crate::api::{FunctionConstant, RefusalKind};
use crate::interp::memory::{Byte, Layout};
use crate::ir::layout::DataLayout;
use crate::ir::{ConstKind, GlobalId, Metadata, Module, Op, Type, Value};

use super::util::{i1_const, remove_inst, replace_uses};
use super::{refusal, refusal_at, values};

/// One `!air.function_constants` record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantRecord {
    /// Index of the `air.fc_initializer` global.
    pub global: u32,
    /// The MSL type name (`int`, `bool2`, `half4`, ...).
    pub type_name: String,
    pub name: String,
    pub index: u64,
}

/// The function constants a module declares.
pub fn records(m: &Module) -> Vec<ConstantRecord> {
    let mut out = Vec::new();
    for nm in m.named_metadata.iter().filter(|n| n.name == "air.function_constants") {
        for &op in &nm.ops {
            let Metadata::Tuple { ops, .. } = m.md(op) else { continue };
            let get = |k: usize| ops.get(k).copied().flatten().map(|x| m.md(x));
            let global = match get(0) {
                Some(Metadata::Value { value: Value::Global(GlobalId::Var(g)), .. }) => *g,
                _ => continue,
            };
            let s = |k: usize| match get(k) {
                Some(Metadata::String(b)) => Some(String::from_utf8_lossy(b).into_owned()),
                _ => None,
            };
            let index = match get(3) {
                Some(Metadata::Value { value: Value::Const(c), .. }) => match &m.constant(*c).kind {
                    ConstKind::Int(w) => w[0],
                    _ => continue,
                },
                _ => continue,
            };
            let (Some(type_name), Some(name)) = (s(1), s(2)) else { continue };
            out.push(ConstantRecord { global, type_name, name, index });
        }
    }
    out
}

/// The `MTLDataType` of an MSL scalar or vector type name (`MTLDataType.h`: Float = 3, Half = 16, Int = 29,
/// UInt = 33, Short = 37, UShort = 41, Char = 45, UChar = 49, Bool = 53, Long = 81, ULong = 85, BFloat = 121, each
/// followed by its 2-, 3- and 4-component vectors).
pub fn data_type(type_name: &str) -> Option<u64> {
    let (base, n) = match type_name.strip_suffix(|c: char| matches!(c, '2' | '3' | '4')) {
        Some(b) => (b, type_name.as_bytes()[type_name.len() - 1] - b'1'),
        None => (type_name, 0),
    };
    let first = match base {
        "float" => 3,
        "half" => 16,
        "int" => 29,
        "uint" => 33,
        "short" => 37,
        "ushort" => 41,
        "char" => 45,
        "uchar" => 49,
        "bool" => 53,
        "long" => 81,
        "ulong" => 85,
        "bfloat" => 121,
        _ => return None,
    };
    Some(first + n as u64)
}

/// Whether an argument predicated on a function constant is kept: a bit of the predicate value is set (undef bits
/// count as clear, as an unset predicate drops the argument).
pub fn predicate_present(bytes: &[Byte]) -> bool {
    bytes.iter().any(|b| !b.poison && b.v & !b.undef != 0)
}

/// Gives the module's function constants their values and resolves `air.is_function_constant_defined`. Returns the
/// names of the globals that got values.
#[allow(clippy::result_large_err)]
pub fn specialize(m: &mut Module, constants: &[FunctionConstant]) -> Result<HashSet<String>, Refusal> {
    let recs = records(m);
    let dl =
        DataLayout::parse(&m.datalayout).map_err(|e| refusal(RefusalKind::RejectedByDevice, "datalayout", "", &e))?;
    let mut defined = HashSet::new();
    for r in &recs {
        let value = constants
            .iter()
            .find(|c| c.name.as_deref() == Some(r.name.as_str()))
            .or_else(|| constants.iter().find(|c| c.name.is_none() && c.index == r.index));
        let g = r.global as usize;
        let gname = m.globals[g].common.name.clone();
        m.globals[g].externally_initialized = false;
        let Some(c) = value else { continue };
        let want = data_type(&r.type_name).ok_or_else(|| {
            refusal(
                RefusalKind::NotImplemented,
                &format!("function constant {} of type {}", r.name, r.type_name),
                "",
                "an MSL type the specialization does not know",
            )
        })?;
        if c.data_type != want {
            return Err(refusal(
                RefusalKind::RejectedByDevice,
                &format!("function constant {} ({})", r.name, r.index),
                "",
                &format!("the constant is of MTLDataType {} but the value has MTLDataType {}", want, c.data_type),
            ));
        }
        let ty = m.globals[g].value_type;
        let layout = Layout { dl: &dl, types: &m.types };
        let size = layout.store_size(ty).unwrap_or(0) as usize;
        if c.value.len() < size {
            return Err(refusal(
                RefusalKind::RejectedByDevice,
                &format!("function constant {} ({})", r.name, r.index),
                "",
                &format!("the value has {} bytes; the constant's type stores {}", c.value.len(), size),
            ));
        }
        let elem = m.types.scalar(ty);
        if r.type_name.starts_with("bool")
            && matches!(m.types.get(elem), Type::Int(8))
            && c.value[..size].iter().any(|&b| b > 1)
        {
            return Err(refusal(
                RefusalKind::NotImplemented,
                &format!("function constant {} ({})", r.name, r.index),
                "",
                "a bool value with a byte other than 0 or 1, which Metal resolves inconsistently",
            ));
        }
        let bytes: Vec<Byte> = c.value[..size].iter().map(|&b| Byte::data(b)).collect();
        let v = layout.decode(ty, &bytes).map_err(|e| refusal(RefusalKind::NotImplemented, &gname, "", &e))?;
        let mem = crate::interp::Memory::new();
        let init =
            values::to_const(m, &mem, ty, &v).map_err(|e| refusal(RefusalKind::NotImplemented, &gname, "", &e))?;
        m.globals[g].initializer = Some(init);
        defined.insert(gname);
    }
    // air.is_function_constant_defined(@g) everywhere.
    let fc_globals: HashSet<u32> = recs.iter().map(|r| r.global).collect();
    let Some(query) = m.function_by_name("air.is_function_constant_defined") else { return Ok(defined) };
    for f in 0..m.functions.len() {
        let calls: Vec<_> = m.functions[f]
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter().copied())
            .filter(|&i| matches!(&m.functions[f].inst(i).op, Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } if *c as usize == query))
            .collect();
        for i in calls {
            let Op::Call { args, .. } = &m.functions[f].inst(i).op else { unreachable!() };
            let g = match args.first() {
                Some(Value::Global(GlobalId::Var(g))) if fc_globals.contains(g) => *g,
                _ => {
                    return Err(refusal_at(
                        m,
                        f,
                        i,
                        RefusalKind::NotImplemented,
                        "air.is_function_constant_defined",
                        "its operand is not a function constant global",
                    ));
                }
            };
            let is = defined.contains(&m.globals[g as usize].common.name);
            let c = i1_const(m, is);
            replace_uses(&mut m.functions[f], Value::Inst(i), c);
            remove_inst(&mut m.functions[f], i);
        }
    }
    Ok(defined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_types_follow_the_header() {
        assert_eq!(data_type("float"), Some(3));
        assert_eq!(data_type("half4"), Some(19));
        assert_eq!(data_type("int"), Some(29));
        assert_eq!(data_type("uint3"), Some(35));
        assert_eq!(data_type("bool"), Some(53));
        assert_eq!(data_type("bool2"), Some(54));
        assert_eq!(data_type("ulong4"), Some(88));
        assert_eq!(data_type("bfloat2"), Some(122));
        assert_eq!(data_type("float2x2"), None);
    }

    #[test]
    fn predicates_count_any_set_bit() {
        let b = |v: &[u8]| v.iter().map(|&x| Byte::data(x)).collect::<Vec<_>>();
        assert!(predicate_present(&b(&[0, 0, 0, 0x80])));
        assert!(!predicate_present(&b(&[0, 0, 0, 0])));
        assert!(predicate_present(&b(&[0, 1])));
        assert!(!predicate_present(&[Byte::UNDEF]));
    }
}
