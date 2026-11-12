//! `TBAAVerifier` (`llvm/lib/IR/Verifier.cpp`, release 21.1.8): the bitcode reader strips every `!tbaa` attachment of
//! a module when one of them fails this check. Results are cached per node across the module, as LLVM's verifier
//! object lives as long as the reader.

use std::collections::{HashMap, HashSet};

use crate::ir::*;

#[derive(Default)]
pub(crate) struct Verifier {
    base_nodes: HashMap<MdId, (bool, u32)>,
    scalar_nodes: HashMap<MdId, bool>,
}

/// An integer constant operand: (bit width, value).
fn const_int(m: &Module, op: Option<MdId>) -> Option<(u32, u64)> {
    match op.map(|o| m.md(o)) {
        Some(Metadata::Value { ty, value: Value::Const(c) }) => match &m.constant(*c).kind {
            ConstKind::Int(w) => {
                let bits = m.types.int_bits(*ty)?;
                if bits > 64 {
                    return None;
                }
                Some((bits, w[0]))
            }
            _ => None,
        },
        _ => None,
    }
}

fn node_ops(m: &Module, id: MdId) -> Option<&Vec<Option<MdId>>> {
    match m.md(id) {
        Metadata::Tuple { ops, .. } => Some(ops),
        _ => None,
    }
}

fn is_node(m: &Module, op: Option<MdId>) -> Option<MdId> {
    let id = op?;
    matches!(m.md(id), Metadata::Tuple { .. }).then_some(id)
}

fn is_string(m: &Module, op: Option<MdId>) -> bool {
    matches!(op.map(|o| m.md(o)), Some(Metadata::String(_)))
}

fn is_root(m: &Module, id: MdId) -> bool {
    node_ops(m, id).map(|o| o.len() < 2).unwrap_or(false)
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

impl Verifier {
    /// `IsScalarTBAANodeImpl`, following the parent chain without recursion.
    fn is_scalar_impl(m: &Module, id: MdId, visited: &mut HashSet<MdId>) -> bool {
        let mut cur = id;
        loop {
            let ops = match node_ops(m, cur) {
                Some(o) => o,
                None => return false,
            };
            if ops.len() != 2 && ops.len() != 3 {
                return false;
            }
            if !is_string(m, ops[0]) {
                return false;
            }
            if ops.len() == 3 {
                match const_int(m, ops[2]) {
                    Some((_, 0)) => {}
                    _ => return false,
                }
            }
            let parent = match is_node(m, ops[1]) {
                Some(p) => p,
                None => return false,
            };
            if !visited.insert(parent) {
                return false;
            }
            if is_root(m, parent) {
                return true;
            }
            cur = parent;
        }
    }

    fn is_valid_scalar(&mut self, m: &Module, id: MdId) -> bool {
        if let Some(&r) = self.scalar_nodes.get(&id) {
            return r;
        }
        let r = Self::is_scalar_impl(m, id, &mut HashSet::new());
        self.scalar_nodes.insert(id, r);
        r
    }

    fn verify_base_node(&mut self, m: &Module, id: MdId, new_format: bool) -> (bool, u32) {
        let n = node_ops(m, id).map(|o| o.len()).unwrap_or(0);
        if n < 2 {
            return (true, u32::MAX);
        }
        if let Some(&r) = self.base_nodes.get(&id) {
            return r;
        }
        let r = self.verify_base_node_impl(m, id, new_format);
        self.base_nodes.insert(id, r);
        r
    }

    fn verify_base_node_impl(&mut self, m: &Module, id: MdId, new_format: bool) -> (bool, u32) {
        const INVALID: (bool, u32) = (true, u32::MAX);
        let ops = node_ops(m, id).unwrap().clone();
        if ops.len() == 2 {
            return if self.is_valid_scalar(m, id) { (false, 0) } else { INVALID };
        }
        if new_format {
            if ops.len() % 3 != 0 {
                return INVALID;
            }
            if const_int(m, ops[1]).is_none() {
                return INVALID;
            }
        } else {
            if ops.len() % 2 != 1 {
                return INVALID;
            }
            if !is_string(m, ops[0]) {
                return INVALID;
            }
        }
        let mut failed = false;
        let mut prev: Option<u64> = None;
        let mut width = u32::MAX;
        let (first, per) = if new_format { (3, 3) } else { (1, 2) };
        let mut idx = first;
        while idx < ops.len() {
            if is_node(m, ops[idx]).is_none() {
                failed = true;
                idx += per;
                continue;
            }
            let (bits, value) = match const_int(m, ops.get(idx + 1).copied().flatten()) {
                Some(x) => x,
                None => {
                    failed = true;
                    idx += per;
                    continue;
                }
            };
            if width == u32::MAX {
                width = bits;
            }
            if bits != width {
                failed = true;
                idx += per;
                continue;
            }
            if let Some(p) = prev {
                if p > value {
                    failed = true;
                }
            }
            prev = Some(value);
            if new_format && const_int(m, ops.get(idx + 2).copied().flatten()).is_none() {
                failed = true;
            }
            idx += per;
        }
        if failed { INVALID } else { (false, width) }
    }

    /// `getFieldNodeFromTBAABaseNode`; `None` when no field contains the offset.
    fn field_node(m: &Module, id: MdId, offset: &mut u64, bits: u32, new_format: bool) -> Option<MdId> {
        let ops = node_ops(m, id)?;
        if ops.len() == 2 {
            return is_node(m, ops[1]);
        }
        let (first, per) = if new_format { (3, 3) } else { (1, 2) };
        let mut idx = first;
        while idx < ops.len() {
            let (_, v) = const_int(m, ops.get(idx + 1).copied().flatten())?;
            if v > *offset {
                if idx == first {
                    return None;
                }
                let prev = idx - per;
                let (_, pv) = const_int(m, ops[prev + 1])?;
                *offset = offset.wrapping_sub(pv) & mask(bits);
                return is_node(m, ops[prev]);
            }
            idx += per;
        }
        let last = ops.len() - per;
        let (_, lv) = const_int(m, ops[last + 1])?;
        *offset = offset.wrapping_sub(lv) & mask(bits);
        is_node(m, ops[last])
    }

    /// `visitTBAAMetadata` for the `!tbaa` attachment `tag` of instruction `inst` of function `f`.
    pub fn visit(&mut self, m: &mut Module, f: usize, inst: usize, tag: MdId) -> bool {
        let ops = match node_ops(m, tag) {
            Some(o) => o.clone(),
            None => return false,
        };
        if ops.is_empty() {
            return false;
        }
        if !matches!(
            m.functions[f].insts[inst].op,
            Op::Load { .. }
                | Op::Store { .. }
                | Op::Call { .. }
                | Op::VaArg { .. }
                | Op::AtomicRmw { .. }
                | Op::CmpXchg { .. }
        ) {
            return false;
        }
        if !(is_node(m, ops[0]).is_some() && ops.len() >= 3) {
            return false;
        }
        let mut base = is_node(m, ops[0]);
        let access = is_node(m, ops[1]);
        let new_format = match access {
            Some(a) => node_ops(m, a).map(|o| o.len() >= 3 && is_node(m, o[0]).is_some()).unwrap_or(false),
            None => false,
        };
        if new_format {
            if ops.len() != 4 && ops.len() != 5 {
                return false;
            }
            if const_int(m, ops[3]).is_none() {
                return false;
            }
        } else if ops.len() >= 5 {
            return false;
        }
        let imm = if new_format { 4 } else { 3 };
        if ops.len() == imm + 1 {
            match const_int(m, ops[imm]) {
                Some((_, 0)) | Some((_, 1)) => {}
                _ => return false,
            }
        }
        let access = match (base, access) {
            (Some(_), Some(a)) => a,
            _ => return false,
        };
        if !new_format && !self.is_valid_scalar(m, access) {
            return false;
        }
        let (offset_bits, mut offset) = match const_int(m, ops[2]) {
            Some(x) => x,
            None => return false,
        };
        let mut seen_access = false;
        let mut path = HashSet::new();
        while let Some(b) = base {
            if is_root(m, b) {
                break;
            }
            if !path.insert(b) {
                return false;
            }
            let (invalid, width) = self.verify_base_node(m, b, new_format);
            if invalid {
                return false;
            }
            seen_access |= b == access;
            if (self.is_valid_scalar(m, b) || b == access) && offset != 0 {
                return false;
            }
            if !(width == offset_bits || (width == 0 && offset == 0) || (new_format && width == u32::MAX)) {
                return false;
            }
            if new_format && seen_access {
                break;
            }
            base = Self::field_node(m, b, &mut offset, offset_bits, new_format);
            if base.is_none() {
                // LLVM reports the failure and ends the walk.
                break;
            }
        }
        seen_access
    }
}
