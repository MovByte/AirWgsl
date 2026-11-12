//! Metadata (`MetadataLoader.cpp`): strings, values, tuples, named metadata, kinds and attachments.
//!
//! Debug-information records are not implemented: a module that has them fails with an error naming the record.

use std::collections::HashMap;

use super::codes::*;
use super::constants::Use;
use super::{Reader, Result, Slot, FWD};
use crate::bitstream::Entry;
use crate::ir::*;

/// Metadata of a block as read, before forward references are resolved.
#[derive(Clone, Debug)]
enum RawMd {
    Ready(MdId),
    /// Operands are bitcode metadata ids plus one; zero is null.
    Tuple { distinct: bool, ops: Vec<u64> },
    ArgList(Vec<u64>),
}

#[derive(Default)]
pub(crate) struct MetadataState {
    /// Bitcode metadata id -> metadata.
    pub list: Vec<MdId>,
    /// Bitcode metadata kind id -> module kind id.
    pub kinds: HashMap<u64, u32>,
    pub struct_unique_id: u32,
    /// Instructions whose `LocalAsMetadata` refers to a value not yet read: (metadata, function, value number).
    pub pending_local: Vec<(MdId, u32, u32)>,
}

impl<'a> Reader<'a> {
    pub(super) fn read_metadata_kinds(&mut self) -> Result<()> {
        self.cur.enter_block(METADATA_KIND_BLOCK_ID)?;
        loop {
            match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block inside the metadata kind block"),
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    if r.code != METADATA_KIND {
                        return self.malformed(format!("unknown metadata kind record {}", r.code));
                    }
                    self.metadata_kind_record(&r.ops)?;
                }
            }
        }
    }

    fn metadata_kind_record(&mut self, ops: &[u64]) -> Result<()> {
        if ops.len() < 2 {
            return self.malformed("invalid METADATA_KIND record");
        }
        let name = self.record_string(&ops[1..])?;
        // `getMDKindID`: fixed kinds first, others numbered in order of first use.
        let id = match self.m.md_kinds.iter().position(|k| *k == name) {
            Some(i) => i as u32,
            None => {
                self.m.md_kinds.push(name);
                (self.m.md_kinds.len() - 1) as u32
            }
        };
        if self.md.kinds.insert(ops[0], id).is_some() {
            return self.malformed("conflicting METADATA_KIND records");
        }
        Ok(())
    }

    /// Reads a metadata block: module-level when `function` is `None`.
    pub(super) fn read_metadata_block(&mut self, function: Option<u32>) -> Result<()> {
        self.cur.enter_block(METADATA_BLOCK_ID)?;
        let first = self.md.list.len();
        let mut raw: Vec<RawMd> = Vec::new();
        // Lazy-loading index: positions of the records that define metadata ids, checked against METADATA_INDEX.
        let mut record_starts: Vec<u64> = Vec::new();
        let mut index_from: Option<(u64, u64)> = None;
        let mut index_checked = false;
        let mut named: Vec<(String, Vec<u64>)> = Vec::new();
        let mut decl_attachments: Vec<Vec<u64>> = Vec::new();
        loop {
            let start = self.cur.position();
            let a = match self.cur.advance()? {
                Entry::EndBlock => break,
                Entry::SubBlock(_) => return self.malformed("sub-block inside a metadata block"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            let ops = &r.ops;
            match r.code {
                METADATA_STRINGS => {
                    if ops.len() != 2 {
                        return self.malformed("invalid metadata strings record");
                    }
                    let blob = match r.blob {
                        Some(b) => b,
                        None => return self.malformed("metadata strings record without a blob"),
                    };
                    let count = ops[0];
                    let offset = ops[1] as usize;
                    if count == 0 {
                        return self.malformed("metadata strings record with no strings");
                    }
                    if offset > blob.len() {
                        return self.malformed("metadata strings record with a corrupt offset");
                    }
                    let mut lengths = crate::bitstream::BitReader::new(&blob[..offset]);
                    let mut chars = &blob[offset..];
                    for _ in 0..count {
                        if lengths.at_end() {
                            return self.malformed("metadata strings record: bad length");
                        }
                        let n = lengths.read_vbr32(6)? as usize;
                        if chars.len() < n {
                            return self.malformed("metadata strings record: truncated characters");
                        }
                        let id = self.m.intern_md(Metadata::String(chars[..n].to_vec()));
                        chars = &chars[n..];
                        raw.push(RawMd::Ready(id));
                    }
                }
                METADATA_STRING_OLD => {
                    let bytes = self.record_bytes(ops)?;
                    if bytes.starts_with(b"llvm.vectorizer.") {
                        return self.unsupported("llvm.vectorizer loop metadata (its upgrade is not implemented)");
                    }
                    let id = self.m.intern_md(Metadata::String(bytes));
                    raw.push(RawMd::Ready(id));
                }
                METADATA_VALUE => {
                    if ops.len() != 2 {
                        return self.malformed("invalid metadata value record");
                    }
                    record_starts.push(start);
                    let ty = self.ty(ops[0])?;
                    if self.m.types.is_metadata(ty) || self.m.types.is_void(ty) {
                        return self.malformed("invalid metadata value type");
                    }
                    let id = self.metadata_value(ops[1], ty, function)?;
                    raw.push(RawMd::Ready(id));
                }
                METADATA_NODE | METADATA_DISTINCT_NODE => {
                    record_starts.push(start);
                    raw.push(RawMd::Tuple { distinct: r.code == METADATA_DISTINCT_NODE, ops: ops.clone() });
                }
                METADATA_ARG_LIST => {
                    record_starts.push(start);
                    raw.push(RawMd::ArgList(ops.clone()));
                }
                METADATA_NAME => {
                    let name = self.record_string(ops)?;
                    // The NAMED_NODE record must follow directly.
                    let a = match self.cur.advance()? {
                        Entry::Record(a) => a,
                        _ => return self.malformed("METADATA_NAME not followed by METADATA_NAMED_NODE"),
                    };
                    let n = self.cur.read_record(a)?;
                    if n.code != METADATA_NAMED_NODE {
                        return self.malformed("METADATA_NAME not followed by METADATA_NAMED_NODE");
                    }
                    if function.is_some() {
                        return self.malformed("named metadata in a function metadata block");
                    }
                    named.push((name, n.ops));
                }
                METADATA_NAMED_NODE => return self.malformed("METADATA_NAMED_NODE without METADATA_NAME"),
                METADATA_KIND => self.metadata_kind_record(ops)?,
                METADATA_INDEX_OFFSET => {
                    if ops.len() != 2 || index_from.is_some() || function.is_some() {
                        return self.malformed("invalid metadata index offset record");
                    }
                    let offset = ops[0] + (ops[1] << 32);
                    let after = self.cur.position();
                    index_from = Some((after, after + offset));
                    record_starts.clear();
                }
                METADATA_INDEX => {
                    let (after, at) = match index_from {
                        Some(x) => x,
                        None => return self.malformed("metadata index without an index offset"),
                    };
                    if at != start || index_checked {
                        return self.malformed("metadata index offset does not point at the metadata index");
                    }
                    let mut pos = after;
                    let mut expected = Vec::with_capacity(ops.len());
                    for d in ops {
                        pos = pos.wrapping_add(*d);
                        expected.push(pos);
                    }
                    if expected != record_starts {
                        return self.malformed("metadata index does not match the metadata records");
                    }
                    index_checked = true;
                }
                METADATA_GLOBAL_DECL_ATTACHMENT => {
                    if ops.len() % 2 == 0 {
                        return self.malformed("invalid global declaration attachment record");
                    }
                    decl_attachments.push(ops.clone());
                }
                METADATA_OLD_NODE | METADATA_OLD_FN_NODE => {
                    return self.unsupported("metadata records of LLVM before 3.6 (OLD_NODE, OLD_FN_NODE)");
                }
                c => {
                    if let Some(name) = debug_metadata_record_name(c) {
                        return self.unsupported(format!("debug information metadata ({})", name));
                    }
                    return self.malformed(format!("unknown metadata record code {}", c));
                }
            }
        }
        if index_from.is_some() && !index_checked {
            return self.malformed("metadata index offset without a metadata index");
        }
        self.resolve_metadata(first, raw)?;
        for (name, ops) in named {
            let mut nodes = Vec::new();
            for id in ops {
                let md = self.md_ref(id)?;
                if !matches!(self.m.md(md), Metadata::Tuple { .. }) {
                    return self.malformed("named metadata operand is not a node");
                }
                nodes.push(md);
            }
            match self.m.named_metadata.iter_mut().find(|n| n.name == name) {
                Some(n) => n.ops.extend(nodes),
                None => self.m.named_metadata.push(NamedMetadata { name, ops: nodes }),
            }
        }
        for ops in decl_attachments {
            let target = match self.values.get(ops[0] as usize).map(|e| e.slot) {
                Some(Slot::Global(g @ (GlobalId::Var(_) | GlobalId::Function(_)))) => g,
                Some(Slot::Global(_)) => continue,
                _ => return self.malformed("global declaration attachment for a value that is not global"),
            };
            let pairs = self.attachment_pairs(&ops[1..])?;
            for (kind, md) in pairs {
                match target {
                    GlobalId::Var(i) => set_attachment(&mut self.m.globals[i as usize].md, kind, md),
                    GlobalId::Function(i) => set_attachment(&mut self.m.functions[i as usize].md, kind, md),
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }

    /// A `ValueAsMetadata` for value number `id` of type `ty`.
    fn metadata_value(&mut self, id: u64, ty: TypeId, function: Option<u32>) -> Result<MdId> {
        let slot = self.values.get(id as usize).map(|e| e.slot);
        let v = match slot {
            Some(Slot::Global(g)) => Value::Global(g),
            Some(Slot::Const(c)) => self.materialize_const_use(c, Use::Metadata)?,
            Some(Slot::Arg(a)) => {
                let f = match function {
                    Some(f) => f,
                    None => return self.malformed("module metadata refers to a function argument"),
                };
                let ty2 = self.m.functions[f as usize].args[a as usize].ty;
                if ty2 != ty {
                    return self.malformed("metadata value has the wrong type");
                }
                return Ok(self.m.intern_md(Metadata::Local { function: f, ty, value: Value::Arg(a) }));
            }
            Some(Slot::Inst(i)) => {
                let f = match function {
                    Some(f) => f,
                    None => return self.malformed("module metadata refers to an instruction"),
                };
                return Ok(self.m.intern_md(Metadata::Local { function: f, ty, value: Value::Inst(i) }));
            }
            None => {
                // A forward reference to an instruction of the function being read.
                let f = match function {
                    Some(f) => f,
                    None => return self.malformed("metadata refers to an undefined value"),
                };
                if id >= FWD as u64 {
                    return self.malformed("metadata value number out of range");
                }
                let md = self.m.push_md(Metadata::Local { function: f, ty, value: Value::Inst(InstId(FWD | id as u32)) });
                self.md.pending_local.push((md, f, id as u32));
                return Ok(md);
            }
        };
        let vt = self.m.module_value_type(v).unwrap();
        if vt != ty {
            return self.malformed("metadata value has the wrong type");
        }
        Ok(self.m.intern_md(Metadata::Value { ty, value: v }))
    }

    /// Bitcode metadata id -> metadata.
    pub fn md_ref(&self, id: u64) -> Result<MdId> {
        match self.md.list.get(id as usize) {
            Some(&m) if m.0 != u32::MAX => Ok(m),
            _ => self.malformed(format!("invalid metadata id {}", id)),
        }
    }

    /// Turns a block's raw metadata into pooled metadata, uniquing non-distinct tuples as LLVM does.
    fn resolve_metadata(&mut self, first: usize, raw: Vec<RawMd>) -> Result<()> {
        const UNSET: MdId = MdId(u32::MAX);
        let n = raw.len();
        self.md.list.resize(first + n, UNSET);
        // Ready entries and distinct tuples get their ids first: distinct tuples are never merged, so their identity
        // does not depend on their operands.
        for (i, r) in raw.iter().enumerate() {
            match r {
                RawMd::Ready(id) => self.md.list[first + i] = *id,
                RawMd::Tuple { distinct: true, .. } => {
                    self.md.list[first + i] = self.m.push_md(Metadata::Tuple { distinct: true, ops: Vec::new() });
                }
                _ => {}
            }
        }
        let resolve_op = |list: &Vec<MdId>, op: u64| -> Option<Option<MdId>> {
            if op == 0 {
                return Some(None);
            }
            match list.get((op - 1) as usize) {
                Some(&m) if m != UNSET => Some(Some(m)),
                _ => None,
            }
        };
        // Uniqued tuples and argument lists: depth-first, operands before users.
        for root in 0..n {
            if self.md.list[first + root] != UNSET {
                continue;
            }
            let mut stack = vec![root];
            let mut on_stack = vec![false; n];
            on_stack[root] = true;
            while let Some(&i) = stack.last() {
                let ops: Vec<u64> = match &raw[i] {
                    RawMd::Tuple { ops, .. } | RawMd::ArgList(ops) => ops.clone(),
                    RawMd::Ready(_) => unreachable!(),
                };
                let is_arglist = matches!(raw[i], RawMd::ArgList(_));
                let mut pending = None;
                for &op in &ops {
                    let target = if is_arglist { op + 1 } else { op };
                    if target == 0 {
                        continue;
                    }
                    let t = (target - 1) as usize;
                    if t >= self.md.list.len() {
                        return self.malformed(format!("metadata refers to undefined metadata {}", t));
                    }
                    if self.md.list[t] == UNSET {
                        if t < first || t >= first + n {
                            return self.malformed("metadata refers to undefined metadata");
                        }
                        if on_stack[t - first] {
                            return self.unsupported("a cycle of uniqued metadata nodes");
                        }
                        pending = Some(t - first);
                        break;
                    }
                }
                if let Some(p) = pending {
                    on_stack[p] = true;
                    stack.push(p);
                    continue;
                }
                let id = if is_arglist {
                    let mut items = Vec::new();
                    for &op in &ops {
                        let m = resolve_op(&self.md.list, op + 1).flatten().unwrap();
                        if !matches!(self.m.md(m), Metadata::Value { .. } | Metadata::Local { .. }) {
                            return self.malformed("DIArgList operand is not a value");
                        }
                        items.push(m);
                    }
                    self.m.intern_md(Metadata::ArgList(items))
                } else {
                    let resolved: Vec<Option<MdId>> = ops.iter().map(|&op| resolve_op(&self.md.list, op).unwrap()).collect();
                    self.m.intern_md(Metadata::Tuple { distinct: false, ops: resolved })
                };
                self.md.list[first + i] = id;
                on_stack[i] = false;
                stack.pop();
            }
        }
        // Distinct tuples' operands.
        for (i, r) in raw.iter().enumerate() {
            if let RawMd::Tuple { distinct: true, ops } = r {
                let mut resolved = Vec::with_capacity(ops.len());
                for &op in ops {
                    match resolve_op(&self.md.list, op) {
                        Some(x) => resolved.push(x),
                        None => return self.malformed("metadata refers to undefined metadata"),
                    }
                }
                let id = self.md.list[first + i];
                self.m.set_md(id, Metadata::Tuple { distinct: true, ops: resolved });
            }
        }
        Ok(())
    }

    /// (kind, metadata) pairs of an attachment record.
    pub(super) fn attachment_pairs(&self, ops: &[u64]) -> Result<Vec<(u32, MdId)>> {
        if ops.len() % 2 != 0 {
            return self.malformed("invalid attachment record");
        }
        let mut out = Vec::new();
        for pair in ops.chunks(2) {
            let kind = match self.md.kinds.get(&pair[0]) {
                Some(&k) => k,
                None => return self.malformed(format!("invalid metadata kind id {}", pair[0])),
            };
            let md = self.md_ref(pair[1])?;
            if !matches!(self.m.md(md), Metadata::Tuple { .. }) {
                return self.malformed("metadata attachment is not a node");
            }
            out.push((kind, md));
        }
        Ok(out)
    }

    /// The function's METADATA_ATTACHMENT block: `insts` lists the instructions in record order.
    pub(super) fn read_attachments(&mut self, f: u32, insts: &[InstId]) -> Result<()> {
        self.cur.enter_block(METADATA_ATTACHMENT_ID)?;
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block inside a metadata attachment block"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            if r.code != METADATA_ATTACHMENT {
                return self.malformed(format!("unknown metadata attachment record {}", r.code));
            }
            let ops = &r.ops;
            if ops.is_empty() {
                return self.malformed("empty metadata attachment record");
            }
            if ops.len() % 2 == 0 {
                let pairs = self.attachment_pairs(ops)?;
                for (kind, md) in pairs {
                    set_attachment(&mut self.m.functions[f as usize].md, kind, md);
                }
                continue;
            }
            let inst = match insts.get(ops[0] as usize) {
                Some(&i) => i,
                None => return self.malformed("metadata attachment for an undefined instruction"),
            };
            let pairs = self.attachment_pairs(&ops[1..])?;
            for (kind, md) in pairs {
                let md = if kind == MD_TBAA { self.upgrade_tbaa(md) } else { md };
                let inst = &mut self.m.functions[f as usize].insts[inst.0 as usize];
                if kind == 0 {
                    inst.dbg = Some(md);
                } else {
                    set_attachment(&mut inst.md, kind, md);
                }
            }
        }
    }

    /// `UpgradeTBAANode`: old scalar TBAA tags become struct-path tags.
    fn upgrade_tbaa(&mut self, md: MdId) -> MdId {
        let ops = match self.m.md(md) {
            Metadata::Tuple { ops, .. } => ops.clone(),
            _ => return md,
        };
        if ops.is_empty() {
            return md;
        }
        let first_is_node = matches!(ops[0].map(|o| self.m.md(o)), Some(Metadata::Tuple { .. }));
        if first_is_node && ops.len() >= 3 {
            return md;
        }
        let i64t = self.m.types.int(64);
        let zero = self.m.intern_const(Constant { ty: i64t, kind: ConstKind::Int(vec![0]) });
        let zero_md = self.m.intern_md(Metadata::Value { ty: i64t, value: Value::Const(zero) });
        if ops.len() == 3 {
            let scalar = self.m.intern_md(Metadata::Tuple { distinct: false, ops: vec![ops[0], ops[1]] });
            return self.m.intern_md(Metadata::Tuple { distinct: false, ops: vec![Some(scalar), Some(scalar), Some(zero_md), ops[2]] });
        }
        self.m.intern_md(Metadata::Tuple { distinct: false, ops: vec![Some(md), Some(md), Some(zero_md)] })
    }
}

/// Sets attachment `kind` (replacing one of the same kind), keeping the list sorted by kind.
pub(crate) fn set_attachment(list: &mut Vec<(u32, MdId)>, kind: u32, md: MdId) {
    match list.binary_search_by_key(&kind, |e| e.0) {
        Ok(i) => list[i].1 = md,
        Err(i) => list.insert(i, (kind, md)),
    }
}
