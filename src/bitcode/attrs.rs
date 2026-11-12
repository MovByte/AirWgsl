//! Attribute groups and attribute lists (`parseAttributeGroupBlock`, `parseAttributeBlock` in `BitcodeReader.cpp`),
//! with the upgrades LLVM applies while reading them.

use super::codes::*;
use super::{Reader, Result};
use crate::bitstream::Entry;
use crate::ir::attrs::AttrClass;
use crate::ir::{AttrKind, AttrList, AttrSet, Attribute, ConstantRange};

/// Bitcode attribute codes LLVM 21 reads as something else.
const ATTR_KIND_NO_CAPTURE: u64 = 11;
const ATTR_KIND_READ_NONE: u64 = 20;
const ATTR_KIND_READ_ONLY: u64 = 21;
const ATTR_KIND_ARGMEMONLY: u64 = 45;
const ATTR_KIND_INACCESSIBLEMEM_ONLY: u64 = 49;
const ATTR_KIND_INACCESSIBLEMEM_OR_ARGMEMONLY: u64 = 50;
const ATTR_KIND_WRITEONLY: u64 = 52;

/// The function attribute index of an attribute group.
const FUNCTION_INDEX: u64 = 0xffff_ffff;

/// `MemoryEffects` with every location at `mr` (two bits per location: argmem, inaccessiblemem, errnomem, other).
fn memory_all(mr: u64) -> u64 {
    mr | mr << 2 | mr << 4 | mr << 6
}

/// `upgradeOldMemoryAttribute`: intersects `me` with the effect an old function memory attribute stands for.
fn upgrade_old_memory(me: &mut u64, code: u64) -> bool {
    let effect = match code {
        ATTR_KIND_READ_NONE => memory_all(0),
        ATTR_KIND_READ_ONLY => memory_all(1),
        ATTR_KIND_WRITEONLY => memory_all(2),
        ATTR_KIND_ARGMEMONLY => 3,
        ATTR_KIND_INACCESSIBLEMEM_ONLY => 3 << 2,
        ATTR_KIND_INACCESSIBLEMEM_OR_ARGMEMONLY => 3 | 3 << 2,
        _ => return false,
    };
    *me &= effect;
    true
}

impl<'a> Reader<'a> {
    fn attr_kind(&self, code: u64) -> Result<AttrKind> {
        match AttrKind::from_bitcode(code) {
            Some(k) => Ok(k),
            None => self.malformed(format!("unknown attribute kind ({})", code)),
        }
    }

    /// `readBitWidthAndConstantRange` / `readConstantRange`.
    fn read_range(&self, ops: &[u64], i: &mut usize, bits: Option<u32>) -> Result<ConstantRange> {
        let bits = match bits {
            Some(b) => b,
            None => {
                if *i >= ops.len() {
                    return self.malformed("truncated constant range");
                }
                let b = ops[*i];
                *i += 1;
                if b == 0 || b > 1 << 23 {
                    return self.malformed("invalid constant range bit width");
                }
                b as u32
            }
        };
        let decode = |v: u64| super::constants::decode_sign_rotated(v);
        if bits > 64 {
            if *i >= ops.len() {
                return self.malformed("truncated constant range");
            }
            let lower_words = (ops[*i] & 0xffff_ffff) as usize;
            let upper_words = (ops[*i] >> 32) as usize;
            *i += 1;
            if ops.len() - *i < lower_words + upper_words {
                return self.malformed("truncated constant range");
            }
            let lower = super::constants::wide_words(&ops[*i..*i + lower_words], bits);
            *i += lower_words;
            let upper = super::constants::wide_words(&ops[*i..*i + upper_words], bits);
            *i += upper_words;
            Ok(ConstantRange { bits, lower, upper })
        } else {
            if ops.len() - *i < 2 {
                return self.malformed("truncated constant range");
            }
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            let lower = decode(ops[*i]) & mask;
            let upper = decode(ops[*i + 1]) & mask;
            *i += 2;
            Ok(ConstantRange { bits, lower: vec![lower], upper: vec![upper] })
        }
    }

    pub(super) fn read_attr_group_block(&mut self) -> Result<()> {
        if self.seen_attr_groups {
            return self.malformed("more than one attribute group block");
        }
        self.seen_attr_groups = true;
        self.cur.enter_block(PARAMATTR_GROUP_BLOCK_ID)?;
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block inside the attribute group block"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            if r.code != PARAMATTR_GRP_CODE_ENTRY {
                return self.malformed(format!("unknown attribute group record {}", r.code));
            }
            let ops = &r.ops;
            if ops.len() < 3 {
                return self.malformed("invalid attribute group record");
            }
            let grp = ops[0];
            let idx = ops[1];
            let mut set = AttrSet::new();
            // MemoryEffects::unknown(): readwrite everywhere.
            let mut me = memory_all(3);
            let mut i = 2;
            while i < ops.len() {
                let tag = ops[i];
                i += 1;
                let next = |i: &mut usize| -> Result<u64> {
                    if *i >= ops.len() {
                        return Err(super::malformed(None, "truncated attribute group record"));
                    }
                    *i += 1;
                    Ok(ops[*i - 1])
                };
                match tag {
                    0 => {
                        let code = next(&mut i)?;
                        if idx == FUNCTION_INDEX && upgrade_old_memory(&mut me, code) {
                            continue;
                        }
                        if code == ATTR_KIND_NO_CAPTURE {
                            set.add(Attribute::Int(AttrKind::Captures, 0));
                            continue;
                        }
                        let kind = self.attr_kind(code)?;
                        match kind {
                            AttrKind::ByVal | AttrKind::StructRet | AttrKind::InAlloca => {
                                set.add(Attribute::Type(kind, None))
                            }
                            // UWTableKind::Default (async).
                            AttrKind::UWTable => set.add(Attribute::Int(kind, 2)),
                            k if k.class() == AttrClass::Enum => set.add(Attribute::Enum(k)),
                            _ => return self.malformed(format!("{} is not an enum attribute", kind.name())),
                        }
                    }
                    1 => {
                        let kind = self.attr_kind(next(&mut i)?)?;
                        if kind.class() != AttrClass::Int {
                            return self.malformed(format!("{} is not an integer attribute", kind.name()));
                        }
                        let v = next(&mut i)?;
                        match kind {
                            AttrKind::Alignment | AttrKind::StackAlignment => {
                                if v == 0 {
                                    continue;
                                }
                                let max = if kind == AttrKind::Alignment { 1u64 << 32 } else { 0x100 };
                                if !v.is_power_of_two() || v > max {
                                    return self.malformed(format!("invalid {} value {}", kind.name(), v));
                                }
                                set.add(Attribute::Int(kind, v));
                            }
                            AttrKind::Dereferenceable | AttrKind::DereferenceableOrNull | AttrKind::VScaleRange => {
                                if v != 0 {
                                    set.add(Attribute::Int(kind, v));
                                }
                            }
                            AttrKind::UWTable => {
                                if v > 2 {
                                    return self.malformed(format!("invalid uwtable kind {}", v));
                                }
                                if v != 0 {
                                    set.add(Attribute::Int(kind, v));
                                }
                            }
                            AttrKind::AllocSize => {
                                if v == 0 {
                                    return self.malformed("allocsize(0, 0)");
                                }
                                set.add(Attribute::Int(kind, v));
                            }
                            AttrKind::AllocKind => set.add(Attribute::Int(kind, v)),
                            AttrKind::Memory => {
                                let version = v >> 56;
                                let encoded = if version == 0 {
                                    // Before errno memory existed "other" covered it.
                                    let arg = v & 3;
                                    let inaccessible = (v >> 2) & 3;
                                    let other = (v >> 4) & 3;
                                    arg | inaccessible << 2 | other << 4 | other << 6
                                } else {
                                    v & 0x00ff_ffff_ffff_ffff
                                };
                                if encoded > 0xff {
                                    return self.malformed(format!("invalid memory effects {:#x}", v));
                                }
                                set.add(Attribute::Int(kind, encoded));
                            }
                            AttrKind::Captures => {
                                if v > 0xff {
                                    return self.malformed(format!("invalid captures value {:#x}", v));
                                }
                                set.add(Attribute::Int(kind, v));
                            }
                            AttrKind::NoFPClass => {
                                let mask = v & 0x3ff;
                                if mask != 0 {
                                    set.add(Attribute::Int(kind, mask));
                                }
                            }
                            _ => return self.malformed(format!("unhandled integer attribute {}", kind.name())),
                        }
                    }
                    3 | 4 => {
                        let has_value = tag == 4;
                        let start = i;
                        while i < ops.len() && ops[i] != 0 {
                            i += 1;
                        }
                        if i >= ops.len() {
                            return self.malformed("attribute key not NUL-terminated");
                        }
                        let key = self.record_bytes(&ops[start..i])?;
                        i += 1;
                        let mut value = Vec::new();
                        if has_value {
                            let vs = i;
                            while i < ops.len() && ops[i] != 0 {
                                i += 1;
                            }
                            if i >= ops.len() {
                                return self.malformed("attribute value not NUL-terminated");
                            }
                            value = self.record_bytes(&ops[vs..i])?;
                            i += 1;
                        }
                        set.add(Attribute::Str(key, value));
                    }
                    5 | 6 => {
                        let has_type = tag == 6;
                        let kind = self.attr_kind(next(&mut i)?)?;
                        if kind.class() != AttrClass::Type {
                            return self.malformed(format!("{} is not a type attribute", kind.name()));
                        }
                        let ty = if has_type { Some(self.ty(next(&mut i)?)?) } else { None };
                        set.add(Attribute::Type(kind, ty));
                    }
                    7 => {
                        let kind = self.attr_kind(next(&mut i)?)?;
                        if kind.class() != AttrClass::ConstantRange {
                            return self.malformed(format!("{} is not a constant range attribute", kind.name()));
                        }
                        let cr = self.read_range(ops, &mut i, None)?;
                        // A full set adds nothing (`addConstantRangeAttr`).
                        if !super::constants::range_is_full(&cr) {
                            set.add(Attribute::Range(kind, cr));
                        }
                    }
                    8 => {
                        let kind = self.attr_kind(next(&mut i)?)?;
                        if kind.class() != AttrClass::ConstantRangeList {
                            return self.malformed(format!("{} is not a constant range list attribute", kind.name()));
                        }
                        if i + 2 > ops.len() {
                            return self.malformed("too few records for constant range list");
                        }
                        let n = ops[i] as usize;
                        let bits = ops[i + 1];
                        i += 2;
                        if bits == 0 || bits > 1 << 23 || n > ops.len() {
                            return self.malformed("invalid constant range list");
                        }
                        let mut list = Vec::new();
                        for _ in 0..n {
                            list.push(self.read_range(ops, &mut i, Some(bits as u32))?);
                        }
                        if !super::constants::ranges_ordered(&list) {
                            return self.malformed("invalid (unordered or overlapping) range list");
                        }
                        set.add(Attribute::RangeList(kind, list));
                    }
                    other => return self.malformed(format!("invalid attribute group entry kind {}", other)),
                }
            }
            if me != memory_all(3) {
                set.add(Attribute::Int(AttrKind::Memory, me));
            }
            upgrade_attributes(&mut set);
            if idx != FUNCTION_INDEX && idx > u32::MAX as u64 {
                return self.malformed("invalid attribute group index");
            }
            let index = if idx == FUNCTION_INDEX { u32::MAX } else { idx as u32 };
            if index != u32::MAX && index > MAX_PARAMS {
                return self.unsupported(format!(
                    "attributes of parameter {} (this reader keeps at most {} parameters)",
                    index - 1,
                    MAX_PARAMS
                ));
            }
            if self.attr_groups.insert(grp, (index, set)).is_some() {
                return self.malformed(format!("attribute group {} defined twice", grp));
            }
        }
    }

    pub(super) fn read_attr_block(&mut self) -> Result<()> {
        if self.seen_attrs {
            return self.malformed("more than one attribute block");
        }
        self.seen_attrs = true;
        self.cur.enter_block(PARAMATTR_BLOCK_ID)?;
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block inside the attribute block"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            match r.code {
                PARAMATTR_CODE_ENTRY => {
                    let mut list = Vec::new();
                    for g in &r.ops {
                        // LLVM looks groups up with `std::map::operator[]`: a group the group block does not define
                        // is an empty attribute list and adds nothing (measured: a system module lists one).
                        if let Some(e) = self.attr_groups.get(g) {
                            list.push(e.clone());
                        }
                    }
                    self.attr_lists.push(list);
                }
                PARAMATTR_CODE_ENTRY_OLD => {
                    return self.unsupported("PARAMATTR_CODE_ENTRY_OLD (attributes of pre-3.3 bitcode)");
                }
                c => return self.malformed(format!("unknown attribute record {}", c)),
            }
        }
    }

    /// The attribute list a record refers to (1-based; 0 is none), with the type attributes still to be completed by
    /// the caller.
    pub fn attr_list(&mut self, id: u64) -> Result<AttrList> {
        if id == 0 {
            return Ok(AttrList::default());
        }
        let groups = match self.attr_lists.get(id as usize - 1) {
            Some(g) => g.clone(),
            None => return self.malformed(format!("invalid attribute list id {}", id)),
        };
        let mut function = AttrSet::new();
        let mut ret = AttrSet::new();
        let mut params: Vec<AttrSet> = Vec::new();
        for (index, set) in groups {
            match index {
                u32::MAX => function.merge(&set),
                0 => ret.merge(&set),
                p => {
                    let p = p as usize - 1;
                    if params.len() <= p {
                        params.resize(p + 1, AttrSet::new());
                    }
                    params[p].merge(&set);
                }
            }
        }
        // Typeless type attributes stay until the caller fills them; they are interned as they are.
        let mut list = AttrList {
            function: self.m.attr_set(function),
            ret: self.m.attr_set(ret),
            params: params.into_iter().map(|s| self.m.attr_set(s)).collect(),
        };
        list.trim();
        Ok(list)
    }

    /// Gives typeless `byval`, `sret` and `inalloca` parameter attributes the pointee type of the parameter
    /// (`parseFunctionRecord`, `propagateAttributeTypes`). `pointee(i)` is the bitcode type id of parameter `i`.
    pub fn upgrade_typed_param_attrs(&mut self, list: &mut AttrList, param_bc_types: &[Option<u32>]) -> Result<()> {
        for i in 0..list.params.len() {
            let set_id = list.params[i];
            let set = self.m.attrs(set_id).clone();
            let mut changed = set.clone();
            for kind in [AttrKind::ByVal, AttrKind::StructRet, AttrKind::InAlloca] {
                if let Some(Attribute::Type(_, None)) = set.get(kind) {
                    let pointee = param_bc_types.get(i).copied().flatten().and_then(|bc| self.contained(bc, 0));
                    let ty = match pointee.and_then(|p| self.bc_types.get(p as usize).copied().flatten()) {
                        Some(t) => t,
                        None => {
                            return self
                                .malformed(format!("missing parameter element type for the {} upgrade", kind.name()));
                        }
                    };
                    changed.add(Attribute::Type(kind, Some(ty)));
                }
            }
            if changed != set {
                list.params[i] = self.m.attr_set(changed);
            }
        }
        for set in [list.function, list.ret] {
            if self.m.attrs(set).iter().any(|a| matches!(a, Attribute::Type(_, None))) {
                return self.malformed("type attribute without a type outside a parameter");
            }
        }
        Ok(())
    }
}

/// How many parameters an attribute list may describe: a limit of this reader, which keeps parameter attributes in a
/// dense list (LLVM's limit is the index type's range).
pub(crate) const MAX_PARAMS: u32 = 1 << 16;

/// `UpgradeAttributes`: old string attributes LLVM replaces while reading a group.
fn upgrade_attributes(set: &mut AttrSet) {
    let mut frame_pointer: Option<&str> = None;
    if let Some(v) = set.get_str(b"no-frame-pointer-elim").map(|v| v.to_vec()) {
        frame_pointer = Some(if v == b"true" { "all" } else { "none" });
        set.remove_str(b"no-frame-pointer-elim");
    }
    if set.get_str(b"no-frame-pointer-elim-non-leaf").is_some() {
        if frame_pointer != Some("all") {
            frame_pointer = Some("non-leaf");
        }
        set.remove_str(b"no-frame-pointer-elim-non-leaf");
    }
    if let Some(fp) = frame_pointer {
        set.add(Attribute::Str(b"frame-pointer".to_vec(), fp.as_bytes().to_vec()));
    }
    if let Some(v) = set.get_str(b"null-pointer-is-valid").map(|v| v.to_vec()) {
        set.remove_str(b"null-pointer-is-valid");
        if v == b"true" {
            set.add(Attribute::Enum(AttrKind::NullPointerIsValid));
        }
    }
}
