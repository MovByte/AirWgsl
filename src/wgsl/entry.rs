//! Entry-point records: the `!air.vertex`, `!air.fragment` and `!air.kernel` metadata of a module, read as far as the
//! lowering needs.
//!
//! This is the seam to package S2 (`src/air`), which owns the AIR semantic layer and its closed grammar. Until it
//! lands, the records are read here into a neutral form ([`Record`]: the argument index, then the operands in order)
//! and interpreted by the stage-interface rules ([`super::io`]); every record those rules do not recognize is refused
//! with a tracked item, so nothing is skipped. When S2 lands, [`entries`] is replaced by its typed entry points and
//! the rules read those.

use crate::api::{MetalStage, Refusal};
use crate::ir::{ConstKind, GlobalId, MdId, Metadata, Module, Value};
use crate::wgsl::tracked;

/// One operand of a record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Str(String),
    Int(u64),
    /// A metadata node (struct type info and the like).
    Node(MdId),
}

/// An argument or output record: `!{i32 index, !"air.kind", ...}` (outputs have no index).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub index: Option<u32>,
    pub items: Vec<Item>,
}

impl Record {
    /// The record's kind: its first string (`air.buffer`, `air.position`, ...).
    pub fn kind(&self) -> &str {
        self.items.iter().find_map(|i| if let Item::Str(s) = i { Some(s.as_str()) } else { None }).unwrap_or("")
    }

    pub fn has(&self, key: &str) -> bool {
        self.items.iter().any(|i| matches!(i, Item::Str(s) if s == key))
    }

    /// The `n`th operand after `key`.
    pub fn after(&self, key: &str, n: usize) -> Option<&Item> {
        let p = self.items.iter().position(|i| matches!(i, Item::Str(s) if s == key))?;
        self.items.get(p + 1 + n)
    }

    pub fn int_after(&self, key: &str, n: usize) -> Option<u64> {
        match self.after(key, n)? {
            Item::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn str_after(&self, key: &str) -> Option<&str> {
        match self.after(key, 0)? {
            Item::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The MSL type name (`air.arg_type_name`).
    pub fn type_name(&self) -> Option<&str> {
        self.str_after("air.arg_type_name")
    }
}

/// An entry point of a module.
#[derive(Clone, Debug)]
pub struct Entry {
    pub stage: MetalStage,
    pub function: usize,
    /// The return value's records, one per member of a returned struct (or one for a single value).
    pub outputs: Vec<Record>,
    /// One record per IR argument, by argument index.
    pub args: Vec<Option<Record>>,
    /// Further operands of the entry record (`early_fragment_tests`, ...).
    pub flags: Vec<Item>,
}

fn md_items(m: &Module, id: MdId) -> Option<Vec<Item>> {
    let Metadata::Tuple { ops, .. } = m.md(id) else {
        return None;
    };
    let mut out = Vec::new();
    for op in ops {
        let Some(op) = op else {
            continue;
        };
        out.push(match m.md(*op) {
            Metadata::String(s) => Item::Str(String::from_utf8_lossy(s).into_owned()),
            Metadata::Value { value: Value::Const(c), .. } => match &m.constant(*c).kind {
                ConstKind::Int(w) => Item::Int(w[0]),
                _ => Item::Node(*op),
            },
            _ => Item::Node(*op),
        });
    }
    Some(out)
}

fn record(m: &Module, id: MdId) -> Option<Record> {
    let items = md_items(m, id)?;
    match items.first() {
        Some(Item::Int(i)) => Some(Record { index: Some(*i as u32), items: items[1..].to_vec() }),
        _ => Some(Record { index: None, items }),
    }
}

fn malformed(what: &str) -> Refusal {
    tracked::refuse("entry-metadata", what.to_string(), None, format!("malformed entry record: {}", what))
}

/// The entry points of a module, from its `!air.vertex`, `!air.fragment` and `!air.kernel` named metadata.
pub fn entries(m: &Module) -> Result<Vec<Entry>, Refusal> {
    let mut out = Vec::new();
    for nm in &m.named_metadata {
        let stage = match nm.name.as_str() {
            "air.vertex" => MetalStage::Vertex,
            "air.fragment" => MetalStage::Fragment,
            "air.kernel" => MetalStage::Kernel,
            _ => continue,
        };
        for &e in &nm.ops {
            let Metadata::Tuple { ops, .. } = m.md(e) else {
                return Err(malformed(&nm.name));
            };
            let function = match ops.first().copied().flatten().map(|f| m.md(f)) {
                Some(Metadata::Value { value: Value::Global(GlobalId::Function(fi)), .. }) => *fi as usize,
                _ => return Err(malformed(&nm.name)),
            };
            let list = |k: usize| -> Result<Vec<Record>, Refusal> {
                let Some(Some(id)) = ops.get(k) else {
                    return Ok(Vec::new());
                };
                let Metadata::Tuple { ops: rs, .. } = m.md(*id) else {
                    return Err(malformed("record list"));
                };
                rs.iter().map(|r| r.and_then(|r| record(m, r)).ok_or_else(|| malformed("record"))).collect()
            };
            // Kernels: {fn, outputs (empty), args}; vertex and fragment: {fn, outputs, args}.
            let outputs = list(1)?;
            let arg_records = list(2)?;
            let nargs = m.functions[function].args.len();
            let mut args = vec![None; nargs];
            for r in arg_records {
                let i = r.index.ok_or_else(|| malformed("argument record without an index"))? as usize;
                if i >= nargs {
                    return Err(malformed("argument record index"));
                }
                args[i] = Some(r);
            }
            let mut flags = Vec::new();
            for op in ops.iter().skip(3).flatten() {
                if let Metadata::String(s) = m.md(*op) {
                    flags.push(Item::Str(String::from_utf8_lossy(s).into_owned()));
                } else {
                    flags.push(Item::Node(*op));
                }
            }
            out.push(Entry { stage, function, outputs, args, flags });
        }
    }
    Ok(out)
}
