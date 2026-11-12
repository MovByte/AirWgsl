//! Reading AIR metadata records: the operands of a tuple as typed values, and a cursor that takes them in order.
//!
//! AIR describes a module in metadata tuples whose operands are keys (`!"air.buffer"`), the values that follow a key
//! (`i32 0`, `!"float4"`, a nested tuple), and references to globals and functions (`ptr @f`). Every decoder in this
//! module reads records through [`Cursor`], which fails with a [`Refusal`] naming the record and the operand whenever
//! an operand is not what the grammar says comes next.

use crate::api::{Location, Refusal, RefusalKind};
use crate::ir::{ConstKind, GlobalId, MdId, Metadata, Module, Type, Value};

/// One operand of a metadata tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand<'m> {
    /// `!"..."`.
    Str(&'m [u8]),
    /// An integer constant: its width in bits and its value, zero-extended (64 bits at most).
    Int { bits: u32, value: u64 },
    /// `ptr @f`: a function, by index into [`Module::functions`].
    Function(u32),
    /// `ptr @g`: a global variable, by index into [`Module::globals`].
    Global(u32),
    /// A nested tuple.
    Node(MdId),
    /// A null operand.
    Null,
    /// Anything else (a float, an aggregate constant, a local value), by its metadata id.
    Other(MdId),
}

impl Operand<'_> {
    /// A short description of the operand for refusal messages.
    pub fn describe(&self, m: &Module) -> String {
        match self {
            Operand::Str(s) => format!("!\"{}\"", String::from_utf8_lossy(s)),
            Operand::Int { bits, value } => format!("i{} {}", bits, value),
            Operand::Function(f) => format!("ptr @{}", m.functions[*f as usize].common.name),
            Operand::Global(g) => format!("ptr @{}", m.globals[*g as usize].common.name),
            Operand::Node(id) => format!("!{}", id.0),
            Operand::Null => "null".into(),
            Operand::Other(id) => format!("metadata {:?}", m.md(*id)),
        }
    }
}

/// The operand `id` of a tuple, classified.
pub fn operand(m: &Module, id: Option<MdId>) -> Operand<'_> {
    let Some(id) = id else { return Operand::Null };
    match m.md(id) {
        Metadata::String(s) => Operand::Str(s),
        Metadata::Tuple { .. } => Operand::Node(id),
        Metadata::Value { ty, value } => match value {
            Value::Const(c) => match (&m.constant(*c).kind, m.types.get(*ty)) {
                (ConstKind::Int(words), Type::Int(bits)) if *bits <= 64 => {
                    Operand::Int { bits: *bits, value: words.first().copied().unwrap_or(0) }
                }
                _ => Operand::Other(id),
            },
            Value::Global(GlobalId::Function(f)) => Operand::Function(*f),
            Value::Global(GlobalId::Var(g)) => Operand::Global(*g),
            _ => Operand::Other(id),
        },
        _ => Operand::Other(id),
    }
}

/// The operands of tuple `id`, or `None` when `id` is not a tuple.
pub fn tuple(m: &Module, id: MdId) -> Option<Vec<Operand<'_>>> {
    match m.md(id) {
        Metadata::Tuple { ops, .. } => Some(ops.iter().map(|o| operand(m, *o)).collect()),
        _ => None,
    }
}

/// Builds a refusal of the metadata grammar: `construct` is the key or record refused.
pub fn refusal(
    kind: RefusalKind,
    construct: impl Into<String>,
    function: Option<&str>,
    reason: impl Into<String>,
) -> Refusal {
    Refusal {
        kind,
        construct: construct.into(),
        location: function.map(|f| Location { function: f.to_string(), instruction: None }),
        reason: reason.into(),
        reference: None,
    }
}

/// Reads the operands of one record in order.
pub struct Cursor<'m> {
    pub m: &'m Module,
    /// What the record is, for refusal messages (`air.buffer argument 2 of f`).
    pub what: String,
    /// The function the record belongs to, when it belongs to one.
    pub function: Option<String>,
    ops: Vec<Operand<'m>>,
    pos: usize,
}

impl<'m> Cursor<'m> {
    pub fn new(m: &'m Module, ops: Vec<Operand<'m>>, what: impl Into<String>, function: Option<&str>) -> Cursor<'m> {
        Cursor { m, what: what.into(), function: function.map(str::to_string), ops, pos: 0 }
    }

    /// A cursor over tuple `id`; a refusal when `id` is not a tuple.
    pub fn node(
        m: &'m Module,
        id: MdId,
        what: impl Into<String>,
        function: Option<&str>,
    ) -> Result<Cursor<'m>, Refusal> {
        let what = what.into();
        match tuple(m, id) {
            Some(ops) => Ok(Cursor::new(m, ops, what, function)),
            None => Err(refusal(
                RefusalKind::NotImplemented,
                what.clone(),
                function,
                format!("{} is {:?}, not a metadata tuple", what, m.md(id)),
            )),
        }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn at_end(&self) -> bool {
        self.pos >= self.ops.len()
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// The operand at position `i`.
    pub fn op_at(&self, i: usize) -> Operand<'m> {
        self.ops[i]
    }

    pub fn peek(&self) -> Option<Operand<'m>> {
        self.ops.get(self.pos).copied()
    }

    /// The next operand when it is a string.
    pub fn peek_str(&self) -> Option<&'m [u8]> {
        match self.peek() {
            Some(Operand::Str(s)) => Some(s),
            _ => None,
        }
    }

    /// The record's operands, as text (for refusal messages).
    pub fn render(&self) -> String {
        let mut s = String::from("{");
        for (i, op) in self.ops.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            if s.len() > 400 {
                s.push_str("...");
                break;
            }
            s.push_str(&op.describe(self.m));
        }
        s.push('}');
        s
    }

    /// A refusal naming this record, the operand position and what was expected there.
    pub fn error(&self, construct: impl Into<String>, expected: &str) -> Refusal {
        let found = match self.peek() {
            Some(op) => op.describe(self.m),
            None => "the end of the record".into(),
        };
        refusal(
            RefusalKind::NotImplemented,
            construct,
            self.function.as_deref(),
            format!(
                "{} {}: operand {} is {}, where the AIR grammar has {}",
                self.what,
                self.render(),
                self.pos,
                found,
                expected
            ),
        )
    }

    /// The kind of record, for refusals of the record as a whole: its description up to the metadata id.
    pub fn construct(&self) -> String {
        self.what.split(" !").next().unwrap_or(&self.what).to_string()
    }

    /// Takes the next operand.
    pub fn advance(&mut self) -> Option<Operand<'m>> {
        let op = self.peek();
        if op.is_some() {
            self.pos += 1;
        }
        op
    }

    /// The next operand, which must be an integer of at most 64 bits.
    pub fn int(&mut self, key: &str) -> Result<u64, Refusal> {
        match self.peek() {
            Some(Operand::Int { value, .. }) => {
                self.pos += 1;
                Ok(value)
            }
            _ => Err(self.error(key, "an integer")),
        }
    }

    /// The next operand, which must be an `i32` (the width AIR writes its numbers in).
    pub fn i32(&mut self, key: &str) -> Result<u32, Refusal> {
        match self.peek() {
            Some(Operand::Int { bits: 32, value }) => {
                self.pos += 1;
                Ok(value as u32)
            }
            _ => Err(self.error(key, "an i32")),
        }
    }

    /// The next operand, which must be a string.
    pub fn bytes(&mut self, key: &str) -> Result<&'m [u8], Refusal> {
        match self.peek() {
            Some(Operand::Str(s)) => {
                self.pos += 1;
                Ok(s)
            }
            _ => Err(self.error(key, "a string")),
        }
    }

    /// The next operand, which must be a UTF-8 string.
    pub fn string(&mut self, key: &str) -> Result<String, Refusal> {
        let at = self.pos;
        let b = self.bytes(key)?;
        match std::str::from_utf8(b) {
            Ok(s) => Ok(s.to_string()),
            Err(_) => {
                self.pos = at;
                Err(self.error(key, "a UTF-8 string"))
            }
        }
    }

    /// The next operand, which must be a tuple.
    pub fn node_id(&mut self, key: &str) -> Result<MdId, Refusal> {
        match self.peek() {
            Some(Operand::Node(id)) => {
                self.pos += 1;
                Ok(id)
            }
            _ => Err(self.error(key, "a metadata tuple")),
        }
    }

    /// The next operand, which must be a function.
    pub fn function_ref(&mut self, key: &str) -> Result<u32, Refusal> {
        match self.peek() {
            Some(Operand::Function(f)) => {
                self.pos += 1;
                Ok(f)
            }
            _ => Err(self.error(key, "a function")),
        }
    }

    /// The next operand, which must be a global variable.
    pub fn global_ref(&mut self, key: &str) -> Result<u32, Refusal> {
        match self.peek() {
            Some(Operand::Global(g)) => {
                self.pos += 1;
                Ok(g)
            }
            _ => Err(self.error(key, "a global variable")),
        }
    }

    /// Fails unless every operand was read.
    pub fn finish(&self) -> Result<(), Refusal> {
        if self.at_end() { Ok(()) } else { Err(self.error(self.construct(), "the end of the record")) }
    }
}
