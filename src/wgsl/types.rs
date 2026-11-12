//! The WGSL carriers of IR values.
//!
//! Values travel as integer bits ([`crate::num`] carriers): `i1` as `bool`; `i8`, `i16`, `i32`, `half` and `float` as
//! `u32` holding the bits zero-extended; `i64` as `vec2<u32>` (low word, high word). A vector of 2 to 4 such lanes is
//! `vecN` of the lane carrier, except that `i64` lanes and vectors of 8 or 16 lanes are `array`s. An array value is
//! `array<carrier, N>`; a struct value is a struct this module declares (`aw_sK`, members `m0`, `m1`, ...). These are
//! the carriers of SSA values only: guest memory is never typed (plan 5.4). Pointers have no type carrier: their
//! representation depends on the regions they may point into ([`crate::mem`]).

use crate::api::Refusal;
use crate::ir::{Type, TypeId, TypeTable};
use crate::num::{Scalar, Ty};
use crate::wgsl::tracked;
use std::collections::BTreeMap;

/// The shape of a type's carrier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    Bool,
    /// One u32 holding `bits` bits; `float` tells binary16/binary32 from integers.
    Word {
        bits: u32,
        float: bool,
    },
    /// vec2<u32>.
    Wide,
    /// A `vecN` (2 to 4 lanes) of a [`Shape::Bool`] or [`Shape::Word`] lane.
    Vector {
        lane: Box<Shape>,
        len: u32,
    },
    /// An `array<carrier, N>` (an array type, or a vector of i64 or of 8 or 16 lanes).
    Array {
        elem: Box<Shape>,
        elem_ty: TypeId,
        len: u64,
        is_vector: bool,
    },
    /// A declared struct.
    Struct {
        ty: TypeId,
        members: Vec<TypeId>,
    },
    Ptr(u32),
    Void,
}

/// Carrier types and the struct declarations they need.
#[derive(Default)]
pub struct Types {
    structs: BTreeMap<TypeId, (String, Vec<String>)>,
    order: Vec<TypeId>,
}

fn refuse_type(id: &str, what: String) -> Refusal {
    tracked::refuse(id, what.clone(), None, format!("type {}", what))
}

impl Types {
    pub fn new() -> Types {
        Types::default()
    }

    /// The shape of `t`'s carrier.
    pub fn shape(&mut self, tt: &TypeTable, t: TypeId) -> Result<Shape, Refusal> {
        Ok(match tt.get(t) {
            Type::Void => Shape::Void,
            Type::Int(1) => Shape::Bool,
            Type::Int(b @ (8 | 16 | 32)) => Shape::Word { bits: *b, float: false },
            Type::Int(64) => Shape::Wide,
            Type::Int(b) => return Err(refuse_type("integer-width", format!("i{}", b))),
            Type::Half => Shape::Word { bits: 16, float: true },
            Type::Float => Shape::Word { bits: 32, float: true },
            Type::Double => return Err(tracked::refuse_family("double", "double", None)),
            Type::BFloat => return Err(tracked::refuse_family("bfloat", "bfloat", None)),
            Type::Ptr(a) => Shape::Ptr(*a),
            Type::Vector { len, elem, scalable: false } => {
                let lane = self.shape(tt, *elem)?;
                match lane {
                    Shape::Bool | Shape::Word { .. } if (2..=4).contains(len) => {
                        Shape::Vector { lane: Box::new(lane), len: *len }
                    }
                    // Vectors WGSL has no vecN for (i64 lanes, more than 4 lanes, as LLVM's vectorizer forms them).
                    Shape::Bool | Shape::Word { .. } | Shape::Wide if *len >= 2 => {
                        Shape::Array { elem: Box::new(lane), elem_ty: *elem, len: *len as u64, is_vector: true }
                    }
                    Shape::Ptr(_) => return Err(refuse_type("pointer-aggregate", format!("<{} x ptr>", len))),
                    _ => return Err(refuse_type("vector-shape", format!("<{} x ...>", len))),
                }
            }
            Type::Array { len, elem } => {
                if *len == 0 {
                    return Err(refuse_type("vector-shape", "[0 x ...]".into()));
                }
                let e = self.shape(tt, *elem)?;
                if matches!(e, Shape::Ptr(_)) {
                    return Err(refuse_type("pointer-aggregate", "an array of pointers".into()));
                }
                Shape::Array { elem: Box::new(e), elem_ty: *elem, len: *len, is_vector: false }
            }
            Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                if elems.is_empty() {
                    return Err(refuse_type("pointer-aggregate", "an empty struct value".into()));
                }
                let elems = elems.clone();
                let mut members = Vec::new();
                for e in &elems {
                    let s = self.shape(tt, *e)?;
                    if matches!(s, Shape::Ptr(_)) {
                        return Err(refuse_type("pointer-aggregate", "a struct with a pointer member".into()));
                    }
                    if s == Shape::Void {
                        return Err(refuse_type("pointer-aggregate", "a struct with a void member".into()));
                    }
                    members.push(self.carrier_of(&s)?);
                }
                if let std::collections::btree_map::Entry::Vacant(e) = self.structs.entry(t) {
                    e.insert((format!("aw_s{}", t.0), members));
                    self.order.push(t);
                }
                Shape::Struct { ty: t, members: elems }
            }
            other => return Err(refuse_type("llvm-instruction", format!("{:?}", other))),
        })
    }

    fn carrier_of(&mut self, s: &Shape) -> Result<String, Refusal> {
        Ok(match s {
            Shape::Bool => "bool".into(),
            Shape::Word { .. } => "u32".into(),
            Shape::Wide => "vec2<u32>".into(),
            Shape::Vector { lane, len } => {
                format!("vec{}<{}>", len, if **lane == Shape::Bool { "bool" } else { "u32" })
            }
            Shape::Array { elem, elem_ty, len, .. } => {
                let _ = elem_ty;
                let e = self.carrier_of(elem)?;
                format!("array<{}, {}>", e, len)
            }
            Shape::Struct { ty, .. } => self.structs[ty].0.clone(),
            Shape::Ptr(_) => return Err(refuse_type("pointer-aggregate", "a pointer carried as a type".into())),
            Shape::Void => return Err(refuse_type("llvm-instruction", "void value".into())),
        })
    }

    /// The carrier type of `t` (not for pointers or void).
    pub fn carrier(&mut self, tt: &TypeTable, t: TypeId) -> Result<String, Refusal> {
        let s = self.shape(tt, t)?;
        self.carrier_of(&s)
    }

    /// The zero value of `t`'s carrier.
    pub fn zero(&mut self, tt: &TypeTable, t: TypeId) -> Result<String, Refusal> {
        let s = self.shape(tt, t)?;
        Ok(match &s {
            Shape::Bool => "false".into(),
            Shape::Word { .. } => "0u".into(),
            Shape::Ptr(_) => "0u".into(),
            _ => format!("{}()", self.carrier_of(&s)?),
        })
    }

    /// The struct declarations, in first-use order.
    pub fn declarations(&self) -> String {
        let mut out = String::new();
        for t in &self.order {
            let (name, members) = &self.structs[t];
            out.push_str(&format!("struct {} {{\n", name));
            for (i, m) in members.iter().enumerate() {
                out.push_str(&format!("    m{}: {},\n", i, m));
            }
            out.push_str("}\n\n");
        }
        out
    }
}

/// The numerics type of a scalar or vector IR type, when it is one the numerics lowering takes.
pub fn num_ty(tt: &TypeTable, t: TypeId) -> Option<Ty> {
    let scalar = |t: TypeId| -> Option<Scalar> {
        Some(match tt.get(t) {
            Type::Int(1) => Scalar::I1,
            Type::Int(8) => Scalar::I8,
            Type::Int(16) => Scalar::I16,
            Type::Int(32) => Scalar::I32,
            Type::Int(64) => Scalar::I64,
            Type::Half => Scalar::F16,
            Type::Float => Scalar::F32,
            _ => return None,
        })
    };
    match tt.get(t) {
        Type::Vector { len, elem, scalable: false } => Some(Ty::vector(scalar(*elem)?, *len)),
        _ => Some(Ty::scalar(scalar(t)?)),
    }
}
