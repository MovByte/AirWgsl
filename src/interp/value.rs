//! Runtime values.
//!
//! A [`Scalar`] is one integer, floating-point value or pointer (an element of a vector, a leaf of an aggregate) with
//! LangRef's three kinds of indefiniteness kept apart:
//!
//! - `poison` (LangRef "Poison Values"): the whole element is poison.
//! - `undef` bits (LangRef "Undefined Values"): each use of the value may see any bit there, independently.
//! - `choice` bits: defined bits whose value was one of several the semantics allowed (the result of `freeze` on an
//!   undef or poison operand, the sign and payload of a NaN result, the sign of a zero under `nsz`, values the Metal
//!   specification leaves unspecified). The interpreter picks one (zero bits for `freeze`, the preferred NaN,
//!   IEEE's sign of zero) and marks the bits so that a comparison can accept any other choice there.
//!
//! Integers and floating-point values keep their encoding's bits in `bits`, zero above the type's width. A pointer
//! keeps its address and its provenance: the memory region it may access ([`super::memory`]), 0 for none.

/// One element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scalar {
    pub bits: u128,
    pub undef: u128,
    pub choice: u128,
    /// For a pointer: the region it was derived from, 0 for none.
    pub prov: u32,
    pub poison: bool,
}

impl Scalar {
    pub const POISON: Scalar = Scalar { bits: 0, undef: 0, choice: 0, prov: 0, poison: true };

    pub fn int(bits: u128) -> Scalar {
        Scalar { bits, undef: 0, choice: 0, prov: 0, poison: false }
    }

    /// Every bit of a `width`-bit value undef.
    pub fn undef(width: u32) -> Scalar {
        Scalar { bits: 0, undef: mask(width), choice: 0, prov: 0, poison: false }
    }

    pub fn ptr(prov: u32, addr: u64) -> Scalar {
        Scalar { bits: addr as u128, undef: 0, choice: 0, prov, poison: false }
    }

    pub fn bool(b: bool) -> Scalar {
        Scalar::int(b as u128)
    }

    /// Not poison and no undef bit: LangRef's "well defined".
    pub fn is_defined(&self) -> bool {
        !self.poison && self.undef == 0
    }

    pub fn has_choice(&self) -> bool {
        !self.poison && self.choice != 0
    }
}

/// The low `width` bits set (`width` up to 128).
pub fn mask(width: u32) -> u128 {
    if width >= 128 { u128::MAX } else { (1u128 << width) - 1 }
}

/// A value of any first-class type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Val {
    /// An integer, floating-point value or pointer.
    S(Scalar),
    /// A vector.
    V(Vec<Scalar>),
    /// A struct or array, member by member.
    A(Vec<Val>),
    /// A metadata operand (`metadata !0`), passed only to intrinsics.
    Md(crate::ir::MdId),
    /// `none` of type `token`.
    Token,
}

impl Val {
    pub fn scalar(&self) -> Option<Scalar> {
        match self {
            Val::S(s) => Some(*s),
            _ => None,
        }
    }

    /// The elements of a scalar (one) or vector.
    pub fn lanes(&self) -> Vec<Scalar> {
        match self {
            Val::S(s) => vec![*s],
            Val::V(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Whether any leaf has a poison, undef or choice part.
    pub fn is_exact(&self) -> bool {
        match self {
            Val::S(s) => s.is_defined() && s.choice == 0,
            Val::V(v) => v.iter().all(|s| s.is_defined() && s.choice == 0),
            Val::A(a) => a.iter().all(Val::is_exact),
            Val::Md(_) | Val::Token => true,
        }
    }
}
