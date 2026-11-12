//! Refinement between two runs: whether every behavior the second run shows is one the first allows (LangRef: poison
//! may be replaced by anything, undef by any defined value, and a value the semantics left open by any other).
//!
//! Values of the two runs are compared bit by bit with their memories: a pointer is its object's label and offset,
//! which name the same thing in both runs of one function.

use super::memory::{Byte, Memory};
use super::value::{Scalar, Val};

/// The first place where the second run does not refine the first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    /// Where: a path into the value, or a region label and byte offset.
    pub at: String,
    pub first: String,
    pub second: String,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: first {}, second {}", self.at, self.first, self.second)
    }
}

/// Whether `b` refines `a` (scalars of the same type).
pub fn scalar_refines(a: &Scalar, ma: &Memory, b: &Scalar, mb: &Memory) -> bool {
    if a.poison {
        return true;
    }
    if b.poison {
        return false;
    }
    // Undef in the second run is allowed only where the first is undef: elsewhere the first had one value for all
    // uses.
    if b.undef & !a.undef != 0 {
        return false;
    }
    // Where the first is defined and not a choice, the second must hold the same bit, without a choice of its own.
    let exact = !a.undef & !a.choice;
    if b.choice & exact != 0 || (a.bits ^ b.bits) & exact != 0 {
        return false;
    }
    // Pointers must name the same object.
    let la = (a.prov != 0).then(|| ma.region(a.prov).label.as_str());
    let lb = (b.prov != 0).then(|| mb.region(b.prov).label.as_str());
    la == lb
}

/// Whether value `b` (second run) refines value `a` (first run).
pub fn refines(a: &Val, ma: &Memory, b: &Val, mb: &Memory) -> Result<(), Mismatch> {
    refine_at(a, ma, b, mb, &mut String::from("value"))
}

fn refine_at(a: &Val, ma: &Memory, b: &Val, mb: &Memory, path: &mut String) -> Result<(), Mismatch> {
    let fail = |path: &str| Mismatch { at: path.to_string(), first: format!("{:?}", a), second: format!("{:?}", b) };
    match (a, b) {
        (Val::S(x), Val::S(y)) => {
            if scalar_refines(x, ma, y, mb) {
                Ok(())
            } else {
                Err(fail(path))
            }
        }
        (Val::V(x), Val::V(y)) if x.len() == y.len() => {
            for (k, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                if !scalar_refines(p, ma, q, mb) {
                    return Err(Mismatch {
                        at: format!("{}[{}]", path, k),
                        first: format!("{:?}", p),
                        second: format!("{:?}", q),
                    });
                }
            }
            Ok(())
        }
        (Val::A(x), Val::A(y)) if x.len() == y.len() => {
            for (k, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                let n = path.len();
                path.push_str(&format!(".{}", k));
                refine_at(p, ma, q, mb, path)?;
                path.truncate(n);
            }
            Ok(())
        }
        (Val::Md(x), Val::Md(y)) if x == y => Ok(()),
        (Val::Token, Val::Token) => Ok(()),
        _ => Err(fail(path)),
    }
}

/// Whether byte `b` refines byte `a`.
pub fn byte_refines(a: &Byte, ma: &Memory, b: &Byte, mb: &Memory) -> bool {
    if a.poison {
        return true;
    }
    if b.poison || b.undef & !a.undef != 0 {
        return false;
    }
    let exact = !a.undef & !a.choice;
    if b.choice & exact != 0 || (a.v ^ b.v) & exact != 0 {
        return false;
    }
    // A pointer byte must be the same byte of a pointer to the same object.
    if a.frag != b.frag {
        return false;
    }
    let la = (a.prov != 0).then(|| ma.region(a.prov).label.as_str());
    let lb = (b.prov != 0).then(|| mb.region(b.prov).label.as_str());
    la == lb
}

/// The first byte at which `b` does not refine `a`.
pub fn bytes_refine(a: &[Byte], ma: &Memory, b: &[Byte], mb: &Memory) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b.iter()).position(|(x, y)| !byte_refines(x, ma, y, mb))
}
