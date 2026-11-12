//! A CPU interpreter for the IR: LLVM IR semantics as LangRef 21.1.8 defines them, with exact integer arithmetic,
//! IEEE floating point (`half`, `bfloat`, `float`, `double`) and reference implementations of intrinsics.
//!
//! It is the oracle of the canonicalization passes (each pass is checked by running a function before and after it,
//! [`crate::passes`]), of the "bit-exact where the specification is exact" comparisons with real Metal, and the
//! executor's CPU evaluation of shader code (plan 5.11). It is never a rendering fallback.
//!
//! # Values and memory
//!
//! A value is a scalar, a vector or an aggregate ([`Val`]) whose scalars carry LangRef's indefiniteness: poison,
//! undef bits, and the bits whose value the semantics left open and the interpreter chose ([`value`]). Memory is a set
//! of allocated objects, each a region of bytes with the same indefiniteness per byte, addressed by pointers that
//! carry their provenance ([`memory`]). The caller creates the regions a function's arguments point to (buffers,
//! threadgroup memory, objects such as textures) and reads them back afterwards; the module's globals get regions on
//! first use, initialized from their initializers.
//!
//! # What a run reports
//!
//! [`run`] executes one function to its return and gives back the returned value and the [`Event`]s: calls whose
//! effects lie outside the modelled memory (a texture write, a barrier), in order. Undefined behavior stops the run
//! with [`Error::Ub`], naming the instruction: everything after it is unconstrained, so a comparison of two runs
//! accepts any second run after it. Constructs the interpreter does not evaluate stop it with
//! [`Error::Unsupported`]; nothing is ever skipped.
//!
//! # Calls of declarations
//!
//! LLVM intrinsics and the AIR functions with exact definitions (conversions, integer functions, atomics on buffers,
//! function constant queries) are evaluated by their definitions ([`intrinsics`]). Other declarations depend on
//! [`Options::externals`]:
//!
//! - [`Externals::Refuse`]: the call is unsupported. This is the mode for comparisons against real Metal, where every
//!   value must be the one the specification defines.
//! - [`Externals::Uninterpreted`]: the call is an uninterpreted function consistent with its memory effects: its
//!   result is a fixed function (a hash) of the callee, the arguments and the state it may read; a call that may
//!   write state the interpreter models as bytes is unsupported; a call that writes only objects or state outside
//!   the module (textures, barriers) is recorded as an event and bumps the written objects' versions. A pass is
//!   correct for every implementation of such a function exactly when it is correct for an uninterpreted one, which
//!   makes this the mode for checking passes.
//!
//! Comparisons of two runs ([`compare`]) accept, per LangRef's refinement, a second run whose values refine the
//! first's: anything where the first is poison, any defined bit where it is undef, any value where its bit was a
//! choice.

pub mod compare;
pub mod float;
pub mod memory;
pub mod ops;
pub mod value;

mod consts;
mod exec;
mod intrinsics;

use std::collections::HashSet;

pub use compare::{Mismatch, refines};
pub use exec::{eval_instruction, global_region, run};
pub use memory::{Byte, Memory, Region, RegionId, RegionKind};
pub use value::{Scalar, Val};

/// How calls of declarations without an evaluated definition are treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Externals {
    /// Unsupported: every value must come from a definition.
    Refuse,
    /// Uninterpreted functions consistent with the declarations' memory effects.
    Uninterpreted,
}

/// What a run may do.
#[derive(Clone, Debug)]
pub struct Options {
    pub externals: Externals,
    /// Instructions executed before the run stops with [`Error::Limit`].
    pub step_limit: u64,
    /// Nested calls before the run stops with [`Error::Limit`].
    pub depth_limit: usize,
    /// The function constants that have values, by the name of their `air.fc_initializer` global (the operand of
    /// `air.is_function_constant_defined`). `None`: such queries are unsupported (canonicalization removes them).
    pub defined_constants: Option<HashSet<String>>,
}

impl Default for Options {
    fn default() -> Options {
        Options { externals: Externals::Refuse, step_limit: 100_000_000, depth_limit: 256, defined_constants: None }
    }
}

/// A call with effects outside the modelled memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub callee: String,
    pub args: Vec<Val>,
}

/// A finished run.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub ret: Option<Val>,
    pub events: Vec<Event>,
    /// Calls evaluated as uninterpreted functions.
    pub uninterpreted: u64,
    /// Whether a branch, an address or a call depended on a choice bit: the run is one of several the semantics
    /// allow, and another choice could have taken a different path.
    pub choice_path: bool,
    pub steps: u64,
}

/// Why a run stopped early.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Undefined behavior.
    Ub { reason: String, location: String },
    /// A construct the interpreter does not evaluate.
    Unsupported { what: String, location: String },
    /// The step or call depth limit.
    Limit(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Ub { reason, location } => write!(f, "undefined behavior: {} ({})", reason, location),
            Error::Unsupported { what, location } => write!(f, "unsupported: {} ({})", what, location),
            Error::Limit(s) => write!(f, "limit: {}", s),
        }
    }
}

impl std::error::Error for Error {}
