//! Numerics: the lowering of AIR's arithmetic intrinsics and floating-point instructions to WGSL.
//!
//! The WGSL emitter calls [`lower`] for each `air.*` or `llvm.*` call of the numeric families (math, geometric,
//! integer, relational, conversion, pack and unpack, derivatives) and for each floating-point LLVM instruction. The
//! result is WGSL text over the value carriers below plus the library items it calls; [`prelude::Prelude`] emits those
//! items once per module.
//!
//! # Carriers
//!
//! Every value is carried as integer bits ([`Ty::carrier`]): binary32 as `u32`, binary16 in the low 16 bits of a
//! `u32`, i8, i16 and i32 as `u32` holding the value's bits zero-extended, i64 as `vec2<u32>` (low word, high word),
//! i1 as `bool`. Vectors of 2 to 4 lanes are `vecN` of those (`array` for i64 and for 8 or 16 lanes). Floating-point
//! values never live in a WGSL `f32` between operations: an implementation may assume NaN and infinity never occur
//! and may flush subnormals (WGSL 15.7.2), which would lose the bits precise MSL code depends on.
//!
//! # Decisions
//!
//! Each floating-point operation is lowered by its numerics.toml row ([`rows`]): fast-math single precision by
//! `fast_f32`, precise (NaN and infinity honored) by `precise_f32`, half precision by `half`. A call is fast when its
//! name is the `fast_` variant, or, for functions without one and for instructions, when its fast-math flags include
//! `nnan` and `ninf`. Polyfill rows use the integer soft-float library unless the [`Rounding`] policy carries a host
//! proof that native WGSL arithmetic rounds to nearest even ([`selftest`]); precise rows always use it.
//!
//! Half precision follows the table's `half` column (native: MSL states no half-precision bound for the target's GPU
//! family) for fast-math calls. A call that must honor NaN and infinity uses the exact binary16 library instead: MSL
//! 8.1 requires NaN and infinity support for half precision with fast math disabled, which native WGSL `f16` does
//! not give, as for single precision (numerics.toml generic precise decision).

pub mod prelude;
pub mod rows;
pub mod selftest;

mod lower;

pub use lower::lower;

use crate::api::{HostCaps, HostRequirements};
use prelude::Prelude;
use std::collections::BTreeSet;

/// Element types of the values the numerics lowering handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
}

impl Scalar {
    pub fn bits(self) -> u32 {
        match self {
            Scalar::I1 => 1,
            Scalar::I8 => 8,
            Scalar::I16 | Scalar::F16 => 16,
            Scalar::I32 | Scalar::F32 => 32,
            Scalar::I64 => 64,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::F32)
    }

    /// The element part of an AIR or LLVM type suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            Scalar::I1 => "i1",
            Scalar::I8 => "i8",
            Scalar::I16 => "i16",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::F16 => "f16",
            Scalar::F32 => "f32",
        }
    }

    /// The WGSL carrier of one element.
    pub fn carrier(self) -> &'static str {
        match self {
            Scalar::I1 => "bool",
            Scalar::I64 => "vec2<u32>",
            _ => "u32",
        }
    }
}

/// A scalar (`lanes` 1) or vector type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ty {
    pub scalar: Scalar,
    pub lanes: u32,
}

impl Ty {
    pub const fn scalar(scalar: Scalar) -> Ty {
        Ty { scalar, lanes: 1 }
    }

    pub const fn vector(scalar: Scalar, lanes: u32) -> Ty {
        Ty { scalar, lanes }
    }

    /// The type as AIR spells it in a function name: `f32`, `v4f32`.
    pub fn suffix(&self) -> String {
        if self.lanes == 1 { self.scalar.suffix().to_string() } else { format!("v{}{}", self.lanes, self.scalar.suffix()) }
    }

    /// Whether the vector form is a WGSL `vecN` (otherwise an `array`).
    pub fn is_wgsl_vector(&self) -> bool {
        self.lanes > 1 && self.lanes <= 4 && self.scalar != Scalar::I64
    }

    /// The WGSL type that carries a value of this type.
    pub fn carrier(&self) -> String {
        let e = self.scalar.carrier();
        if self.lanes == 1 {
            e.to_string()
        } else if self.is_wgsl_vector() {
            format!("vec{}<{}>", self.lanes, e)
        } else {
            format!("array<{}, {}>", e, self.lanes)
        }
    }

    pub fn element(&self) -> Ty {
        Ty::scalar(self.scalar)
    }
}

/// LLVM fast-math flags of an instruction or call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FastMath {
    pub nnan: bool,
    pub ninf: bool,
    pub nsz: bool,
    pub arcp: bool,
    pub contract: bool,
    pub afn: bool,
    pub reassoc: bool,
}

impl FastMath {
    /// No flags: IEEE semantics, NaN and infinity honored.
    pub const NONE: FastMath =
        FastMath { nnan: false, ninf: false, nsz: false, arcp: false, contract: false, afn: false, reassoc: false };
    /// LLVM's `fast`: every flag.
    pub const FAST: FastMath =
        FastMath { nnan: true, ninf: true, nsz: true, arcp: true, contract: true, afn: true, reassoc: true };

    /// NaN and infinity may be assumed absent: MSL's fast-math semantics (MSL 8.1).
    pub fn finite_only(&self) -> bool {
        self.nnan && self.ninf
    }
}

/// How the fast-math single-precision rows the numerics table marks polyfill are lowered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// The default, exact on every host: integer soft-float for every polyfill row.
    SoftFloat,
    /// The host's native binary32 arithmetic was shown to round to nearest even ([`selftest`]); the rows whose
    /// polyfill exists only for the rounding direction ([`rows::ROUNDING_ONLY`]) use it.
    Native(NativeRounding),
}

/// What a passing rounding self-test showed. Only [`selftest::Report::native_rounding`] makes one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeRounding {
    /// `fma` rounded once, to nearest even, in every case: native `fma` may serve MSL's correctly rounded fma.
    pub(crate) fused_fma: bool,
    /// The self-test summary, for the executor's log.
    pub(crate) evidence: String,
}

impl NativeRounding {
    pub fn fused_fma(&self) -> bool {
        self.fused_fma
    }

    pub fn evidence(&self) -> &str {
        &self.evidence
    }
}

/// What the lowering targets: the rounding policy and the device's features (`shader-f16`).
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    pub rounding: &'a Rounding,
    pub host: &'a HostCaps,
}

impl Target<'_> {
    pub fn has_f16(&self) -> bool {
        self.host.features.contains("shader-f16")
    }
}

/// An operand: a WGSL expression in the carrier of `ty`. `constant` holds the bits of a constant scalar operand
/// (a `1.0` numerator makes an `fdiv` the reciprocal row).
#[derive(Clone, Copy, Debug)]
pub struct Operand<'a> {
    pub expr: &'a str,
    pub ty: Ty,
    pub constant: Option<u64>,
}

/// LLVM floating-point comparison predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FPred {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

/// LLVM instructions with numeric semantics the lowering owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Inst {
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    FNeg,
    FCmp(FPred),
    FPToSI,
    FPToUI,
    SIToFP,
    UIToFP,
    FPTrunc,
    FPExt,
}

/// What to lower.
#[derive(Clone, Copy, Debug)]
pub enum Op<'a> {
    /// A call to an `air.*` or `llvm.*` function, by name.
    Call(&'a str),
    Inst(Inst),
}

/// A store the lowered operation makes through a pointer operand (`sincos` returns the cosine that way).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    /// The pointer operand's index.
    pub operand: usize,
    /// The value, in the carrier of `ty`.
    pub value: String,
    pub ty: Ty,
}

/// How the lowering was decided, for the executor's log and the tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// The numerics.toml row, for floating-point operations.
    pub row: Option<&'static str>,
    pub column: Column,
    pub how: How,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Column {
    FastF32,
    PreciseF32,
    Half,
    /// Integer, conversion, pack and relational operations and bit operations on floats: exact by definition.
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum How {
    /// WGSL's own operation.
    Native,
    /// The integer soft-float library.
    SoftFloat,
    /// Native WGSL arithmetic under a host rounding proof.
    NativeRne,
    /// Exact integer code (integers, conversions, packing, selections and other bit operations).
    Exact,
}

/// A lowered operation.
#[derive(Clone, Debug, PartialEq)]
pub struct Lowered {
    /// Statements to place before the value is used, in order (`let` bindings named from the caller's prefix).
    pub stmts: Vec<String>,
    /// The result in the carrier of the result type; `None` for a call without a result.
    pub value: Option<String>,
    pub stores: Vec<Store>,
    /// Library items the code calls.
    pub prelude: Prelude,
    /// WebGPU features and WGSL language features the code needs.
    pub requirements: HostRequirements,
    /// WGSL `enable` directives the module needs (`f16`).
    pub enables: BTreeSet<String>,
    pub rule: Rule,
}
