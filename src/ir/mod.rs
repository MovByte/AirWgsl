//! The in-memory IR: an LLVM module as LLVM 21 represents it after reading bitcode.
//!
//! Every later stage of the compiler works on this form, so it is LLVM's semantics without LLVM's object graph: plain
//! data in vectors, referenced by index.
//!
//! # Shape
//!
//! - A [`Module`] owns a [`TypeTable`] (interned types; pointers are opaque, `ptr addrspace(N)`), an attribute-set
//!   pool, a constant pool, the global variables, functions, aliases and ifuncs, the metadata pool and named
//!   metadata, and the context tables bitcode refers to by number (metadata kind names, sync scope names, operand
//!   bundle tags).
//! - A [`Value`] names an operand: a pooled constant ([`ConstId`]), a global ([`GlobalId`]), a function argument, an
//!   instruction result ([`InstId`]), metadata used as a value (call arguments of type `metadata`), or inline
//!   assembly. Argument and instruction values are local to the function whose operands they are.
//! - Constants are uniqued: equal constants have equal [`ConstId`]s. They are canonical in LLVM's way: an aggregate of
//!   zeros is [`ConstKind::Zero`] (`zeroinitializer`), an aggregate of simple integers or floats is a data array or
//!   vector, and constant expressions are folded the way LLVM folds them when it creates them (a bitcast to the same
//!   type is its operand, a cast of `undef` is `undef` or zero, a GEP with all-zero indices is its pointer).
//! - A [`Function`] holds its arguments, its blocks in layout order, and an instruction arena; a block lists its
//!   instructions by [`InstId`]. A declaration has no blocks.
//! - Metadata is pooled and uniqued like LLVM's: strings by content, value wrappers by value, and non-distinct tuples
//!   by operands; distinct tuples are never merged.
//!
//! # What the reader adds and removes
//!
//! The reader applies the upgrades LLVM 21's bitcode reader applies, so that the IR is in the current form whatever
//! LLVM release wrote the bitcode: typed pointers become opaque (the pointee types stay available in
//! [`Module::bitcode_types`]), `nocapture` becomes `captures(none)`, function-level `readnone`/`readonly`/
//! `writeonly`/`argmemonly`/`inaccessiblememonly` become `memory(...)`, typeless `byval`/`sret`/`inalloca` get their
//! pointee type, attributes invalid for a value's type are dropped, and calls whose `strictfp` the caller lacks get
//! `nobuiltin`. Where LLVM 21 would apply an upgrade this reader does not implement, reading fails with an error that
//! names it.
//!
//! Use-list order (`USELIST_BLOCK`) is not part of a program's meaning and the IR does not keep it, with one exception:
//! a block's use order, which decides the order of the predecessors `llvm-dis` lists, is kept in [`Block::use_order`].
//!
//! # Address spaces
//!
//! The IR keeps pointer address spaces as numbers; their meaning is the AIR layer's business. Measured on the system
//! and generated corpora (pointer types, and the `air.address_space` of kernel argument metadata): 0 (thread-private
//! memory), 1 (`device`), 2 (`constant`), 3 (`threadgroup`), 4 (imageblock data, the pointer inside
//! `metal::_imageblock_base`), 6 (the object-to-mesh payload, `air.payload`) and 7 (the mesh output, `air.mesh`).
//! Address space 64 appears only in the x86_64 host modules that ship beside AIR.

pub mod attrs;
pub mod layout;
pub mod print;
pub mod types;

use std::collections::HashMap;

pub use attrs::{AttrKind, AttrList, AttrSet, AttrSetId, Attribute, ConstantRange};
pub use types::{FloatKind, Type, TypeId, TypeTable};

/// Index of a constant in [`Module::consts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstId(pub u32);

/// Index of metadata in [`Module::metadata`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MdId(pub u32);

/// Index of an instruction in its function's [`Function::insts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(pub u32);

/// Index of a block in its function's [`Function::blocks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A global value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GlobalId {
    /// Index into [`Module::globals`].
    Var(u32),
    /// Index into [`Module::functions`].
    Function(u32),
    /// Index into [`Module::aliases`].
    Alias(u32),
    /// Index into [`Module::ifuncs`].
    IFunc(u32),
}

/// An operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    Const(ConstId),
    Global(GlobalId),
    /// An argument of the enclosing function.
    Arg(u32),
    /// The result of an instruction of the enclosing function.
    Inst(InstId),
    /// Metadata passed as a value (`metadata !0`, `metadata i32 1`).
    Metadata(MdId),
    /// Index into [`Module::inline_asms`].
    InlineAsm(u32),
}

// Global values --------------------------------------------------------------------------------------------------------

/// `GlobalValue::LinkageTypes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Linkage {
    External,
    AvailableExternally,
    LinkOnceAny,
    LinkOnceOdr,
    WeakAny,
    WeakOdr,
    Appending,
    Internal,
    Private,
    ExternalWeak,
    Common,
}

impl Linkage {
    pub fn is_local(self) -> bool {
        matches!(self, Linkage::Internal | Linkage::Private)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Visibility {
    #[default]
    Default,
    Hidden,
    Protected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DllStorage {
    #[default]
    Default,
    Import,
    Export,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum UnnamedAddr {
    #[default]
    None,
    Global,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ThreadLocal {
    #[default]
    NotThreadLocal,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}

/// The properties every global value has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalCommon {
    /// Empty for an unnamed global.
    pub name: String,
    pub linkage: Linkage,
    pub visibility: Visibility,
    pub dll_storage: DllStorage,
    pub unnamed_addr: UnnamedAddr,
    pub thread_local: ThreadLocal,
    pub dso_local: bool,
    pub addrspace: u32,
    /// `partition "..."`, empty for none.
    pub partition: String,
}

/// A global variable.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalVariable {
    pub common: GlobalCommon,
    /// The type of the variable's contents.
    pub value_type: TypeId,
    pub constant: bool,
    pub initializer: Option<Value>,
    pub externally_initialized: bool,
    /// Alignment in bytes.
    pub align: Option<u64>,
    pub section: Option<String>,
    /// Index into [`Module::comdats`].
    pub comdat: Option<u32>,
    pub attrs: AttrSetId,
    /// `code_model "..."` (`CodeModel::Model` as written: tiny, small, kernel, medium, large).
    pub code_model: Option<CodeModel>,
    /// `GlobalValue::SanitizerMetadata` bits (no_sanitize_address, no_sanitize_hwaddress, sanitize_memtag,
    /// sanitize_address_dyninit), when any is set.
    pub sanitizer: u32,
    /// Metadata attachments, sorted by kind.
    pub md: Vec<(u32, MdId)>,
    /// Bitcode type id of the variable's value type (typed-pointer bitcode keeps the pointee types there).
    pub bitcode_type: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodeModel {
    Tiny,
    Small,
    Kernel,
    Medium,
    Large,
}

/// A function argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Argument {
    pub ty: TypeId,
    /// `None` when unnamed.
    pub name: Option<String>,
}

/// A basic block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Block {
    pub name: Option<String>,
    pub insts: Vec<InstId>,
    /// The order of the block's uses from a use-list record (`USELIST_CODE_BB`), when the bitcode has one that
    /// applies. LLVM keeps a value's uses newest first; entry `k` is the position, in that default order, of the use
    /// that comes `k`th. Use order has no meaning for the program; it orders the predecessors that `; preds =` lists.
    pub use_order: Option<Vec<u32>>,
}

/// A function: a declaration when `blocks` is empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub common: GlobalCommon,
    /// The function type.
    pub ty: TypeId,
    /// `CallingConv::ID`.
    pub cc: u32,
    pub attrs: AttrList,
    pub align: Option<u64>,
    pub section: Option<String>,
    pub comdat: Option<u32>,
    pub gc: Option<String>,
    pub prefix: Option<Value>,
    pub prologue: Option<Value>,
    pub personality: Option<Value>,
    pub args: Vec<Argument>,
    pub blocks: Vec<Block>,
    pub insts: Vec<Instruction>,
    /// Metadata attachments, sorted by kind.
    pub md: Vec<(u32, MdId)>,
    /// Bitcode type id of the function type; its contained ids are the bitcode types of the return value and
    /// parameters, with the pointee types typed-pointer bitcode recorded.
    pub bitcode_type: Option<u32>,
}

impl Function {
    pub fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn inst(&self, id: InstId) -> &Instruction {
        &self.insts[id.0 as usize]
    }
}

/// A global alias.
#[derive(Clone, Debug, PartialEq)]
pub struct Alias {
    pub common: GlobalCommon,
    pub value_type: TypeId,
    pub aliasee: Value,
}

/// A global ifunc.
#[derive(Clone, Debug, PartialEq)]
pub struct IFunc {
    pub common: GlobalCommon,
    pub value_type: TypeId,
    pub resolver: Value,
}

/// `Comdat::SelectionKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComdatKind {
    Any,
    ExactMatch,
    Largest,
    NoDeduplicate,
    SameSize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comdat {
    pub name: String,
    pub kind: ComdatKind,
}

/// Inline assembly used as a callee.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InlineAsm {
    pub fn_ty: TypeId,
    pub asm: Vec<u8>,
    pub constraints: Vec<u8>,
    pub side_effects: bool,
    pub align_stack: bool,
    /// 0 AT&T, 1 Intel.
    pub dialect: u8,
    pub can_throw: bool,
}

// Constants -----------------------------------------------------------------------------------------------------------

/// A constant: its type and content.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constant {
    pub ty: TypeId,
    pub kind: ConstKind,
}

/// The content of a constant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstKind {
    /// An integer (`ConstantInt`): little-endian 64-bit words, high bits beyond the width zero.
    Int(Vec<u64>),
    /// A floating-point value (`ConstantFP`): the encoding's bits, in the low bits.
    Float(u128),
    /// `zeroinitializer` of an aggregate or vector type (`ConstantAggregateZero`); also the zero of a target
    /// extension type.
    Zero,
    /// `null` (`ConstantPointerNull`).
    NullPtr,
    /// `none` of type token (`ConstantTokenNone`).
    TokenNone,
    Undef,
    Poison,
    /// A struct whose members are not all zero, undef or poison alike.
    Struct(Vec<Value>),
    /// An array that is not data (`ConstantArray`).
    Array(Vec<Value>),
    /// A vector that is not data (`ConstantVector`).
    Vector(Vec<Value>),
    /// An array of `i8`/`i16`/`i32`/`i64`/`half`/`bfloat`/`float`/`double` elements (`ConstantDataArray`): each
    /// element's bits.
    DataArray(Vec<u64>),
    /// A vector of such elements (`ConstantDataVector`).
    DataVector(Vec<u64>),
    Expr(ConstExpr),
    /// `blockaddress(@function, %block)`.
    BlockAddress { function: u32, block: BlockId },
    /// `dso_local_equivalent @g`.
    DsoLocalEquivalent(GlobalId),
    /// `no_cfi @g`.
    NoCfi(GlobalId),
    /// `ptrauth (ptr, i32 key, i64 disc, ptr addrdisc)`.
    PtrAuth { ptr: Value, key: Value, disc: Value, addr_disc: Value },
}

/// A constant expression. Its operands are constants or globals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstExpr {
    /// A cast to the constant's type.
    Cast { op: CastOp, value: Value },
    Binary { op: BinOp, lhs: Value, rhs: Value, flags: u8 },
    Gep { source: TypeId, flags: GepFlags, inrange: Option<ConstantRange>, ptr: Value, indices: Vec<Value> },
    ExtractElement { vec: Value, idx: Value },
    InsertElement { vec: Value, elt: Value, idx: Value },
    ShuffleVector { a: Value, b: Value, mask: Vec<i32> },
}

// Instructions --------------------------------------------------------------------------------------------------------

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    FAdd,
    Sub,
    FSub,
    Mul,
    FMul,
    UDiv,
    SDiv,
    FDiv,
    URem,
    SRem,
    FRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
}

impl BinOp {
    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::FAdd => "fadd",
            BinOp::Sub => "sub",
            BinOp::FSub => "fsub",
            BinOp::Mul => "mul",
            BinOp::FMul => "fmul",
            BinOp::UDiv => "udiv",
            BinOp::SDiv => "sdiv",
            BinOp::FDiv => "fdiv",
            BinOp::URem => "urem",
            BinOp::SRem => "srem",
            BinOp::FRem => "frem",
            BinOp::Shl => "shl",
            BinOp::LShr => "lshr",
            BinOp::AShr => "ashr",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem)
    }
}

/// Integer flags of binary operators and casts.
pub mod flags {
    /// `nuw` (add, sub, mul, shl, trunc).
    pub const NUW: u8 = 1;
    /// `nsw` (add, sub, mul, shl, trunc).
    pub const NSW: u8 = 2;
    /// `exact` (udiv, sdiv, lshr, ashr).
    pub const EXACT: u8 = 4;
    /// `disjoint` (or).
    pub const DISJOINT: u8 = 8;
    /// `nneg` (zext, uitofp).
    pub const NNEG: u8 = 16;
}

/// Cast operators, in `Instruction::CastOps` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CastOp {
    Trunc,
    ZExt,
    SExt,
    FPToUI,
    FPToSI,
    UIToFP,
    SIToFP,
    FPTrunc,
    FPExt,
    PtrToInt,
    IntToPtr,
    BitCast,
    AddrSpaceCast,
}

impl CastOp {
    pub fn name(self) -> &'static str {
        match self {
            CastOp::Trunc => "trunc",
            CastOp::ZExt => "zext",
            CastOp::SExt => "sext",
            CastOp::FPToUI => "fptoui",
            CastOp::FPToSI => "fptosi",
            CastOp::UIToFP => "uitofp",
            CastOp::SIToFP => "sitofp",
            CastOp::FPTrunc => "fptrunc",
            CastOp::FPExt => "fpext",
            CastOp::PtrToInt => "ptrtoint",
            CastOp::IntToPtr => "inttoptr",
            CastOp::BitCast => "bitcast",
            CastOp::AddrSpaceCast => "addrspacecast",
        }
    }

    /// Position in `Instruction::CastOps` (trunc = 0).
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Fast-math flags (`FastMathFlags`): reassoc 1, nnan 2, ninf 4, nsz 8, arcp 16, contract 32, afn 64.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FastMath(pub u8);

impl FastMath {
    pub const REASSOC: u8 = 1;
    pub const NNAN: u8 = 2;
    pub const NINF: u8 = 4;
    pub const NSZ: u8 = 8;
    pub const ARCP: u8 = 16;
    pub const CONTRACT: u8 = 32;
    pub const AFN: u8 = 64;
    pub const ALL: u8 = 127;

    pub fn is_fast(self) -> bool {
        self.0 & FastMath::ALL == FastMath::ALL
    }

    pub fn any(self) -> bool {
        self.0 != 0
    }
}

/// `getelementptr` no-wrap flags (`GEPNoWrapFlags`): inbounds 1, nusw 2, nuw 4; inbounds implies nusw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GepFlags(pub u8);

impl GepFlags {
    pub const INBOUNDS: u8 = 1;
    pub const NUSW: u8 = 2;
    pub const NUW: u8 = 4;

    pub fn inbounds(self) -> bool {
        self.0 & GepFlags::INBOUNDS != 0
    }
}

/// Integer comparison predicates, with `CmpInst::Predicate` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntPredicate {
    Eq = 32,
    Ne = 33,
    Ugt = 34,
    Uge = 35,
    Ult = 36,
    Ule = 37,
    Sgt = 38,
    Sge = 39,
    Slt = 40,
    Sle = 41,
}

impl IntPredicate {
    pub fn from_raw(v: u64) -> Option<IntPredicate> {
        use IntPredicate::*;
        Some(match v {
            32 => Eq,
            33 => Ne,
            34 => Ugt,
            35 => Uge,
            36 => Ult,
            37 => Ule,
            38 => Sgt,
            39 => Sge,
            40 => Slt,
            41 => Sle,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use IntPredicate::*;
        match self {
            Eq => "eq",
            Ne => "ne",
            Ugt => "ugt",
            Uge => "uge",
            Ult => "ult",
            Ule => "ule",
            Sgt => "sgt",
            Sge => "sge",
            Slt => "slt",
            Sle => "sle",
        }
    }
}

/// Floating-point comparison predicates, with `CmpInst::Predicate` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatPredicate {
    False = 0,
    Oeq = 1,
    Ogt = 2,
    Oge = 3,
    Olt = 4,
    Ole = 5,
    One = 6,
    Ord = 7,
    Uno = 8,
    Ueq = 9,
    Ugt = 10,
    Uge = 11,
    Ult = 12,
    Ule = 13,
    Une = 14,
    True = 15,
}

impl FloatPredicate {
    pub fn from_raw(v: u64) -> Option<FloatPredicate> {
        use FloatPredicate::*;
        Some(match v {
            0 => False,
            1 => Oeq,
            2 => Ogt,
            3 => Oge,
            4 => Olt,
            5 => Ole,
            6 => One,
            7 => Ord,
            8 => Uno,
            9 => Ueq,
            10 => Ugt,
            11 => Uge,
            12 => Ult,
            13 => Ule,
            14 => Une,
            15 => True,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use FloatPredicate::*;
        match self {
            False => "false",
            Oeq => "oeq",
            Ogt => "ogt",
            Oge => "oge",
            Olt => "olt",
            Ole => "ole",
            One => "one",
            Ord => "ord",
            Uno => "uno",
            Ueq => "ueq",
            Ugt => "ugt",
            Uge => "uge",
            Ult => "ult",
            Ule => "ule",
            Une => "une",
            True => "true",
        }
    }
}

/// `AtomicOrdering`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    Unordered,
    Monotonic,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl Ordering {
    pub fn name(self) -> &'static str {
        match self {
            Ordering::Unordered => "unordered",
            Ordering::Monotonic => "monotonic",
            Ordering::Acquire => "acquire",
            Ordering::Release => "release",
            Ordering::AcqRel => "acq_rel",
            Ordering::SeqCst => "seq_cst",
        }
    }
}

/// The atomic part of a load or store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Atomic {
    pub ordering: Ordering,
    /// Index into [`Module::sync_scopes`].
    pub scope: u32,
}

/// `AtomicRMWInst::BinOp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RmwOp {
    Xchg,
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Max,
    Min,
    UMax,
    UMin,
    FAdd,
    FSub,
    FMax,
    FMin,
    FMaximum,
    FMinimum,
    UIncWrap,
    UDecWrap,
    USubCond,
    USubSat,
}

impl RmwOp {
    pub fn name(self) -> &'static str {
        use RmwOp::*;
        match self {
            Xchg => "xchg",
            Add => "add",
            Sub => "sub",
            And => "and",
            Nand => "nand",
            Or => "or",
            Xor => "xor",
            Max => "max",
            Min => "min",
            UMax => "umax",
            UMin => "umin",
            FAdd => "fadd",
            FSub => "fsub",
            FMax => "fmax",
            FMin => "fmin",
            FMaximum => "fmaximum",
            FMinimum => "fminimum",
            UIncWrap => "uinc_wrap",
            UDecWrap => "udec_wrap",
            USubCond => "usub_cond",
            USubSat => "usub_sat",
        }
    }
}

/// `CallInst::TailCallKind`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TailKind {
    #[default]
    None,
    Tail,
    MustTail,
    NoTail,
}

/// An operand bundle of a call.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bundle {
    /// Index into [`Module::bundle_tags`].
    pub tag: u32,
    pub inputs: Vec<Value>,
}

/// An instruction.
#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    /// The result type; `void` when there is none.
    pub ty: TypeId,
    pub name: Option<String>,
    pub op: Op,
    /// Metadata attachments other than `!dbg`, sorted by kind.
    pub md: Vec<(u32, MdId)>,
    /// The `!dbg` location.
    pub dbg: Option<MdId>,
}

/// What an instruction does, with its operands.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    FNeg { value: Value, fmf: FastMath },
    Binary { op: BinOp, lhs: Value, rhs: Value, flags: u8, fmf: FastMath },
    /// A cast to the instruction's type. `flags`: [`flags::NNEG`] (zext, uitofp), [`flags::NUW`]/[`flags::NSW`]
    /// (trunc).
    Cast { op: CastOp, value: Value, flags: u8, fmf: FastMath },
    ICmp { pred: IntPredicate, lhs: Value, rhs: Value, samesign: bool },
    FCmp { pred: FloatPredicate, lhs: Value, rhs: Value, fmf: FastMath },
    Select { cond: Value, t: Value, f: Value, fmf: FastMath },
    Phi { incoming: Vec<(Value, BlockId)>, fmf: FastMath },
    Alloca { allocated: TypeId, count: Value, align: u64, addrspace: u32, inalloca: bool, swifterror: bool },
    /// A load of the instruction's type.
    Load { ptr: Value, align: u64, volatile: bool, atomic: Option<Atomic> },
    Store { value: Value, ptr: Value, align: u64, volatile: bool, atomic: Option<Atomic> },
    Gep { source: TypeId, ptr: Value, indices: Vec<Value>, flags: GepFlags },
    ExtractElement { vec: Value, idx: Value },
    InsertElement { vec: Value, elt: Value, idx: Value },
    /// `mask` elements are lane numbers or -1 for poison.
    ShuffleVector { a: Value, b: Value, mask: Vec<i32> },
    ExtractValue { agg: Value, indices: Vec<u32> },
    InsertValue { agg: Value, value: Value, indices: Vec<u32> },
    Call {
        callee: Value,
        fn_ty: TypeId,
        args: Vec<Value>,
        attrs: AttrList,
        cc: u32,
        tail: TailKind,
        fmf: FastMath,
        bundles: Vec<Bundle>,
    },
    Freeze { value: Value },
    AtomicRmw { op: RmwOp, ptr: Value, value: Value, align: u64, volatile: bool, ordering: Ordering, scope: u32 },
    CmpXchg {
        ptr: Value,
        cmp: Value,
        new: Value,
        align: u64,
        volatile: bool,
        weak: bool,
        success: Ordering,
        failure: Ordering,
        scope: u32,
    },
    Fence { ordering: Ordering, scope: u32 },
    VaArg { list: Value },
    Ret { value: Option<Value> },
    Br { dest: BlockId },
    CondBr { cond: Value, t: BlockId, f: BlockId },
    Switch { cond: Value, default: BlockId, cases: Vec<(ConstId, BlockId)> },
    IndirectBr { addr: Value, dests: Vec<BlockId> },
    Unreachable,
}

impl Op {
    pub fn is_terminator(&self) -> bool {
        matches!(self, Op::Ret { .. } | Op::Br { .. } | Op::CondBr { .. } | Op::Switch { .. } | Op::IndirectBr { .. } | Op::Unreachable)
    }

    /// The instruction's value operands in LLVM's operand order, without block operands.
    pub fn operands(&self) -> Vec<Value> {
        match self {
            Op::FNeg { value, .. } | Op::Cast { value, .. } | Op::Freeze { value } => vec![*value],
            Op::Binary { lhs, rhs, .. } | Op::ICmp { lhs, rhs, .. } | Op::FCmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Select { cond, t, f, .. } => vec![*cond, *t, *f],
            Op::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
            Op::Alloca { count, .. } => vec![*count],
            Op::Load { ptr, .. } => vec![*ptr],
            Op::Store { value, ptr, .. } => vec![*value, *ptr],
            Op::Gep { ptr, indices, .. } => std::iter::once(*ptr).chain(indices.iter().copied()).collect(),
            Op::ExtractElement { vec, idx } => vec![*vec, *idx],
            Op::InsertElement { vec, elt, idx } => vec![*vec, *elt, *idx],
            Op::ShuffleVector { a, b, .. } => vec![*a, *b],
            Op::ExtractValue { agg, .. } => vec![*agg],
            Op::InsertValue { agg, value, .. } => vec![*agg, *value],
            Op::Call { callee, args, bundles, .. } => {
                let mut v: Vec<Value> = args.clone();
                for b in bundles {
                    v.extend(b.inputs.iter().copied());
                }
                v.push(*callee);
                v
            }
            Op::AtomicRmw { ptr, value, .. } => vec![*ptr, *value],
            Op::CmpXchg { ptr, cmp, new, .. } => vec![*ptr, *cmp, *new],
            Op::Fence { .. } | Op::Unreachable | Op::Br { .. } => vec![],
            Op::VaArg { list } => vec![*list],
            Op::Ret { value } => value.iter().copied().collect(),
            Op::CondBr { cond, .. } => vec![*cond],
            Op::Switch { cond, cases, .. } => {
                std::iter::once(*cond).chain(cases.iter().map(|(c, _)| Value::Const(*c))).collect()
            }
            Op::IndirectBr { addr, .. } => vec![*addr],
        }
    }

    /// The successor blocks of a terminator, in LLVM's order.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Op::Br { dest } => vec![*dest],
            Op::CondBr { t, f, .. } => vec![*t, *f],
            Op::Switch { default, cases, .. } => std::iter::once(*default).chain(cases.iter().map(|(_, b)| *b)).collect(),
            Op::IndirectBr { dests, .. } => dests.clone(),
            _ => vec![],
        }
    }
}

// Metadata ------------------------------------------------------------------------------------------------------------

/// A metadata entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Metadata {
    /// `!"..."` (`MDString`), arbitrary bytes.
    String(Vec<u8>),
    /// A constant or global as metadata (`ConstantAsMetadata`), with its type.
    Value { ty: TypeId, value: Value },
    /// An argument or instruction of function `function` as metadata (`LocalAsMetadata`).
    Local { function: u32, ty: TypeId, value: Value },
    /// `!{...}` or `distinct !{...}` (`MDTuple`); `None` operands are null.
    Tuple { distinct: bool, ops: Vec<Option<MdId>> },
    /// `!DIArgList(...)` of value metadata.
    ArgList(Vec<MdId>),
}

/// Named metadata (`!name = !{...}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedMetadata {
    pub name: String,
    pub ops: Vec<MdId>,
}

/// The wrapper header that precedes bitcode in a metallib or an `.air` file (`BitcodeWrapperHeader`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrapperHeader {
    pub version: u32,
    pub offset: u32,
    pub size: u32,
    pub cputype: u32,
}

/// The metadata kinds LLVM 21 registers in every context (`FixedMetadataKinds.def`), id = position.
pub const FIXED_MD_KINDS: &[&str] = &[
    "dbg",
    "tbaa",
    "prof",
    "fpmath",
    "range",
    "tbaa.struct",
    "invariant.load",
    "alias.scope",
    "noalias",
    "nontemporal",
    "llvm.mem.parallel_loop_access",
    "nonnull",
    "dereferenceable",
    "dereferenceable_or_null",
    "make.implicit",
    "unpredictable",
    "invariant.group",
    "align",
    "llvm.loop",
    "type",
    "section_prefix",
    "absolute_symbol",
    "associated",
    "callees",
    "irr_loop",
    "llvm.access.group",
    "callback",
    "llvm.preserve.access.index",
    "vcall_visibility",
    "noundef",
    "annotation",
    "nosanitize",
    "func_sanitize",
    "exclude",
    "memprof",
    "callsite",
    "kcfi_type",
    "pcsections",
    "DIAssignID",
    "coro.outside.frame",
    "mmra",
    "noalias.addrspace",
];

/// The `!tbaa` kind id.
pub const MD_TBAA: u32 = 1;

/// A type of the bitcode's own type table: the IR type it became and, for typed-pointer bitcode, the contained
/// bitcode type ids (a pointer's pointee; a function's return and parameter types; aggregate members).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitcodeType {
    pub ty: TypeId,
    pub contained: Vec<u32>,
}

/// A module.
#[derive(Clone, Debug)]
pub struct Module {
    /// The producer string of the identification block, when there is one.
    pub producer: Option<String>,
    pub wrapper: Option<WrapperHeader>,
    pub source_filename: Option<Vec<u8>>,
    pub triple: String,
    pub datalayout: String,
    pub module_asm: Vec<u8>,
    pub types: TypeTable,
    pub attr_sets: Vec<AttrSet>,
    pub consts: Vec<Constant>,
    pub globals: Vec<GlobalVariable>,
    pub functions: Vec<Function>,
    pub aliases: Vec<Alias>,
    pub ifuncs: Vec<IFunc>,
    pub comdats: Vec<Comdat>,
    pub inline_asms: Vec<InlineAsm>,
    pub metadata: Vec<Metadata>,
    pub named_metadata: Vec<NamedMetadata>,
    /// Metadata kind names by id: [`FIXED_MD_KINDS`] then the module's own in order of first use.
    pub md_kinds: Vec<String>,
    /// Sync scope names by id: 0 is `singlethread`, 1 is the system scope (empty), then the module's own.
    pub sync_scopes: Vec<String>,
    pub bundle_tags: Vec<String>,
    /// The bitcode's type table, by bitcode type id (pointer pointee types survive here).
    pub bitcode_types: Vec<BitcodeType>,
    attr_set_ids: HashMap<AttrSet, AttrSetId>,
    const_ids: HashMap<Constant, ConstId>,
    md_unique: HashMap<Metadata, MdId>,
}

impl Default for Module {
    fn default() -> Module {
        Module::new()
    }
}

impl Module {
    pub fn new() -> Module {
        let mut m = Module {
            producer: None,
            wrapper: None,
            source_filename: None,
            triple: String::new(),
            datalayout: String::new(),
            module_asm: Vec::new(),
            types: TypeTable::new(),
            attr_sets: Vec::new(),
            consts: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            aliases: Vec::new(),
            ifuncs: Vec::new(),
            comdats: Vec::new(),
            inline_asms: Vec::new(),
            metadata: Vec::new(),
            named_metadata: Vec::new(),
            md_kinds: FIXED_MD_KINDS.iter().map(|s| s.to_string()).collect(),
            sync_scopes: vec!["singlethread".into(), String::new()],
            bundle_tags: Vec::new(),
            bitcode_types: Vec::new(),
            attr_set_ids: HashMap::new(),
            const_ids: HashMap::new(),
            md_unique: HashMap::new(),
        };
        m.attr_set(AttrSet::new());
        m
    }

    /// Interns an attribute set.
    pub fn attr_set(&mut self, set: AttrSet) -> AttrSetId {
        if let Some(&id) = self.attr_set_ids.get(&set) {
            return id;
        }
        let id = AttrSetId(self.attr_sets.len() as u32);
        self.attr_sets.push(set.clone());
        self.attr_set_ids.insert(set, id);
        id
    }

    pub fn attrs(&self, id: AttrSetId) -> &AttrSet {
        &self.attr_sets[id.0 as usize]
    }

    /// Interns a constant as given (canonicalization is the constructor's job, see `bitcode::constants`).
    pub fn intern_const(&mut self, c: Constant) -> ConstId {
        if let Some(&id) = self.const_ids.get(&c) {
            return id;
        }
        let id = ConstId(self.consts.len() as u32);
        self.consts.push(c.clone());
        self.const_ids.insert(c, id);
        id
    }

    /// Replaces the constant pool (same ids, rewritten contents) and rebuilds its uniquing table.
    pub fn replace_consts(&mut self, consts: Vec<Constant>) {
        self.consts = consts;
        self.const_ids.clear();
        for (i, c) in self.consts.iter().enumerate() {
            self.const_ids.entry(c.clone()).or_insert(ConstId(i as u32));
        }
    }

    pub fn constant(&self, id: ConstId) -> &Constant {
        &self.consts[id.0 as usize]
    }

    /// Adds metadata; strings, values and non-distinct tuples are uniqued, distinct tuples always added.
    pub fn intern_md(&mut self, md: Metadata) -> MdId {
        let unique = !matches!(md, Metadata::Tuple { distinct: true, .. });
        if unique {
            if let Some(&id) = self.md_unique.get(&md) {
                return id;
            }
        }
        let id = MdId(self.metadata.len() as u32);
        self.metadata.push(md.clone());
        if unique {
            self.md_unique.insert(md, id);
        }
        id
    }

    /// Adds metadata without uniquing (for forward-reference placeholders the reader fills in later).
    pub fn push_md(&mut self, md: Metadata) -> MdId {
        let id = MdId(self.metadata.len() as u32);
        self.metadata.push(md);
        id
    }

    /// Replaces the content of metadata `id` (used by the reader to fill placeholders) without uniquing.
    pub fn set_md(&mut self, id: MdId, md: Metadata) {
        self.metadata[id.0 as usize] = md;
    }

    /// Rebuilds the uniquing table after the reader rewrote metadata.
    pub fn rebuild_md_unique(&mut self) {
        self.md_unique.clear();
        for (i, md) in self.metadata.iter().enumerate() {
            if !matches!(md, Metadata::Tuple { distinct: true, .. }) {
                self.md_unique.entry(md.clone()).or_insert(MdId(i as u32));
            }
        }
    }

    pub fn md(&self, id: MdId) -> &Metadata {
        &self.metadata[id.0 as usize]
    }

    /// The common properties of a global.
    pub fn global(&self, g: GlobalId) -> &GlobalCommon {
        match g {
            GlobalId::Var(i) => &self.globals[i as usize].common,
            GlobalId::Function(i) => &self.functions[i as usize].common,
            GlobalId::Alias(i) => &self.aliases[i as usize].common,
            GlobalId::IFunc(i) => &self.ifuncs[i as usize].common,
        }
    }

    /// The type of a module-level value (constant, global, metadata or inline asm); `None` for function-local values.
    pub fn module_value_type(&mut self, v: Value) -> Option<TypeId> {
        match v {
            Value::Const(c) => Some(self.constant(c).ty),
            Value::Global(g) => {
                let a = self.global(g).addrspace;
                Some(self.types.ptr(a))
            }
            Value::Metadata(_) => Some(self.types.intern(Type::Metadata)),
            Value::InlineAsm(_) => Some(self.types.ptr(0)),
            Value::Arg(_) | Value::Inst(_) => None,
        }
    }

    /// The type of `v` used in function `f`.
    pub fn value_type(&mut self, f: usize, v: Value) -> TypeId {
        match v {
            Value::Arg(i) => self.functions[f].args[i as usize].ty,
            Value::Inst(i) => self.functions[f].insts[i.0 as usize].ty,
            other => self.module_value_type(other).unwrap(),
        }
    }

    /// The function named `name`.
    pub fn function_by_name(&self, name: &str) -> Option<usize> {
        self.functions.iter().position(|f| f.common.name == name)
    }
}
