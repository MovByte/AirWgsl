//! Constants (`parseConstants`, `materializeValue` in `BitcodeReader.cpp`) and the canonical forms and folds LLVM
//! applies when it creates them (`Constants.cpp`, `ConstantFold.cpp`, `CastInst::isEliminableCastPair` in
//! `Instructions.cpp`).
//!
//! Records are read into [`RawConst`]s first, because aggregates and expressions may refer to constants defined later
//! in the block; they become pooled [`Constant`]s when first used.

use super::codes::*;
use super::{Reader, Result, Slot, VEntry};
use crate::bitstream::Entry;
use crate::ir::*;

/// `decodeSignRotatedValue`: sign in the low bit.
pub fn decode_sign_rotated(v: u64) -> u64 {
    if v & 1 == 0 {
        v >> 1
    } else if v != 1 {
        (v >> 1).wrapping_neg()
    } else {
        1u64 << 63
    }
}

/// `readWideAPInt`: sign-rotated words, low word first, as a `bits`-bit value.
pub fn wide_words(vals: &[u64], bits: u32) -> Vec<u64> {
    let n = bits.div_ceil(64) as usize;
    let mut words = vec![0u64; n];
    for (i, v) in vals.iter().enumerate().take(n) {
        words[i] = decode_sign_rotated(*v);
    }
    mask_words(&mut words, bits);
    words
}

/// Clears the bits above `bits`.
pub fn mask_words(words: &mut [u64], bits: u32) {
    let n = bits.div_ceil(64) as usize;
    for w in words.iter_mut().skip(n) {
        *w = 0;
    }
    if !bits.is_multiple_of(64) && n > 0 {
        words[n - 1] &= (1u64 << (bits % 64)) - 1;
    }
}

fn words_all_ones(words: &[u64], bits: u32) -> bool {
    let mut full = vec![u64::MAX; words.len()];
    mask_words(&mut full, bits);
    words == full.as_slice()
}

fn words_zero(words: &[u64]) -> bool {
    words.iter().all(|&w| w == 0)
}

pub fn range_is_full(cr: &ConstantRange) -> bool {
    cr.lower == cr.upper && words_all_ones(&cr.lower, cr.bits)
}

/// `ConstantRangeList::isOrderedRanges`: non-empty, non-wrapping ranges in increasing order that do not touch.
pub fn ranges_ordered(list: &[ConstantRange]) -> bool {
    let signed_lt = |a: &[u64], b: &[u64], bits: u32| -> bool {
        // Compare as signed bits-wide integers.
        let sa = sign_of(a, bits);
        let sb = sign_of(b, bits);
        if sa != sb {
            return sa;
        }
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return a[i] < b[i];
            }
        }
        false
    };
    for (i, r) in list.iter().enumerate() {
        if r.lower == r.upper {
            return false;
        }
        // Non-wrapping as signed: lower < upper.
        if !signed_lt(&r.lower, &r.upper, r.bits) {
            return false;
        }
        if i > 0 {
            let prev = &list[i - 1];
            // Each range must start strictly after the previous one ends.
            if !signed_lt(&prev.upper, &r.lower, r.bits) {
                return false;
            }
        }
    }
    true
}

fn sign_of(words: &[u64], bits: u32) -> bool {
    let top = (bits - 1) as usize;
    (words[top / 64] >> (top % 64)) & 1 == 1
}

/// A constant as read, before its operands are resolved.
#[derive(Clone, Debug)]
pub(crate) struct RawConst {
    pub ty: TypeId,
    pub kind: RawKind,
    /// The pooled constant once made.
    pub done: Option<Value>,
    pub busy: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum RawKind {
    /// Already canonical.
    Ready(ConstKind),
    Aggregate(Vec<u64>),
    Cast {
        op: CastOp,
        operand: u64,
    },
    Binary {
        op: BinOp,
        flags: u8,
        lhs: u64,
        rhs: u64,
    },
    FNeg(u64),
    Gep {
        source: TypeId,
        flags: u8,
        inrange: Option<ConstantRange>,
        operands: Vec<u64>,
    },
    Select(u64, u64, u64),
    ExtractElement(u64, u64),
    InsertElement(u64, u64, u64),
    ShuffleVector(u64, u64, u64),
    Cmp {
        pred: u64,
        lhs: u64,
        rhs: u64,
    },
    InlineAsm(u32),
}

pub(crate) fn cast_op(raw: u64) -> Option<CastOp> {
    use CastOp::*;
    Some(match raw {
        0 => Trunc,
        1 => ZExt,
        2 => SExt,
        3 => FPToUI,
        4 => FPToSI,
        5 => UIToFP,
        6 => SIToFP,
        7 => FPTrunc,
        8 => FPExt,
        9 => PtrToInt,
        10 => IntToPtr,
        11 => BitCast,
        12 => AddrSpaceCast,
        _ => return None,
    })
}

/// `getDecodedBinaryOpcode`.
pub(crate) fn binary_op(raw: u64, is_float: bool) -> Option<BinOp> {
    use BinOp::*;
    Some(match (raw, is_float) {
        (0, false) => Add,
        (0, true) => FAdd,
        (1, false) => Sub,
        (1, true) => FSub,
        (2, false) => Mul,
        (2, true) => FMul,
        (3, false) => UDiv,
        (4, false) => SDiv,
        (4, true) => FDiv,
        (5, false) => URem,
        (6, false) => SRem,
        (6, true) => FRem,
        (7, false) => Shl,
        (8, false) => LShr,
        (9, false) => AShr,
        (10, false) => And,
        (11, false) => Or,
        (12, false) => Xor,
        _ => return None,
    })
}

/// `CastInst::castIsValid`.
pub(crate) fn cast_is_valid(types: &TypeTable, op: CastOp, src: TypeId, dst: TypeId) -> bool {
    use CastOp::*;
    let sv = types.vector_len(src);
    let dv = types.vector_len(dst);
    let sscalable = matches!(types.get(src), Type::Vector { scalable: true, .. });
    let dscalable = matches!(types.get(dst), Type::Vector { scalable: true, .. });
    let same_count = sv == dv && sscalable == dscalable;
    let sbits = types.scalar_bits(src);
    let dbits = types.scalar_bits(dst);
    match op {
        Trunc => types.is_int_or_int_vector(src) && types.is_int_or_int_vector(dst) && same_count && sbits > dbits,
        ZExt | SExt => {
            types.is_int_or_int_vector(src) && types.is_int_or_int_vector(dst) && same_count && sbits < dbits
        }
        FPTrunc => types.is_fp_or_fp_vector(src) && types.is_fp_or_fp_vector(dst) && same_count && sbits > dbits,
        FPExt => types.is_fp_or_fp_vector(src) && types.is_fp_or_fp_vector(dst) && same_count && sbits < dbits,
        UIToFP | SIToFP => types.is_int_or_int_vector(src) && types.is_fp_or_fp_vector(dst) && same_count,
        FPToUI | FPToSI => types.is_fp_or_fp_vector(src) && types.is_int_or_int_vector(dst) && same_count,
        PtrToInt => types.is_ptr_or_ptr_vector(src) && types.is_int_or_int_vector(dst) && same_count,
        IntToPtr => types.is_int_or_int_vector(src) && types.is_ptr_or_ptr_vector(dst) && same_count,
        BitCast => {
            let sp = types.is_ptr_or_ptr_vector(src);
            let dp = types.is_ptr_or_ptr_vector(dst);
            if sp != dp {
                return false;
            }
            if sp {
                // Pointers: same address space, and vector-ness preserved with equal counts.
                return types.pointer_addrspace(src) == types.pointer_addrspace(dst) && same_count;
            }
            if !types.is_first_class(src)
                || !types.is_first_class(dst)
                || types.is_aggregate(src)
                || types.is_aggregate(dst)
            {
                return false;
            }
            // Primitive sizes must match.
            let size = |t: TypeId| -> Option<u64> {
                match types.get(t) {
                    Type::Vector { len, elem, .. } => Some(*len as u64 * types.scalar_bits(*elem) as u64),
                    Type::Int(b) => Some(*b as u64),
                    _ => types.float_kind(t).map(|k| k.bits() as u64),
                }
            };
            matches!((size(src), size(dst)), (Some(a), Some(b)) if a == b && a != 0)
        }
        AddrSpaceCast => {
            types.is_ptr_or_ptr_vector(src)
                && types.is_ptr_or_ptr_vector(dst)
                && types.pointer_addrspace(src) != types.pointer_addrspace(dst)
                && same_count
        }
    }
}

/// `ConstantExpr::isSupportedCastOp` (also `isDesirableCastOp`) in LLVM 21.
fn supported_cast(op: CastOp) -> bool {
    matches!(op, CastOp::Trunc | CastOp::PtrToInt | CastOp::IntToPtr | CastOp::BitCast | CastOp::AddrSpaceCast)
}

/// `CastInst::isEliminableCastPair` with a 64-bit fake integer pointer type in the middle and none at the ends, as
/// `foldConstantCastPair` calls it. Returns the opcode of the single replacing cast.
fn eliminable_cast_pair(
    types: &TypeTable,
    first: CastOp,
    second: CastOp,
    src: TypeId,
    mid: TypeId,
    dst: TypeId,
) -> Option<CastOp> {
    use CastOp::*;
    const TABLE: [[u8; 13]; 13] = [
        [1, 0, 0, 99, 99, 0, 0, 99, 99, 99, 0, 3, 0],
        [8, 1, 9, 99, 99, 2, 17, 99, 99, 99, 2, 3, 0],
        [8, 0, 1, 99, 99, 0, 2, 99, 99, 99, 0, 3, 0],
        [0, 0, 0, 99, 99, 0, 0, 99, 99, 99, 0, 3, 0],
        [0, 0, 0, 99, 99, 0, 0, 99, 99, 99, 0, 3, 0],
        [99, 99, 99, 0, 0, 99, 99, 0, 0, 99, 99, 4, 0],
        [99, 99, 99, 0, 0, 99, 99, 0, 0, 99, 99, 4, 0],
        [99, 99, 99, 0, 0, 99, 99, 0, 0, 99, 99, 4, 0],
        [99, 99, 99, 2, 2, 99, 99, 8, 2, 99, 99, 4, 0],
        [1, 0, 0, 99, 99, 0, 0, 99, 99, 99, 7, 3, 0],
        [99, 99, 99, 99, 99, 99, 99, 99, 99, 11, 99, 15, 0],
        [5, 5, 5, 0, 0, 5, 5, 0, 0, 16, 5, 1, 14],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 12],
    ];
    let first_bitcast = first == BitCast;
    let second_bitcast = second == BitCast;
    if ((first_bitcast && types.is_vector(src) != types.is_vector(mid))
        || (second_bitcast && types.is_vector(mid) != types.is_vector(dst)))
        && !(first_bitcast && second_bitcast)
    {
        return None;
    }
    match TABLE[first.index()][second.index()] {
        0 | 99 => None,
        1 => Some(first),
        2 => Some(second),
        3 => (!types.is_vector(src) && types.is_int(dst)).then_some(first),
        4 => (dst == mid).then_some(first),
        5 => types.is_int(src).then_some(second),
        7 => {
            if types.pointer_addrspace(src) != types.pointer_addrspace(dst) {
                return None;
            }
            // With the middle type the largest possible pointer size (64 bits) the pair is a bitcast; there is no
            // pointer-size information otherwise.
            (types.scalar_bits(mid) == 64).then_some(BitCast)
        }
        8 => {
            let s = types.scalar_bits(src);
            let d = types.scalar_bits(dst);
            if src == dst {
                Some(BitCast)
            } else if s < d {
                Some(first)
            } else if s > d {
                Some(second)
            } else {
                None
            }
        }
        9 => Some(ZExt),
        11 => {
            let s = types.scalar_bits(src);
            let d = types.scalar_bits(dst);
            (s <= 64 && s == d).then_some(BitCast)
        }
        12 => Some(if types.pointer_addrspace(src) != types.pointer_addrspace(dst) { AddrSpaceCast } else { BitCast }),
        13 => Some(first),
        14 => Some(AddrSpaceCast),
        15 => Some(first),
        16 => Some(second),
        17 => Some(UIToFP),
        _ => None,
    }
}

/// How deep constant expressions and aggregates may nest: a limit of this reader (the printer and later stages
/// walk constants recursively).
pub(crate) const MAX_CONST_DEPTH: u32 = 1024;

/// Where a constant is being used: what LLVM does with a constant expression it no longer supports depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Use {
    /// A global initializer or another place that needs a constant: an error.
    Initializer,
    /// A metadata operand: the reference becomes `poison` (`getValueFwdRef` in `MetadataLoader.cpp`).
    Metadata,
    /// An instruction operand: LLVM expands the expression into instructions.
    Instruction,
}

impl<'a> Reader<'a> {
    pub(super) fn read_constants(&mut self, function: Option<u32>) -> Result<()> {
        let _ = function;
        self.cur.enter_block(CONSTANTS_BLOCK_ID)?;
        let mut cur_ty = self.m.types.int(32);
        let mut cur_bc = self.virtual_type(cur_ty, vec![]);
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => break,
                Entry::SubBlock(_) => return self.malformed("sub-block inside a constants block"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            let ops = &r.ops;
            let types = &self.m.types;
            let kind: RawKind = match r.code {
                CST_CODE_SETTYPE => {
                    let t = match ops.first() {
                        Some(&t) => t,
                        None => return self.malformed("invalid SETTYPE record"),
                    };
                    let ty = self.ty(t)?;
                    if self.m.types.is_void(ty) {
                        return self.malformed("constant of type void");
                    }
                    cur_ty = ty;
                    cur_bc = t as u32;
                    continue;
                }
                CST_CODE_UNDEF => RawKind::Ready(ConstKind::Undef),
                CST_CODE_POISON => RawKind::Ready(ConstKind::Poison),
                CST_CODE_NULL => {
                    if matches!(types.get(cur_ty), Type::Void | Type::Function { .. } | Type::Label | Type::Metadata) {
                        return self.malformed("invalid type for a null constant");
                    }
                    if matches!(types.get(cur_ty), Type::TargetExt { .. }) {
                        return self.unsupported("null constant of a target extension type");
                    }
                    RawKind::Ready(null_kind(types, cur_ty))
                }
                CST_CODE_INTEGER => {
                    let bits = match types.get(types.scalar(cur_ty)) {
                        Type::Int(b) => *b,
                        _ => return self.malformed("integer constant of a non-integer type"),
                    };
                    let v = match ops.first() {
                        Some(&v) => decode_sign_rotated(v),
                        None => return self.malformed("invalid integer constant record"),
                    };
                    if bits < 64 {
                        // The writer stores the sign-extended value; anything else does not fit the type.
                        let sext = ((v << (64 - bits)) as i64 >> (64 - bits)) as u64;
                        let zext = v & ((1u64 << bits) - 1);
                        if v != sext && v != zext {
                            return self.malformed(format!("integer constant {:#x} does not fit i{}", v, bits));
                        }
                    }
                    let mut words = vec![v];
                    words.resize(bits.div_ceil(64) as usize, if bits > 64 && (v as i64) < 0 { u64::MAX } else { 0 });
                    mask_words(&mut words, bits);
                    self.int_or_splat(cur_ty, words)?
                }
                CST_CODE_WIDE_INTEGER => {
                    let bits = match types.get(types.scalar(cur_ty)) {
                        Type::Int(b) => *b,
                        _ => return self.malformed("integer constant of a non-integer type"),
                    };
                    if ops.is_empty() {
                        return self.malformed("invalid wide integer constant record");
                    }
                    let words = wide_words(ops, bits);
                    self.int_or_splat(cur_ty, words)?
                }
                CST_CODE_FLOAT => {
                    let fk = match types.float_kind(types.scalar(cur_ty)) {
                        Some(k) => k,
                        None => return self.malformed("floating-point constant of a non-floating-point type"),
                    };
                    let bits: u128 = match (fk, ops.len()) {
                        (FloatKind::Half | FloatKind::BFloat, 1) if ops[0] <= 0xffff => ops[0] as u128,
                        (FloatKind::Float, 1) if ops[0] <= 0xffff_ffff => ops[0] as u128,
                        (FloatKind::Double, 1) => ops[0] as u128,
                        (FloatKind::X86Fp80, 2) if ops[1] <= 0xffff => {
                            // Bits are stored rearranged: word 0 holds the high 64 bits.
                            let lo = (ops[1] & 0xffff) | (ops[0] << 16);
                            let hi = ops[0] >> 48;
                            lo as u128 | (hi as u128) << 64
                        }
                        (FloatKind::Fp128 | FloatKind::PpcFp128, 2) => ops[0] as u128 | (ops[1] as u128) << 64,
                        _ => return self.malformed("invalid floating-point constant record"),
                    };
                    match types.get(cur_ty) {
                        Type::Vector { len, scalable: false, .. } if fk.bits() <= 64 => {
                            RawKind::Ready(data_or_splat_vector(*len, bits as u64, true))
                        }
                        Type::Vector { .. } => {
                            return self
                                .unsupported("vector constants of 80- or 128-bit floating-point or scalable type");
                        }
                        _ => RawKind::Ready(ConstKind::Float(bits)),
                    }
                }
                CST_CODE_AGGREGATE => {
                    if ops.is_empty() {
                        return self.malformed("invalid aggregate record");
                    }
                    if !matches!(
                        types.get(cur_ty),
                        Type::Struct { .. } | Type::LiteralStruct { .. } | Type::Array { .. } | Type::Vector { .. }
                    ) {
                        return self.malformed("aggregate constant of a non-aggregate type");
                    }
                    RawKind::Aggregate(ops.clone())
                }
                CST_CODE_STRING | CST_CODE_CSTRING => {
                    if ops.is_empty() {
                        return self.malformed("invalid string record");
                    }
                    let mut bytes = self.record_bytes(ops)?.into_iter().map(|b| b as u64).collect::<Vec<_>>();
                    if r.code == CST_CODE_CSTRING {
                        bytes.push(0);
                    }
                    let i8t = self.m.types.int(8);
                    let want = self.m.types.intern(Type::Array { len: bytes.len() as u64, elem: i8t });
                    if want != cur_ty {
                        return self.malformed("string constant does not match its type");
                    }
                    RawKind::Ready(data_kind(bytes, false))
                }
                CST_CODE_DATA => {
                    if ops.is_empty() {
                        return self.malformed("invalid data record");
                    }
                    let (elem, len, is_vector) = match types.get(cur_ty) {
                        Type::Array { len, elem } => (*elem, *len, false),
                        Type::Vector { len, elem, scalable: false } => (*elem, *len as u64, true),
                        _ => return self.malformed("data constant of a non-sequential type"),
                    };
                    let width = match types.get(elem) {
                        Type::Int(b @ (8 | 16 | 32 | 64)) => *b,
                        Type::Half | Type::BFloat => 16,
                        Type::Float => 32,
                        Type::Double => 64,
                        _ => return self.malformed("data constant of an incompatible element type"),
                    };
                    if ops.len() as u64 != len {
                        return self.malformed("data constant element count does not match its type");
                    }
                    if width < 64 && ops.iter().any(|&v| v >> width != 0) {
                        return self.malformed("data constant element wider than its type");
                    }
                    RawKind::Ready(data_kind(ops.clone(), is_vector))
                }
                CST_CODE_CE_UNOP => {
                    if ops.len() < 2 {
                        return self.malformed("invalid unary constant expression record");
                    }
                    if ops[0] != 0 || !types.is_fp_or_fp_vector(cur_ty) {
                        return self.malformed("unknown unary constant expression");
                    }
                    RawKind::FNeg(ops[1])
                }
                CST_CODE_CE_BINOP => {
                    if ops.len() < 3 {
                        return self.malformed("invalid binary constant expression record");
                    }
                    let is_float = types.is_fp_or_fp_vector(cur_ty);
                    if !is_float && !types.is_int_or_int_vector(cur_ty) {
                        return self.malformed("binary constant expression of a non-arithmetic type");
                    }
                    let op = match binary_op(ops[0], is_float) {
                        Some(o) => o,
                        None => return self.malformed("unknown binary constant expression"),
                    };
                    let flags = if ops.len() >= 4 { binop_flags(op, ops[3]) } else { 0 };
                    RawKind::Binary { op, flags, lhs: ops[1], rhs: ops[2] }
                }
                CST_CODE_CE_CAST => {
                    if ops.len() < 3 {
                        return self.malformed("invalid cast constant expression record");
                    }
                    let op = match cast_op(ops[0]) {
                        Some(o) => o,
                        None => return self.malformed("unknown cast constant expression"),
                    };
                    self.ty(ops[1])?;
                    RawKind::Cast { op, operand: ops[2] }
                }
                CST_CODE_CE_INBOUNDS_GEP
                | CST_CODE_CE_GEP_OLD
                | CST_CODE_CE_GEP_WITH_INRANGE_INDEX_OLD
                | CST_CODE_CE_GEP
                | CST_CODE_CE_GEP_WITH_INRANGE => {
                    if ops.len() < 2 {
                        return self.malformed("constant GEP record must have at least two elements");
                    }
                    let mut i = 0;
                    let mut pointee: Option<TypeId> = None;
                    if matches!(
                        r.code,
                        CST_CODE_CE_GEP_WITH_INRANGE_INDEX_OLD | CST_CODE_CE_GEP_WITH_INRANGE | CST_CODE_CE_GEP
                    ) || ops.len() % 2 == 1
                    {
                        pointee = Some(self.ty(ops[0])?);
                        i += 1;
                    }
                    let mut flags = 0u8;
                    let mut inrange = None;
                    match r.code {
                        CST_CODE_CE_GEP_WITH_INRANGE_INDEX_OLD => {
                            // The old inrange index is dropped, as LLVM does.
                            flags = if ops[i] & 1 != 0 { GepFlags::INBOUNDS | GepFlags::NUSW } else { 0 };
                            i += 1;
                        }
                        CST_CODE_CE_GEP_WITH_INRANGE => {
                            flags = gep_flags(ops[i]);
                            i += 1;
                            inrange = Some(self.read_range_public(ops, &mut i)?);
                        }
                        CST_CODE_CE_GEP => {
                            flags = gep_flags(ops[i]);
                            i += 1;
                        }
                        CST_CODE_CE_INBOUNDS_GEP => flags = GepFlags::INBOUNDS | GepFlags::NUSW,
                        _ => {}
                    }
                    if i >= ops.len() || (ops.len() - i) % 2 != 0 {
                        return self.malformed("invalid constant GEP record");
                    }
                    let base_bc = ops[i];
                    let mut operands = Vec::new();
                    while i < ops.len() {
                        self.ty(ops[i])?;
                        operands.push(ops[i + 1]);
                        i += 2;
                    }
                    let mut base_bc32 = base_bc as u32;
                    let base_ty = self.ty(base_bc)?;
                    if self.m.types.is_vector(base_ty) {
                        base_bc32 = match self.contained(base_bc32, 0) {
                            Some(c) => c,
                            None => return self.malformed("vector GEP base without element type"),
                        };
                    }
                    if !self.m.types.is_ptr(self.ty(base_bc32 as u64)?) {
                        return self.malformed("GEP base operand must be pointer or vector of pointer");
                    }
                    let source = match pointee {
                        Some(p) => p,
                        None => match self
                            .contained(base_bc32, 0)
                            .and_then(|c| self.bc_types.get(c as usize).copied().flatten())
                        {
                            Some(p) => p,
                            None => return self.malformed("missing element type for an old-style constant GEP"),
                        },
                    };
                    RawKind::Gep { source, flags, inrange, operands }
                }
                CST_CODE_CE_SELECT => {
                    if ops.len() < 3 {
                        return self.malformed("invalid select constant expression record");
                    }
                    RawKind::Select(ops[0], ops[1], ops[2])
                }
                CST_CODE_CE_EXTRACTELT => {
                    if ops.len() < 3 {
                        return self.malformed("invalid extractelement constant expression record");
                    }
                    if !self.m.types.is_vector(self.ty(ops[0])?) {
                        return self.malformed("invalid extractelement constant expression record");
                    }
                    let idx = if ops.len() == 4 {
                        self.ty(ops[2])?;
                        ops[3]
                    } else {
                        ops[2]
                    };
                    RawKind::ExtractElement(ops[1], idx)
                }
                CST_CODE_CE_INSERTELT => {
                    if ops.len() < 3 || !types.is_vector(cur_ty) {
                        return self.malformed("invalid insertelement constant expression record");
                    }
                    let idx = if ops.len() == 4 {
                        self.ty(ops[2])?;
                        ops[3]
                    } else {
                        ops[2]
                    };
                    RawKind::InsertElement(ops[0], ops[1], idx)
                }
                CST_CODE_CE_SHUFFLEVEC => {
                    if ops.len() < 3 || !types.is_vector(cur_ty) {
                        return self.malformed("invalid shufflevector constant expression record");
                    }
                    RawKind::ShuffleVector(ops[0], ops[1], ops[2])
                }
                CST_CODE_CE_SHUFVEC_EX => {
                    if ops.len() < 4 || !types.is_vector(cur_ty) || !self.m.types.is_vector(self.ty(ops[0])?) {
                        return self.malformed("invalid shufflevector constant expression record");
                    }
                    RawKind::ShuffleVector(ops[1], ops[2], ops[3])
                }
                CST_CODE_CE_CMP => {
                    if ops.len() < 4 {
                        return self.malformed("invalid compare constant expression record");
                    }
                    self.ty(ops[0])?;
                    RawKind::Cmp { pred: ops[3], lhs: ops[1], rhs: ops[2] }
                }
                CST_CODE_INLINEASM => {
                    if ops.len() < 3 {
                        return self.malformed("invalid inline asm record");
                    }
                    let fn_ty = self.ty(ops[0])?;
                    if !self.m.types.is_function(fn_ty) {
                        return self.malformed("invalid inline asm record");
                    }
                    let flags = ops[1];
                    let asm_len = ops[2] as usize;
                    if 3 + asm_len >= ops.len() {
                        return self.malformed("invalid inline asm record");
                    }
                    let asm = self.record_bytes(&ops[3..3 + asm_len])?;
                    let c_len = ops[3 + asm_len] as usize;
                    if 4 + asm_len + c_len > ops.len() {
                        return self.malformed("invalid inline asm record");
                    }
                    let constraints = self.record_bytes(&ops[4 + asm_len..4 + asm_len + c_len])?;
                    let ia = InlineAsm {
                        fn_ty,
                        asm,
                        constraints,
                        side_effects: flags & 1 != 0,
                        align_stack: (flags >> 1) & 1 != 0,
                        dialect: ((flags >> 2) & 1) as u8,
                        can_throw: (flags >> 3) & 1 != 0,
                    };
                    let idx = match self.m.inline_asms.iter().position(|x| *x == ia) {
                        Some(i) => i,
                        None => {
                            self.m.inline_asms.push(ia);
                            self.m.inline_asms.len() - 1
                        }
                    };
                    RawKind::InlineAsm(idx as u32)
                }
                CST_CODE_INLINEASM_OLD | CST_CODE_INLINEASM_OLD2 | CST_CODE_INLINEASM_OLD3 => {
                    return self.unsupported(
                        "inline asm records of LLVM before 13 (the asm string upgrade is not implemented)",
                    );
                }
                CST_CODE_BLOCKADDRESS => return self.unsupported("blockaddress constants"),
                CST_CODE_DSO_LOCAL_EQUIVALENT => return self.unsupported("dso_local_equivalent constants"),
                CST_CODE_NO_CFI_VALUE => return self.unsupported("no_cfi constants"),
                CST_CODE_PTRAUTH => return self.unsupported("ptrauth constants"),
                c => return self.malformed(format!("unknown constant record code {}", c)),
            };
            let idx = self.raw_consts.len() as u32;
            self.raw_consts.push(RawConst { ty: cur_ty, kind, done: None, busy: false });
            self.values.push(VEntry { slot: Slot::Const(idx), bc_ty: cur_bc });
        }
        Ok(())
    }

    fn read_range_public(&self, ops: &[u64], i: &mut usize) -> Result<ConstantRange> {
        // [bitwidth, lower, upper] with the same encoding as attribute ranges.
        if *i >= ops.len() {
            return self.malformed("truncated constant range");
        }
        let bits = ops[*i];
        *i += 1;
        if bits == 0 || bits > 64 {
            return self.unsupported("inrange of more than 64 bits");
        }
        if ops.len() - *i < 2 {
            return self.malformed("truncated constant range");
        }
        let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let lower = decode_sign_rotated(ops[*i]) & mask;
        let upper = decode_sign_rotated(ops[*i + 1]) & mask;
        *i += 2;
        Ok(ConstantRange { bits: bits as u32, lower: vec![lower], upper: vec![upper] })
    }

    fn int_or_splat(&mut self, ty: TypeId, words: Vec<u64>) -> Result<RawKind> {
        Ok(match self.m.types.get(ty).clone() {
            Type::Vector { len, scalable: false, elem } => {
                let bits = self.m.types.int_bits(elem).unwrap();
                if matches!(bits, 8 | 16 | 32 | 64) {
                    RawKind::Ready(data_or_splat_vector(len, words[0], false))
                } else {
                    let e = self.m.intern_const(Constant { ty: elem, kind: ConstKind::Int(words) });
                    RawKind::Ready(vector_splat_kind(&self.m, len, e))
                }
            }
            Type::Vector { scalable: true, .. } => return self.unsupported("constants of scalable vector type"),
            _ => RawKind::Ready(ConstKind::Int(words)),
        })
    }

    /// The value of value number `id` where a constant is needed.
    pub(crate) fn const_operand(&mut self, id: u64, use_: Use) -> Result<Value> {
        match self.values.get(id as usize).map(|e| e.slot) {
            Some(Slot::Global(g)) => Ok(Value::Global(g)),
            Some(Slot::Const(c)) => self.materialize_const_use(c, use_),
            _ => self.malformed(format!("constant operand {} is not a constant", id)),
        }
    }

    pub(crate) fn materialize_const(&mut self, idx: u32, _ctx: Option<()>) -> Result<Value> {
        self.materialize_const_use(idx, Use::Initializer)
    }

    /// Makes raw constant `idx` a pooled constant (`materializeValue`), with LLVM's canonicalization and folding.
    pub(crate) fn materialize_const_use(&mut self, idx: u32, use_: Use) -> Result<Value> {
        if let Some(v) = self.raw_consts[idx as usize].done {
            return Ok(v);
        }
        if self.raw_consts[idx as usize].busy {
            return self.malformed("constant refers to itself");
        }
        if self.const_depth >= MAX_CONST_DEPTH {
            return self.unsupported(format!("constants nested more than {} deep", MAX_CONST_DEPTH));
        }
        self.raw_consts[idx as usize].busy = true;
        self.const_depth += 1;
        let r = self.build_const(idx, use_);
        self.const_depth -= 1;
        self.raw_consts[idx as usize].busy = false;
        let v = r?;
        // Constants that became something else than a constant (expansions) are not cached; this reader expands
        // nothing, so every result is a constant.
        self.raw_consts[idx as usize].done = Some(v);
        Ok(v)
    }

    fn build_const(&mut self, idx: u32, use_: Use) -> Result<Value> {
        let rc = self.raw_consts[idx as usize].clone();
        let ty = rc.ty;
        match rc.kind {
            RawKind::Ready(k) => Ok(Value::Const(self.m.intern_const(Constant { ty, kind: k }))),
            RawKind::InlineAsm(i) => Ok(Value::InlineAsm(i)),
            RawKind::Aggregate(ids) => {
                let mut elems = Vec::with_capacity(ids.len());
                for id in ids {
                    elems.push(self.const_operand(id, use_)?);
                }
                let t = self.m.types.get(ty).clone();
                match t {
                    Type::Struct { .. } | Type::LiteralStruct { .. } => {
                        let members = self.m.types.struct_elems(ty).map(|e| e.to_vec()).unwrap_or_default();
                        if members.len() != elems.len() {
                            return self.malformed("invalid number of elements in struct initializer");
                        }
                        for (m, e) in members.iter().zip(&elems) {
                            if self.m.module_value_type(*e) != Some(*m) {
                                return self.malformed("incorrect type in struct initializer");
                            }
                        }
                        Ok(const_struct(&mut self.m, ty, elems))
                    }
                    Type::Array { len, elem } => {
                        if len != elems.len() as u64 {
                            return self.malformed("invalid number of elements in array initializer");
                        }
                        for e in &elems {
                            if self.m.module_value_type(*e) != Some(elem) {
                                return self.malformed("incorrect type in array initializer");
                            }
                        }
                        Ok(const_array(&mut self.m, ty, elems))
                    }
                    Type::Vector { len, elem, scalable: false } => {
                        if len as usize != elems.len() {
                            return self.malformed("invalid number of elements in vector initializer");
                        }
                        for e in &elems {
                            if self.m.module_value_type(*e) != Some(elem) {
                                return self.malformed("incorrect type in vector initializer");
                            }
                        }
                        Ok(const_vector(&mut self.m, ty, elems))
                    }
                    _ => self.malformed("aggregate constant of a non-aggregate type"),
                }
            }
            RawKind::Cast { op, operand } => {
                let v = self.const_operand(operand, use_)?;
                let src = self.m.module_value_type(v).unwrap();
                if !supported_cast(op) {
                    return self.unsupported_expr(op.name(), ty, use_);
                }
                if !cast_is_valid(&self.m.types, op, src, ty) {
                    return self.malformed(format!("invalid {} constant expression", op.name()));
                }
                // `UpgradeBitCastExpr`: a bitcast between address spaces is ptrtoint to i64 and inttoptr.
                if op == CastOp::BitCast && self.m.types.pointer_addrspace(src) != self.m.types.pointer_addrspace(ty) {
                    let i64t = self.m.types.int(64);
                    let mid = get_cast(&mut self.m, CastOp::PtrToInt, v, i64t);
                    return Ok(get_cast(&mut self.m, CastOp::IntToPtr, mid, ty));
                }
                Ok(get_cast(&mut self.m, op, v, ty))
            }
            RawKind::Gep { source, flags, inrange, operands } => {
                if operands.is_empty() {
                    return self.malformed("invalid GEP with no operands");
                }
                let mut vals = Vec::new();
                for id in &operands {
                    vals.push(self.const_operand(*id, use_)?);
                }
                if matches!(self.m.types.get(source), Type::Vector { scalable: true, .. }) {
                    return self.unsupported_expr("getelementptr", ty, use_);
                }
                let ptr = vals[0];
                let indices = vals[1..].to_vec();
                self.check_const_gep(source, ptr, &indices, ty)?;
                Ok(get_gep(&mut self.m, ty, source, flags, inrange, ptr, indices))
            }
            RawKind::Binary { op, flags, lhs, rhs } => {
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Xor) {
                    self.unsupported(format!(
                        "constant expression {} (flags {:#x}) of constants {} and {}: folding binary constant \
                         expressions as LLVM does is not implemented",
                        op.name(),
                        flags,
                        lhs,
                        rhs
                    ))
                } else {
                    self.unsupported_expr(&format!("{} of constants {} and {}", op.name(), lhs, rhs), ty, use_)
                }
            }
            RawKind::FNeg(v) => self.unsupported_expr(&format!("fneg of constant {}", v), ty, use_),
            RawKind::Select(c, a, b) => {
                self.unsupported_expr(&format!("select of constants {}, {}, {}", c, a, b), ty, use_)
            }
            RawKind::Cmp { pred, lhs, rhs } => self.unsupported_expr(
                &format!("compare (predicate {}) of constants {} and {}", pred, lhs, rhs),
                ty,
                use_,
            ),
            RawKind::ExtractElement(a, b) => self.unsupported_vector_expr("extractelement", &[a, b]),
            RawKind::InsertElement(a, b, c) => self.unsupported_vector_expr("insertelement", &[a, b, c]),
            RawKind::ShuffleVector(a, b, c) => self.unsupported_vector_expr("shufflevector", &[a, b, c]),
        }
    }

    fn unsupported_vector_expr(&mut self, name: &str, operands: &[u64]) -> Result<Value> {
        self.unsupported(format!(
            "constant expression {} of constants {:?}: folding vector constant expressions as LLVM does is not \
             implemented",
            name, operands
        ))
    }

    /// A constant expression LLVM 21 no longer has as a constant.
    fn unsupported_expr(&mut self, name: &str, ty: TypeId, use_: Use) -> Result<Value> {
        match use_ {
            Use::Metadata => Ok(Value::Const(self.m.intern_const(Constant { ty, kind: ConstKind::Poison }))),
            Use::Initializer => self.malformed(format!(
                "value referenced by an initializer is an unsupported constant expression ({})",
                name
            )),
            Use::Instruction => self.unsupported(format!(
                "constant expression {} used by an instruction (LLVM 21 expands it into instructions; not implemented)",
                name
            )),
        }
    }

    fn check_const_gep(&mut self, source: TypeId, ptr: Value, indices: &[Value], ty: TypeId) -> Result<()> {
        let pt = self.m.module_value_type(ptr).unwrap();
        if !self.m.types.is_ptr_or_ptr_vector(pt) {
            return self.malformed("GEP base operand must be pointer or vector of pointer");
        }
        let mut cur = source;
        for (k, idx) in indices.iter().enumerate() {
            let it = self.m.module_value_type(*idx).unwrap();
            if !self.m.types.is_int_or_int_vector(it) {
                return self.malformed("GEP index is not an integer");
            }
            if k == 0 {
                continue;
            }
            cur = match self.m.types.get(cur).clone() {
                Type::Struct { .. } | Type::LiteralStruct { .. } => {
                    let field = match const_index_value(&self.m, *idx) {
                        Some(f) => f,
                        None => return self.malformed("struct GEP index is not a constant"),
                    };
                    match self.m.types.struct_elems(cur).and_then(|e| e.get(field as usize)).copied() {
                        Some(t) => t,
                        None => return self.malformed("struct GEP index out of range"),
                    }
                }
                Type::Array { elem, .. } | Type::Vector { elem, .. } => elem,
                _ => return self.malformed("GEP indexes into a non-aggregate type"),
            };
        }
        let want = gep_result_type(&mut self.m, pt, indices);
        if want != ty {
            return self.malformed("constant GEP has the wrong type");
        }
        Ok(())
    }
}

fn binop_flags(op: BinOp, raw: u64) -> u8 {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => {
            let mut f = 0;
            if raw & 2 != 0 {
                f |= flags::NSW;
            }
            if raw & 1 != 0 {
                f |= flags::NUW;
            }
            f
        }
        BinOp::SDiv | BinOp::UDiv | BinOp::LShr | BinOp::AShr if raw & 1 != 0 => flags::EXACT,
        _ => 0,
    }
}

/// `toGEPNoWrapFlags`: bitcode bits 0 inbounds, 1 nusw, 2 nuw; inbounds implies nusw.
pub(crate) fn gep_flags(raw: u64) -> u8 {
    let mut f = 0u8;
    if raw & 1 != 0 {
        f |= GepFlags::INBOUNDS | GepFlags::NUSW;
    }
    if raw & 2 != 0 {
        f |= GepFlags::NUSW;
    }
    if raw & 4 != 0 {
        f |= GepFlags::NUW;
    }
    f
}

/// The zero of a type (`Constant::getNullValue`).
pub(crate) fn null_kind(types: &TypeTable, ty: TypeId) -> ConstKind {
    match types.get(ty) {
        Type::Int(b) => ConstKind::Int(vec![0; b.div_ceil(64) as usize]),
        Type::Ptr(_) => ConstKind::NullPtr,
        Type::Token => ConstKind::TokenNone,
        _ if types.is_float(ty) => ConstKind::Float(0),
        _ => ConstKind::Zero,
    }
}

pub(crate) fn null_value(m: &mut Module, ty: TypeId) -> Value {
    let k = null_kind(&m.types, ty);
    Value::Const(m.intern_const(Constant { ty, kind: k }))
}

/// `ConstantDataSequential::getImpl` with the element bits: all-zero data is `zeroinitializer`.
fn data_kind(elems: Vec<u64>, is_vector: bool) -> ConstKind {
    if elems.iter().all(|&e| e == 0) {
        ConstKind::Zero
    } else if is_vector {
        ConstKind::DataVector(elems)
    } else {
        ConstKind::DataArray(elems)
    }
}

/// A splat of a data-compatible element: data vector, or zero.
fn data_or_splat_vector(len: u32, bits: u64, _float: bool) -> ConstKind {
    data_kind(vec![bits; len as usize], true)
}

fn vector_splat_kind(m: &Module, len: u32, e: ConstId) -> ConstKind {
    if is_null(m, Value::Const(e)) { ConstKind::Zero } else { ConstKind::Vector(vec![Value::Const(e); len as usize]) }
}

/// `Constant::isNullValue`.
pub(crate) fn is_null(m: &Module, v: Value) -> bool {
    match v {
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Int(w) => words_zero(w),
            ConstKind::Float(b) => *b == 0,
            ConstKind::Zero | ConstKind::NullPtr | ConstKind::TokenNone => true,
            _ => false,
        },
        _ => false,
    }
}

fn is_undef(m: &Module, v: Value) -> bool {
    matches!(v, Value::Const(c) if matches!(m.constant(c).kind, ConstKind::Undef | ConstKind::Poison))
}

fn is_poison(m: &Module, v: Value) -> bool {
    matches!(v, Value::Const(c) if matches!(m.constant(c).kind, ConstKind::Poison))
}

/// The element bits of a simple element (`ConstantInt` or `ConstantFP`) for data sequences.
fn simple_bits(m: &Module, v: Value) -> Option<u64> {
    match v {
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Int(w) => Some(w[0]),
            ConstKind::Float(b) => Some(*b as u64),
            _ => None,
        },
        _ => None,
    }
}

fn data_compatible(types: &TypeTable, elem: TypeId) -> bool {
    matches!(types.get(elem), Type::Int(8 | 16 | 32 | 64) | Type::Half | Type::BFloat | Type::Float | Type::Double)
}

/// `ConstantStruct::get`.
pub(crate) fn const_struct(m: &mut Module, ty: TypeId, elems: Vec<Value>) -> Value {
    let mut zero = true;
    let mut undef = false;
    let mut poison = false;
    if !elems.is_empty() {
        undef = is_undef(m, elems[0]);
        poison = is_poison(m, elems[0]);
        zero = is_null(m, elems[0]);
        if undef || zero {
            for &e in &elems {
                if !is_null(m, e) {
                    zero = false;
                }
                if !is_poison(m, e) {
                    poison = false;
                }
                if is_poison(m, e) || !is_undef(m, e) {
                    undef = false;
                }
            }
        }
    }
    let kind = if zero {
        ConstKind::Zero
    } else if poison {
        ConstKind::Poison
    } else if undef {
        ConstKind::Undef
    } else {
        ConstKind::Struct(elems)
    };
    Value::Const(m.intern_const(Constant { ty, kind }))
}

/// `ConstantArray::get`.
pub(crate) fn const_array(m: &mut Module, ty: TypeId, elems: Vec<Value>) -> Value {
    let kind = sequence_kind(m, ty, elems, false);
    Value::Const(m.intern_const(Constant { ty, kind }))
}

/// `ConstantVector::get`.
pub(crate) fn const_vector(m: &mut Module, ty: TypeId, elems: Vec<Value>) -> Value {
    let kind = sequence_kind(m, ty, elems, true);
    Value::Const(m.intern_const(Constant { ty, kind }))
}

fn sequence_kind(m: &Module, ty: TypeId, elems: Vec<Value>, is_vector: bool) -> ConstKind {
    if elems.is_empty() {
        return ConstKind::Zero;
    }
    let c = elems[0];
    let all_same = elems.iter().all(|&e| e == c);
    if all_same {
        if is_poison(m, c) {
            return ConstKind::Poison;
        }
        if is_undef(m, c) {
            return ConstKind::Undef;
        }
        if is_null(m, c) {
            return ConstKind::Zero;
        }
    }
    let elem = match m.types.get(ty) {
        Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
        _ => unreachable!(),
    };
    if data_compatible(&m.types, elem) {
        let bits: Option<Vec<u64>> = elems.iter().map(|&e| simple_bits(m, e)).collect();
        if let Some(bits) = bits {
            return data_kind(bits, is_vector);
        }
    }
    if is_vector { ConstKind::Vector(elems) } else { ConstKind::Array(elems) }
}

/// The value of a constant integer index (`ConstantInt::getZExtValue`), including a splat vector of one.
pub(crate) fn const_index_value(m: &Module, v: Value) -> Option<u64> {
    match v {
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Int(w) => Some(w[0]),
            ConstKind::Zero => Some(0),
            ConstKind::DataVector(e) if e.iter().all(|x| *x == e[0]) => Some(e[0]),
            _ => None,
        },
        _ => None,
    }
}

/// `GetElementPtrInst::getGEPReturnType`: a pointer in the base's address space, or a vector of them when the base
/// or an index is a vector.
pub(crate) fn gep_result_type(m: &mut Module, base: TypeId, indices: &[Value]) -> TypeId {
    gep_result_type_with(m, base, indices.iter().map(|_| None).collect(), indices)
}

pub(crate) fn gep_result_type_with(
    m: &mut Module,
    base: TypeId,
    local: Vec<Option<TypeId>>,
    indices: &[Value],
) -> TypeId {
    let addrspace = m.types.pointer_addrspace(base).unwrap_or(0);
    let ptr = m.types.ptr(addrspace);
    if let Some(n) = m.types.vector_len(base) {
        return m.types.vector(n, ptr);
    }
    for (k, idx) in indices.iter().enumerate() {
        let t = match local.get(k).copied().flatten() {
            Some(t) => t,
            None => match m.module_value_type(*idx) {
                Some(t) => t,
                None => continue,
            },
        };
        if let Some(n) = m.types.vector_len(t) {
            return m.types.vector(n, ptr);
        }
    }
    ptr
}

/// The splat element of a constant vector, if it is one (`Constant::getSplatValue`).
fn splat_value(m: &mut Module, v: Value) -> Option<Value> {
    let c = match v {
        Value::Const(c) => c,
        _ => return None,
    };
    let con = m.constant(c).clone();
    let elem = match m.types.get(con.ty) {
        Type::Vector { elem, .. } => *elem,
        _ => return None,
    };
    match con.kind {
        ConstKind::Zero => Some(null_value(m, elem)),
        ConstKind::DataVector(e) => {
            if e.iter().all(|x| *x == e[0]) {
                Some(data_element(m, elem, e[0]))
            } else {
                None
            }
        }
        ConstKind::Vector(e) => {
            if e.iter().all(|x| *x == e[0]) {
                Some(e[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A data sequence element as a constant.
pub(crate) fn data_element(m: &mut Module, elem: TypeId, bits: u64) -> Value {
    let kind = if m.types.is_int(elem) { ConstKind::Int(vec![bits]) } else { ConstKind::Float(bits as u128) };
    Value::Const(m.intern_const(Constant { ty: elem, kind }))
}

/// The elements of a constant of fixed vector type, as constants.
fn vector_elements(m: &mut Module, v: Value) -> Option<Vec<Value>> {
    let c = match v {
        Value::Const(c) => c,
        _ => return None,
    };
    let con = m.constant(c).clone();
    let (len, elem) = match m.types.get(con.ty) {
        Type::Vector { len, elem, scalable: false } => (*len, *elem),
        _ => return None,
    };
    match con.kind {
        ConstKind::Zero => {
            let z = null_value(m, elem);
            Some(vec![z; len as usize])
        }
        ConstKind::Undef | ConstKind::Poison => {
            let k = con.kind.clone();
            let e = Value::Const(m.intern_const(Constant { ty: elem, kind: k }));
            Some(vec![e; len as usize])
        }
        ConstKind::DataVector(e) => Some(e.iter().map(|&b| data_element(m, elem, b)).collect()),
        ConstKind::Vector(e) => Some(e),
        _ => None,
    }
}

/// `Constant::isAllOnesValue`.
fn is_all_ones(m: &mut Module, v: Value) -> bool {
    let c = match v {
        Value::Const(c) => c,
        _ => return false,
    };
    let con = m.constant(c).clone();
    match &con.kind {
        ConstKind::Int(w) => words_all_ones(w, m.types.int_bits(con.ty).unwrap()),
        ConstKind::Float(b) => {
            let bits = m.types.float_kind(con.ty).unwrap().bits();
            *b == if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 }
        }
        _ => {
            if m.types.is_vector(con.ty) {
                match splat_value(m, v) {
                    Some(s) => is_all_ones(m, s),
                    None => false,
                }
            } else {
                false
            }
        }
    }
}

/// `Constant::getAllOnesValue`.
fn all_ones(m: &mut Module, ty: TypeId) -> Option<Value> {
    match m.types.get(ty).clone() {
        Type::Int(b) => {
            let mut w = vec![u64::MAX; b.div_ceil(64) as usize];
            mask_words(&mut w, b);
            Some(Value::Const(m.intern_const(Constant { ty, kind: ConstKind::Int(w) })))
        }
        Type::Vector { len, elem, scalable: false } => {
            let e = all_ones(m, elem)?;
            Some(splat(m, ty, len, e))
        }
        _ => {
            let k = m.types.float_kind(ty)?;
            let bits = k.bits();
            let b = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
            Some(Value::Const(m.intern_const(Constant { ty, kind: ConstKind::Float(b) })))
        }
    }
}

/// `ConstantVector::getSplat` for a fixed vector type.
pub(crate) fn splat(m: &mut Module, ty: TypeId, len: u32, e: Value) -> Value {
    const_vector(m, ty, vec![e; len as usize])
}

/// `ConstantFoldCastInstruction` for the cast operators LLVM 21 keeps as constant expressions.
fn fold_cast(m: &mut Module, op: CastOp, v: Value, dst: TypeId) -> Option<Value> {
    if is_poison(m, v) {
        return Some(Value::Const(m.intern_const(Constant { ty: dst, kind: ConstKind::Poison })));
    }
    if is_undef(m, v) {
        if matches!(op, CastOp::ZExt | CastOp::SExt | CastOp::UIToFP | CastOp::SIToFP) {
            return Some(null_value(m, dst));
        }
        return Some(Value::Const(m.intern_const(Constant { ty: dst, kind: ConstKind::Undef })));
    }
    if is_null(m, v) && !matches!(m.types.get(dst), Type::X86Amx) && op != CastOp::AddrSpaceCast {
        return Some(null_value(m, dst));
    }
    // A cast of a cast.
    if let Value::Const(c) = v
        && let ConstKind::Expr(ConstExpr::Cast { op: inner, value: inner_v }) = m.constant(c).kind.clone()
    {
        let src = m.module_value_type(inner_v).unwrap();
        let mid = m.constant(c).ty;
        if let Some(new_op) = eliminable_cast_pair(&m.types, inner, op, src, mid, dst) {
            // `foldMaybeUndesirableCast`: desirable opcodes go through getCast (and may stay an expression).
            return if supported_cast(new_op) { Some(get_cast(m, new_op, inner_v, dst)) } else { None };
        }
    }
    let src = m.module_value_type(v).unwrap();
    if let (Some(dn), Some(sn)) = (m.types.vector_len(dst), m.types.vector_len(src))
        && dn == sn
    {
        let delem = m.types.scalar(dst);
        if let Some(s) = splat_value(m, v) {
            let r = fold_maybe_undesirable(m, op, s, delem)?;
            return Some(splat(m, dst, dn, r));
        }
        let elems = vector_elements(m, v)?;
        let mut out = Vec::new();
        for e in elems {
            out.push(fold_maybe_undesirable(m, op, e, delem)?);
        }
        return Some(const_vector(m, dst, out));
    }
    match op {
        CastOp::Trunc => {
            if let Value::Const(c) = v
                && let ConstKind::Int(w) = m.constant(c).kind.clone()
            {
                let bits = m.types.int_bits(dst)?;
                let mut w = w;
                w.truncate(bits.div_ceil(64) as usize);
                mask_words(&mut w, bits);
                return Some(Value::Const(m.intern_const(Constant { ty: dst, kind: ConstKind::Int(w) })));
            }
            None
        }
        CastOp::BitCast => fold_bitcast(m, v, dst),
        _ => None,
    }
}

fn fold_maybe_undesirable(m: &mut Module, op: CastOp, v: Value, dst: TypeId) -> Option<Value> {
    if supported_cast(op) { Some(get_cast(m, op, v, dst)) } else { fold_cast(m, op, v, dst) }
}

/// `FoldBitCast`.
fn fold_bitcast(m: &mut Module, v: Value, dst: TypeId) -> Option<Value> {
    let src = m.module_value_type(v).unwrap();
    if src == dst {
        return Some(v);
    }
    if is_all_ones(m, v)
        && let Some(a) = all_ones(m, dst)
    {
        return Some(a);
    }
    let c = match v {
        Value::Const(c) => c,
        _ => return None,
    };
    match m.constant(c).kind.clone() {
        ConstKind::Int(w) => {
            if m.types.is_vector(dst) && !m.types.is_vector(src) {
                let vt = m.types.vector(1, src);
                let vec = const_vector(m, vt, vec![v]);
                return Some(get_cast(m, CastOp::BitCast, vec, dst));
            }
            if !m.types.is_fp_or_fp_vector(dst)
                || matches!(m.types.get(dst), Type::PpcFp128)
                || m.types.scalar_bits(dst) != m.types.scalar_bits(src)
            {
                return None;
            }
            if m.types.is_vector(dst) {
                return None;
            }
            let bits = w[0] as u128 | (w.get(1).copied().unwrap_or(0) as u128) << 64;
            Some(Value::Const(m.intern_const(Constant { ty: dst, kind: ConstKind::Float(bits) })))
        }
        ConstKind::Float(b) => {
            if m.types.is_vector(dst) && !m.types.is_vector(src) {
                let vt = m.types.vector(1, src);
                let vec = const_vector(m, vt, vec![v]);
                return Some(get_cast(m, CastOp::BitCast, vec, dst));
            }
            if matches!(m.types.get(src), Type::PpcFp128) {
                return None;
            }
            if !m.types.is_int_or_int_vector(dst)
                || m.types.scalar_bits(dst) != m.types.scalar_bits(src)
                || m.types.is_vector(dst)
            {
                return None;
            }
            let bits = m.types.int_bits(dst)?;
            let mut w = vec![b as u64, (b >> 64) as u64];
            w.truncate(bits.div_ceil(64) as usize);
            Some(Value::Const(m.intern_const(Constant { ty: dst, kind: ConstKind::Int(w) })))
        }
        _ => None,
    }
}

/// `ConstantExpr::getCast` for a supported cast: folded when LLVM folds it, an expression otherwise.
pub(crate) fn get_cast(m: &mut Module, op: CastOp, v: Value, dst: TypeId) -> Value {
    if let Some(f) = fold_cast(m, op, v, dst) {
        return f;
    }
    Value::Const(m.intern_const(Constant { ty: dst, kind: ConstKind::Expr(ConstExpr::Cast { op, value: v }) }))
}

/// `ConstantExpr::getGetElementPtr` with `ConstantFoldGetElementPtr`.
pub(crate) fn get_gep(
    m: &mut Module,
    ty: TypeId,
    source: TypeId,
    flags: u8,
    inrange: Option<ConstantRange>,
    ptr: Value,
    indices: Vec<Value>,
) -> Value {
    if indices.is_empty() {
        return ptr;
    }
    if is_poison(m, ptr) {
        return Value::Const(m.intern_const(Constant { ty, kind: ConstKind::Poison }));
    }
    if is_undef(m, ptr) {
        return Value::Const(m.intern_const(Constant { ty, kind: ConstKind::Undef }));
    }
    if inrange.is_none() && indices.iter().all(|&i| is_null(m, i) || is_undef(m, i)) {
        let pt = m.module_value_type(ptr).unwrap();
        if let (Some(n), false) = (m.types.vector_len(ty), m.types.is_vector(pt)) {
            return splat(m, ty, n, ptr);
        }
        return ptr;
    }
    let _ = source;
    Value::Const(m.intern_const(Constant {
        ty,
        kind: ConstKind::Expr(ConstExpr::Gep { source, flags: GepFlags(flags), inrange, ptr, indices }),
    }))
}
