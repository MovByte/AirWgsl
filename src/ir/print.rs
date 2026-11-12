//! The textual form of a module, as LLVM 21's `AsmWriter` (`llvm/lib/IR/AsmWriter.cpp`, release 21.1.8) prints it.
//!
//! The output follows `llvm-dis`: slot numbering (`SlotTracker`), the order named and numbered types are printed in
//! (`TypeFinder`), attribute groups, metadata numbering, floating-point formatting (`APFloat::toString` with the
//! round-trip check), escaping, and the comments (`; Function Attrs:`, `; preds =`). It omits the `; ModuleID` line,
//! which names the file `llvm-dis` read.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use super::attrs::{captures_string, memory_string, nofpclass_string};
use super::*;

/// The module's text.
pub fn print_module(m: &Module) -> String {
    let mut p = Printer::new(m);
    p.module();
    p.out
}

/// One instruction's text (for diagnostics; slots are those of the whole function).
pub fn print_instruction(m: &Module, f: usize, i: InstId) -> String {
    let mut p = Printer::new(m);
    p.process_module();
    p.incorporate_function(f);
    let mut s = std::mem::take(&mut p.out);
    p.instruction(f, i);
    std::mem::swap(&mut s, &mut p.out);
    s.trim().to_string()
}

struct Printer<'m> {
    m: &'m Module,
    out: String,
    global_slots: HashMap<GlobalId, u32>,
    md_slots: HashMap<MdId, u32>,
    md_order: Vec<MdId>,
    attr_slots: HashMap<AttrSetId, u32>,
    attr_order: Vec<AttrSetId>,
    numbered_types: HashMap<TypeId, u32>,
    numbered_list: Vec<TypeId>,
    named_list: Vec<TypeId>,
    /// Local slots of the function being printed.
    local_slots: HashMap<LocalKey, u32>,
    cur_fn: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LocalKey {
    Arg(u32),
    Inst(u32),
    Block(u32),
}

/// `printEscapedString`.
pub fn escape(out: &mut String, bytes: &[u8]) {
    for &c in bytes {
        if c == b'\\' {
            out.push_str("\\\\");
        } else if (0x20..0x7f).contains(&c) && c != b'"' {
            out.push(c as char);
        } else {
            let _ = write!(out, "\\{:X}{:X}", c >> 4, c & 0xf);
        }
    }
}

/// `printLLVMNameWithoutPrefix`.
fn llvm_name(out: &mut String, name: &str) {
    let b = name.as_bytes();
    let needs_quotes = b[0].is_ascii_digit() || b.iter().any(|&c| !(c.is_ascii_alphanumeric() || c == b'-' || c == b'.' || c == b'_'));
    if needs_quotes {
        out.push('"');
        escape(out, b);
        out.push('"');
    } else {
        out.push_str(name);
    }
}

/// `printMetadataIdentifier`.
fn md_identifier(out: &mut String, name: &str) {
    if name.is_empty() {
        out.push_str("<empty name> ");
        return;
    }
    for (i, &c) in name.as_bytes().iter().enumerate() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic() || c == b'-' || c == b'$' || c == b'.' || c == b'_'
        } else {
            c.is_ascii_alphanumeric() || c == b'-' || c == b'$' || c == b'.' || c == b'_'
        };
        if ok {
            out.push(c as char);
        } else {
            let _ = write!(out, "\\{:X}{:X}", c >> 4, c & 0xf);
        }
    }
}

fn linkage_name(l: Linkage) -> &'static str {
    match l {
        Linkage::External => "external",
        Linkage::Private => "private",
        Linkage::Internal => "internal",
        Linkage::LinkOnceAny => "linkonce",
        Linkage::LinkOnceOdr => "linkonce_odr",
        Linkage::WeakAny => "weak",
        Linkage::WeakOdr => "weak_odr",
        Linkage::Common => "common",
        Linkage::Appending => "appending",
        Linkage::ExternalWeak => "extern_weak",
        Linkage::AvailableExternally => "available_externally",
    }
}

/// `PrintCallingConv`.
pub fn calling_conv(cc: u32) -> String {
    match cc {
        8 => "fastcc".into(),
        9 => "coldcc".into(),
        13 => "anyregcc".into(),
        14 => "preserve_mostcc".into(),
        15 => "preserve_allcc".into(),
        21 => "preserve_nonecc".into(),
        17 => "cxx_fast_tlscc".into(),
        10 => "ghccc".into(),
        18 => "tailcc".into(),
        107 => "graalcc".into(),
        19 => "cfguard_checkcc".into(),
        64 => "x86_stdcallcc".into(),
        65 => "x86_fastcallcc".into(),
        70 => "x86_thiscallcc".into(),
        92 => "x86_regcallcc".into(),
        80 => "x86_vectorcallcc".into(),
        77 => "intel_ocl_bicc".into(),
        66 => "arm_apcscc".into(),
        67 => "arm_aapcscc".into(),
        68 => "arm_aapcs_vfpcc".into(),
        97 => "aarch64_vector_pcs".into(),
        98 => "aarch64_sve_vector_pcs".into(),
        102 => "aarch64_sme_preservemost_from_x0".into(),
        111 => "aarch64_sme_preservemost_from_x1".into(),
        103 => "aarch64_sme_preservemost_from_x2".into(),
        69 => "msp430_intrcc".into(),
        84 => "avr_intrcc ".into(),
        85 => "avr_signalcc ".into(),
        71 => "ptx_kernel".into(),
        72 => "ptx_device".into(),
        78 => "x86_64_sysvcc".into(),
        79 => "win64cc".into(),
        75 => "spir_func".into(),
        76 => "spir_kernel".into(),
        16 => "swiftcc".into(),
        20 => "swifttailcc".into(),
        83 => "x86_intrcc".into(),
        81 => "hhvmcc".into(),
        82 => "hhvm_ccc".into(),
        87 => "amdgpu_vs".into(),
        95 => "amdgpu_ls".into(),
        93 => "amdgpu_hs".into(),
        96 => "amdgpu_es".into(),
        88 => "amdgpu_gs".into(),
        89 => "amdgpu_ps".into(),
        90 => "amdgpu_cs".into(),
        104 => "amdgpu_cs_chain".into(),
        105 => "amdgpu_cs_chain_preserve".into(),
        91 => "amdgpu_kernel".into(),
        100 => "amdgpu_gfx".into(),
        106 => "m68k_rtdcc".into(),
        110 => "riscv_vector_cc".into(),
        112..=123 => format!("riscv_vls_cc({})", 32u64 << (cc - 112)),
        other => format!("cc{}", other),
    }
}

/// Signed decimal of a `bits`-wide integer in little-endian words (`APInt::print` with `isSigned`).
pub fn int_decimal(words: &[u64], bits: u32) -> String {
    let n = bits.div_ceil(64) as usize;
    let mut w: Vec<u64> = words.iter().copied().chain(std::iter::repeat(0)).take(n).collect();
    let top = (bits - 1) as usize;
    let negative = (w[top / 64] >> (top % 64)) & 1 == 1;
    if negative {
        // Two's complement negation within the width.
        let mut carry = 1u64;
        for x in w.iter_mut() {
            let (v, c) = (!*x).overflowing_add(carry);
            *x = v;
            carry = c as u64;
        }
        if bits % 64 != 0 {
            let last = w.len() - 1;
            w[last] &= (1u64 << (bits % 64)) - 1;
        }
    }
    let mut big = BigUint::from_words(&w);
    let mut digits = Vec::new();
    if big.is_zero() {
        digits.push(b'0');
    }
    while !big.is_zero() {
        digits.push(b'0' + big.div_small(10) as u8);
    }
    let mut s = String::new();
    if negative {
        s.push('-');
    }
    for d in digits.iter().rev() {
        s.push(*d as char);
    }
    s
}

/// An arbitrary-precision unsigned integer, for decimal formatting.
#[derive(Clone, Debug)]
struct BigUint {
    limbs: Vec<u32>,
}

impl BigUint {
    fn from_words(w: &[u64]) -> BigUint {
        let mut limbs = Vec::new();
        for x in w {
            limbs.push(*x as u32);
            limbs.push((*x >> 32) as u32);
        }
        let mut b = BigUint { limbs };
        b.trim();
        b
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn div_small(&mut self, d: u32) -> u32 {
        let mut rem = 0u64;
        for x in self.limbs.iter_mut().rev() {
            let cur = (rem << 32) | *x as u64;
            *x = (cur / d as u64) as u32;
            rem = cur % d as u64;
        }
        self.trim();
        rem as u32
    }

    fn mul_small(&mut self, m: u32) {
        let mut carry = 0u64;
        for x in self.limbs.iter_mut() {
            let cur = *x as u64 * m as u64 + carry;
            *x = cur as u32;
            carry = cur >> 32;
        }
        if carry != 0 {
            self.limbs.push(carry as u32);
        }
    }

    fn shl(&mut self, n: u32) {
        let words = (n / 32) as usize;
        let bits = n % 32;
        let mut out = vec![0u32; words];
        let mut carry = 0u32;
        for &x in &self.limbs {
            if bits == 0 {
                out.push(x);
            } else {
                out.push((x << bits) | carry);
                carry = x >> (32 - bits);
            }
        }
        if carry != 0 {
            out.push(carry);
        }
        self.limbs = out;
        self.trim();
    }

    fn active_bits(&self) -> u32 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() as u32 - 1) * 32 + (32 - top.leading_zeros()),
        }
    }

    fn trailing_zeros(&self) -> u32 {
        let mut n = 0;
        for &x in &self.limbs {
            if x == 0 {
                n += 32;
            } else {
                return n + x.trailing_zeros();
            }
        }
        n
    }

    fn shr(&mut self, n: u32) {
        let words = (n / 32) as usize;
        let bits = n % 32;
        if words >= self.limbs.len() {
            self.limbs.clear();
            return;
        }
        let src: Vec<u32> = self.limbs[words..].to_vec();
        let mut out = Vec::with_capacity(src.len());
        for i in 0..src.len() {
            let lo = src[i] >> bits;
            let hi = if bits != 0 && i + 1 < src.len() { src[i + 1] << (32 - bits) } else { 0 };
            out.push(lo | hi);
        }
        self.limbs = out;
        self.trim();
    }
}

/// `IEEEFloat::toString(Str, FormatPrecision, FormatMaxPadding = 0, TruncateZero = false)` for a finite non-zero
/// value `significand * 2^exp` (the `toStringImpl` path), with `FormatPrecision` given.
fn apfloat_to_string(negative: bool, mut exp: i32, significand: u64, precision_bits: u32, format_precision: u32) -> String {
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    let _ = precision_bits;
    let mut sig = BigUint::from_words(&[significand]);
    // Ignore trailing binary zeros.
    let tz = sig.trailing_zeros();
    exp += tz as i32;
    sig.shr(tz);
    // Change the exponent from 2^e to 10^e.
    if exp > 0 {
        sig.shl(exp as u32);
        exp = 0;
    } else if exp < 0 {
        let texp = (-exp) as u32;
        for _ in 0..texp {
            sig.mul_small(5);
        }
    }
    // AdjustToPrecision on the integer: drop the decimal digits that cannot matter.
    let bits = sig.active_bits();
    let bits_required = (format_precision * 196 + 58) / 59;
    if bits > bits_required {
        let tens = (bits - bits_required) * 59 / 196;
        if tens > 0 {
            exp += tens as i32;
            for _ in 0..tens {
                sig.div_small(10);
            }
        }
    }
    // Digits, least significant first, dropping trailing zeros.
    let mut buffer: Vec<u8> = Vec::new();
    let mut in_trail = true;
    while !sig.is_zero() {
        let d = sig.div_small(10) as u8;
        if in_trail && d == 0 {
            exp += 1;
        } else {
            buffer.push(b'0' + d);
            in_trail = false;
        }
    }
    // AdjustToPrecision on the digits (round half up).
    let n = buffer.len() as u32;
    if n > format_precision {
        let mut first = (n - format_precision) as usize;
        if buffer[first - 1] < b'5' {
            while first < buffer.len() && buffer[first] == b'0' {
                first += 1;
            }
            exp += first as i32;
            buffer.drain(..first);
        } else {
            let mut carried_all = true;
            for i in first..buffer.len() {
                if buffer[i] == b'9' {
                    first += 1;
                } else {
                    buffer[i] += 1;
                    carried_all = false;
                    break;
                }
            }
            if carried_all || first == buffer.len() {
                exp += first as i32;
                buffer.clear();
                buffer.push(b'1');
            } else {
                exp += first as i32;
                buffer.drain(..first);
            }
        }
    }
    let nd = buffer.len() as i32;
    // FormatMaxPadding = 0: always scientific.
    exp += nd - 1;
    out.push(buffer[nd as usize - 1] as char);
    out.push('.');
    for i in 1..nd as usize {
        out.push(buffer[nd as usize - 1 - i] as char);
    }
    if format_precision as i32 > nd - 1 {
        for _ in 0..(format_precision as i32 - nd + 1) {
            out.push('0');
        }
    }
    out.push('e');
    out.push(if exp >= 0 { '+' } else { '-' });
    let e = exp.unsigned_abs();
    if e < 10 {
        out.push('0');
    }
    let _ = write!(out, "{}", e);
    out
}

/// `WriteAPFloatInternal`: a floating-point constant's text.
pub fn float_text(kind: FloatKind, bits: u128) -> String {
    match kind {
        FloatKind::Float | FloatKind::Double => {
            let (negative, exp_bits, mant, ebits, mbits) = match kind {
                FloatKind::Float => {
                    let b = bits as u32;
                    (b >> 31 == 1, ((b >> 23) & 0xff) as i32, (b & 0x7f_ffff) as u64, 8u32, 23u32)
                }
                _ => {
                    let b = bits as u64;
                    (b >> 63 == 1, ((b >> 52) & 0x7ff) as i32, b & 0xf_ffff_ffff_ffff, 11u32, 52u32)
                }
            };
            let max_exp = (1 << ebits) - 1;
            let bias = (1 << (ebits - 1)) - 1;
            let value: f64 = if kind == FloatKind::Float { f32::from_bits(bits as u32) as f64 } else { f64::from_bits(bits as u64) };
            if exp_bits != max_exp {
                let s = if exp_bits == 0 && mant == 0 {
                    // Zero: "0.000000e+00".
                    format!("{}0.000000e+00", if negative { "-" } else { "" })
                } else {
                    let (sig, e) = if exp_bits == 0 {
                        (mant, 1 - bias - mbits as i32)
                    } else {
                        (mant | 1 << mbits, exp_bits - bias - mbits as i32)
                    };
                    apfloat_to_string(negative, e, sig, mbits + 1, 6)
                };
                if let Ok(back) = s.parse::<f64>() {
                    if back == value {
                        return s;
                    }
                }
            }
            // Hexadecimal, as a double.
            let d: u64 = if kind == FloatKind::Double {
                bits as u64
            } else {
                float_to_double_bits(bits as u32)
            };
            format!("0x{:X}", d)
        }
        FloatKind::Half => format!("0xH{:04X}", bits as u16),
        FloatKind::BFloat => format!("0xR{:04X}", bits as u16),
        FloatKind::X86Fp80 => format!("0xK{:04X}{:016X}", (bits >> 64) as u16, bits as u64),
        FloatKind::Fp128 => format!("0xL{:016X}{:016X}", bits as u64, (bits >> 64) as u64),
        FloatKind::PpcFp128 => format!("0xM{:016X}{:016X}", bits as u64, (bits >> 64) as u64),
    }
}

/// `APFloat::convert` from single to double, with LLVM's signaling-NaN handling in `WriteAPFloatInternal`.
fn float_to_double_bits(b: u32) -> u64 {
    let sign = ((b >> 31) as u64) << 63;
    let exp = (b >> 23) & 0xff;
    let mant = (b & 0x7f_ffff) as u64;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7ff0_0000_0000_0000;
        }
        let payload = mant << 29;
        let signaling = mant & 0x40_0000 == 0;
        if !signaling {
            return sign | 0x7ff0_0000_0000_0000 | payload;
        }
        // Converting quiets the NaN; LLVM then rebuilds a signaling NaN from that payload.
        let mut m = (payload | 1 << 51) & ((1u64 << 52) - 1);
        m &= !(1u64 << 51);
        if m == 0 {
            m = 1 << 50;
        }
        return sign | 0x7ff0_0000_0000_0000 | m;
    }
    // Finite values convert exactly.
    (f32::from_bits(b) as f64).to_bits()
}

impl<'m> Printer<'m> {
    fn new(m: &'m Module) -> Printer<'m> {
        Printer {
            m,
            out: String::new(),
            global_slots: HashMap::new(),
            md_slots: HashMap::new(),
            md_order: Vec::new(),
            attr_slots: HashMap::new(),
            attr_order: Vec::new(),
            numbered_types: HashMap::new(),
            numbered_list: Vec::new(),
            named_list: Vec::new(),
            local_slots: HashMap::new(),
            cur_fn: None,
        }
    }

    // Slots ------------------------------------------------------------------------------------------------------

    fn md_slot(&mut self, id: MdId) {
        // `CreateMetadataSlot`: pre-order over node operands. Only nodes get slots.
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            if !matches!(self.m.md(n), Metadata::Tuple { .. }) {
                continue;
            }
            if self.md_slots.contains_key(&n) {
                continue;
            }
            self.md_slots.insert(n, self.md_order.len() as u32);
            self.md_order.push(n);
            if let Metadata::Tuple { ops, .. } = self.m.md(n) {
                for op in ops.iter().rev().flatten() {
                    if matches!(self.m.md(*op), Metadata::Tuple { .. }) && !self.md_slots.contains_key(op) {
                        stack.push(*op);
                    }
                }
            }
        }
    }

    fn attr_slot(&mut self, id: AttrSetId) {
        if id == AttrSetId::EMPTY || self.attr_slots.contains_key(&id) {
            return;
        }
        self.attr_slots.insert(id, self.attr_order.len() as u32);
        self.attr_order.push(id);
    }

    /// `SlotTracker::processModule`.
    fn process_module(&mut self) {
        let m = self.m;
        let mut next = 0u32;
        for (i, g) in m.globals.iter().enumerate() {
            if g.common.name.is_empty() {
                self.global_slots.insert(GlobalId::Var(i as u32), next);
                next += 1;
            }
            for (_, md) in &g.md {
                self.md_slot(*md);
            }
            self.attr_slot(g.attrs);
        }
        for (i, a) in m.aliases.iter().enumerate() {
            if a.common.name.is_empty() {
                self.global_slots.insert(GlobalId::Alias(i as u32), next);
                next += 1;
            }
        }
        for (i, a) in m.ifuncs.iter().enumerate() {
            if a.common.name.is_empty() {
                self.global_slots.insert(GlobalId::IFunc(i as u32), next);
                next += 1;
            }
        }
        for n in &m.named_metadata {
            for op in &n.ops {
                self.md_slot(*op);
            }
        }
        for (i, f) in m.functions.iter().enumerate() {
            if f.common.name.is_empty() {
                self.global_slots.insert(GlobalId::Function(i as u32), next);
                next += 1;
            }
            self.attr_slot(f.attrs.function);
        }
    }

    fn is_intrinsic_name(name: &str) -> bool {
        name.starts_with("llvm.")
    }

    /// `SlotTracker::processFunction` (with its function metadata).
    fn incorporate_function(&mut self, fi: usize) {
        let m = self.m;
        let f = &m.functions[fi];
        self.cur_fn = Some(fi);
        self.local_slots.clear();
        for (_, md) in &f.md {
            self.md_slot(*md);
        }
        for b in &f.blocks {
            for &i in &b.insts {
                let inst = f.inst(i);
                if let Op::Call { callee: Value::Global(GlobalId::Function(c)), args, .. } = &inst.op {
                    if Self::is_intrinsic_name(&m.functions[*c as usize].common.name) {
                        for a in args {
                            if let Value::Metadata(md) = a {
                                if matches!(m.md(*md), Metadata::Tuple { .. }) {
                                    self.md_slot(*md);
                                }
                            }
                        }
                    }
                }
                if let Some(d) = inst.dbg {
                    self.md_slot(d);
                }
                for (_, md) in &inst.md {
                    self.md_slot(*md);
                }
            }
        }
        let mut next = 0u32;
        for (i, a) in f.args.iter().enumerate() {
            if a.name.is_none() {
                self.local_slots.insert(LocalKey::Arg(i as u32), next);
                next += 1;
            }
        }
        for (bi, b) in f.blocks.iter().enumerate() {
            if b.name.is_none() {
                self.local_slots.insert(LocalKey::Block(bi as u32), next);
                next += 1;
            }
            for &i in &b.insts {
                let inst = f.inst(i);
                if !m.types.is_void(inst.ty) && inst.name.is_none() {
                    self.local_slots.insert(LocalKey::Inst(i.0), next);
                    next += 1;
                }
                if let Op::Call { attrs, .. } = &inst.op {
                    self.attr_slot(attrs.function);
                }
            }
        }
    }

    // Types ------------------------------------------------------------------------------------------------------

    /// `TypeFinder::run` and `TypePrinting::incorporateTypes`.
    fn find_types(&mut self) {
        let m = self.m;
        let mut tf = TypeFinder { m, visited_types: HashSet::new(), visited_consts: HashSet::new(), visited_md: HashSet::new(), visited_attrs: HashSet::new(), structs: Vec::new() };
        for g in &m.globals {
            tf.ty(g.value_type);
            if let Some(i) = g.initializer {
                tf.value(i, None);
            }
        }
        for a in &m.aliases {
            tf.ty(a.value_type);
            tf.value(a.aliasee, None);
        }
        for a in &m.ifuncs {
            tf.ty(a.value_type);
        }
        for (fi, f) in m.functions.iter().enumerate() {
            tf.ty(f.ty);
            tf.attrs(&f.attrs);
            let ptr0 = m.types.iter().find(|(_, t)| matches!(t, Type::Ptr(0))).map(|(i, _)| i);
            if f.prefix.is_some() || f.prologue.is_some() || f.personality.is_some() {
                for v in [f.personality, f.prefix, f.prologue] {
                    match v {
                        Some(v) => tf.value(v, Some(fi)),
                        // Absent hung-off operands are `ptr null` placeholders.
                        None => {
                            if let Some(p) = ptr0 {
                                tf.ty(p);
                            }
                        }
                    }
                }
            }
            for b in &f.blocks {
                for &i in &b.insts {
                    let inst = f.inst(i);
                    tf.ty(inst.ty);
                    for op in inst.op.operands() {
                        if !matches!(op, Value::Inst(_) | Value::Arg(_)) {
                            tf.value(op, Some(fi));
                        }
                    }
                    match &inst.op {
                        Op::Gep { source, .. } => tf.ty(*source),
                        Op::Alloca { allocated, .. } => tf.ty(*allocated),
                        Op::Call { attrs, .. } => tf.attrs(attrs),
                        _ => {}
                    }
                    for (_, md) in &inst.md {
                        tf.md(*md);
                    }
                }
            }
        }
        for n in &m.named_metadata {
            for op in &n.ops {
                tf.md(*op);
            }
        }
        let mut next = 0;
        for t in tf.structs {
            if let Type::Struct { name, .. } = m.types.get(t) {
                if name.is_empty() {
                    self.numbered_types.insert(t, next);
                    self.numbered_list.push(t);
                    next += 1;
                } else {
                    self.named_list.push(t);
                }
            }
        }
    }

    fn type_to(&self, out: &mut String, t: TypeId) {
        let m = self.m;
        match m.types.get(t) {
            Type::Void => out.push_str("void"),
            Type::Half => out.push_str("half"),
            Type::BFloat => out.push_str("bfloat"),
            Type::Float => out.push_str("float"),
            Type::Double => out.push_str("double"),
            Type::X86Fp80 => out.push_str("x86_fp80"),
            Type::Fp128 => out.push_str("fp128"),
            Type::PpcFp128 => out.push_str("ppc_fp128"),
            Type::Label => out.push_str("label"),
            Type::Metadata => out.push_str("metadata"),
            Type::X86Amx => out.push_str("x86_amx"),
            Type::Token => out.push_str("token"),
            Type::Int(b) => {
                let _ = write!(out, "i{}", b);
            }
            Type::Function { ret, params, vararg } => {
                self.type_to(out, *ret);
                out.push_str(" (");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.type_to(out, *p);
                }
                if *vararg {
                    if !params.is_empty() {
                        out.push_str(", ");
                    }
                    out.push_str("...");
                }
                out.push(')');
            }
            Type::LiteralStruct { .. } => self.struct_body_to(out, t),
            Type::Struct { name, .. } => {
                if !name.is_empty() {
                    out.push('%');
                    llvm_name(out, name);
                } else {
                    match self.numbered_types.get(&t) {
                        Some(n) => {
                            let _ = write!(out, "%{}", n);
                        }
                        None => out.push_str("%\"type <unnumbered>\""),
                    }
                }
            }
            Type::Ptr(a) => {
                out.push_str("ptr");
                if *a != 0 {
                    let _ = write!(out, " addrspace({})", a);
                }
            }
            Type::Array { len, elem } => {
                let _ = write!(out, "[{} x ", len);
                self.type_to(out, *elem);
                out.push(']');
            }
            Type::Vector { len, elem, scalable } => {
                out.push('<');
                if *scalable {
                    out.push_str("vscale x ");
                }
                let _ = write!(out, "{} x ", len);
                self.type_to(out, *elem);
                out.push('>');
            }
            Type::TargetExt { name, types, ints } => {
                out.push_str("target(\"");
                escape(out, name.as_bytes());
                out.push('"');
                for t in types {
                    out.push_str(", ");
                    self.type_to(out, *t);
                }
                for i in ints {
                    let _ = write!(out, ", {}", i);
                }
                out.push(')');
            }
        }
    }

    fn struct_body_to(&self, out: &mut String, t: TypeId) {
        let (packed, elems) = match self.m.types.get(t) {
            Type::LiteralStruct { packed, elems } => (*packed, Some(elems)),
            Type::Struct { packed, body, .. } => (*packed, body.as_ref()),
            _ => unreachable!(),
        };
        let elems = match elems {
            Some(e) => e,
            None => {
                out.push_str("opaque");
                return;
            }
        };
        if packed {
            out.push('<');
        }
        if elems.is_empty() {
            out.push_str("{}");
        } else {
            out.push_str("{ ");
            for (i, e) in elems.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                self.type_to(out, *e);
            }
            out.push_str(" }");
        }
        if packed {
            out.push('>');
        }
    }

    // Attributes -------------------------------------------------------------------------------------------------

    fn attr_to(&self, out: &mut String, a: &Attribute, in_group: bool) {
        match a {
            Attribute::Enum(k) => out.push_str(k.name()),
            Attribute::Type(k, t) => {
                out.push_str(k.name());
                if let Some(t) = t {
                    out.push('(');
                    self.type_to(out, *t);
                    out.push(')');
                }
            }
            Attribute::Int(k, v) => {
                let v = *v;
                match k {
                    AttrKind::Alignment => {
                        let _ = if in_group { write!(out, "align={}", v) } else { write!(out, "align {}", v) };
                    }
                    AttrKind::StackAlignment | AttrKind::Dereferenceable | AttrKind::DereferenceableOrNull => {
                        let _ = if in_group { write!(out, "{}={}", k.name(), v) } else { write!(out, "{}({})", k.name(), v) };
                    }
                    AttrKind::AllocSize => {
                        // Packed as (elem size << 32) | num elems, with u32::MAX for none.
                        let elem = v >> 32;
                        let num = v & 0xffff_ffff;
                        if num == 0xffff_ffff {
                            let _ = write!(out, "allocsize({})", elem);
                        } else {
                            let _ = write!(out, "allocsize({},{})", elem, num);
                        }
                    }
                    AttrKind::VScaleRange => {
                        let min = v >> 32;
                        let max = v & 0xffff_ffff;
                        let _ = write!(out, "vscale_range({},{})", min, max);
                    }
                    AttrKind::UWTable => out.push_str(if v == 2 { "uwtable" } else { "uwtable(sync)" }),
                    AttrKind::AllocKind => {
                        let names = ["alloc", "realloc", "free", "uninitialized", "zeroed", "aligned"];
                        let parts: Vec<&str> = names.iter().enumerate().filter(|(i, _)| v & (1 << i) != 0).map(|(_, n)| *n).collect();
                        let _ = write!(out, "allockind(\"{}\")", parts.join(","));
                    }
                    AttrKind::Memory => out.push_str(&memory_string(v)),
                    AttrKind::Captures => out.push_str(&captures_string(v)),
                    AttrKind::NoFPClass => out.push_str(&nofpclass_string(v)),
                    _ => {
                        let _ = write!(out, "{}({})", k.name(), v);
                    }
                }
            }
            Attribute::Range(_, cr) => {
                let _ = write!(out, "range(i{} {}, {})", cr.bits, int_decimal(&cr.lower, cr.bits), int_decimal(&cr.upper, cr.bits));
            }
            Attribute::RangeList(_, list) => {
                out.push_str("initializes(");
                for (i, cr) in list.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "({}, {})", int_decimal(&cr.lower, cr.bits), int_decimal(&cr.upper, cr.bits));
                }
                out.push(')');
            }
            Attribute::Str(k, v) => {
                out.push('"');
                out.push_str(&String::from_utf8_lossy(k));
                out.push('"');
                if !v.is_empty() {
                    out.push_str("=\"");
                    escape(out, v);
                    out.push('"');
                }
            }
        }
    }

    fn attr_set_to(&self, out: &mut String, id: AttrSetId, in_group: bool) {
        for (i, a) in self.m.attrs(id).iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            self.attr_to(out, a, in_group);
        }
    }

    // Values -----------------------------------------------------------------------------------------------------

    fn value_type(&self, v: Value) -> TypeId {
        let m = self.m;
        match v {
            Value::Const(c) => m.constant(c).ty,
            Value::Global(g) => {
                let a = m.global(g).addrspace;
                m.types.iter().find(|(_, t)| **t == Type::Ptr(a)).map(|(i, _)| i).unwrap()
            }
            Value::Arg(i) => m.functions[self.cur_fn.unwrap()].args[i as usize].ty,
            Value::Inst(i) => m.functions[self.cur_fn.unwrap()].insts[i.0 as usize].ty,
            Value::Metadata(_) => m.types.iter().find(|(_, t)| matches!(t, Type::Metadata)).map(|(i, _)| i).unwrap(),
            Value::InlineAsm(_) => m.types.iter().find(|(_, t)| **t == Type::Ptr(0)).map(|(i, _)| i).unwrap(),
        }
    }

    /// `WriteAsOperandInternal` for a value.
    fn value_to(&self, out: &mut String, v: Value) {
        let m = self.m;
        match v {
            Value::Global(g) => {
                let c = m.global(g);
                if !c.name.is_empty() {
                    out.push('@');
                    llvm_name(out, &c.name);
                } else {
                    match self.global_slots.get(&g) {
                        Some(s) => {
                            let _ = write!(out, "@{}", s);
                        }
                        None => out.push_str("<badref>"),
                    }
                }
            }
            Value::Arg(i) => {
                let f = &m.functions[self.cur_fn.unwrap()];
                match &f.args[i as usize].name {
                    Some(n) => {
                        out.push('%');
                        llvm_name(out, n);
                    }
                    None => {
                        let _ = write!(out, "%{}", self.local_slots[&LocalKey::Arg(i)]);
                    }
                }
            }
            Value::Inst(i) => {
                let f = &m.functions[self.cur_fn.unwrap()];
                match &f.insts[i.0 as usize].name {
                    Some(n) => {
                        out.push('%');
                        llvm_name(out, n);
                    }
                    None => match self.local_slots.get(&LocalKey::Inst(i.0)) {
                        Some(s) => {
                            let _ = write!(out, "%{}", s);
                        }
                        None => out.push_str("<badref>"),
                    },
                }
            }
            Value::Const(c) => self.constant_to(out, c),
            Value::Metadata(md) => self.md_operand_to(out, md, true),
            Value::InlineAsm(i) => {
                let ia = &m.inline_asms[i as usize];
                out.push_str("asm ");
                if ia.side_effects {
                    out.push_str("sideeffect ");
                }
                if ia.align_stack {
                    out.push_str("alignstack ");
                }
                if ia.dialect == 1 {
                    out.push_str("inteldialect ");
                }
                if ia.can_throw {
                    out.push_str("unwind ");
                }
                out.push('"');
                escape(out, &ia.asm);
                out.push_str("\", \"");
                escape(out, &ia.constraints);
                out.push('"');
            }
        }
    }

    fn block_to(&self, out: &mut String, b: BlockId) {
        let f = &self.m.functions[self.cur_fn.unwrap()];
        match &f.blocks[b.0 as usize].name {
            Some(n) => {
                out.push('%');
                llvm_name(out, n);
            }
            None => match self.local_slots.get(&LocalKey::Block(b.0)) {
                Some(s) => {
                    let _ = write!(out, "%{}", s);
                }
                None => out.push_str("<badref>"),
            },
        }
    }

    /// `writeOperand(V, PrintType)`.
    fn operand_to(&self, out: &mut String, v: Value, with_type: bool) {
        if with_type {
            let t = self.value_type(v);
            self.type_to(out, t);
            out.push(' ');
        }
        self.value_to(out, v);
    }

    fn typed_element_to(&self, out: &mut String, ty: TypeId, v: Value) {
        self.type_to(out, ty);
        out.push(' ');
        self.value_to(out, v);
    }

    /// The splat element of a vector constant if it is an integer or float (`getSplatValue`).
    fn splat_element(&self, c: &Constant) -> Option<(TypeId, ConstKind)> {
        let elem = match self.m.types.get(c.ty) {
            Type::Vector { elem, .. } => *elem,
            _ => return None,
        };
        match &c.kind {
            ConstKind::DataVector(e) => {
                if e.iter().all(|x| *x == e[0]) {
                    let k = if self.m.types.is_int(elem) { ConstKind::Int(vec![e[0]]) } else { ConstKind::Float(e[0] as u128) };
                    Some((elem, k))
                } else {
                    None
                }
            }
            ConstKind::Vector(e) => {
                if e.iter().all(|x| *x == e[0]) {
                    if let Value::Const(x) = e[0] {
                        let k = &self.m.constant(x).kind;
                        if matches!(k, ConstKind::Int(_) | ConstKind::Float(_)) {
                            return Some((elem, k.clone()));
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn scalar_text(&self, out: &mut String, ty: TypeId, k: &ConstKind) {
        match k {
            ConstKind::Int(w) => {
                let bits = self.m.types.int_bits(ty).unwrap();
                if bits == 1 {
                    out.push_str(if w[0] & 1 == 1 { "true" } else { "false" });
                } else {
                    out.push_str(&int_decimal(w, bits));
                }
            }
            ConstKind::Float(b) => out.push_str(&float_text(self.m.types.float_kind(ty).unwrap(), *b)),
            _ => unreachable!(),
        }
    }

    /// `WriteConstantInternal`.
    fn constant_to(&self, out: &mut String, id: ConstId) {
        let m = self.m;
        let c = m.constant(id);
        match &c.kind {
            ConstKind::Int(_) | ConstKind::Float(_) => self.scalar_text(out, c.ty, &c.kind),
            ConstKind::Zero => out.push_str("zeroinitializer"),
            ConstKind::NullPtr => out.push_str("null"),
            ConstKind::TokenNone => out.push_str("none"),
            ConstKind::Poison => out.push_str("poison"),
            ConstKind::Undef => out.push_str("undef"),
            ConstKind::Array(elems) => {
                let et = match m.types.get(c.ty) {
                    Type::Array { elem, .. } => *elem,
                    _ => unreachable!(),
                };
                out.push('[');
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.typed_element_to(out, et, *e);
                }
                out.push(']');
            }
            ConstKind::DataArray(elems) => {
                let et = match m.types.get(c.ty) {
                    Type::Array { elem, .. } => *elem,
                    _ => unreachable!(),
                };
                if m.types.int_bits(et) == Some(8) {
                    out.push_str("c\"");
                    let bytes: Vec<u8> = elems.iter().map(|x| *x as u8).collect();
                    escape(out, &bytes);
                    out.push('"');
                    return;
                }
                out.push('[');
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.type_to(out, et);
                    out.push(' ');
                    let k = if m.types.is_int(et) { ConstKind::Int(vec![*e]) } else { ConstKind::Float(*e as u128) };
                    self.scalar_text(out, et, &k);
                }
                out.push(']');
            }
            ConstKind::Struct(elems) => {
                let packed = matches!(m.types.get(c.ty), Type::LiteralStruct { packed: true, .. } | Type::Struct { packed: true, .. });
                if packed {
                    out.push('<');
                }
                out.push('{');
                if !elems.is_empty() {
                    out.push(' ');
                    for (i, e) in elems.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let t = self.value_type(*e);
                        self.typed_element_to(out, t, *e);
                    }
                    out.push(' ');
                }
                out.push('}');
                if packed {
                    out.push('>');
                }
            }
            ConstKind::Vector(_) | ConstKind::DataVector(_) => {
                let et = m.types.scalar(c.ty);
                if let Some((t, k)) = self.splat_element(c) {
                    out.push_str("splat (");
                    self.type_to(out, t);
                    out.push(' ');
                    self.scalar_text(out, t, &k);
                    out.push(')');
                    return;
                }
                out.push('<');
                match &c.kind {
                    ConstKind::Vector(elems) => {
                        for (i, e) in elems.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            self.typed_element_to(out, et, *e);
                        }
                    }
                    ConstKind::DataVector(elems) => {
                        for (i, e) in elems.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            self.type_to(out, et);
                            out.push(' ');
                            let k = if m.types.is_int(et) { ConstKind::Int(vec![*e]) } else { ConstKind::Float(*e as u128) };
                            self.scalar_text(out, et, &k);
                        }
                    }
                    _ => unreachable!(),
                }
                out.push('>');
            }
            ConstKind::Expr(e) => self.const_expr_to(out, c.ty, e),
            ConstKind::BlockAddress { .. } | ConstKind::DsoLocalEquivalent(_) | ConstKind::NoCfi(_) | ConstKind::PtrAuth { .. } => {
                out.push_str("<unsupported constant>");
            }
        }
    }

    fn const_expr_to(&self, out: &mut String, ty: TypeId, e: &ConstExpr) {
        match e {
            ConstExpr::Cast { op, value } => {
                out.push_str(op.name());
                out.push_str(" (");
                self.operand_to(out, *value, true);
                out.push_str(" to ");
                self.type_to(out, ty);
                out.push(')');
            }
            ConstExpr::Binary { op, lhs, rhs, flags: fl } => {
                out.push_str(op.name());
                if fl & flags::NUW != 0 {
                    out.push_str(" nuw");
                }
                if fl & flags::NSW != 0 {
                    out.push_str(" nsw");
                }
                if fl & flags::EXACT != 0 {
                    out.push_str(" exact");
                }
                out.push_str(" (");
                self.operand_to(out, *lhs, true);
                out.push_str(", ");
                self.operand_to(out, *rhs, true);
                out.push(')');
            }
            ConstExpr::Gep { source, flags: gf, inrange, ptr, indices } => {
                out.push_str("getelementptr");
                self.gep_flags_to(out, *gf);
                if let Some(r) = inrange {
                    let _ = write!(out, " inrange({}, {})", int_decimal(&r.lower, r.bits), int_decimal(&r.upper, r.bits));
                }
                out.push_str(" (");
                self.type_to(out, *source);
                out.push_str(", ");
                self.operand_to(out, *ptr, true);
                for i in indices {
                    out.push_str(", ");
                    self.operand_to(out, *i, true);
                }
                out.push(')');
            }
            ConstExpr::ExtractElement { vec, idx } => {
                out.push_str("extractelement (");
                self.operand_to(out, *vec, true);
                out.push_str(", ");
                self.operand_to(out, *idx, true);
                out.push(')');
            }
            ConstExpr::InsertElement { vec, elt, idx } => {
                out.push_str("insertelement (");
                self.operand_to(out, *vec, true);
                out.push_str(", ");
                self.operand_to(out, *elt, true);
                out.push_str(", ");
                self.operand_to(out, *idx, true);
                out.push(')');
            }
            ConstExpr::ShuffleVector { a, b, mask } => {
                out.push_str("shufflevector (");
                self.operand_to(out, *a, true);
                out.push_str(", ");
                self.operand_to(out, *b, true);
                shuffle_mask_to(out, mask);
                out.push(')');
            }
        }
    }

    fn gep_flags_to(&self, out: &mut String, f: GepFlags) {
        if f.0 & GepFlags::INBOUNDS != 0 {
            out.push_str(" inbounds");
        } else if f.0 & GepFlags::NUSW != 0 {
            out.push_str(" nusw");
        }
        if f.0 & GepFlags::NUW != 0 {
            out.push_str(" nuw");
        }
    }

    // Metadata ---------------------------------------------------------------------------------------------------

    /// `WriteAsOperandInternal` for metadata.
    fn md_operand_to(&self, out: &mut String, md: MdId, from_value: bool) {
        let _ = from_value;
        match self.m.md(md) {
            Metadata::Tuple { .. } => match self.md_slots.get(&md) {
                Some(s) => {
                    let _ = write!(out, "!{}", s);
                }
                None => out.push_str("<unnumbered metadata>"),
            },
            Metadata::String(s) => {
                out.push_str("!\"");
                escape(out, s);
                out.push('"');
            }
            Metadata::Value { ty, value } | Metadata::Local { ty, value, .. } => {
                self.type_to(out, *ty);
                out.push(' ');
                self.value_to(out, *value);
            }
            Metadata::ArgList(items) => {
                out.push_str("!DIArgList(");
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.md_operand_to(out, *it, true);
                }
                out.push(')');
            }
        }
    }

    fn md_node_body_to(&self, out: &mut String, md: MdId) {
        if let Metadata::Tuple { distinct, ops } = self.m.md(md) {
            if *distinct {
                out.push_str("distinct ");
            }
            out.push_str("!{");
            for (i, op) in ops.iter().enumerate() {
                match op {
                    None => out.push_str("null"),
                    Some(o) => self.md_operand_to(out, *o, false),
                }
                if i + 1 != ops.len() {
                    out.push_str(", ");
                }
            }
            out.push('}');
        }
    }

    fn attachments_to(&self, out: &mut String, dbg: Option<MdId>, md: &[(u32, MdId)], sep: &str) {
        let all = dbg.map(|d| (0u32, d)).into_iter().chain(md.iter().copied());
        for (kind, n) in all {
            out.push_str(sep);
            out.push('!');
            match self.m.md_kinds.get(kind as usize) {
                Some(name) => md_identifier(out, name),
                None => {
                    let _ = write!(out, "<unknown kind #{}>", kind);
                }
            }
            out.push(' ');
            self.md_operand_to(out, n, false);
        }
    }

    // Module -----------------------------------------------------------------------------------------------------

    fn is_implicit_dso_local(c: &GlobalCommon) -> bool {
        c.linkage.is_local() || (c.visibility != Visibility::Default && c.linkage != Linkage::ExternalWeak)
    }

    fn global_prefix_to(out: &mut String, c: &GlobalCommon) {
        if c.linkage != Linkage::External {
            out.push_str(linkage_name(c.linkage));
            out.push(' ');
        }
        if c.dso_local && !Self::is_implicit_dso_local(c) {
            out.push_str("dso_local ");
        }
        match c.visibility {
            Visibility::Default => {}
            Visibility::Hidden => out.push_str("hidden "),
            Visibility::Protected => out.push_str("protected "),
        }
    }

    fn dll_to(out: &mut String, d: DllStorage) {
        match d {
            DllStorage::Default => {}
            DllStorage::Import => out.push_str("dllimport "),
            DllStorage::Export => out.push_str("dllexport "),
        }
    }

    fn thread_local_to(out: &mut String, t: ThreadLocal) {
        out.push_str(match t {
            ThreadLocal::NotThreadLocal => "",
            ThreadLocal::GeneralDynamic => "thread_local ",
            ThreadLocal::LocalDynamic => "thread_local(localdynamic) ",
            ThreadLocal::InitialExec => "thread_local(initialexec) ",
            ThreadLocal::LocalExec => "thread_local(localexec) ",
        });
    }

    fn unnamed_addr_str(u: UnnamedAddr) -> &'static str {
        match u {
            UnnamedAddr::None => "",
            UnnamedAddr::Local => "local_unnamed_addr",
            UnnamedAddr::Global => "unnamed_addr",
        }
    }

    fn comdat_to(&self, out: &mut String, comdat: Option<u32>, obj_name: &str, is_var: bool) {
        if let Some(c) = comdat {
            let c = &self.m.comdats[c as usize];
            if is_var {
                out.push(',');
            }
            out.push_str(" comdat");
            if c.name != obj_name {
                out.push_str("($");
                llvm_name(out, &c.name);
                out.push(')');
            }
        }
    }

    fn module(&mut self) {
        let m = self.m;
        self.process_module();
        self.find_types();
        let mut out = std::mem::take(&mut self.out);
        // An empty name is not printed. A module without the record has no name here; LLVM names it after the module
        // identifier (the input file), like the `; ModuleID` line this printer does not write.
        if let Some(s) = m.source_filename.as_ref().filter(|s| !s.is_empty()) {
            out.push_str("source_filename = \"");
            escape(&mut out, s);
            out.push_str("\"\n");
        }
        if !m.datalayout.is_empty() {
            let _ = writeln!(out, "target datalayout = \"{}\"", m.datalayout);
        }
        if !m.triple.is_empty() {
            let _ = writeln!(out, "target triple = \"{}\"", m.triple);
        }
        if !m.module_asm.is_empty() {
            out.push('\n');
            let mut rest: &[u8] = &m.module_asm;
            loop {
                let (front, tail) = match rest.iter().position(|c| *c == b'\n') {
                    Some(i) => (&rest[..i], &rest[i + 1..]),
                    None => (rest, &rest[rest.len()..]),
                };
                out.push_str("module asm \"");
                escape(&mut out, front);
                out.push_str("\"\n");
                rest = tail;
                if rest.is_empty() {
                    break;
                }
            }
        }
        // Type identities.
        if !self.numbered_list.is_empty() || !self.named_list.is_empty() {
            out.push('\n');
            for (i, t) in self.numbered_list.iter().enumerate() {
                let _ = write!(out, "%{} = type ", i);
                self.struct_body_to(&mut out, *t);
                out.push('\n');
            }
            for t in &self.named_list {
                if let Type::Struct { name, .. } = m.types.get(*t) {
                    out.push('%');
                    llvm_name(&mut out, name);
                    out.push_str(" = type ");
                    self.struct_body_to(&mut out, *t);
                    out.push('\n');
                }
            }
        }
        // Comdats, in the order of the global objects using them (functions, then variables).
        let mut comdats: Vec<u32> = Vec::new();
        for c in m.functions.iter().filter_map(|f| f.comdat).chain(m.globals.iter().filter_map(|g| g.comdat)) {
            if !comdats.contains(&c) {
                comdats.push(c);
            }
        }
        if !comdats.is_empty() {
            out.push('\n');
        }
        for (i, c) in comdats.iter().enumerate() {
            let c = &m.comdats[*c as usize];
            out.push('$');
            llvm_name(&mut out, &c.name);
            out.push_str(" = comdat ");
            out.push_str(match c.kind {
                ComdatKind::Any => "any",
                ComdatKind::ExactMatch => "exactmatch",
                ComdatKind::Largest => "largest",
                ComdatKind::NoDeduplicate => "nodeduplicate",
                ComdatKind::SameSize => "samesize",
            });
            if i + 1 != comdats.len() {
                out.push('\n');
            }
        }
        if !comdats.is_empty() {
            out.push('\n');
        }
        if !m.globals.is_empty() {
            out.push('\n');
        }
        for i in 0..m.globals.len() {
            self.global_var_to(&mut out, i);
            out.push('\n');
        }
        if !m.aliases.is_empty() {
            out.push('\n');
        }
        for a in &m.aliases {
            self.value_to(&mut out, Value::Global(GlobalId::Alias(m.aliases.iter().position(|x| std::ptr::eq(x, a)).unwrap() as u32)));
            out.push_str(" = ");
            Self::global_prefix_to(&mut out, &a.common);
            Self::dll_to(&mut out, a.common.dll_storage);
            Self::thread_local_to(&mut out, a.common.thread_local);
            let ua = Self::unnamed_addr_str(a.common.unnamed_addr);
            if !ua.is_empty() {
                out.push_str(ua);
                out.push(' ');
            }
            out.push_str("alias ");
            self.type_to(&mut out, a.value_type);
            out.push_str(", ");
            let is_expr = matches!(a.aliasee, Value::Const(c) if matches!(m.constant(c).kind, ConstKind::Expr(_)));
            self.operand_to(&mut out, a.aliasee, !is_expr);
            if !a.common.partition.is_empty() {
                out.push_str(", partition \"");
                escape(&mut out, a.common.partition.as_bytes());
                out.push('"');
            }
            out.push('\n');
        }
        if !m.ifuncs.is_empty() {
            out.push('\n');
        }
        for (i, a) in m.ifuncs.iter().enumerate() {
            self.value_to(&mut out, Value::Global(GlobalId::IFunc(i as u32)));
            out.push_str(" = ");
            Self::global_prefix_to(&mut out, &a.common);
            out.push_str("ifunc ");
            self.type_to(&mut out, a.value_type);
            out.push_str(", ");
            let is_expr = matches!(a.resolver, Value::Const(c) if matches!(m.constant(c).kind, ConstKind::Expr(_)));
            self.operand_to(&mut out, a.resolver, !is_expr);
            if !a.common.partition.is_empty() {
                out.push_str(", partition \"");
                escape(&mut out, a.common.partition.as_bytes());
                out.push('"');
            }
            out.push('\n');
        }
        self.out = out;
        for f in 0..m.functions.len() {
            self.out.push('\n');
            self.function(f);
        }
        let mut out = std::mem::take(&mut self.out);
        if !self.attr_order.is_empty() {
            out.push('\n');
            for (i, a) in self.attr_order.iter().enumerate() {
                let _ = write!(out, "attributes #{} = {{ ", i);
                self.attr_set_to(&mut out, *a, true);
                out.push_str(" }\n");
            }
        }
        if !m.named_metadata.is_empty() {
            out.push('\n');
        }
        for n in &m.named_metadata {
            out.push('!');
            md_identifier(&mut out, &n.name);
            out.push_str(" = !{");
            for (i, op) in n.ops.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "!{}", self.md_slots[op]);
            }
            out.push_str("}\n");
        }
        if !self.md_order.is_empty() {
            out.push('\n');
            for (i, md) in self.md_order.iter().enumerate() {
                let _ = write!(out, "!{} = ", i);
                self.md_node_body_to(&mut out, *md);
                out.push('\n');
            }
        }
        self.out = out;
    }

    fn global_var_to(&self, out: &mut String, i: usize) {
        let m = self.m;
        let g = &m.globals[i];
        self.value_to(out, Value::Global(GlobalId::Var(i as u32)));
        out.push_str(" = ");
        if g.initializer.is_none() && g.common.linkage == Linkage::External {
            out.push_str("external ");
        }
        Self::global_prefix_to(out, &g.common);
        Self::dll_to(out, g.common.dll_storage);
        Self::thread_local_to(out, g.common.thread_local);
        let ua = Self::unnamed_addr_str(g.common.unnamed_addr);
        if !ua.is_empty() {
            out.push_str(ua);
            out.push(' ');
        }
        if g.common.addrspace != 0 {
            let _ = write!(out, "addrspace({}) ", g.common.addrspace);
        }
        if g.externally_initialized {
            out.push_str("externally_initialized ");
        }
        out.push_str(if g.constant { "constant " } else { "global " });
        self.type_to(out, g.value_type);
        if let Some(init) = g.initializer {
            out.push(' ');
            self.value_to(out, init);
        }
        if let Some(s) = &g.section {
            out.push_str(", section \"");
            escape(out, s.as_bytes());
            out.push('"');
        }
        if !g.common.partition.is_empty() {
            out.push_str(", partition \"");
            escape(out, g.common.partition.as_bytes());
            out.push('"');
        }
        if let Some(cm) = g.code_model {
            out.push_str(", code_model \"");
            out.push_str(match cm {
                CodeModel::Tiny => "tiny",
                CodeModel::Small => "small",
                CodeModel::Kernel => "kernel",
                CodeModel::Medium => "medium",
                CodeModel::Large => "large",
            });
            out.push('"');
        }
        if g.sanitizer & 1 != 0 {
            out.push_str(", no_sanitize_address");
        }
        if g.sanitizer & 2 != 0 {
            out.push_str(", no_sanitize_hwaddress");
        }
        if g.sanitizer & 4 != 0 {
            out.push_str(", sanitize_memtag");
        }
        if g.sanitizer & 8 != 0 {
            out.push_str(", sanitize_address_dyninit");
        }
        self.comdat_to(out, g.comdat, &g.common.name, true);
        if let Some(a) = g.align {
            let _ = write!(out, ", align {}", a);
        }
        self.attachments_to(out, None, &g.md, ", ");
        if g.attrs != AttrSetId::EMPTY {
            let _ = write!(out, " #{}", self.attr_slots[&g.attrs]);
        }
    }

    fn function(&mut self, fi: usize) {
        let m = self.m;
        let f = &m.functions[fi];
        let mut out = std::mem::take(&mut self.out);
        let fn_attrs = m.attrs(f.attrs.function);
        let non_string: Vec<String> = fn_attrs
            .iter()
            .filter(|a| !matches!(a, Attribute::Str(..)))
            .map(|a| {
                let mut s = String::new();
                self.attr_to_plain(&mut s, a);
                s
            })
            .collect();
        if !non_string.is_empty() {
            let _ = writeln!(out, "; Function Attrs: {}", non_string.join(" "));
        }
        if f.common.name.starts_with("llvm.") && !crate::bitcode::is_known_intrinsic(&f.common.name) {
            out.push_str("; Unknown intrinsic\n");
        }
        self.out = String::new();
        self.incorporate_function(fi);
        if f.is_declaration() {
            out.push_str("declare");
            self.attachments_to(&mut out, None, &f.md, " ");
            out.push(' ');
        } else {
            out.push_str("define ");
        }
        Self::global_prefix_to(&mut out, &f.common);
        Self::dll_to(&mut out, f.common.dll_storage);
        if f.cc != 0 {
            out.push_str(&calling_conv(f.cc));
            out.push(' ');
        }
        let (ret, params, vararg) = {
            let (r, p, v) = m.types.function(f.ty).unwrap();
            (r, p.to_vec(), v)
        };
        if f.attrs.ret != AttrSetId::EMPTY {
            self.attr_set_to_plain(&mut out, f.attrs.ret);
            out.push(' ');
        }
        self.type_to(&mut out, ret);
        out.push(' ');
        self.value_to(&mut out, Value::Global(GlobalId::Function(fi as u32)));
        out.push('(');
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            self.type_to(&mut out, *p);
            let pa = f.attrs.param(i);
            if pa != AttrSetId::EMPTY {
                out.push(' ');
                self.attr_set_to(&mut out, pa, false);
            }
            if !f.is_declaration() {
                out.push(' ');
                self.value_to(&mut out, Value::Arg(i as u32));
            }
        }
        if vararg {
            if !params.is_empty() {
                out.push_str(", ");
            }
            out.push_str("...");
        }
        out.push(')');
        let ua = Self::unnamed_addr_str(f.common.unnamed_addr);
        if !ua.is_empty() {
            out.push(' ');
            out.push_str(ua);
        }
        let program_as = crate::ir::layout::DataLayout::parse(&m.datalayout).map(|d| d.program_addrspace).unwrap_or(0);
        if f.common.addrspace != 0 || program_as != 0 {
            let _ = write!(out, " addrspace({})", f.common.addrspace);
        }
        if f.attrs.function != AttrSetId::EMPTY {
            let _ = write!(out, " #{}", self.attr_slots[&f.attrs.function]);
        }
        if let Some(s) = &f.section {
            out.push_str(" section \"");
            escape(&mut out, s.as_bytes());
            out.push('"');
        }
        if !f.common.partition.is_empty() {
            out.push_str(" partition \"");
            escape(&mut out, f.common.partition.as_bytes());
            out.push('"');
        }
        self.comdat_to(&mut out, f.comdat, &f.common.name, false);
        if let Some(a) = f.align {
            let _ = write!(out, " align {}", a);
        }
        if let Some(gc) = &f.gc {
            let _ = write!(out, " gc \"{}\"", gc);
        }
        if let Some(v) = f.prefix {
            out.push_str(" prefix ");
            self.operand_to(&mut out, v, true);
        }
        if let Some(v) = f.prologue {
            out.push_str(" prologue ");
            self.operand_to(&mut out, v, true);
        }
        if let Some(v) = f.personality {
            out.push_str(" personality ");
            self.operand_to(&mut out, v, true);
        }
        if f.is_declaration() {
            out.push('\n');
            self.out = out;
            self.cur_fn = None;
            return;
        }
        self.attachments_to(&mut out, None, &f.md, " ");
        out.push_str(" {");
        // Predecessors, in LLVM's use-list order: the most recently created use first.
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); f.blocks.len()];
        for (bi, b) in f.blocks.iter().enumerate() {
            for &i in &b.insts {
                let uses: Vec<BlockId> = match &f.inst(i).op {
                    // BranchInst sets the false destination before the true one.
                    Op::CondBr { t, f: fb, .. } => vec![*fb, *t],
                    Op::Br { dest } => vec![*dest],
                    Op::Switch { default, cases, .. } => std::iter::once(*default).chain(cases.iter().map(|c| c.1)).collect(),
                    Op::IndirectBr { dests, .. } => dests.clone(),
                    _ => vec![],
                };
                for u in uses {
                    preds[u.0 as usize].push(BlockId(bi as u32));
                }
            }
        }
        self.out = out;
        for bi in 0..f.blocks.len() {
            let mut out = std::mem::take(&mut self.out);
            let b = &f.blocks[bi];
            let entry = bi == 0;
            if let Some(n) = &b.name {
                out.push('\n');
                llvm_name(&mut out, n);
                out.push(':');
            } else if !entry {
                out.push('\n');
                let _ = write!(out, "{}:", self.local_slots[&LocalKey::Block(bi as u32)]);
            }
            if !entry {
                // PadToColumn(50): spaces up to column 50, and at least one. The label is printable ASCII (names are
                // escaped), so its column is its length.
                let col = out.len() - out.rfind('\n').map(|p| p + 1).unwrap_or(0);
                for _ in 0..50usize.saturating_sub(col).max(1) {
                    out.push(' ');
                }
                out.push(';');
                // Newest use first, then any order a use-list record gave.
                let mut p: Vec<BlockId> = preds[bi].iter().rev().copied().collect();
                if let Some(order) = &b.use_order {
                    p = order.iter().map(|&k| p[k as usize]).collect();
                }
                if p.is_empty() {
                    out.push_str(" No predecessors!");
                } else {
                    out.push_str(" preds = ");
                    for (k, pb) in p.iter().enumerate() {
                        if k > 0 {
                            out.push_str(", ");
                        }
                        self.block_to(&mut out, *pb);
                    }
                }
            }
            out.push('\n');
            self.out = out;
            for &i in &b.insts {
                self.instruction(fi, i);
                self.out.push('\n');
            }
        }
        self.out.push_str("}\n");
        self.cur_fn = None;
    }

    /// Attributes outside groups (`Attribute::getAsString(false)`), for return attributes and comments.
    fn attr_to_plain(&self, out: &mut String, a: &Attribute) {
        self.attr_to(out, a, false)
    }

    fn attr_set_to_plain(&self, out: &mut String, id: AttrSetId) {
        self.attr_set_to(out, id, false)
    }

    fn fmf_to(out: &mut String, f: FastMath) {
        if f.is_fast() {
            out.push_str(" fast");
            return;
        }
        for (bit, name) in [
            (FastMath::REASSOC, " reassoc"),
            (FastMath::NNAN, " nnan"),
            (FastMath::NINF, " ninf"),
            (FastMath::NSZ, " nsz"),
            (FastMath::ARCP, " arcp"),
            (FastMath::CONTRACT, " contract"),
            (FastMath::AFN, " afn"),
        ] {
            if f.0 & bit != 0 {
                out.push_str(name);
            }
        }
    }

    fn sync_scope_to(&self, out: &mut String, scope: u32) {
        if scope != 1 {
            out.push_str(" syncscope(\"");
            escape(out, self.m.sync_scopes[scope as usize].as_bytes());
            out.push_str("\")");
        }
    }

    fn is_fp_math_type(&self, ty: TypeId) -> bool {
        let t = &self.m.types;
        if t.is_fp_or_fp_vector(ty) {
            return true;
        }
        match t.get(ty) {
            Type::LiteralStruct { elems, .. } => !elems.is_empty() && elems.iter().all(|e| *e == elems[0]) && t.is_fp_or_fp_vector(elems[0]),
            Type::Array { .. } => {
                let mut e = ty;
                while let Type::Array { elem, .. } = t.get(e) {
                    e = *elem;
                }
                t.is_fp_or_fp_vector(e)
            }
            _ => false,
        }
    }

    /// `printInstruction`.
    fn instruction(&mut self, fi: usize, id: InstId) {
        let m = self.m;
        let f = &m.functions[fi];
        let inst = f.inst(id);
        let mut out = std::mem::take(&mut self.out);
        out.push_str("  ");
        if let Some(n) = &inst.name {
            out.push('%');
            llvm_name(&mut out, n);
            out.push_str(" = ");
        } else if !m.types.is_void(inst.ty) {
            let _ = write!(out, "%{} = ", self.local_slots[&LocalKey::Inst(id.0)]);
        }
        let op = &inst.op;
        if let Op::Call { tail, .. } = op {
            match tail {
                TailKind::MustTail => out.push_str("musttail "),
                TailKind::Tail => out.push_str("tail "),
                TailKind::NoTail => out.push_str("notail "),
                TailKind::None => {}
            }
        }
        // Opcode.
        let opcode: &str = match op {
            Op::FNeg { .. } => "fneg",
            Op::Binary { op, .. } => op.name(),
            Op::Cast { op, .. } => op.name(),
            Op::ICmp { .. } => "icmp",
            Op::FCmp { .. } => "fcmp",
            Op::Select { .. } => "select",
            Op::Phi { .. } => "phi",
            Op::Alloca { .. } => "alloca",
            Op::Load { .. } => "load",
            Op::Store { .. } => "store",
            Op::Gep { .. } => "getelementptr",
            Op::ExtractElement { .. } => "extractelement",
            Op::InsertElement { .. } => "insertelement",
            Op::ShuffleVector { .. } => "shufflevector",
            Op::ExtractValue { .. } => "extractvalue",
            Op::InsertValue { .. } => "insertvalue",
            Op::Call { .. } => "call",
            Op::Freeze { .. } => "freeze",
            Op::AtomicRmw { .. } => "atomicrmw",
            Op::CmpXchg { .. } => "cmpxchg",
            Op::Fence { .. } => "fence",
            Op::VaArg { .. } => "va_arg",
            Op::Ret { .. } => "ret",
            Op::Br { .. } | Op::CondBr { .. } => "br",
            Op::Switch { .. } => "switch",
            Op::IndirectBr { .. } => "indirectbr",
            Op::Unreachable => "unreachable",
        };
        out.push_str(opcode);
        match op {
            Op::Load { atomic: Some(_), .. } | Op::Store { atomic: Some(_), .. } => out.push_str(" atomic"),
            _ => {}
        }
        if let Op::CmpXchg { weak: true, .. } = op {
            out.push_str(" weak");
        }
        match op {
            Op::Load { volatile: true, .. }
            | Op::Store { volatile: true, .. }
            | Op::CmpXchg { volatile: true, .. }
            | Op::AtomicRmw { volatile: true, .. } => out.push_str(" volatile"),
            _ => {}
        }
        // Optimization information.
        match op {
            Op::FNeg { fmf, .. } | Op::FCmp { fmf, .. } => Self::fmf_to(&mut out, *fmf),
            Op::Binary { op: b, flags: fl, fmf, .. } => {
                if b.is_float() {
                    Self::fmf_to(&mut out, *fmf);
                }
                match b {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => {
                        if fl & flags::NUW != 0 {
                            out.push_str(" nuw");
                        }
                        if fl & flags::NSW != 0 {
                            out.push_str(" nsw");
                        }
                    }
                    BinOp::UDiv | BinOp::SDiv | BinOp::LShr | BinOp::AShr => {
                        if fl & flags::EXACT != 0 {
                            out.push_str(" exact");
                        }
                    }
                    BinOp::Or => {
                        if fl & flags::DISJOINT != 0 {
                            out.push_str(" disjoint");
                        }
                    }
                    _ => {}
                }
            }
            Op::Cast { op: c, flags: fl, fmf, .. } => {
                if matches!(c, CastOp::FPTrunc | CastOp::FPExt) {
                    Self::fmf_to(&mut out, *fmf);
                }
                match c {
                    CastOp::ZExt | CastOp::UIToFP => {
                        if fl & flags::NNEG != 0 {
                            out.push_str(" nneg");
                        }
                    }
                    CastOp::Trunc => {
                        if fl & flags::NUW != 0 {
                            out.push_str(" nuw");
                        }
                        if fl & flags::NSW != 0 {
                            out.push_str(" nsw");
                        }
                    }
                    _ => {}
                }
            }
            Op::Select { fmf, .. } | Op::Phi { fmf, .. } | Op::Call { fmf, .. } => {
                if self.is_fp_math_type(inst.ty) {
                    Self::fmf_to(&mut out, *fmf);
                }
            }
            Op::Gep { flags: gf, .. } => self.gep_flags_to(&mut out, *gf),
            Op::ICmp { samesign: true, .. } => out.push_str(" samesign"),
            _ => {}
        }
        match op {
            Op::ICmp { pred, .. } => {
                out.push(' ');
                out.push_str(pred.name());
            }
            Op::FCmp { pred, .. } => {
                out.push(' ');
                out.push_str(pred.name());
            }
            Op::AtomicRmw { op: r, .. } => {
                out.push(' ');
                out.push_str(r.name());
            }
            _ => {}
        }
        match op {
            Op::CondBr { cond, t, f: fb } => {
                out.push(' ');
                self.operand_to(&mut out, *cond, true);
                out.push_str(", label ");
                self.block_to(&mut out, *t);
                out.push_str(", label ");
                self.block_to(&mut out, *fb);
            }
            Op::Br { dest } => {
                out.push_str(" label ");
                self.block_to(&mut out, *dest);
            }
            Op::Switch { cond, default, cases } => {
                out.push(' ');
                self.operand_to(&mut out, *cond, true);
                out.push_str(", label ");
                self.block_to(&mut out, *default);
                out.push_str(" [");
                for (c, b) in cases {
                    out.push_str("\n    ");
                    self.operand_to(&mut out, Value::Const(*c), true);
                    out.push_str(", label ");
                    self.block_to(&mut out, *b);
                }
                out.push_str("\n  ]");
            }
            Op::IndirectBr { addr, dests } => {
                out.push(' ');
                self.operand_to(&mut out, *addr, true);
                out.push_str(", [");
                for (i, d) in dests.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str("label ");
                    self.block_to(&mut out, *d);
                }
                out.push(']');
            }
            Op::Phi { incoming, .. } => {
                out.push(' ');
                self.type_to(&mut out, inst.ty);
                out.push(' ');
                for (k, (v, b)) in incoming.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    out.push_str("[ ");
                    self.value_to(&mut out, *v);
                    out.push_str(", ");
                    self.block_to(&mut out, *b);
                    out.push_str(" ]");
                }
            }
            Op::ExtractValue { agg, indices } => {
                out.push(' ');
                self.operand_to(&mut out, *agg, true);
                for i in indices {
                    let _ = write!(out, ", {}", i);
                }
            }
            Op::InsertValue { agg, value, indices } => {
                out.push(' ');
                self.operand_to(&mut out, *agg, true);
                out.push_str(", ");
                self.operand_to(&mut out, *value, true);
                for i in indices {
                    let _ = write!(out, ", {}", i);
                }
            }
            Op::Ret { value: None } => out.push_str(" void"),
            Op::Call { callee, fn_ty, args, attrs, cc, bundles, .. } => {
                if *cc != 0 {
                    out.push(' ');
                    out.push_str(&calling_conv(*cc));
                }
                if attrs.ret != AttrSetId::EMPTY {
                    out.push(' ');
                    self.attr_set_to_plain(&mut out, attrs.ret);
                }
                let callee_as = m.types.pointer_addrspace(self.value_type(*callee)).unwrap_or(0);
                let program_as = crate::ir::layout::DataLayout::parse(&m.datalayout).map(|d| d.program_addrspace).unwrap_or(0);
                if callee_as != 0 || program_as != 0 {
                    let _ = write!(out, " addrspace({})", callee_as);
                }
                let (ret, _, vararg) = m.types.function(*fn_ty).unwrap();
                out.push(' ');
                self.type_to(&mut out, if vararg { *fn_ty } else { ret });
                out.push(' ');
                self.value_to(&mut out, *callee);
                out.push('(');
                for (k, a) in args.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    let t = self.value_type(*a);
                    self.type_to(&mut out, t);
                    let pa = attrs.param(k);
                    if pa != AttrSetId::EMPTY {
                        out.push(' ');
                        self.attr_set_to(&mut out, pa, false);
                    }
                    out.push(' ');
                    self.value_to(&mut out, *a);
                }
                if let TailKind::MustTail = inst.op_tail() {
                    let fvararg = m.types.function(f.ty).unwrap().2;
                    if fvararg {
                        if !args.is_empty() {
                            out.push_str(", ");
                        }
                        out.push_str("...");
                    }
                }
                out.push(')');
                if attrs.function != AttrSetId::EMPTY {
                    let _ = write!(out, " #{}", self.attr_slots[&attrs.function]);
                }
                if !bundles.is_empty() {
                    out.push_str(" [ ");
                    for (k, b) in bundles.iter().enumerate() {
                        if k > 0 {
                            out.push_str(", ");
                        }
                        out.push('"');
                        escape(&mut out, m.bundle_tags[b.tag as usize].as_bytes());
                        out.push_str("\"(");
                        for (j, v) in b.inputs.iter().enumerate() {
                            if j > 0 {
                                out.push_str(", ");
                            }
                            self.operand_to(&mut out, *v, true);
                        }
                        out.push(')');
                    }
                    out.push_str(" ]");
                }
            }
            Op::Alloca { allocated, count, align, addrspace, inalloca, swifterror } => {
                out.push(' ');
                if *inalloca {
                    out.push_str("inalloca ");
                }
                if *swifterror {
                    out.push_str("swifterror ");
                }
                self.type_to(&mut out, *allocated);
                let count_ty = self.value_type(*count);
                let is_one = matches!(count, Value::Const(c) if matches!(&m.constant(*c).kind, ConstKind::Int(w) if w.len() == 1 && w[0] == 1));
                if !is_one || m.types.int_bits(count_ty) != Some(32) {
                    out.push_str(", ");
                    self.operand_to(&mut out, *count, true);
                }
                let _ = write!(out, ", align {}", align);
                if *addrspace != 0 {
                    let _ = write!(out, ", addrspace({})", addrspace);
                }
            }
            Op::Cast { value, .. } => {
                out.push(' ');
                self.operand_to(&mut out, *value, true);
                out.push_str(" to ");
                self.type_to(&mut out, inst.ty);
            }
            Op::VaArg { list } => {
                out.push(' ');
                self.operand_to(&mut out, *list, true);
                out.push_str(", ");
                self.type_to(&mut out, inst.ty);
            }
            Op::Unreachable | Op::Fence { .. } => {}
            _ => {
                if let Op::Gep { source, .. } = op {
                    out.push(' ');
                    self.type_to(&mut out, *source);
                    out.push(',');
                } else if let Op::Load { .. } = op {
                    out.push(' ');
                    self.type_to(&mut out, inst.ty);
                    out.push(',');
                }
                let operands = op.operands();
                let first_ty = self.value_type(operands[0]);
                let print_all = matches!(op, Op::Select { .. } | Op::Store { .. } | Op::ShuffleVector { .. } | Op::Ret { .. } | Op::CmpXchg { .. } | Op::AtomicRmw { .. })
                    || operands[1..].iter().any(|o| self.value_type(*o) != first_ty);
                if !print_all {
                    out.push(' ');
                    self.type_to(&mut out, first_ty);
                }
                out.push(' ');
                for (k, o) in operands.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    self.operand_to(&mut out, *o, print_all);
                }
            }
        }
        match op {
            Op::Load { atomic, align, .. } | Op::Store { atomic, align, .. } => {
                if let Some(a) = atomic {
                    self.sync_scope_to(&mut out, a.scope);
                    out.push(' ');
                    out.push_str(a.ordering.name());
                }
                let _ = write!(out, ", align {}", align);
            }
            Op::CmpXchg { success, failure, scope, align, .. } => {
                self.sync_scope_to(&mut out, *scope);
                let _ = write!(out, " {} {}", success.name(), failure.name());
                let _ = write!(out, ", align {}", align);
            }
            Op::AtomicRmw { ordering, scope, align, .. } => {
                self.sync_scope_to(&mut out, *scope);
                let _ = write!(out, " {}", ordering.name());
                let _ = write!(out, ", align {}", align);
            }
            Op::Fence { ordering, scope } => {
                self.sync_scope_to(&mut out, *scope);
                let _ = write!(out, " {}", ordering.name());
            }
            Op::ShuffleVector { mask, .. } => shuffle_mask_to(&mut out, mask),
            _ => {}
        }
        self.attachments_to(&mut out, inst.dbg, &inst.md, ", ");
        self.out = out;
    }
}

impl Instruction {
    fn op_tail(&self) -> TailKind {
        match &self.op {
            Op::Call { tail, .. } => *tail,
            _ => TailKind::None,
        }
    }
}

/// `PrintShuffleMask`.
fn shuffle_mask_to(out: &mut String, mask: &[i32]) {
    let _ = write!(out, ", <{} x i32> ", mask.len());
    if mask.iter().all(|&e| e == 0) {
        out.push_str("zeroinitializer");
    } else if mask.iter().all(|&e| e == -1) {
        out.push_str("poison");
    } else {
        out.push('<');
        for (i, e) in mask.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            if *e == -1 {
                out.push_str("i32 poison");
            } else {
                let _ = write!(out, "i32 {}", e);
            }
        }
        out.push('>');
    }
}

/// `TypeFinder`.
struct TypeFinder<'m> {
    m: &'m Module,
    visited_types: HashSet<TypeId>,
    visited_consts: HashSet<ConstId>,
    visited_md: HashSet<MdId>,
    visited_attrs: HashSet<AttrList>,
    structs: Vec<TypeId>,
}

impl<'m> TypeFinder<'m> {
    fn ty(&mut self, t: TypeId) {
        if !self.visited_types.insert(t) {
            return;
        }
        let mut work = vec![t];
        while let Some(t) = work.pop() {
            if matches!(self.m.types.get(t), Type::Struct { .. } | Type::LiteralStruct { .. }) {
                self.structs.push(t);
            }
            let subs: Vec<TypeId> = match self.m.types.get(t) {
                Type::Function { ret, params, .. } => std::iter::once(*ret).chain(params.iter().copied()).collect(),
                Type::LiteralStruct { elems, .. } => elems.clone(),
                Type::Struct { body: Some(e), .. } => e.clone(),
                Type::Array { elem, .. } | Type::Vector { elem, .. } => vec![*elem],
                Type::TargetExt { types, .. } => types.clone(),
                _ => vec![],
            };
            for s in subs.into_iter().rev() {
                if self.visited_types.insert(s) {
                    work.push(s);
                }
            }
        }
    }

    fn attrs(&mut self, a: &AttrList) {
        if !self.visited_attrs.insert(a.clone()) {
            return;
        }
        // AttributeList iteration order: function, return, parameters.
        let sets: Vec<AttrSetId> = std::iter::once(a.function).chain(std::iter::once(a.ret)).chain(a.params.iter().copied()).collect();
        for s in sets {
            for at in self.m.attrs(s).iter() {
                if let Attribute::Type(_, Some(t)) = at {
                    self.ty(*t);
                }
            }
        }
    }

    fn value(&mut self, v: Value, f: Option<usize>) {
        match v {
            Value::Metadata(md) => match self.m.md(md) {
                Metadata::Tuple { .. } => self.md(md),
                Metadata::Value { value, .. } | Metadata::Local { value, .. } => {
                    let value = *value;
                    self.value(value, f)
                }
                Metadata::ArgList(items) => {
                    for it in items.clone() {
                        if let Metadata::Value { value, .. } | Metadata::Local { value, .. } = self.m.md(it) {
                            let value = *value;
                            self.value(value, f);
                        }
                    }
                }
                Metadata::String(_) => {}
            },
            Value::Const(c) => {
                if !self.visited_consts.insert(c) {
                    return;
                }
                let con = self.m.constant(c);
                self.ty(con.ty);
                let ops: Vec<Value> = match &con.kind {
                    ConstKind::Struct(e) | ConstKind::Array(e) | ConstKind::Vector(e) => e.clone(),
                    ConstKind::Expr(ConstExpr::Gep { source, ptr, indices, .. }) => {
                        self.ty(*source);
                        std::iter::once(*ptr).chain(indices.iter().copied()).collect()
                    }
                    ConstKind::Expr(ConstExpr::Cast { value, .. }) => vec![*value],
                    ConstKind::Expr(ConstExpr::Binary { lhs, rhs, .. }) => vec![*lhs, *rhs],
                    ConstKind::Expr(ConstExpr::ExtractElement { vec, idx }) => vec![*vec, *idx],
                    ConstKind::Expr(ConstExpr::InsertElement { vec, elt, idx }) => vec![*vec, *elt, *idx],
                    ConstKind::Expr(ConstExpr::ShuffleVector { a, b, .. }) => vec![*a, *b],
                    _ => vec![],
                };
                for o in ops {
                    self.value(o, f);
                }
            }
            // Inline asm is a value of pointer type with no operands to walk.
            Value::InlineAsm(_) => {
                if let Some((t, _)) = self.m.types.iter().find(|(_, t)| **t == Type::Ptr(0)) {
                    self.ty(t);
                }
            }
            Value::Global(_) | Value::Arg(_) | Value::Inst(_) => {}
        }
    }

    fn md(&mut self, id: MdId) {
        // `incorporateMDNode`, depth first over node operands, without recursion.
        if !self.visited_md.insert(id) {
            return;
        }
        let mut stack: Vec<(MdId, usize)> = vec![(id, 0)];
        while let Some((node, next)) = stack.pop() {
            let ops = match self.m.md(node) {
                Metadata::Tuple { ops, .. } => ops,
                _ => continue,
            };
            if next >= ops.len() {
                continue;
            }
            stack.push((node, next + 1));
            let op = match ops[next] {
                Some(op) => op,
                None => continue,
            };
            match self.m.md(op) {
                Metadata::Tuple { .. } => {
                    if self.visited_md.insert(op) {
                        stack.push((op, 0));
                    }
                }
                Metadata::Value { value, .. } => {
                    let value = *value;
                    self.value(value, None);
                }
                Metadata::ArgList(items) => {
                    for it in items.clone() {
                        if let Metadata::Value { value, .. } | Metadata::Local { value, .. } = self.m.md(it) {
                            let value = *value;
                            self.value(value, None);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_like_llvm() {
        assert_eq!(float_text(FloatKind::Float, 1.0f32.to_bits() as u128), "1.000000e+00");
        assert_eq!(float_text(FloatKind::Float, 0.5f32.to_bits() as u128), "5.000000e-01");
        assert_eq!(float_text(FloatKind::Float, (-2.0f32).to_bits() as u128), "-2.000000e+00");
        assert_eq!(float_text(FloatKind::Float, 0.0f32.to_bits() as u128), "0.000000e+00");
        assert_eq!(float_text(FloatKind::Float, (-0.0f32).to_bits() as u128), "-0.000000e+00");
        assert_eq!(float_text(FloatKind::Float, 0.1f32.to_bits() as u128), "0x3FB99999A0000000");
        assert_eq!(float_text(FloatKind::Double, 0.1f64.to_bits() as u128), "1.000000e-01");
        assert_eq!(float_text(FloatKind::Float, 255.0f32.to_bits() as u128), "2.550000e+02");
        assert_eq!(float_text(FloatKind::Float, f32::INFINITY.to_bits() as u128), "0x7FF0000000000000");
        assert_eq!(float_text(FloatKind::Float, f32::NAN.to_bits() as u128), "0x7FF8000000000000");
        assert_eq!(float_text(FloatKind::Half, 0x3c00), "0xH3C00");
        assert_eq!(float_text(FloatKind::Double, 1e300f64.to_bits() as u128), "1.000000e+300");
    }

    #[test]
    fn integers_print_signed() {
        assert_eq!(int_decimal(&[255], 8), "-1");
        assert_eq!(int_decimal(&[127], 8), "127");
        assert_eq!(int_decimal(&[u64::MAX, u64::MAX], 128), "-1");
        assert_eq!(int_decimal(&[0, 1], 128), "18446744073709551616");
        assert_eq!(int_decimal(&[0], 32), "0");
    }

    #[test]
    fn names_quote_when_needed() {
        let mut s = String::new();
        llvm_name(&mut s, "struct.Espresso::split_params");
        assert_eq!(s, "\"struct.Espresso::split_params\"");
        let mut s = String::new();
        llvm_name(&mut s, "air.read_texture_2d.v4f16");
        assert_eq!(s, "air.read_texture_2d.v4f16");
    }
}
