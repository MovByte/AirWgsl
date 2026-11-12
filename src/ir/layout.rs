//! The data layout string and the sizes and alignments it gives types (`llvm/lib/IR/DataLayout.cpp`, release
//! 21.1.8).
//!
//! AIR modules state their layout explicitly (`e-p:64:64:64-i1:8:8-...-v96:128:128-...-n8:16:32`); memory layout
//! downstream comes from here, never from WGSL's rules.

use super::types::{Type, TypeId, TypeTable};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrimitiveSpec {
    bits: u32,
    abi: u64,
    pref: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointerSpec {
    pub addrspace: u32,
    pub bits: u32,
    pub abi: u64,
    pub pref: u64,
    pub index_bits: u32,
    pub non_integral: bool,
}

/// A parsed data layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataLayout {
    pub big_endian: bool,
    pub alloca_addrspace: u32,
    pub program_addrspace: u32,
    pub globals_addrspace: u32,
    /// `S<size>`, in bytes.
    pub stack_natural_align: Option<u64>,
    /// Native integer widths (`n...`).
    pub legal_int_widths: Vec<u32>,
    int_specs: Vec<PrimitiveSpec>,
    float_specs: Vec<PrimitiveSpec>,
    vector_specs: Vec<PrimitiveSpec>,
    pointer_specs: Vec<PointerSpec>,
    struct_abi: u64,
    struct_pref: u64,
}

impl Default for DataLayout {
    /// LLVM's defaults, which an empty layout string leaves in place.
    fn default() -> DataLayout {
        let p = |bits, abi, pref| PrimitiveSpec { bits, abi, pref };
        DataLayout {
            big_endian: false,
            alloca_addrspace: 0,
            program_addrspace: 0,
            globals_addrspace: 0,
            stack_natural_align: None,
            legal_int_widths: Vec::new(),
            int_specs: vec![p(1, 1, 1), p(8, 1, 1), p(16, 2, 2), p(32, 4, 4), p(64, 4, 8)],
            float_specs: vec![p(16, 2, 2), p(32, 4, 4), p(64, 8, 8), p(128, 16, 16)],
            vector_specs: vec![p(64, 8, 8), p(128, 16, 16)],
            pointer_specs: vec![PointerSpec {
                addrspace: 0,
                bits: 64,
                abi: 8,
                pref: 8,
                index_bits: 64,
                non_integral: false,
            }],
            struct_abi: 1,
            struct_pref: 8,
        }
    }
}

fn parse_u32(s: &str, what: &str, limit_bits: u32) -> Result<u32, String> {
    if s.is_empty() {
        return Err(format!("{} component cannot be empty", what));
    }
    if !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{} must be a decimal integer", what));
    }
    match s.parse::<u64>() {
        Ok(v) if v < (1u64 << limit_bits) => Ok(v as u32),
        _ => Err(format!("{} must be a {}-bit integer", what, limit_bits)),
    }
}

fn parse_size(s: &str, what: &str) -> Result<u32, String> {
    let v = parse_u32(s, what, 24)?;
    if v == 0 {
        return Err(format!("{} must be a non-zero 24-bit integer", what));
    }
    Ok(v)
}

/// An alignment component in bits, returned in bytes.
fn parse_align(s: &str, what: &str, allow_zero: bool) -> Result<u64, String> {
    let v = parse_u32(s, &format!("{} alignment", what), 16)?;
    if v == 0 {
        if !allow_zero {
            return Err(format!("{} alignment must be non-zero", what));
        }
        return Ok(1);
    }
    if v % 8 != 0 || !(v / 8).is_power_of_two() {
        return Err(format!("{} alignment must be a power of two times the byte width", what));
    }
    Ok((v / 8) as u64)
}

fn parse_addrspace(s: &str) -> Result<u32, String> {
    parse_u32(s, "address space", 24)
}

impl DataLayout {
    pub fn parse(s: &str) -> Result<DataLayout, String> {
        let mut dl = DataLayout::default();
        if s.is_empty() {
            return Ok(dl);
        }
        let mut non_integral = Vec::new();
        for spec in s.split('-') {
            if spec.is_empty() {
                return Err("empty specification is not allowed".into());
            }
            dl.parse_spec(spec, &mut non_integral)?;
        }
        for a in non_integral {
            let p = *dl.pointer_spec(a);
            dl.set_pointer_spec(PointerSpec { addrspace: a, non_integral: true, ..p });
        }
        Ok(dl)
    }

    fn parse_spec(&mut self, spec: &str, non_integral: &mut Vec<u32>) -> Result<(), String> {
        if let Some(rest) = spec.strip_prefix("ni") {
            let rest = rest
                .strip_prefix(':')
                .ok_or("malformed specification, must be of the form \"ni:<address space>[:<address space>]...\"")?;
            for a in rest.split(':') {
                let a = parse_addrspace(a)?;
                if a == 0 {
                    return Err("address space 0 cannot be non-integral".into());
                }
                non_integral.push(a);
            }
            return Ok(());
        }
        let (c, rest) = spec.split_at(1);
        match c {
            "i" | "f" | "v" => {
                let parts: Vec<&str> = rest.split(':').collect();
                if parts.len() < 2 || parts.len() > 3 {
                    return Err(format!("malformed specification, must be of the form \"{}<size>:<abi>[:<pref>]\"", c));
                }
                let bits = parse_size(parts[0], "size")?;
                let abi = parse_align(parts[1], "ABI", false)?;
                if c == "i" && bits == 8 && abi != 1 {
                    return Err("i8 must be 8-bit aligned".into());
                }
                let pref = if parts.len() > 2 { parse_align(parts[2], "preferred", false)? } else { abi };
                if pref < abi {
                    return Err("preferred alignment cannot be less than the ABI alignment".into());
                }
                let specs = match c {
                    "i" => &mut self.int_specs,
                    "f" => &mut self.float_specs,
                    _ => &mut self.vector_specs,
                };
                match specs.binary_search_by_key(&bits, |s| s.bits) {
                    Ok(i) => {
                        specs[i].abi = abi;
                        specs[i].pref = pref;
                    }
                    Err(i) => specs.insert(i, PrimitiveSpec { bits, abi, pref }),
                }
            }
            "a" => {
                let parts: Vec<&str> = rest.split(':').collect();
                if parts.len() < 2 || parts.len() > 3 {
                    return Err("malformed specification, must be of the form \"a:<abi>[:<pref>]\"".into());
                }
                if !parts[0].is_empty() && parts[0].parse::<u64>().ok() != Some(0) {
                    return Err("size must be zero".into());
                }
                let abi = parse_align(parts[1], "ABI", true)?;
                let pref = if parts.len() > 2 { parse_align(parts[2], "preferred", false)? } else { abi };
                if pref < abi {
                    return Err("preferred alignment cannot be less than the ABI alignment".into());
                }
                self.struct_abi = abi;
                self.struct_pref = pref;
            }
            "p" => {
                let parts: Vec<&str> = rest.split(':').collect();
                if parts.len() < 3 || parts.len() > 5 {
                    return Err(
                        "malformed specification, must be of the form \"p[<n>]:<size>:<abi>[:<pref>[:<idx>]]\"".into(),
                    );
                }
                let addrspace = if parts[0].is_empty() { 0 } else { parse_addrspace(parts[0])? };
                let bits = parse_size(parts[1], "pointer size")?;
                let abi = parse_align(parts[2], "ABI", false)?;
                let pref = if parts.len() > 3 { parse_align(parts[3], "preferred", false)? } else { abi };
                if pref < abi {
                    return Err("preferred alignment cannot be less than the ABI alignment".into());
                }
                let index_bits = if parts.len() > 4 { parse_size(parts[4], "index size")? } else { bits };
                if index_bits > bits {
                    return Err("index size cannot be larger than the pointer size".into());
                }
                self.set_pointer_spec(PointerSpec { addrspace, bits, abi, pref, index_bits, non_integral: false });
            }
            "s" => {}
            "e" | "E" => {
                if !rest.is_empty() {
                    return Err("malformed specification, must be just 'e' or 'E'".into());
                }
                self.big_endian = c == "E";
            }
            "n" => {
                for w in rest.split(':') {
                    self.legal_int_widths.push(parse_size(w, "size")?);
                }
            }
            "S" => {
                if rest.is_empty() {
                    return Err("malformed specification, must be of the form \"S<size>\"".into());
                }
                self.stack_natural_align = Some(parse_align(rest, "stack natural", false)?);
            }
            "F" => {
                let mut ch = rest.chars();
                match ch.next() {
                    Some('i') | Some('n') => {}
                    Some(t) => return Err(format!("unknown function pointer alignment type '{}'", t)),
                    None => return Err("malformed specification, must be of the form \"F<type><abi>\"".into()),
                }
                parse_align(&rest[1..], "ABI", false)?;
            }
            "P" => self.program_addrspace = parse_addrspace(rest)?,
            "A" => self.alloca_addrspace = parse_addrspace(rest)?,
            "G" => self.globals_addrspace = parse_addrspace(rest)?,
            "m" => {
                let mode =
                    rest.strip_prefix(':').ok_or("malformed specification, must be of the form \"m:<mangling>\"")?;
                if !matches!(mode, "e" | "l" | "o" | "m" | "w" | "x" | "a") {
                    return Err("unknown mangling mode".into());
                }
            }
            other => return Err(format!("unknown specifier '{}'", other)),
        }
        Ok(())
    }

    fn set_pointer_spec(&mut self, p: PointerSpec) {
        match self.pointer_specs.binary_search_by_key(&p.addrspace, |s| s.addrspace) {
            Ok(i) => self.pointer_specs[i] = p,
            Err(i) => self.pointer_specs.insert(i, p),
        }
    }

    /// The pointer specification of an address space (that of address space 0 when it has none).
    pub fn pointer_spec(&self, addrspace: u32) -> &PointerSpec {
        if addrspace != 0
            && let Ok(i) = self.pointer_specs.binary_search_by_key(&addrspace, |s| s.addrspace)
        {
            return &self.pointer_specs[i];
        }
        &self.pointer_specs[0]
    }

    fn int_align(&self, bits: u32, abi: bool) -> u64 {
        let i = self.int_specs.partition_point(|s| s.bits < bits);
        let s = if i == self.int_specs.len() { &self.int_specs[i - 1] } else { &self.int_specs[i] };
        if abi { s.abi } else { s.pref }
    }

    /// Size in bits (`getTypeSizeInBits`). `None` for unsized types.
    pub fn size_bits(&self, types: &TypeTable, t: TypeId) -> Option<u64> {
        Some(match types.get(t) {
            Type::Label => self.pointer_spec(0).bits as u64,
            Type::Ptr(a) => self.pointer_spec(*a).bits as u64,
            Type::Array { len, elem } => len.checked_mul(self.alloc_size(types, *elem)? * 8)?,
            Type::LiteralStruct { .. } | Type::Struct { .. } => self.struct_layout(types, t)?.size * 8,
            Type::Int(b) => *b as u64,
            Type::Half | Type::BFloat => 16,
            Type::Float => 32,
            Type::Double => 64,
            Type::Fp128 | Type::PpcFp128 => 128,
            Type::X86Fp80 => 80,
            Type::X86Amx => 8192,
            Type::Vector { len, elem, scalable: false } => *len as u64 * self.size_bits(types, *elem)?,
            _ => return None,
        })
    }

    /// `getTypeStoreSize`, in bytes.
    pub fn store_size(&self, types: &TypeTable, t: TypeId) -> Option<u64> {
        Some(self.size_bits(types, t)?.div_ceil(8))
    }

    /// `getTypeAllocSize`, in bytes.
    pub fn alloc_size(&self, types: &TypeTable, t: TypeId) -> Option<u64> {
        let s = self.store_size(types, t)?;
        let a = self.abi_align(types, t)?;
        Some(s.div_ceil(a) * a)
    }

    pub fn abi_align(&self, types: &TypeTable, t: TypeId) -> Option<u64> {
        self.align(types, t, true)
    }

    pub fn pref_align(&self, types: &TypeTable, t: TypeId) -> Option<u64> {
        self.align(types, t, false)
    }

    fn align(&self, types: &TypeTable, t: TypeId, abi: bool) -> Option<u64> {
        Some(match types.get(t) {
            Type::Label => {
                let p = self.pointer_spec(0);
                if abi { p.abi } else { p.pref }
            }
            Type::Ptr(a) => {
                let p = self.pointer_spec(*a);
                if abi { p.abi } else { p.pref }
            }
            Type::Array { elem, .. } => self.align(types, *elem, abi)?,
            Type::LiteralStruct { packed, .. } | Type::Struct { packed, .. } => {
                if *packed && abi {
                    return Some(1);
                }
                let l = self.struct_layout(types, t)?;
                let base = if abi { self.struct_abi } else { self.struct_pref };
                base.max(l.align)
            }
            Type::Int(b) => self.int_align(*b, abi),
            Type::Half | Type::BFloat | Type::Float | Type::Double | Type::PpcFp128 | Type::Fp128 | Type::X86Fp80 => {
                let bits = self.size_bits(types, t)? as u32;
                match self.float_specs.binary_search_by_key(&bits, |s| s.bits) {
                    Ok(i) => {
                        if abi {
                            self.float_specs[i].abi
                        } else {
                            self.float_specs[i].pref
                        }
                    }
                    Err(_) => ((bits / 8) as u64).next_power_of_two(),
                }
            }
            Type::Vector { scalable: false, .. } => {
                let bits = self.size_bits(types, t)?;
                match self.vector_specs.binary_search_by_key(&(bits as u32), |s| s.bits) {
                    Ok(i) if bits <= u32::MAX as u64 => {
                        if abi {
                            self.vector_specs[i].abi
                        } else {
                            self.vector_specs[i].pref
                        }
                    }
                    _ => self.store_size(types, t)?.next_power_of_two(),
                }
            }
            Type::X86Amx => 64,
            _ => return None,
        })
    }

    /// Member offsets, size and alignment of a struct (`StructLayout`).
    pub fn struct_layout(&self, types: &TypeTable, t: TypeId) -> Option<StructLayout> {
        let (packed, elems) = match types.get(t) {
            Type::LiteralStruct { packed, elems } => (*packed, elems.clone()),
            Type::Struct { packed, body: Some(elems), .. } => (*packed, elems.clone()),
            _ => return None,
        };
        let mut size = 0u64;
        let mut align = 1u64;
        let mut offsets = Vec::with_capacity(elems.len());
        for e in elems {
            let a = if packed { 1 } else { self.abi_align(types, e)? };
            size = size.div_ceil(a) * a;
            align = align.max(a);
            offsets.push(size);
            size = size.checked_add(self.alloc_size(types, e)?)?;
        }
        size = size.div_ceil(align) * align;
        Some(StructLayout { offsets, size, align })
    }
}

/// Where a struct's members are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout {
    /// Byte offset of each member.
    pub offsets: Vec<u64>,
    /// Size in bytes, padded to the alignment.
    pub size: u64,
    /// The largest member alignment (not counting the layout's aggregate alignment).
    pub align: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: &str = concat!(
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-",
        "v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-",
        "v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    );

    #[test]
    fn air_layout_sizes() {
        let dl = DataLayout::parse(AIR).unwrap();
        let mut t = TypeTable::new();
        let f = t.intern(Type::Float);
        let v3 = t.vector(3, f);
        let h = t.intern(Type::Half);
        let v3h = t.vector(3, h);
        assert_eq!(dl.store_size(&t, v3), Some(12));
        assert_eq!(dl.abi_align(&t, v3), Some(16));
        assert_eq!(dl.alloc_size(&t, v3), Some(16));
        assert_eq!(dl.abi_align(&t, v3h), Some(8));
        assert_eq!(dl.alloc_size(&t, v3h), Some(8));
        let i1 = t.int(1);
        assert_eq!(dl.alloc_size(&t, i1), Some(1));
        let s = t.intern(Type::LiteralStruct { packed: false, elems: vec![i1, v3] });
        let l = dl.struct_layout(&t, s).unwrap();
        assert_eq!(l.offsets, vec![0, 16]);
        assert_eq!(l.size, 32);
    }

    #[test]
    fn rejects_bad_specs() {
        assert!(DataLayout::parse("i8:16").is_err());
        assert!(DataLayout::parse("e--p:64:64").is_err());
        assert!(DataLayout::parse("q").is_err());
    }
}
