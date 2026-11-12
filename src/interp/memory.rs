//! Memory: allocated objects (LangRef "Allocated Objects") as regions of bytes in one flat address space.
//!
//! Every region has a distinct base address; regions never overlap and are separated by unmapped gaps, so an address
//! names at most one region. A pointer value carries the region it is based on (its provenance); an access through
//! a pointer must fall inside that region, and the region must be alive (LangRef "Object Lifetime"). A pointer
//! without provenance (from `inttoptr`, or loaded from bytes that were not a stored pointer) accesses the region its
//! address falls in.
//!
//! A byte holds eight bits with LangRef's indefiniteness ([`super::value`]): undef and choice bits, or poison for the
//! whole byte. A byte written as part of a pointer also remembers the pointer's provenance and its position in the
//! pointer, so that loading the pointer back restores the provenance.

use std::collections::HashMap;

use super::value::{Scalar, Val, mask};
use crate::ir::layout::DataLayout;
use crate::ir::{Type, TypeId, TypeTable};

/// A region's index, from 1 (0 is "no provenance").
pub type RegionId = u32;

/// One byte of memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Byte {
    pub v: u8,
    pub undef: u8,
    pub choice: u8,
    pub poison: bool,
    /// For a byte of a stored pointer: the byte's index in the pointer plus one; 0 otherwise.
    pub frag: u8,
    /// For a byte of a stored pointer: the pointer's provenance.
    pub prov: u32,
}

impl Byte {
    pub const UNDEF: Byte = Byte { v: 0, undef: 0xff, choice: 0, poison: false, frag: 0, prov: 0 };
    pub const POISON: Byte = Byte { v: 0, undef: 0, choice: 0, poison: true, frag: 0, prov: 0 };

    pub fn data(v: u8) -> Byte {
        Byte { v, undef: 0, choice: 0, poison: false, frag: 0, prov: 0 }
    }

    /// Defined, with no choice bits and not part of a pointer.
    pub fn is_plain(&self) -> bool {
        !self.poison && self.undef == 0 && self.choice == 0 && self.frag == 0
    }
}

/// What a region is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionKind {
    /// Memory the caller provides (a bound buffer, threadgroup memory, an argument buffer).
    Buffer,
    /// A global variable of the module, by index.
    Global(u32),
    /// Memory of an `alloca`.
    Alloca,
    /// An object the interpreter does not model as bytes (a texture, a sampler, an acceleration structure): pointers
    /// to it are handles; the calls that use them are evaluated as functions of the handle and of the object's
    /// write count.
    Object,
    /// A function of the module, by index (a function pointer's target).
    Function(u32),
}

/// An allocated object.
#[derive(Clone, Debug)]
pub struct Region {
    pub kind: RegionKind,
    pub addrspace: u32,
    pub base: u64,
    pub bytes: Vec<Byte>,
    pub writable: bool,
    pub live: bool,
    /// A name that identifies the region across runs (comparisons of two runs match regions by it).
    pub label: String,
    /// Writes to an [`RegionKind::Object`] region so far (each call that may write it counts one).
    pub version: u64,
}

/// All regions.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    regions: Vec<Region>,
    next_base: u64,
    globals: HashMap<u32, RegionId>,
    functions: HashMap<u32, RegionId>,
}

/// Base addresses are multiples of this, and consecutive regions leave at least one such unmapped block between
/// them.
const GRANULE: u64 = 1 << 16;

impl Memory {
    pub fn new() -> Memory {
        Memory { regions: Vec::new(), next_base: GRANULE, globals: HashMap::new(), functions: HashMap::new() }
    }

    /// Adds a region with the given contents.
    pub fn add(
        &mut self,
        kind: RegionKind,
        addrspace: u32,
        bytes: Vec<Byte>,
        writable: bool,
        label: String,
    ) -> RegionId {
        let base = self.next_base;
        let len = bytes.len() as u64;
        self.next_base = (base + len).div_ceil(GRANULE) * GRANULE + GRANULE;
        self.regions.push(Region { kind, addrspace, base, bytes, writable, live: true, label, version: 0 });
        self.regions.len() as RegionId
    }

    /// Adds a region of defined bytes.
    pub fn add_data(
        &mut self,
        kind: RegionKind,
        addrspace: u32,
        data: &[u8],
        writable: bool,
        label: String,
    ) -> RegionId {
        self.add(kind, addrspace, data.iter().map(|&b| Byte::data(b)).collect(), writable, label)
    }

    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id as usize - 1]
    }

    pub fn region_mut(&mut self, id: RegionId) -> &mut Region {
        &mut self.regions[id as usize - 1]
    }

    pub fn regions(&self) -> impl Iterator<Item = (RegionId, &Region)> {
        self.regions.iter().enumerate().map(|(i, r)| ((i + 1) as RegionId, r))
    }

    /// The pointer to `offset` in region `id`.
    pub fn pointer(&self, id: RegionId, offset: u64) -> Scalar {
        Scalar::ptr(id, self.region(id).base.wrapping_add(offset))
    }

    /// The region of global variable `index`, if it was created.
    pub fn global(&self, index: u32) -> Option<RegionId> {
        self.globals.get(&index).copied()
    }

    pub fn set_global(&mut self, index: u32, id: RegionId) {
        self.globals.insert(index, id);
    }

    /// The region standing for function `index` (created on first use).
    pub fn function(&mut self, index: u32, name: &str) -> RegionId {
        if let Some(&r) = self.functions.get(&index) {
            return r;
        }
        let r = self.add(RegionKind::Function(index), 0, Vec::new(), false, format!("function {}", name));
        self.functions.insert(index, r);
        r
    }

    /// The region whose range `[base, base + len]` contains `addr` (one past the end included).
    pub fn region_at(&self, addr: u64) -> Option<RegionId> {
        let i = self.regions.partition_point(|r| r.base <= addr);
        if i == 0 {
            return None;
        }
        let r = &self.regions[i - 1];
        if addr <= r.base + r.bytes.len() as u64 { Some(i as RegionId) } else { None }
    }

    /// The region and offset an access of `size` bytes through `p` touches, or why the access is undefined.
    pub fn resolve(&self, p: &Scalar, size: u64) -> Result<(RegionId, u64), String> {
        if p.poison {
            return Err("access through a poison pointer".into());
        }
        if p.undef != 0 {
            return Err("access through an undef pointer".into());
        }
        let addr = p.bits as u64;
        let id = if p.prov != 0 {
            p.prov
        } else {
            match self.region_at(addr) {
                Some(r) => r,
                None => return Err(format!("access at address {:#x}, which no allocated object holds", addr)),
            }
        };
        let r = self.region(id);
        if !r.live {
            return Err(format!("access to {}, which is not alive", r.label));
        }
        let off = addr.wrapping_sub(r.base);
        if off > r.bytes.len() as u64 || size > r.bytes.len() as u64 - off {
            return Err(format!(
                "access of {} bytes at offset {} of {} ({} bytes)",
                size,
                off as i64,
                r.label,
                r.bytes.len()
            ));
        }
        Ok((id, off))
    }

    pub fn read(&self, p: &Scalar, size: u64) -> Result<Vec<Byte>, String> {
        let (id, off) = self.resolve(p, size)?;
        Ok(self.region(id).bytes[off as usize..(off + size) as usize].to_vec())
    }

    pub fn write(&mut self, p: &Scalar, bytes: &[Byte]) -> Result<(), String> {
        let (id, off) = self.resolve(p, bytes.len() as u64)?;
        let r = self.region_mut(id);
        if !r.writable {
            return Err(format!("store to {}, which is read-only", r.label));
        }
        r.bytes[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

/// Byte layout of values: what a store writes and a load reads (little-endian, as AIR's data layout states).
pub struct Layout<'a> {
    pub dl: &'a DataLayout,
    pub types: &'a TypeTable,
}

impl Layout<'_> {
    /// Bytes a store of `ty` writes.
    pub fn store_size(&self, ty: TypeId) -> Option<u64> {
        self.dl.store_size(self.types, ty)
    }

    pub fn alloc_size(&self, ty: TypeId) -> Option<u64> {
        self.dl.alloc_size(self.types, ty)
    }

    fn ptr_bytes(&self, addrspace: u32) -> u64 {
        (self.dl.pointer_spec(addrspace).bits / 8) as u64
    }

    /// The bytes of `v`, a value of type `ty` (`store_size(ty)` of them; aggregate padding undef).
    pub fn encode(&self, ty: TypeId, v: &Val) -> Result<Vec<Byte>, String> {
        let n = self.store_size(ty).ok_or("a store of an unsized type")? as usize;
        let mut out = vec![Byte::UNDEF; n];
        self.encode_into(ty, v, &mut out)?;
        Ok(out)
    }

    fn encode_into(&self, ty: TypeId, v: &Val, out: &mut [Byte]) -> Result<(), String> {
        match (self.types.get(ty), v) {
            (Type::Ptr(a), Val::S(s)) => {
                let n = self.ptr_bytes(*a) as usize;
                for (i, b) in out.iter_mut().take(n).enumerate() {
                    *b = scalar_byte(s, i);
                    if !s.poison {
                        b.frag = i as u8 + 1;
                        b.prov = s.prov;
                    }
                }
            }
            (Type::Vector { len, elem, .. }, Val::V(lanes)) => {
                let eb = self.types.scalar_bits(*elem);
                if self.types.is_ptr(*elem) {
                    let es = self.ptr_bytes(self.types.pointer_addrspace(*elem).unwrap_or(0)) as usize;
                    for (k, s) in lanes.iter().enumerate() {
                        self.encode_into(*elem, &Val::S(*s), &mut out[k * es..(k + 1) * es])?;
                    }
                } else if eb.is_multiple_of(8) {
                    let es = (eb / 8) as usize;
                    for (k, s) in lanes.iter().enumerate() {
                        for i in 0..es {
                            out[k * es + i] = scalar_byte(s, i);
                        }
                    }
                } else {
                    // Elements narrower than a byte or not byte-sized: the vector is stored as its bitcast to an
                    // integer of len * eb bits, element 0 in the low bits (LangRef "Vector Type", memory layout).
                    let total = *len * eb;
                    if total > 128 {
                        return Err(format!("a store of a {}-bit packed vector", total));
                    }
                    let s = pack_lanes(lanes, eb);
                    write_int(&s, total, out);
                }
            }
            (Type::Array { elem, .. }, Val::A(items)) => {
                let es = self.alloc_size(*elem).ok_or("an array of unsized elements")? as usize;
                for (k, item) in items.iter().enumerate() {
                    let sz = self.store_size(*elem).unwrap() as usize;
                    self.encode_into(*elem, item, &mut out[k * es..k * es + sz])?;
                }
            }
            (Type::Struct { .. } | Type::LiteralStruct { .. }, Val::A(items)) => {
                let l = self.dl.struct_layout(self.types, ty).ok_or("a store of an opaque struct")?;
                let elems = self.types.struct_elems(ty).unwrap();
                for (k, item) in items.iter().enumerate() {
                    let off = l.offsets[k] as usize;
                    let sz = self.store_size(elems[k]).ok_or("an unsized struct member")? as usize;
                    self.encode_into(elems[k], item, &mut out[off..off + sz])?;
                }
            }
            (Type::Int(w), Val::S(s)) => write_int(s, *w, out),
            (t, Val::S(s)) if float_width(t).is_some() => write_int(s, float_width(t).unwrap(), out),
            (t, v) => return Err(format!("a store of {:?} as {:?}", v, t)),
        }
        Ok(())
    }

    /// The value of type `ty` that `bytes` (`store_size(ty)` of them) hold.
    pub fn decode(&self, ty: TypeId, bytes: &[Byte]) -> Result<Val, String> {
        Ok(match self.types.get(ty) {
            Type::Ptr(a) => Val::S(read_ptr(bytes, self.ptr_bytes(*a) as usize)),
            Type::Vector { len, elem, .. } => {
                let eb = self.types.scalar_bits(*elem);
                if self.types.is_ptr(*elem) {
                    let es = self.ptr_bytes(self.types.pointer_addrspace(*elem).unwrap_or(0)) as usize;
                    Val::V((0..*len as usize).map(|k| read_ptr(&bytes[k * es..(k + 1) * es], es)).collect())
                } else if eb.is_multiple_of(8) {
                    let es = (eb / 8) as usize;
                    Val::V((0..*len as usize).map(|k| read_int(&bytes[k * es..(k + 1) * es], eb)).collect())
                } else {
                    let total = *len * eb;
                    if total > 128 {
                        return Err(format!("a load of a {}-bit packed vector", total));
                    }
                    let s = read_int(bytes, total);
                    Val::V(unpack_lanes(&s, eb, *len))
                }
            }
            Type::Array { len, elem } => {
                let es = self.alloc_size(*elem).ok_or("an array of unsized elements")? as usize;
                let sz = self.store_size(*elem).unwrap() as usize;
                let mut items = Vec::with_capacity(*len as usize);
                for k in 0..*len as usize {
                    items.push(self.decode(*elem, &bytes[k * es..k * es + sz])?);
                }
                Val::A(items)
            }
            Type::Struct { .. } | Type::LiteralStruct { .. } => {
                let l = self.dl.struct_layout(self.types, ty).ok_or("a load of an opaque struct")?;
                let elems = self.types.struct_elems(ty).unwrap().to_vec();
                let mut items = Vec::with_capacity(elems.len());
                for (k, e) in elems.iter().enumerate() {
                    let off = l.offsets[k] as usize;
                    let sz = self.store_size(*e).ok_or("an unsized struct member")? as usize;
                    items.push(self.decode(*e, &bytes[off..off + sz])?);
                }
                Val::A(items)
            }
            Type::Int(w) => Val::S(read_int(bytes, *w)),
            t if float_width(t).is_some() => Val::S(read_int(bytes, float_width(t).unwrap())),
            t => return Err(format!("a load of type {:?}", t)),
        })
    }
}

/// Width of a floating-point type's encoding.
pub fn float_width(t: &Type) -> Option<u32> {
    match t {
        Type::Half | Type::BFloat => Some(16),
        Type::Float => Some(32),
        Type::Double => Some(64),
        _ => None,
    }
}

fn scalar_byte(s: &Scalar, i: usize) -> Byte {
    if s.poison {
        return Byte::POISON;
    }
    let sh = 8 * i as u32;
    if sh >= 128 {
        return Byte::UNDEF;
    }
    Byte {
        v: (s.bits >> sh) as u8,
        undef: (s.undef >> sh) as u8,
        choice: (s.choice >> sh) as u8,
        poison: false,
        frag: 0,
        prov: 0,
    }
}

/// Writes a `width`-bit scalar to `width.div_ceil(8)` bytes. The bits of the last byte above the width are undef:
/// LangRef leaves unspecified what a store of a type like `i20` does to them.
fn write_int(s: &Scalar, width: u32, out: &mut [Byte]) {
    let n = width.div_ceil(8) as usize;
    for (i, b) in out.iter_mut().take(n).enumerate() {
        *b = scalar_byte(s, i);
        let top = 8 * (i as u32 + 1);
        if top > width && !b.poison {
            let keep = (1u16 << (8 - (top - width))) as u8 - 1;
            b.v &= keep;
            b.choice &= keep;
            b.undef |= !keep;
        }
    }
}

/// Reads a `width`-bit scalar from `width.div_ceil(8)` bytes: poison if any byte is (LangRef "Poison Values": a load
/// that reads a poison byte returns poison); the bits above the width are dropped.
fn read_int(bytes: &[Byte], width: u32) -> Scalar {
    let n = width.div_ceil(8) as usize;
    let mut s = Scalar::int(0);
    for (i, b) in bytes.iter().take(n).enumerate() {
        if b.poison {
            return Scalar::POISON;
        }
        let sh = 8 * i as u32;
        s.bits |= (b.v as u128) << sh;
        s.undef |= (b.undef as u128) << sh;
        s.choice |= (b.choice as u128) << sh;
    }
    let m = mask(width);
    s.bits &= m;
    s.undef &= m;
    s.choice &= m;
    s
}

/// Reads a pointer: the bytes of one stored pointer give back its provenance; any other bytes give an address
/// without provenance.
fn read_ptr(bytes: &[Byte], n: usize) -> Scalar {
    let mut s = read_int(bytes, 8 * n as u32);
    if s.poison {
        return s;
    }
    let prov = bytes[0].prov;
    if bytes.iter().take(n).enumerate().all(|(i, b)| b.frag == i as u8 + 1 && b.prov == prov) {
        s.prov = prov;
    }
    s
}

/// Packs vector lanes into one scalar of `lanes * eb` bits, lane 0 lowest; poison if any lane is.
pub fn pack_lanes(lanes: &[Scalar], eb: u32) -> Scalar {
    let mut s = Scalar::int(0);
    for (k, l) in lanes.iter().enumerate() {
        if l.poison {
            return Scalar::POISON;
        }
        let sh = k as u32 * eb;
        let m = mask(eb);
        s.bits |= (l.bits & m) << sh;
        s.undef |= (l.undef & m) << sh;
        s.choice |= (l.choice & m) << sh;
    }
    s
}

/// The inverse of [`pack_lanes`].
pub fn unpack_lanes(s: &Scalar, eb: u32, len: u32) -> Vec<Scalar> {
    (0..len)
        .map(|k| {
            if s.poison {
                return Scalar::POISON;
            }
            let sh = k * eb;
            let m = mask(eb);
            Scalar {
                bits: (s.bits >> sh) & m,
                undef: (s.undef >> sh) & m,
                choice: (s.choice >> sh) & m,
                prov: 0,
                poison: false,
            }
        })
        .collect()
}

/// Reinterprets lanes of `sw` bits as `n` lanes of `dw` bits (lane 0 in the low bits, as a little-endian bitcast
/// orders them). A destination lane is poison when a source lane it takes bits from is.
pub fn bitcast_lanes(lanes: &[Scalar], sw: u32, dw: u32, n: u32) -> Vec<Scalar> {
    (0..n)
        .map(|k| {
            let (lo, hi) = (k * dw, (k + 1) * dw);
            let mut out = Scalar::int(0);
            for (j, l) in lanes.iter().enumerate() {
                let (slo, shi) = (j as u32 * sw, (j as u32 + 1) * sw);
                if shi <= lo || slo >= hi {
                    continue;
                }
                if l.poison {
                    return Scalar::POISON;
                }
                for b in slo.max(lo)..shi.min(hi) {
                    let (from, to) = (b - slo, b - lo);
                    out.bits |= ((l.bits >> from) & 1) << to;
                    out.undef |= ((l.undef >> from) & 1) << to;
                    out.choice |= ((l.choice >> from) & 1) << to;
                }
            }
            out
        })
        .collect()
}
