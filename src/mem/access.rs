//! Loads, stores and byte copies as word accesses on the region arrays.
//!
//! A value of any type is a list of leaves (scalars at byte offsets, from the data layout). Reading it reads the
//! words that cover its bytes and extracts each leaf with shifts and masks; writing it assembles the bytes of each
//! covered word and writes whole words where every byte is the value's, and masked words elsewhere. Padding bytes
//! inside the value are left as they are (LLVM leaves them undefined after a store). The offset's residue modulo 4 is
//! taken from the analysis or the instruction's alignment (an `align` that the address does not have is undefined
//! behavior in LLVM); where it is unknown, each 4-byte group of the value is moved through the two words it may span,
//! with the shift computed at run time.

use super::{Memory, NULL, RegionId};
use crate::api::Refusal;
use crate::ir::layout::DataLayout;
use crate::ir::{Type, TypeId, TypeTable};
use crate::wgsl::tracked;
use crate::wgsl::types::{Shape, Types};

/// Where generated statements go.
pub trait Code {
    /// A fresh identifier.
    fn fresh(&mut self) -> String;
    /// One statement.
    fn line(&mut self, s: String);
    /// Open a compound statement (`if c {`, `loop {`).
    fn open(&mut self, head: String);
    fn close(&mut self);
}

/// A pointer at the point of an access.
#[derive(Clone, Debug)]
pub struct Ptr {
    /// The candidate regions.
    pub regions: Vec<RegionId>,
    /// The byte offset (a `u32` expression).
    pub off: String,
    /// The offset when it is a compile-time constant.
    pub konst: Option<u64>,
    /// The region tag (`u32` expression) when there are several candidates.
    pub tag: Option<String>,
    /// The offset modulo 4 when known.
    pub mod4: Option<u32>,
}

impl Ptr {
    /// The pointer `bytes` further on.
    pub fn plus(&self, bytes: u64) -> Ptr {
        if bytes == 0 {
            return self.clone();
        }
        Ptr {
            regions: self.regions.clone(),
            off: match self.konst {
                Some(k) => format!("{}u", (k.wrapping_add(bytes)) as u32),
                None => format!("({} + {}u)", self.off, bytes as u32),
            },
            konst: self.konst.map(|k| (k.wrapping_add(bytes)) & 0xffff_ffff),
            tag: self.tag.clone(),
            mod4: self.mod4.map(|r| ((r as u64 + bytes) % 4) as u32),
        }
    }

    /// The pointer `expr * stride` further on (`expr` a `u32` expression).
    pub fn plus_scaled(&self, expr: &str, stride: u64) -> Ptr {
        Ptr {
            regions: self.regions.clone(),
            off: format!("({} + {} * {}u)", self.off, expr, stride as u32),
            konst: None,
            tag: self.tag.clone(),
            mod4: if stride.is_multiple_of(4) { self.mod4 } else { None },
        }
    }

    /// The pointer with what an access's `align` tells about its offset.
    pub fn with_align(&self, align: u64) -> Ptr {
        let mut p = self.clone();
        if p.mod4.is_none() && align >= 4 {
            p.mod4 = Some(0);
        }
        if let Some(k) = p.konst {
            p.mod4 = Some((k % 4) as u32);
        }
        p
    }
}

/// The context of an access.
pub struct Access<'a> {
    pub tt: &'a TypeTable,
    pub dl: &'a DataLayout,
    pub types: &'a mut Types,
    pub mem: &'a Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeafKind {
    Bool,
    Word,
    Wide,
}

#[derive(Clone, Debug)]
struct Leaf {
    off: u64,
    bytes: u32,
    kind: LeafKind,
    path: String,
}

/// Largest number of leaves moved as one group of words; larger arrays are moved element by element in a loop.
const SMALL_LEAVES: usize = 64;

fn mem_refusal(what: &str) -> Refusal {
    tracked::refuse("pointer-in-memory", what.to_string(), None, format!("{} in memory", what))
}

impl Access<'_> {
    fn leaves(&self, t: TypeId, off: u64, path: &str, out: &mut Vec<Leaf>) -> Result<(), Refusal> {
        match self.tt.get(t) {
            Type::Int(1) => out.push(Leaf { off, bytes: 1, kind: LeafKind::Bool, path: path.into() }),
            Type::Int(8) => out.push(Leaf { off, bytes: 1, kind: LeafKind::Word, path: path.into() }),
            Type::Int(16) | Type::Half => out.push(Leaf { off, bytes: 2, kind: LeafKind::Word, path: path.into() }),
            Type::Int(32) | Type::Float => out.push(Leaf { off, bytes: 4, kind: LeafKind::Word, path: path.into() }),
            Type::Int(64) => out.push(Leaf { off, bytes: 8, kind: LeafKind::Wide, path: path.into() }),
            Type::Int(b) => return Err(tracked::refuse("integer-width", format!("i{}", b), None, "in memory")),
            Type::Double => return Err(tracked::refuse_family("double", "double", None)),
            Type::BFloat => return Err(tracked::refuse_family("bfloat", "bfloat", None)),
            Type::Ptr(_) => return Err(mem_refusal("a pointer")),
            Type::Vector { len, elem, scalable: false } => {
                if self.tt.int_bits(*elem) == Some(1) {
                    return Err(tracked::refuse("vector-shape", "<N x i1>", None, "a vector of i1 in memory"));
                }
                // Vector elements are packed at their size in bits.
                let es = self.dl.size_bits(self.tt, *elem).unwrap_or(0) / 8;
                for i in 0..*len as u64 {
                    self.leaves(*elem, off + i * es, &format!("{}[{}]", path, i), out)?;
                }
            }
            Type::Array { len, elem } => {
                let es = self.dl.alloc_size(self.tt, *elem).unwrap_or(0);
                for i in 0..*len {
                    self.leaves(*elem, off + i * es, &format!("{}[{}]", path, i), out)?;
                    if out.len() > SMALL_LEAVES * 4 {
                        return Ok(());
                    }
                }
            }
            Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                let l = self.dl.struct_layout(self.tt, t).ok_or_else(|| mem_refusal("an unsized struct"))?;
                for (i, e) in elems.iter().enumerate() {
                    self.leaves(*e, off + l.offsets[i], &format!("{}.m{}", path, i), out)?;
                }
            }
            other => return Err(mem_refusal(&format!("{:?}", other))),
        }
        Ok(())
    }

    fn is_small(&self, t: TypeId) -> Result<bool, Refusal> {
        let mut v = Vec::new();
        self.leaves(t, 0, "", &mut v)?;
        Ok(v.len() <= SMALL_LEAVES)
    }

    // Words ---------------------------------------------------------------------------------------------------------

    fn fixed_out_of_range(&self, r: RegionId, w: &WordIx) -> bool {
        matches!((w.konst, self.mem.regions[r].words), (Some(k), Some(n)) if k >= n)
    }

    fn read_word(&self, r: RegionId, w: &WordIx) -> String {
        let reg = &self.mem.regions[r];
        if r == NULL || self.fixed_out_of_range(r, w) {
            // Outside the object: undefined in LLVM; WGSL rejects a constant index past a fixed-size array.
            return "0u".into();
        }
        if reg.atomic { format!("atomicLoad(&{}[{}])", reg.name, w.expr) } else { format!("{}[{}]", reg.name, w.expr) }
    }

    /// Replace the bits of `mask` in word `w` by those of `value` (which has no bits outside `mask`). `full` is a
    /// static promise that the mask is all ones.
    fn write_word(&self, c: &mut dyn Code, r: RegionId, w: &WordIx, mask: &str, value: &str, full: bool) {
        let reg = &self.mem.regions[r];
        if r == NULL || self.fixed_out_of_range(r, w) {
            return;
        }
        let at = format!("{}[{}]", reg.name, w.expr);
        match (reg.atomic, full) {
            (true, true) => c.line(format!("atomicStore(&{}, {});", at, value)),
            (true, false) => {
                c.line(format!("atomicAnd(&{}, ~({}));", at, mask));
                c.line(format!("atomicOr(&{}, {});", at, value));
            }
            (false, true) => c.line(format!("{} = {};", at, value)),
            (false, false) => c.line(format!("{} = ({} & ~({})) | ({});", at, at, mask, value)),
        }
    }

    // Small values: groups of words -----------------------------------------------------------------------------------

    /// The value chunks (bytes 4c..4c+3 of the value, as u32) of `size` bytes at `p` in region `r`.
    fn read_chunks(&self, c: &mut dyn Code, r: RegionId, p: &Ptr, size: u64) -> Vec<String> {
        let nchunks = size.div_ceil(4) as usize;
        let mut chunks = Vec::with_capacity(nchunks);
        match p.mod4 {
            Some(k) => {
                let base = WordIx::of(p);
                let k = k as u64;
                let nwords = (k + size).div_ceil(4);
                let mut words = Vec::new();
                for i in 0..nwords {
                    let w = c.fresh();
                    c.line(format!("let {} = {};", w, self.read_word(r, &base.plus(i))));
                    words.push(w);
                }
                for ch in 0..nchunks as u64 {
                    if k == 0 {
                        chunks.push(words[ch as usize].clone());
                    } else {
                        let last = (4 * ch + 3).min(size - 1);
                        let lo = format!("({} >> {}u)", words[ch as usize], 8 * k);
                        if (k + last) / 4 > ch {
                            chunks.push(format!("({} | ({} << {}u))", lo, words[ch as usize + 1], 32 - 8 * k));
                        } else {
                            chunks.push(lo);
                        }
                    }
                }
            }
            None => {
                for ch in 0..nchunks as u64 {
                    let q = p.plus(4 * ch);
                    let (o, w, sh, lo, hi) = (c.fresh(), c.fresh(), c.fresh(), c.fresh(), c.fresh());
                    c.line(format!("let {} = {};", o, q.off));
                    c.line(format!("let {} = {} >> 2u;", w, o));
                    c.line(format!("let {} = ({} & 3u) * 8u;", sh, o));
                    let wi = WordIx { expr: w.clone(), konst: None };
                    let wi1 = WordIx { expr: format!("({} + 1u)", w), konst: None };
                    c.line(format!("let {} = {};", lo, self.read_word(r, &wi)));
                    let last = (4 * ch + 3).min(size - 1) - 4 * ch;
                    if last == 0 {
                        // One byte never spans two words.
                        chunks.push(format!("({} >> {})", lo, sh));
                        continue;
                    }
                    c.line(format!("let {} = {};", hi, self.read_word(r, &wi1)));
                    chunks
                        .push(format!("select(({} >> {}) | ({} << (32u - {})), {}, {} == 0u)", lo, sh, hi, sh, lo, sh));
                }
            }
        }
        chunks
    }

    /// Write value chunks with their byte masks (bit i of `bytes[c]` set when byte i of chunk c is the value's).
    fn write_chunks(&self, c: &mut dyn Code, r: RegionId, p: &Ptr, chunks: &[String], bytes: &[u8]) {
        let size = 4 * chunks.len() as u64;
        match p.mod4 {
            Some(k) => {
                let base = WordIx::of(p);
                let k = k as u64;
                let nwords = (k + size).div_ceil(4);
                for i in 0..nwords {
                    // Memory word i holds value bytes j with (k + j) / 4 == i.
                    let mut parts = Vec::new();
                    let mut mask: u32 = 0;
                    for (ch, chunk) in chunks.iter().enumerate() {
                        let ch = ch as u64;
                        for b in 0..4u64 {
                            let j = 4 * ch + b;
                            if bytes[ch as usize] & (1 << b) == 0 || (k + j) / 4 != i {
                                continue;
                            }
                            mask |= 0xff << (8 * ((k + j) % 4));
                        }
                        // The part of chunk ch that lands in word i.
                        if k == 0 && ch == i {
                            parts.push(chunk.clone());
                        } else if k != 0 && ch == i {
                            parts.push(format!("({} << {}u)", chunk, 8 * k));
                        } else if k != 0 && ch + 1 == i {
                            parts.push(format!("({} >> {}u)", chunk, 32 - 8 * k));
                        }
                    }
                    if mask == 0 {
                        continue;
                    }
                    let value = format!("(({}) & {}u)", parts.join(" | "), mask);
                    let full = mask == u32::MAX;
                    let value = if full { format!("({})", parts.join(" | ")) } else { value };
                    self.write_word(c, r, &base.plus(i), &format!("{}u", mask), &value, full);
                }
            }
            None => {
                for (ch, chunk) in chunks.iter().enumerate() {
                    let m = bytes[ch];
                    if m == 0 {
                        continue;
                    }
                    let mut bm: u32 = 0;
                    for b in 0..4 {
                        if m & (1 << b) != 0 {
                            bm |= 0xff << (8 * b);
                        }
                    }
                    let q = p.plus(4 * ch as u64);
                    let (o, w, sh, v) = (c.fresh(), c.fresh(), c.fresh(), c.fresh());
                    c.line(format!("let {} = {};", o, q.off));
                    c.line(format!("let {} = {} >> 2u;", w, o));
                    c.line(format!("let {} = ({} & 3u) * 8u;", sh, o));
                    c.line(format!("let {} = ({}) & {}u;", v, chunk, bm));
                    let wi = WordIx { expr: w.clone(), konst: None };
                    let wi1 = WordIx { expr: format!("({} + 1u)", w), konst: None };
                    self.write_word(c, r, &wi, &format!("({}u << {})", bm, sh), &format!("({} << {})", v, sh), false);
                    let (mh, vh) = (c.fresh(), c.fresh());
                    c.line(format!("let {} = select({}u >> (32u - {}), 0u, {} == 0u);", mh, bm, sh, sh));
                    c.line(format!("let {} = select({} >> (32u - {}), 0u, {} == 0u);", vh, v, sh, sh));
                    c.open(format!("if {} != 0u {{", mh));
                    self.write_word(c, r, &wi1, &mh, &vh, false);
                    c.close();
                }
            }
        }
    }

    /// Extract leaf bits from value chunks: `bytes` bytes at value byte `off`, as u32 (bytes <= 4).
    fn extract(chunks: &[String], off: u64, bytes: u32) -> String {
        let i = (off / 4) as usize;
        let s = (off % 4) * 8;
        let span = s + 8 * bytes as u64;
        let raw = if s == 0 {
            chunks[i].clone()
        } else if span <= 32 || i + 1 >= chunks.len() {
            format!("({} >> {}u)", chunks[i], s)
        } else {
            format!("(({} >> {}u) | ({} << {}u))", chunks[i], s, chunks[i + 1], 32 - s)
        };
        if bytes == 4 { raw } else { format!("({} & {}u)", raw, (1u64 << (8 * bytes)) - 1) }
    }

    fn build(&mut self, t: TypeId, leaves: &mut std::slice::Iter<'_, (Leaf, String)>) -> Result<String, Refusal> {
        let shape = self.types.shape(self.tt, t)?;
        Ok(match shape {
            Shape::Bool | Shape::Word { .. } | Shape::Wide => leaves.next().map(|(_, e)| e.clone()).unwrap_or_default(),
            Shape::Vector { len, .. } => {
                let c = self.types.carrier(self.tt, t)?;
                let elem = match self.tt.get(t) {
                    Type::Vector { elem, .. } => *elem,
                    _ => unreachable!(),
                };
                let parts: Result<Vec<String>, Refusal> = (0..len).map(|_| self.build(elem, leaves)).collect();
                format!("{}({})", c, parts?.join(", "))
            }
            Shape::Array { elem_ty, len, .. } => {
                let c = self.types.carrier(self.tt, t)?;
                let parts: Result<Vec<String>, Refusal> = (0..len).map(|_| self.build(elem_ty, leaves)).collect();
                format!("{}({})", c, parts?.join(", "))
            }
            Shape::Struct { members, .. } => {
                let c = self.types.carrier(self.tt, t)?;
                let parts: Result<Vec<String>, Refusal> = members.iter().map(|m| self.build(*m, leaves)).collect();
                format!("{}({})", c, parts?.join(", "))
            }
            Shape::Ptr(_) | Shape::Void => return Err(mem_refusal("a pointer")),
        })
    }

    fn load_small(&mut self, c: &mut dyn Code, r: RegionId, p: &Ptr, t: TypeId) -> Result<String, Refusal> {
        let mut leaves = Vec::new();
        self.leaves(t, 0, "", &mut leaves)?;
        let size = self.dl.store_size(self.tt, t).unwrap_or(0);
        if size == 0 {
            return self.types.zero(self.tt, t);
        }
        let chunks = self.read_chunks(c, r, p, size);
        let mut vals = Vec::new();
        for l in &leaves {
            let e = match l.kind {
                LeafKind::Bool => format!("(({} & 1u) != 0u)", Self::extract(&chunks, l.off, 1)),
                LeafKind::Word => Self::extract(&chunks, l.off, l.bytes),
                LeafKind::Wide => {
                    format!("vec2<u32>({}, {})", Self::extract(&chunks, l.off, 4), Self::extract(&chunks, l.off + 4, 4))
                }
            };
            vals.push((l.clone(), e));
        }
        let mut it = vals.iter();
        let v = self.build(t, &mut it)?;
        let name = c.fresh();
        c.line(format!("let {} = {};", name, v));
        Ok(name)
    }

    fn store_small(&mut self, c: &mut dyn Code, r: RegionId, p: &Ptr, t: TypeId, value: &str) -> Result<(), Refusal> {
        let mut leaves = Vec::new();
        self.leaves(t, 0, "", &mut leaves)?;
        let size = self.dl.store_size(self.tt, t).unwrap_or(0);
        if size == 0 {
            return Ok(());
        }
        let nchunks = size.div_ceil(4) as usize;
        let mut parts: Vec<Vec<String>> = vec![Vec::new(); nchunks];
        let mut bytes = vec![0u8; nchunks];
        let v = c.fresh();
        c.line(format!("let {} = {};", v, value));
        for l in &leaves {
            let words: Vec<(u64, String, u32)> = match l.kind {
                LeafKind::Bool => vec![(l.off, format!("select(0u, 1u, {}{})", v, l.path), 1)],
                LeafKind::Word => vec![(l.off, format!("{}{}", v, l.path), l.bytes)],
                LeafKind::Wide => {
                    vec![(l.off, format!("{}{}.x", v, l.path), 4), (l.off + 4, format!("{}{}.y", v, l.path), 4)]
                }
            };
            for (off, e, n) in words {
                let i = (off / 4) as usize;
                let s = (off % 4) * 8;
                parts[i].push(if s == 0 { e.clone() } else { format!("({} << {}u)", e, s) });
                for b in 0..n as u64 {
                    let j = off + b;
                    bytes[(j / 4) as usize] |= 1 << (j % 4);
                }
                if s + 8 * n as u64 > 32 {
                    parts[i + 1].push(format!("({} >> {}u)", e, 32 - s));
                }
            }
        }
        let chunks: Vec<String> = parts
            .iter()
            .map(|p| if p.is_empty() { "0u".to_string() } else { format!("({})", p.join(" | ")) })
            .collect();
        self.write_chunks(c, r, p, &chunks, &bytes);
        Ok(())
    }

    // Values of any size ----------------------------------------------------------------------------------------------

    fn load_in(&mut self, c: &mut dyn Code, r: RegionId, p: &Ptr, t: TypeId) -> Result<String, Refusal> {
        if self.is_small(t)? {
            return self.load_small(c, r, p, t);
        }
        match self.types.shape(self.tt, t)? {
            Shape::Struct { members, .. } => {
                let l = self.dl.struct_layout(self.tt, t).ok_or_else(|| mem_refusal("an unsized struct"))?;
                let mut parts = Vec::new();
                for (i, m) in members.iter().enumerate() {
                    parts.push(self.load_in(c, r, &p.plus(l.offsets[i]), *m)?);
                }
                let ct = self.types.carrier(self.tt, t)?;
                let n = c.fresh();
                c.line(format!("let {} = {}({});", n, ct, parts.join(", ")));
                Ok(n)
            }
            Shape::Array { elem_ty, len, is_vector, .. } => {
                let stride = if is_vector {
                    self.dl.size_bits(self.tt, elem_ty).unwrap_or(0) / 8
                } else {
                    self.dl.alloc_size(self.tt, elem_ty).unwrap_or(0)
                };
                let ct = self.types.carrier(self.tt, t)?;
                let (n, i) = (c.fresh(), c.fresh());
                c.line(format!("var {}: {};", n, ct));
                c.open(format!("for (var {}: u32 = 0u; {} < {}u; {}++) {{", i, i, len, i));
                let e = self.load_in(c, r, &p.plus_scaled(&i, stride), elem_ty)?;
                c.line(format!("{}[{}] = {};", n, i, e));
                c.close();
                Ok(n)
            }
            _ => self.load_small(c, r, p, t),
        }
    }

    fn store_in(&mut self, c: &mut dyn Code, r: RegionId, p: &Ptr, t: TypeId, value: &str) -> Result<(), Refusal> {
        if self.is_small(t)? {
            return self.store_small(c, r, p, t, value);
        }
        match self.types.shape(self.tt, t)? {
            Shape::Struct { members, .. } => {
                let l = self.dl.struct_layout(self.tt, t).ok_or_else(|| mem_refusal("an unsized struct"))?;
                for (i, m) in members.iter().enumerate() {
                    self.store_in(c, r, &p.plus(l.offsets[i]), *m, &format!("{}.m{}", value, i))?;
                }
                Ok(())
            }
            Shape::Array { elem_ty, len, is_vector, .. } => {
                let stride = if is_vector {
                    self.dl.size_bits(self.tt, elem_ty).unwrap_or(0) / 8
                } else {
                    self.dl.alloc_size(self.tt, elem_ty).unwrap_or(0)
                };
                let v = c.fresh();
                c.line(format!("let {} = {};", v, value));
                let i = c.fresh();
                c.open(format!("for (var {}: u32 = 0u; {} < {}u; {}++) {{", i, i, len, i));
                self.store_in(c, r, &p.plus_scaled(&i, stride), elem_ty, &format!("{}[{}]", v, i))?;
                c.close();
                Ok(())
            }
            _ => self.store_small(c, r, p, t, value),
        }
    }

    /// Run `f` for each candidate region of `p`: directly for one, under a test of the tag for several.
    fn per_region(
        &mut self,
        c: &mut dyn Code,
        p: &Ptr,
        mut f: impl FnMut(&mut Self, &mut dyn Code, RegionId) -> Result<(), Refusal>,
    ) -> Result<(), Refusal> {
        if p.regions.len() == 1 {
            return f(self, c, p.regions[0]);
        }
        let tag = p.tag.clone().expect("a pointer into several regions without its tag");
        for &r in &p.regions {
            if r == NULL {
                continue;
            }
            c.open(format!("if {} == {}u {{", tag, self.mem.regions[r].tag));
            f(self, c, r)?;
            c.close();
        }
        Ok(())
    }

    /// Load a value of type `t` through `p` (alignment `align`); returns the WGSL expression of the value.
    pub fn load(&mut self, c: &mut dyn Code, p: &Ptr, t: TypeId, align: u64) -> Result<String, Refusal> {
        let p = p.with_align(align);
        if p.regions.len() == 1 {
            return self.load_in(c, p.regions[0], &p, t);
        }
        let ct = self.types.carrier(self.tt, t)?;
        let res = c.fresh();
        c.line(format!("var {}: {};", res, ct));
        let pp = p.clone();
        self.per_region(c, &p, |a, c, r| {
            let v = a.load_in(c, r, &pp, t)?;
            c.line(format!("{} = {};", res, v));
            Ok(())
        })?;
        Ok(res)
    }

    /// Store `value` (a WGSL expression of type `t`'s carrier) through `p`.
    pub fn store(&mut self, c: &mut dyn Code, p: &Ptr, t: TypeId, align: u64, value: &str) -> Result<(), Refusal> {
        let p = p.with_align(align);
        let v = c.fresh();
        c.line(format!("let {} = {};", v, value));
        let pp = p.clone();
        self.per_region(c, &p, |a, c, r| a.store_in(c, r, &pp, t, &v))
    }

    /// Copy `len` bytes (a constant) from `src` to `dst` (each with its alignment applied, [`Ptr::with_align`]). With
    /// `overlap`, the source is read completely before the
    /// destination is written when the length is small, and otherwise copied in the direction that preserves it.
    pub fn copy(&mut self, c: &mut dyn Code, dst: &Ptr, src: &Ptr, len: u64, overlap: bool) -> Result<(), Refusal> {
        if len == 0 {
            return Ok(());
        }
        let (dst, src) = (dst.clone(), src.clone());
        // Moved in groups of up to 64 bytes: a group is read completely before it is written.
        const GROUP: u64 = 64;
        let groups = len.div_ceil(GROUP);
        let group_chunks = |a: &mut Self, c: &mut dyn Code, s: &Ptr, n: u64| -> Result<Vec<String>, Refusal> {
            let res: Vec<String> = (0..n.div_ceil(4)).map(|_| c.fresh()).collect();
            for r in &res {
                c.line(format!("var {}: u32;", r));
            }
            let rs = res.clone();
            let sp = s.clone();
            a.per_region(c, s, |a, c, reg| {
                let ch = a.read_chunks(c, reg, &sp, n);
                for (d, e) in rs.iter().zip(ch) {
                    c.line(format!("{} = {};", d, e));
                }
                Ok(())
            })?;
            Ok(res)
        };
        let put = |a: &mut Self, c: &mut dyn Code, d: &Ptr, chunks: &[String], n: u64| -> Result<(), Refusal> {
            let mut bytes = vec![0xfu8; chunks.len()];
            let tail = n % 4;
            if tail != 0 {
                *bytes.last_mut().unwrap() = (1u8 << tail) - 1;
            }
            let dp = d.clone();
            a.per_region(c, d, |a, c, reg| {
                a.write_chunks(c, reg, &dp, chunks, &bytes);
                Ok(())
            })
        };
        if groups == 1 {
            let ch = group_chunks(self, c, &src, len)?;
            return put(self, c, &dst, &ch, len);
        }
        // Whole groups in a loop, forward or (for an overlapping copy to a higher address) backward, then the tail.
        let full = len / GROUP;
        let tail = len % GROUP;
        let body = |a: &mut Self, c: &mut dyn Code, g: &str| -> Result<(), Refusal> {
            let s = src.plus_scaled(g, GROUP);
            let d = dst.plus_scaled(g, GROUP);
            let ch = group_chunks(a, c, &s, GROUP)?;
            put(a, c, &d, &ch, GROUP)
        };
        let tail_copy = |a: &mut Self, c: &mut dyn Code| -> Result<(), Refusal> {
            if tail == 0 {
                return Ok(());
            }
            let s = src.plus(full * GROUP);
            let d = dst.plus(full * GROUP);
            let ch = group_chunks(a, c, &s, tail)?;
            put(a, c, &d, &ch, tail)
        };
        let forward = |a: &mut Self, c: &mut dyn Code| -> Result<(), Refusal> {
            let g = c.fresh();
            c.open(format!("for (var {}: u32 = 0u; {} < {}u; {}++) {{", g, g, full, g));
            body(a, c, &g)?;
            c.close();
            tail_copy(a, c)
        };
        if !overlap {
            return forward(self, c);
        }
        let backward = |a: &mut Self, c: &mut dyn Code| -> Result<(), Refusal> {
            tail_copy(a, c)?;
            let g = c.fresh();
            c.open(format!("for (var {}: u32 = {}u; {} > 0u; {}--) {{", g, full, g, g));
            body(a, c, &format!("({} - 1u)", g))?;
            c.close();
            Ok(())
        };
        // Overlap is possible only within one region; a destination above the source copies backward.
        let dtag = dst.tag.clone().unwrap_or_else(|| format!("{}u", self.mem.regions[dst.regions[0]].tag));
        let stag = src.tag.clone().unwrap_or_else(|| format!("{}u", self.mem.regions[src.regions[0]].tag));
        c.open(format!("if {} == {} && {} > {} {{", dtag, stag, dst.off, src.off));
        backward(self, c)?;
        c.close();
        c.open("else {".into());
        forward(self, c)?;
        c.close();
        Ok(())
    }

    /// Set `len` bytes (a constant) at `dst` to the byte `value` (a u32 expression holding it).
    pub fn fill(&mut self, c: &mut dyn Code, dst: &Ptr, value: &str, len: u64, align: u64) -> Result<(), Refusal> {
        if len == 0 {
            return Ok(());
        }
        let dst = dst.with_align(align);
        let word = c.fresh();
        c.line(format!("let {} = ({} & 0xffu) * 0x01010101u;", word, value));
        const GROUP: u64 = 64;
        let put = |a: &mut Self, c: &mut dyn Code, d: &Ptr, n: u64| -> Result<(), Refusal> {
            let nch = n.div_ceil(4) as usize;
            let chunks = vec![word.clone(); nch];
            let mut bytes = vec![0xfu8; nch];
            if !n.is_multiple_of(4) {
                bytes[nch - 1] = (1u8 << (n % 4)) - 1;
            }
            let dp = d.clone();
            a.per_region(c, d, |a, c, reg| {
                a.write_chunks(c, reg, &dp, &chunks, &bytes);
                Ok(())
            })
        };
        if len <= GROUP {
            return put(self, c, &dst, len);
        }
        let full = len / GROUP;
        let g = c.fresh();
        c.open(format!("for (var {}: u32 = 0u; {} < {}u; {}++) {{", g, g, full, g));
        put(self, c, &dst.plus_scaled(&g, GROUP), GROUP)?;
        c.close();
        if !len.is_multiple_of(GROUP) {
            put(self, c, &dst.plus(full * GROUP), len % GROUP)?;
        }
        Ok(())
    }

    /// Copy or fill with a length known only at run time: one byte at a time, in the direction an overlapping copy
    /// needs. `src` `None` fills with `value`.
    pub fn bytes_dynamic(
        &mut self,
        c: &mut dyn Code,
        dst: &Ptr,
        src: Option<&Ptr>,
        value: &str,
        len: &str,
        overlap: bool,
    ) -> Result<(), Refusal> {
        let i = c.fresh();
        let byte_at = |a: &mut Self, c: &mut dyn Code, idx: &str| -> Result<(), Refusal> {
            let d = dst.plus_scaled(idx, 1);
            let b = c.fresh();
            c.line(format!("var {}: u32;", b));
            match src {
                Some(s) => {
                    let sp = s.plus_scaled(idx, 1);
                    let spp = sp.clone();
                    let bb = b.clone();
                    a.per_region(c, &sp, |a, c, r| {
                        let ch = a.read_chunks(c, r, &spp, 1);
                        c.line(format!("{} = {} & 0xffu;", bb, ch[0]));
                        Ok(())
                    })?;
                }
                None => c.line(format!("{} = {} & 0xffu;", b, value)),
            }
            let dp = d.clone();
            a.per_region(c, &d, |a, c, r| {
                a.write_chunks(c, r, &dp, std::slice::from_ref(&b), &[1]);
                Ok(())
            })
        };
        if overlap && let Some(s) = src {
            let dtag = dst.tag.clone().unwrap_or_else(|| format!("{}u", self.mem.regions[dst.regions[0]].tag));
            let stag = s.tag.clone().unwrap_or_else(|| format!("{}u", self.mem.regions[s.regions[0]].tag));
            c.open(format!("if {} == {} && {} > {} {{", dtag, stag, dst.off, s.off));
            c.open(format!("for (var {}: u32 = {}; {} > 0u; {}--) {{", i, len, i, i));
            byte_at(self, c, &format!("({} - 1u)", i))?;
            c.close();
            c.close();
            c.open("else {".into());
        }
        let j = c.fresh();
        c.open(format!("for (var {}: u32 = 0u; {} < {}; {}++) {{", j, j, len, j));
        byte_at(self, c, &j)?;
        c.close();
        if overlap && src.is_some() {
            c.close();
        }
        Ok(())
    }
}

/// A word index: an expression, and its value when constant.
#[derive(Clone, Debug)]
struct WordIx {
    expr: String,
    konst: Option<u64>,
}

impl WordIx {
    fn of(p: &Ptr) -> WordIx {
        match p.konst {
            Some(k) => WordIx { expr: format!("{}u", (k >> 2) as u32), konst: Some(k >> 2) },
            None => WordIx { expr: format!("({} >> 2u)", p.off), konst: None },
        }
    }

    fn plus(&self, i: u64) -> WordIx {
        if i == 0 {
            return self.clone();
        }
        match self.konst {
            Some(k) => WordIx { expr: format!("{}u", (k + i) as u32), konst: Some(k + i) },
            None => WordIx { expr: format!("({} + {}u)", self.expr, i), konst: None },
        }
    }
}

/// Which bytes of each word of a store of `t` at an offset with residue `mod4` the store covers only partly; the
/// memory model makes a shared region atomic when any store into it is partial.
pub fn store_is_partial(tt: &TypeTable, dl: &DataLayout, t: TypeId, mod4: Option<u32>) -> bool {
    let size = dl.store_size(tt, t).unwrap_or(0);
    if mod4 != Some(0) {
        return true;
    }
    if !size.is_multiple_of(4) {
        return true;
    }
    // Padding inside the value leaves words partly written.
    let mut covered = vec![false; size as usize];
    fn mark(tt: &TypeTable, dl: &DataLayout, t: TypeId, off: u64, covered: &mut [bool]) {
        match tt.get(t) {
            Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                if let Some(l) = dl.struct_layout(tt, t) {
                    for (i, e) in elems.iter().enumerate() {
                        mark(tt, dl, *e, off + l.offsets[i], covered);
                    }
                }
            }
            Type::Array { len, elem } => {
                let es = dl.alloc_size(tt, *elem).unwrap_or(0);
                for i in 0..*len {
                    mark(tt, dl, *elem, off + i * es, covered);
                }
            }
            _ => {
                let n = dl.store_size(tt, t).unwrap_or(0);
                for b in off..off + n {
                    if (b as usize) < covered.len() {
                        covered[b as usize] = true;
                    }
                }
            }
        }
    }
    mark(tt, dl, t, 0, &mut covered);
    covered.iter().any(|c| !c)
}
