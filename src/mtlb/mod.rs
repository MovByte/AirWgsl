//! Metal library containers: metallibs (`MTLB`) and the files that carry them.
//!
//! The layouts are Apple's and undocumented; every field here was measured on the metallibs of iOS 17.7.11 (21H461)
//! and the Metal toolchain on macOS, and is checked against `metal-readobj` (see `examples/mtlb_check.rs`).
//!
//! A metallib starts with an 88-byte header (little-endian): `MTLB`; `u16` file major version with bit 15 set for a
//! macOS target; `u16` minor; `u16` bug-fix; `u8` file type with bit 7 marking a stub library (the descriptor of
//! precompiled GPU code, with no bitcode); `u8` platform with bit 7 marking a 64-bit target; `u16` platform major
//! version; `u8` minor; `u8` update; then `u64` pairs (offset, size) relative to the metallib's start: file size,
//! function list (the size excludes the list's leading `u32` count), public metadata, private metadata, and the
//! module list (the bitcode).
//!
//! The function list holds a `u32` count and one entry per function: a `u32` entry size (counting itself), then tags
//! (four bytes, a `u16` length, the payload) up to `ENDT`. After the entries comes the header extension: more tags up
//! to `ENDT`, naming the optional sections (`UUID`, and (offset, size) pairs: `RLST` reflection list, `SLST` script
//! list, `HDYN` dynamic header).
//!
//! The public and private metadata are sequences of records, one per function, each a `u32` size and tags up to
//! `ENDT`. The size counts itself in libraries with a non-empty header extension; in libraries whose extension is
//! only `ENDT` (an older writer, which also orders `MDSZ` before `OFFT` and writes no `RFLT`) it counts only the tags.
//! The reflection and script lists start with a `u32` record count; their records' sizes count themselves, and
//! their payload tags (`RBUF`, `SBUF`) have a `u32` length. The dynamic header is one tag list (`NAME`, the install
//! name) up to `ENDT`, with no size word.

mod sha256;

use std::fmt;

pub use sha256::sha256;

/// Why a container could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Byte offset in the input, when tied to one.
    pub offset: Option<usize>,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.offset {
            Some(o) => write!(f, "metallib error at {:#x}: {}", o, self.message),
            None => write!(f, "metallib error: {}", self.message),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

fn err<T>(offset: usize, message: impl Into<String>) -> Result<T> {
    Err(Error { offset: Some(offset), message: message.into() })
}

fn u16_at(d: &[u8], o: usize) -> Option<u16> {
    d.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    d.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn u64_at(d: &[u8], o: usize) -> Option<u64> {
    d.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
}

/// The magic of a metallib.
pub const MTLB_MAGIC: &[u8; 4] = b"MTLB";
/// The module offset of a function entry that names no module (stub libraries).
pub const NO_MODULE: u64 = u64::MAX;
const HEADER_SIZE: usize = 88;

/// A (offset, size) section of a metallib, relative to its start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Range {
    pub offset: u64,
    pub size: u64,
}

/// The metallib header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub macos_target: bool,
    pub file_major: u16,
    pub file_minor: u16,
    pub file_bugfix: u16,
    /// `METALLIB_EXECUTE` 0, `METALLIB_CI` 1, `METALLIB_DYLIB` 2 (as `metal-readobj` names them).
    pub file_type: u8,
    pub stub: bool,
    pub arch64: bool,
    /// `METALLIB_PLATFORM_*`: iOS is 2.
    pub platform: u8,
    pub platform_major: u16,
    pub platform_minor: u8,
    pub platform_update: u8,
    pub file_size: u64,
    pub function_list: Range,
    pub public_metadata: Range,
    pub private_metadata: Range,
    pub module_list: Range,
}

/// One tag of a tag list: four-character name and payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tag<'a> {
    pub name: [u8; 4],
    pub data: &'a [u8],
}

/// The `TESS` tag of a post-tessellation vertex function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tessellation {
    /// `METALLIB_TRIANGLE_PATCH_TYPE` 1, `METALLIB_QUAD_PATCH_TYPE` 2.
    pub patch_type: u8,
    pub control_points: u8,
}

/// The `SPCI` tag of a specialized function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Specialization {
    /// The payload's first byte; 0 in every measured library.
    pub kind: u8,
    pub unspecialized_hash: [u8; 32],
    /// Offset of the specialization script in the script list.
    pub script_offset: u64,
}

/// One function-list entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function<'a> {
    pub name: String,
    /// For an `ALIS` entry: the function it aliases. The tag holds the alias's own name (`name`), then this name;
    /// `metal-readobj` prints them as `Alias` and `Name`.
    pub alias_of: Option<String>,
    /// `METALLIB_*_FUNCTION`: vertex 0, fragment 1, compute 2 (others as stored).
    pub function_type: u8,
    pub hash: [u8; 32],
    pub public_metadata_offset: u64,
    pub private_metadata_offset: u64,
    /// [`NO_MODULE`] in a stub library.
    pub module_offset: u64,
    /// The module (wrapper header and bitcode), absent in a stub library.
    pub module: Option<&'a [u8]>,
    pub air_version: (u16, u16),
    pub metal_version: (u16, u16),
    /// The `RFLT` tag: the function's record in the reflection list. Stub libraries carry it as 0 with no list.
    pub reflection_offset: Option<u64>,
    pub tessellation: Option<Tessellation>,
    pub specialization: Option<Specialization>,
    /// The `LAYR` tag: the data type of the render target array index output (`METALLIB_UINT_DATA_TYPE` 0x21,
    /// `METALLIB_UCHAR_DATA_TYPE` 0x31, as `metal-readobj` names them).
    pub render_target_array_index_type: Option<u8>,
    /// The entry's size in the function list, including its size word.
    pub entry_size: u32,
    /// Every tag of the entry, in order.
    pub tags: Vec<Tag<'a>>,
}

/// A metallib.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metallib<'a> {
    /// The metallib's bytes (exactly `header.file_size` of them).
    pub data: &'a [u8],
    pub header: Header,
    pub functions: Vec<Function<'a>>,
    /// The header extension's tags, in order.
    pub extension: Vec<Tag<'a>>,
    pub uuid: Option<[u8; 16]>,
    pub reflection_list: Option<Range>,
    pub script_list: Option<Range>,
    pub dynamic_header: Option<Range>,
}

fn read_tags<'a>(d: &'a [u8], mut p: usize, end: usize, what: &str) -> Result<(Vec<Tag<'a>>, usize)> {
    let mut tags = Vec::new();
    loop {
        if p + 4 > end {
            return err(p, format!("{} runs past its end without ENDT", what));
        }
        let name: [u8; 4] = d[p..p + 4].try_into().unwrap();
        p += 4;
        if &name == b"ENDT" {
            return Ok((tags, p));
        }
        let len = match u16_at(d, p) {
            Some(l) if p + 2 <= end => l as usize,
            _ => return err(p, format!("{} tag {} is truncated", what, String::from_utf8_lossy(&name))),
        };
        p += 2;
        if p + len > end {
            return err(p, format!("{} tag {} runs past its end", what, String::from_utf8_lossy(&name)));
        }
        tags.push(Tag { name, data: &d[p..p + len] });
        p += len;
    }
}

fn cstr(b: &[u8], what: &str, at: usize) -> Result<String> {
    if b.last() != Some(&0) || b[..b.len() - 1].contains(&0) {
        return err(at, format!("malformed {}", what));
    }
    match std::str::from_utf8(&b[..b.len() - 1]) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => err(at, format!("{} is not UTF-8", what)),
    }
}

/// Reads a metallib at the start of `data`; bytes past its file size are ignored (the caller checks them).
pub fn parse_metallib(data: &[u8]) -> Result<Metallib<'_>> {
    if data.len() < HEADER_SIZE {
        return err(0, "truncated metallib header");
    }
    if &data[..4] != MTLB_MAGIC {
        return err(0, "not a metallib (no MTLB magic)");
    }
    let major_raw = u16_at(data, 4).unwrap();
    let ftype = data[10];
    let platform = data[11];
    let r = |o: usize| Range { offset: u64_at(data, o).unwrap(), size: u64_at(data, o + 8).unwrap() };
    let header = Header {
        macos_target: major_raw >> 15 == 1,
        file_major: major_raw & 0x7fff,
        file_minor: u16_at(data, 6).unwrap(),
        file_bugfix: u16_at(data, 8).unwrap(),
        file_type: ftype & 0x7f,
        stub: ftype & 0x80 != 0,
        arch64: platform & 0x80 != 0,
        platform: platform & 0x7f,
        platform_major: u16_at(data, 12).unwrap(),
        platform_minor: data[14],
        platform_update: data[15],
        file_size: u64_at(data, 16).unwrap(),
        function_list: r(24),
        public_metadata: r(40),
        private_metadata: r(56),
        module_list: r(72),
    };
    let size = header.file_size;
    if size < HEADER_SIZE as u64 || size > data.len() as u64 {
        return err(16, format!("file size {:#x} does not fit the {} bytes given", size, data.len()));
    }
    let data = &data[..size as usize];
    let fl = header.function_list;
    for (name, rg, extra) in [
        ("function list", fl, 4u64),
        ("public metadata", header.public_metadata, 0),
        ("private metadata", header.private_metadata, 0),
        ("module list", header.module_list, 0),
    ] {
        let end = rg.offset.checked_add(rg.size).and_then(|e| e.checked_add(extra));
        if rg.offset < HEADER_SIZE as u64 || end.map(|e| e > size).unwrap_or(true) {
            return err(24, format!("{} [{:#x}, +{:#x}) outside the metallib", name, rg.offset, rg.size));
        }
    }
    let count = u32_at(data, fl.offset as usize).unwrap() as usize;
    let mut p = fl.offset as usize + 4;
    let list_end = p + fl.size as usize;
    let mut functions = Vec::with_capacity(count.min(1 << 16));
    let modules = header.module_list;
    for index in 0..count {
        let entry_size = match u32_at(data, p) {
            Some(s) if p + 4 <= list_end => s as usize,
            _ => return err(p, format!("function {} runs past the function list", index)),
        };
        if entry_size < 4 || p + entry_size > list_end {
            return err(p, format!("function {} entry size {} does not fit the function list", index, entry_size));
        }
        let (tags, q) = read_tags(data, p + 4, p + entry_size, "function entry")?;
        if q != p + entry_size {
            return err(p, format!("function {}: entry size {}, tags end at {}", index, entry_size, q - p));
        }
        let mut f = function_entry(data, p, tags, &header, modules)?;
        f.entry_size = entry_size as u32;
        functions.push(f);
        p += entry_size;
    }
    if p != list_end {
        return err(
            p,
            format!("function entries span {:#x} bytes, the header says {:#x}", p - fl.offset as usize - 4, fl.size),
        );
    }
    let (extension, q) = read_tags(data, p, header.public_metadata.offset as usize, "header extension")?;
    if q != header.public_metadata.offset as usize {
        return err(q, "bytes between the header extension and the public metadata");
    }
    let mut lib = Metallib {
        data,
        header,
        functions,
        extension: Vec::new(),
        uuid: None,
        reflection_list: None,
        script_list: None,
        dynamic_header: None,
    };
    for t in &extension {
        let range = |t: &Tag| -> Result<Range> {
            if t.data.len() != 16 {
                return err(p, format!("{} tag of {} bytes", String::from_utf8_lossy(&t.name), t.data.len()));
            }
            let rg = Range { offset: u64_at(t.data, 0).unwrap(), size: u64_at(t.data, 8).unwrap() };
            if rg.offset.checked_add(rg.size).map(|e| e > size).unwrap_or(true) || rg.offset < HEADER_SIZE as u64 {
                return err(p, format!("{} section outside the metallib", String::from_utf8_lossy(&t.name)));
            }
            Ok(rg)
        };
        let dup = |o: &Option<Range>| o.is_some();
        match &t.name {
            b"UUID" => {
                if t.data.len() != 16 || lib.uuid.is_some() {
                    return err(p, "malformed UUID tag");
                }
                lib.uuid = Some(t.data.try_into().unwrap());
            }
            b"RLST" => {
                if dup(&lib.reflection_list) {
                    return err(p, "two RLST tags");
                }
                lib.reflection_list = Some(range(t)?);
            }
            b"SLST" => {
                if dup(&lib.script_list) {
                    return err(p, "two SLST tags");
                }
                lib.script_list = Some(range(t)?);
            }
            b"HDYN" => {
                if dup(&lib.dynamic_header) {
                    return err(p, "two HDYN tags");
                }
                lib.dynamic_header = Some(range(t)?);
            }
            other => return err(p, format!("unknown header extension tag {}", String::from_utf8_lossy(other))),
        }
    }
    lib.extension = extension;
    check_modules(&lib)?;
    check_records(&lib)?;
    Ok(lib)
}

fn function_entry<'a>(
    data: &'a [u8],
    at: usize,
    tags: Vec<Tag<'a>>,
    header: &Header,
    modules: Range,
) -> Result<Function<'a>> {
    let mut f = Function {
        name: String::new(),
        alias_of: None,
        function_type: 0,
        hash: [0; 32],
        public_metadata_offset: 0,
        private_metadata_offset: 0,
        module_offset: NO_MODULE,
        module: None,
        air_version: (0, 0),
        metal_version: (0, 0),
        reflection_offset: None,
        tessellation: None,
        specialization: None,
        render_target_array_index_type: None,
        entry_size: 0,
        tags: Vec::new(),
    };
    let mut seen: Vec<[u8; 4]> = Vec::new();
    let mut module_size: Option<u64> = None;
    let mut have = |n: &[u8; 4]| -> bool {
        let d = seen.contains(n);
        seen.push(*n);
        d
    };
    for t in &tags {
        if have(&t.name) {
            return err(at, format!("function tag {} repeated", String::from_utf8_lossy(&t.name)));
        }
        let len = |n: usize| -> Result<()> {
            if t.data.len() != n {
                return err(
                    at,
                    format!(
                        "function tag {} of {} bytes (expected {})",
                        String::from_utf8_lossy(&t.name),
                        t.data.len(),
                        n
                    ),
                );
            }
            Ok(())
        };
        match &t.name {
            b"NAME" => f.name = cstr(t.data, "NAME tag", at)?,
            b"ALIS" => {
                let parts: Vec<&[u8]> = t.data.split(|c| *c == 0).collect();
                if parts.len() != 3 || !parts[2].is_empty() {
                    return err(at, "malformed ALIS tag");
                }
                let s = |b: &[u8]| std::str::from_utf8(b).map(|s| s.to_string());
                match (s(parts[0]), s(parts[1])) {
                    (Ok(a), Ok(b)) => {
                        f.name = a;
                        f.alias_of = Some(b);
                    }
                    _ => return err(at, "ALIS tag is not UTF-8"),
                }
            }
            b"TYPE" => {
                len(1)?;
                f.function_type = t.data[0];
            }
            b"HASH" => {
                len(32)?;
                f.hash = t.data.try_into().unwrap();
            }
            b"OFFT" => {
                len(24)?;
                f.public_metadata_offset = u64_at(t.data, 0).unwrap();
                f.private_metadata_offset = u64_at(t.data, 8).unwrap();
                f.module_offset = u64_at(t.data, 16).unwrap();
            }
            b"MDSZ" => {
                len(8)?;
                module_size = Some(u64_at(t.data, 0).unwrap());
            }
            b"VERS" => {
                len(8)?;
                f.air_version = (u16_at(t.data, 0).unwrap(), u16_at(t.data, 2).unwrap());
                f.metal_version = (u16_at(t.data, 4).unwrap(), u16_at(t.data, 6).unwrap());
            }
            b"RFLT" => {
                len(8)?;
                f.reflection_offset = Some(u64_at(t.data, 0).unwrap());
            }
            b"TESS" => {
                len(1)?;
                f.tessellation = Some(Tessellation { patch_type: t.data[0] & 3, control_points: t.data[0] >> 2 });
            }
            b"SPCI" => {
                len(41)?;
                if t.data[0] != 0 {
                    return err(at, format!("SPCI tag with leading byte {} (only 0 is measured)", t.data[0]));
                }
                f.specialization = Some(Specialization {
                    kind: t.data[0],
                    unspecialized_hash: t.data[1..33].try_into().unwrap(),
                    script_offset: u64_at(t.data, 33).unwrap(),
                });
            }
            b"LAYR" => {
                len(1)?;
                f.render_target_array_index_type = Some(t.data[0]);
            }
            other => return err(at, format!("unknown function tag {}", String::from_utf8_lossy(other))),
        }
    }
    let has = |n: &[u8; 4]| tags.iter().any(|t| &t.name == n);
    if has(b"NAME") == has(b"ALIS") {
        return err(at, "function entry needs exactly one of NAME and ALIS");
    }
    for required in [b"TYPE", b"HASH", b"OFFT", b"VERS"] {
        if !has(required) {
            return err(at, format!("function {} has no {} tag", f.name, String::from_utf8_lossy(required)));
        }
    }
    match module_size {
        None => {
            if f.module_offset != NO_MODULE || !header.stub {
                return err(at, format!("function {}: no MDSZ outside a stub library", f.name));
            }
        }
        Some(sz) => {
            if f.module_offset == NO_MODULE {
                return err(at, format!("function {}: MDSZ without a module offset", f.name));
            }
            let end = f.module_offset.checked_add(sz);
            if end.map(|e| e > modules.size).unwrap_or(true) {
                return err(at, format!("function {}: module outside the module list", f.name));
            }
            let start = (modules.offset + f.module_offset) as usize;
            let bytes = &data[start..start + sz as usize];
            if bytes.len() < 4
                || (bytes[..4] != crate::bitcode::WRAPPER_MAGIC && bytes[..4] != crate::bitcode::BITCODE_MAGIC)
            {
                return err(start, format!("function {}: no bitcode at its module offset", f.name));
            }
            if sha256(bytes) != f.hash {
                return err(start, format!("function {}: HASH is not the SHA-256 of its module", f.name));
            }
            f.module = Some(bytes);
        }
    }
    f.tags = tags;
    Ok(f)
}

/// Modules must not overlap, and no bitcode may hide in the module list outside them.
fn check_modules(lib: &Metallib) -> Result<()> {
    let m = lib.header.module_list;
    let mut spans: Vec<(u64, u64)> = lib
        .functions
        .iter()
        .filter(|f| f.module.is_some())
        .map(|f| (f.module_offset, f.module.unwrap().len() as u64))
        .collect();
    spans.sort();
    spans.dedup();
    let mut cursor = 0u64;
    for (o, s) in &spans {
        if *o < cursor {
            return err((m.offset + o) as usize, "overlapping modules");
        }
        let gap = &lib.data[(m.offset + cursor) as usize..(m.offset + o) as usize];
        if gap.windows(4).any(|w| w == crate::bitcode::WRAPPER_MAGIC) {
            return err((m.offset + cursor) as usize, "unreferenced module in the module list");
        }
        cursor = o + s;
    }
    let tail = &lib.data[(m.offset + cursor) as usize..(m.offset + m.size) as usize];
    if tail.windows(4).any(|w| w == crate::bitcode::WRAPPER_MAGIC) {
        return err((m.offset + cursor) as usize, "unreferenced module at the end of the module list");
    }
    Ok(())
}

/// A record of a per-function section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionRecord<'a> {
    /// Offset from the start of the section (the offset a function entry names).
    pub offset: u64,
    /// The record's size word as stored (see the module documentation for whether it counts itself).
    pub size: u32,
    pub tags: Vec<Tag<'a>>,
}

/// The per-function sections of a metallib.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    PublicMetadata,
    PrivateMetadata,
    ReflectionList,
    ScriptList,
}

/// Tags whose length is a `u32` (the flatbuffer payloads of the reflection and script lists).
const WIDE_TAGS: [&[u8; 4]; 2] = [b"RBUF", b"SBUF"];

fn read_wide_tags<'a>(d: &'a [u8], mut p: usize, end: usize, what: &str) -> Result<(Vec<Tag<'a>>, usize)> {
    let mut tags = Vec::new();
    loop {
        if p + 4 > end {
            return err(p, format!("{} runs past its end without ENDT", what));
        }
        let name: [u8; 4] = d[p..p + 4].try_into().unwrap();
        p += 4;
        if &name == b"ENDT" {
            return Ok((tags, p));
        }
        if !WIDE_TAGS.contains(&&name) {
            return err(p - 4, format!("unknown {} tag {}", what, String::from_utf8_lossy(&name)));
        }
        let len = match u32_at(d, p) {
            Some(l) if p + 4 <= end => l as usize,
            _ => return err(p, format!("{} tag {} is truncated", what, String::from_utf8_lossy(&name))),
        };
        p += 4;
        if len > end - p {
            return err(p, format!("{} tag {} runs past its end", what, String::from_utf8_lossy(&name)));
        }
        tags.push(Tag { name, data: &d[p..p + len] });
        p += len;
    }
}

impl<'a> Metallib<'a> {
    /// Whether the metadata records' size words count themselves: true unless the header extension is empty.
    pub fn metadata_sizes_count_themselves(&self) -> bool {
        !self.extension.is_empty()
    }

    /// The records of a per-function section, or an empty list when the library does not have it.
    pub fn section_records(&self, section: Section) -> Result<Vec<SectionRecord<'a>>> {
        let (range, counted, what) = match section {
            Section::PublicMetadata => (Some(self.header.public_metadata), false, "public metadata record"),
            Section::PrivateMetadata => (Some(self.header.private_metadata), false, "private metadata record"),
            Section::ReflectionList => (self.reflection_list, true, "reflection record"),
            Section::ScriptList => (self.script_list, true, "script record"),
        };
        let r = match range {
            Some(r) => r,
            None => return Ok(Vec::new()),
        };
        let d = self.data;
        let start = r.offset as usize;
        let end = start + r.size as usize;
        let mut p = start;
        let mut count = None;
        if counted {
            count = match u32_at(d, p) {
                Some(c) if p + 4 <= end => Some(c as usize),
                _ => return err(p, format!("{} count runs past its section", what)),
            };
            p += 4;
        }
        let self_counted = counted || self.metadata_sizes_count_themselves();
        let mut out = Vec::new();
        while p < end {
            let word = match u32_at(d, p) {
                Some(n) if p + 4 <= end => n,
                _ => return err(p, format!("{} size runs past its section", what)),
            };
            let total = if self_counted { word as usize } else { word as usize + 4 };
            if total < 4 || total > end - p {
                return err(p, format!("{} of size {:#x} does not fit its section", what, word));
            }
            let (tags, q) = if counted {
                read_wide_tags(d, p + 4, p + total, what)?
            } else {
                read_tags(d, p + 4, p + total, what)?
            };
            if q != p + total {
                return err(p, format!("{} size {:#x} does not match its tags", what, word));
            }
            out.push(SectionRecord { offset: (p - start) as u64, size: word, tags });
            p += total;
        }
        if let Some(c) = count {
            if c != out.len() {
                return err(start, format!("{} count {} with {} records", what, c, out.len()));
            }
        }
        Ok(out)
    }

    /// The dynamic header's tags, or an empty list when the library has none.
    pub fn dynamic_header_tags(&self) -> Result<Vec<Tag<'a>>> {
        let r = match self.dynamic_header {
            Some(r) => r,
            None => return Ok(Vec::new()),
        };
        let (start, end) = (r.offset as usize, (r.offset + r.size) as usize);
        let (tags, q) = read_tags(self.data, start, end, "dynamic header")?;
        if q != end {
            return err(q, "bytes after the dynamic header's ENDT");
        }
        Ok(tags)
    }
}

/// The per-function sections must be sequences of well-formed records, every function must name a record of each it
/// uses, and the dynamic header must be a well-formed tag list.
fn check_records(lib: &Metallib) -> Result<()> {
    let starts = |s: Section| -> Result<Vec<u64>> { Ok(lib.section_records(s)?.iter().map(|x| x.offset).collect()) };
    let public = starts(Section::PublicMetadata)?;
    let private = starts(Section::PrivateMetadata)?;
    let reflection = starts(Section::ReflectionList)?;
    let scripts = starts(Section::ScriptList)?;
    for f in &lib.functions {
        if !public.contains(&f.public_metadata_offset) {
            return err(0, format!("function {}: public metadata offset not at a record", f.name));
        }
        if !private.contains(&f.private_metadata_offset) {
            return err(0, format!("function {}: private metadata offset not at a record", f.name));
        }
        match (f.reflection_offset, lib.reflection_list) {
            (Some(o), Some(_)) if !reflection.contains(&o) => {
                return err(0, format!("function {}: reflection offset not at a record", f.name));
            }
            // Stub libraries carry RFLT 0 and no reflection list.
            (Some(o), None) if o != 0 => {
                return err(0, format!("function {}: reflection offset {:#x} without a reflection list", f.name, o));
            }
            _ => {}
        }
        if let Some(s) = &f.specialization {
            if lib.script_list.is_none() {
                return err(0, format!("function {}: specialization script without a script list", f.name));
            }
            if !scripts.contains(&s.script_offset) {
                return err(0, format!("function {}: script offset not at a record", f.name));
            }
        }
    }
    let dynamic = lib.dynamic_header_tags()?;
    if lib.dynamic_header.is_some() {
        match dynamic.as_slice() {
            [t] if &t.name == b"NAME" => {
                cstr(t.data, "dynamic header NAME tag", lib.dynamic_header.unwrap().offset as usize)?;
            }
            _ => return err(lib.dynamic_header.unwrap().offset as usize, "dynamic header is not one NAME tag"),
        }
    }
    Ok(())
}

// Containers ----------------------------------------------------------------------------------------------------------

const FAT_MAGIC: [u8; 4] = [0xca, 0xfe, 0xba, 0xbe];
const FAT_MAGIC_64: [u8; 4] = [0xca, 0xfe, 0xba, 0xbf];
const MACHO_MAGIC_64: [u8; 4] = [0xcf, 0xfa, 0xed, 0xfe];
const LPPM_MAGIC: &[u8; 4] = b"LPPM";
const AR_MAGIC: &[u8; 8] = b"!<arch>\n";
const LC_SEGMENT_64: u32 = 0x19;

/// Where a library or module sits in a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item<'a> {
    Metallib {
        location: String,
        offset: usize,
        lib: Metallib<'a>,
    },
    /// One bitcode module that is not inside a metallib (a raw or wrapped bitcode file, an archive member, an entry of
    /// the GPU driver's runtime tables).
    Module {
        location: String,
        name: String,
        offset: usize,
        bytes: &'a [u8],
    },
    /// A pipeline-library entry, archive member or Mach-O section that holds no AIR, counted.
    Other {
        location: String,
        offset: usize,
        size: usize,
    },
}

/// Reads a file of any of the forms AIR ships in (the forms measured on the system root): a metallib, a big-endian
/// universal file of them, a 64-bit Mach-O with `__metallib` sections, a pipeline-library container (`LPPM`, whose
/// `functions` entries are metallibs), a BSD archive of bitcode members, a bitcode file (raw or wrapped), or the GPU
/// driver's runtime table (`.metallib` files without a magic). `path` decides the last form.
pub fn parse_container<'a>(data: &'a [u8], path: &str) -> Result<Vec<Item<'a>>> {
    let mut out = Vec::new();
    structure(data, 0, data.len(), "", path, &mut out)?;
    if out.is_empty() {
        return err(0, "no AIR container format matches");
    }
    Ok(out)
}

fn structure<'a>(
    data: &'a [u8],
    base: usize,
    limit: usize,
    loc: &str,
    path: &str,
    out: &mut Vec<Item<'a>>,
) -> Result<()> {
    let head = &data[base..limit.min(base + 8)];
    if head.starts_with(MTLB_MAGIC) {
        let lib = parse_metallib(&data[base..limit])?;
        if base + (lib.header.file_size as usize) != limit {
            return err(base, format!("{} bytes after the metallib", limit - base - lib.header.file_size as usize));
        }
        out.push(Item::Metallib { location: format!("{}mtlb", loc), offset: base, lib });
        return Ok(());
    }
    if head.starts_with(&FAT_MAGIC_64) {
        return err(base, "64-bit universal file (not measured on the system root)");
    }
    if head.starts_with(&FAT_MAGIC) {
        return fat(data, base, limit, loc, path, out);
    }
    if head.starts_with(&MACHO_MAGIC_64) {
        return macho(data, base, limit, loc, out);
    }
    if head.starts_with(LPPM_MAGIC) {
        return lppm(data, base, limit, loc, out);
    }
    if head == AR_MAGIC {
        return archive(data, base, limit, loc, out);
    }
    if head.starts_with(&crate::bitcode::WRAPPER_MAGIC) || head.starts_with(&crate::bitcode::BITCODE_MAGIC) {
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        out.push(Item::Module { location: format!("{}bitcode", loc), name, offset: base, bytes: &data[base..limit] });
        return Ok(());
    }
    if base == 0 && path.ends_with(".metallib") {
        return agx_runtime(data, loc, out);
    }
    Ok(())
}

fn fat<'a>(data: &'a [u8], base: usize, limit: usize, loc: &str, path: &str, out: &mut Vec<Item<'a>>) -> Result<()> {
    let be = |o: usize| -> Option<u32> { data.get(o..o + 4).map(|b| u32::from_be_bytes(b.try_into().unwrap())) };
    let count =
        be(base + 4).ok_or(Error { offset: Some(base), message: "truncated universal header".into() })? as usize;
    for i in 0..count {
        let e = base + 8 + 20 * i;
        let (cputype, sub, off, size) = match (be(e), be(e + 4), be(e + 8), be(e + 12)) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c as usize, d as usize),
            _ => return err(e, "truncated universal slice table"),
        };
        let start = base + off;
        if start + size > limit {
            return err(e, format!("universal slice {} outside the file", i));
        }
        let before = out.len();
        structure(data, start, start + size, &format!("{}fat:{:08x}/{:08x}/", loc, cputype, sub), path, out)?;
        if out.len() == before {
            return err(start, format!("universal slice {} has an unknown format", i));
        }
    }
    Ok(())
}

fn macho<'a>(data: &'a [u8], base: usize, limit: usize, loc: &str, out: &mut Vec<Item<'a>>) -> Result<()> {
    let ncmds = u32_at(data, base + 16)
        .ok_or(Error { offset: Some(base), message: "truncated Mach-O header".into() })? as usize;
    let mut p = base + 32;
    for _ in 0..ncmds {
        let (cmd, size) = match (u32_at(data, p), u32_at(data, p + 4)) {
            (Some(c), Some(s)) if s >= 8 => (c, s as usize),
            _ => return err(p, "malformed Mach-O load command"),
        };
        if p + size > limit {
            return err(p, "Mach-O load command runs past the file");
        }
        if cmd == LC_SEGMENT_64 {
            let nsects = u32_at(data, p + 64).unwrap_or(0) as usize;
            if 72 + 80 * nsects > size {
                return err(p, "Mach-O segment command too small for its sections");
            }
            for s in 0..nsects {
                let sp = p + 72 + 80 * s;
                let name = |o: usize| String::from_utf8_lossy(&data[o..o + 16]).trim_end_matches('\0').to_string();
                let sect = name(sp);
                let seg = name(sp + 16);
                let ssize = u64_at(data, sp + 40).unwrap();
                let soff = u32_at(data, sp + 48).unwrap() as usize;
                if ssize == 0 || soff == 0 {
                    continue;
                }
                let start = base + soff;
                let ssize = match usize::try_from(ssize)
                    .ok()
                    .filter(|n| start.checked_add(*n).map(|e| e <= limit).unwrap_or(false))
                {
                    Some(n) => n,
                    None => return err(sp, format!("section {},{} outside the file", seg, sect)),
                };
                if sect != "__metallib" {
                    continue;
                }
                metallib_section(data, start, start + ssize, &format!("{}macho:{},{}", loc, seg, sect), out)?;
            }
        }
        p += size;
    }
    Ok(())
}

/// A Mach-O `__metallib` section: metallibs back to back, each starting at a 16-byte boundary of the section, with
/// zero bytes between them and after the last (measured: 0 to 16 bytes).
fn metallib_section<'a>(data: &'a [u8], start: usize, end: usize, loc: &str, out: &mut Vec<Item<'a>>) -> Result<()> {
    let mut p = start;
    let mut index = 0;
    while p < end {
        if !data[p..end].starts_with(MTLB_MAGIC) {
            return err(p, format!("{} holds no metallib at {:#x}", loc, p - start));
        }
        let lib = parse_metallib(&data[p..end])?;
        let next = p + lib.header.file_size as usize;
        out.push(Item::Metallib { location: format!("{}[{}]/", loc, index), offset: p, lib });
        index += 1;
        p = next;
        while p < end && data[p] == 0 {
            p += 1;
        }
        if p < end && (p - start) % 16 != 0 {
            return err(
                p,
                format!("{}: bytes after a metallib that are neither zero nor a metallib at a 16-byte boundary", loc),
            );
        }
    }
    Ok(())
}

fn lppm<'a>(data: &'a [u8], base: usize, limit: usize, loc: &str, out: &mut Vec<Item<'a>>) -> Result<()> {
    // `LPPM`, three u32 (a version), u64 file size, u64 table offset, u64 data offset; the table: u64 count, then per
    // entry a 32-byte key, u64 offset (from the data offset) and u64 size. The entries tile the data region.
    if limit - base < 40 {
        return err(base, "truncated LPPM header");
    }
    let version = (u32_at(data, base + 4).unwrap(), u32_at(data, base + 8).unwrap(), u32_at(data, base + 12).unwrap());
    if version != (1, 1, 0) {
        return err(base, format!("LPPM version {}.{}.{} (only 1.1.0 is measured)", version.0, version.1, version.2));
    }
    let file_size = u64_at(data, base + 16).unwrap();
    let table = u64_at(data, base + 24).unwrap();
    let data_off = u64_at(data, base + 32).unwrap();
    if file_size != (limit - base) as u64 {
        return err(base, "LPPM file size does not match the container");
    }
    // Every offset below is checked against the container's size first, so the arithmetic cannot overflow.
    let size = (limit - base) as u64;
    if table > size || data_off > size {
        return err(base, "LPPM table or data offset outside the file");
    }
    let (table, data_off) = (table as usize, data_off as usize);
    let count = match u64_at(data, base + table) {
        Some(c) if base + table + 8 <= limit => c,
        _ => return err(base, "LPPM table outside the file"),
    };
    if count.checked_mul(48).and_then(|n| n.checked_add(8 + table as u64)) != Some(data_off as u64) {
        return err(base, "LPPM table does not end at the data offset");
    }
    let count = count as usize;
    let mut spans = Vec::new();
    for i in 0..count {
        let e = base + table + 8 + 48 * i;
        let off = u64_at(data, e + 32).unwrap();
        let size = u64_at(data, e + 40).unwrap();
        let region = (limit - base - data_off) as u64;
        if off.checked_add(size).map(|x| x > region).unwrap_or(true) {
            return err(e, format!("LPPM entry {} outside the file", i));
        }
        let (start, size) = (base + data_off + off as usize, size as usize);
        spans.push((start, size));
        if data[start..start + size].starts_with(MTLB_MAGIC) {
            let lib = parse_metallib(&data[start..start + size])?;
            if lib.header.file_size as usize != size {
                return err(start, format!("LPPM entry {}: metallib size differs from the entry size", i));
            }
            out.push(Item::Metallib { location: format!("{}lppm[{}]/", loc, i), offset: start, lib });
        } else {
            out.push(Item::Other { location: format!("{}lppm[{}]/", loc, i), offset: start, size });
        }
    }
    spans.sort();
    let mut cursor = base + data_off;
    for (s, n) in spans {
        if s != cursor {
            return err(cursor, "LPPM data region has a gap or an overlap");
        }
        cursor = s + n;
    }
    if cursor != limit {
        return err(cursor, "LPPM data region does not reach the end of the file");
    }
    Ok(())
}

fn archive<'a>(data: &'a [u8], base: usize, limit: usize, loc: &str, out: &mut Vec<Item<'a>>) -> Result<()> {
    // BSD archive: member header name[16] date[12] uid[6] gid[6] mode[8] size[10] "`\n"; a name "#1/N" puts the real
    // name in the first N bytes of the member; members start at even offsets.
    let mut p = base + 8;
    while p < limit {
        if p + 60 > limit {
            return err(p, "truncated archive member header");
        }
        let h = &data[p..p + 60];
        if &h[58..60] != b"`\n" {
            return err(p, "bad archive member header");
        }
        let field = |a: usize, b: usize| String::from_utf8_lossy(&h[a..b]).trim_end().to_string();
        let mut name = field(0, 16);
        let size: usize = match field(48, 58).parse() {
            Ok(s) => s,
            Err(_) => return err(p, "bad archive member size"),
        };
        let start = p + 60;
        if start + size > limit {
            return err(p, format!("archive member {} runs past the archive", name));
        }
        let (body, body_size) = if let Some(n) = name.strip_prefix("#1/") {
            let n: usize = match n.parse() {
                Ok(n) if n <= size => n,
                _ => return err(p, "bad extended archive member name"),
            };
            name = String::from_utf8_lossy(&data[start..start + n]).trim_end_matches('\0').to_string();
            (start + n, size - n)
        } else if name.starts_with('/') || name.ends_with('/') {
            return err(p, format!("GNU archive member {} (not measured on the system root)", name));
        } else {
            (start, size)
        };
        let bytes = &data[body..body + body_size];
        if name.starts_with("__.SYMDEF") {
            out.push(Item::Other { location: format!("{}ar:{}", loc, name), offset: body, size: body_size });
        } else if bytes.starts_with(&crate::bitcode::WRAPPER_MAGIC) || bytes.starts_with(&crate::bitcode::BITCODE_MAGIC)
        {
            out.push(Item::Module { location: format!("{}ar", loc), name, offset: body, bytes });
        } else {
            out.push(Item::Other { location: format!("{}ar:{}", loc, name), offset: body, size: body_size });
        }
        p = start + size;
        p += p & 1;
    }
    if p != limit {
        return err(p, "archive does not end on a member boundary");
    }
    Ok(())
}

fn agx_runtime<'a>(data: &'a [u8], loc: &str, out: &mut Vec<Item<'a>>) -> Result<()> {
    // The GPU driver's runtime libraries: u32 count; per entry a NUL-terminated name and a u32 file offset; at each
    // offset a u32 length and one module of that length, the entries tiling the file.
    let count = match u32_at(data, 0) {
        Some(c) => c as usize,
        None => return err(0, "too short for a driver runtime table"),
    };
    let mut p = 4;
    let mut table = Vec::new();
    for _ in 0..count {
        let end = match data[p.min(data.len())..].iter().position(|c| *c == 0) {
            Some(e) => p + e,
            None => return err(p, "truncated driver runtime table"),
        };
        let off = match u32_at(data, end + 1) {
            Some(o) => o as usize,
            None => return err(p, "truncated driver runtime table"),
        };
        table.push((String::from_utf8_lossy(&data[p..end]).to_string(), off));
        p = end + 5;
    }
    let mut cursor = p;
    for (name, off) in table {
        if off != cursor {
            return err(off, format!("runtime entry {} at {:#x}, expected {:#x}", name, off, cursor));
        }
        let size = match u32_at(data, off) {
            Some(s) => s as usize,
            None => return err(off, "truncated runtime entry"),
        };
        let start = off + 4;
        if start + size > data.len() {
            return err(off, format!("runtime entry {} outside the file", name));
        }
        let bytes = &data[start..start + size];
        if !(bytes.starts_with(&crate::bitcode::WRAPPER_MAGIC) || bytes.starts_with(&crate::bitcode::BITCODE_MAGIC)) {
            return err(start, format!("runtime entry {} is not a module", name));
        }
        out.push(Item::Module { location: format!("{}agx-rtlib", loc), name, offset: start, bytes });
        cursor = start + size;
    }
    if cursor != data.len() {
        return err(cursor, "bytes after the last runtime entry");
    }
    Ok(())
}

/// Metallibs embedded in `data` outside the given ranges (the shared cache keeps some in dylib data): every `MTLB`
/// occurrence whose header, function list and modules decode. Occurrences that do not decode are not metallibs (the
/// text "MTLBuffer", for one) and are skipped.
pub fn embedded_metallibs<'a>(data: &'a [u8], exclude: &[(usize, usize)]) -> Vec<(usize, Metallib<'a>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = data[pos..].windows(4).position(|w| w == MTLB_MAGIC) {
        let at = pos + i;
        pos = at + 1;
        if exclude.iter().any(|(s, n)| at >= *s && at < s + n)
            || out.iter().any(|(s, l): &(usize, Metallib)| at >= *s && at < s + l.data.len())
        {
            continue;
        }
        if let Ok(lib) = parse_metallib(&data[at..]) {
            out.push((at, lib));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(name);
        out.extend_from_slice(&(data.len() as u16).to_le_bytes());
        out.extend_from_slice(data);
    }

    /// A stub library with one compute function `f` and the newer layout (a header extension with a UUID).
    fn stub_library() -> Vec<u8> {
        let mut entry = Vec::new();
        tag(&mut entry, b"NAME", b"f\0");
        tag(&mut entry, b"TYPE", &[2]);
        tag(&mut entry, b"HASH", &[0x11; 32]);
        let mut offt = Vec::new();
        offt.extend_from_slice(&0u64.to_le_bytes());
        offt.extend_from_slice(&0u64.to_le_bytes());
        offt.extend_from_slice(&u64::MAX.to_le_bytes());
        tag(&mut entry, b"OFFT", &offt);
        tag(&mut entry, b"VERS", &[2, 0, 6, 0, 3, 0, 1, 0]);
        entry.extend_from_slice(b"ENDT");
        let mut list = Vec::new();
        list.extend_from_slice(&((entry.len() + 4) as u32).to_le_bytes());
        list.extend_from_slice(&entry);
        let mut ext = Vec::new();
        tag(&mut ext, b"UUID", &[0x22; 16]);
        ext.extend_from_slice(b"ENDT");
        let record = [8u8, 0, 0, 0, b'E', b'N', b'D', b'T'];
        let fl_off = 88u64;
        let public = fl_off + 4 + list.len() as u64 + ext.len() as u64;
        let private = public + 8;
        let modules = private + 8;
        let size = modules;
        let mut d = Vec::new();
        d.extend_from_slice(b"MTLB");
        d.extend_from_slice(&1u16.to_le_bytes());
        d.extend_from_slice(&2u16.to_le_bytes());
        d.extend_from_slice(&7u16.to_le_bytes());
        d.push(0x80);
        d.push(0x82);
        d.extend_from_slice(&17u16.to_le_bytes());
        d.push(7);
        d.push(0);
        for v in [size, fl_off, list.len() as u64, public, 8, private, 8, modules, 0] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(d.len(), 88);
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&list);
        d.extend_from_slice(&ext);
        d.extend_from_slice(&record);
        d.extend_from_slice(&record);
        assert_eq!(d.len() as u64, size);
        d
    }

    #[test]
    fn reads_a_stub_library() {
        let d = stub_library();
        let lib = parse_metallib(&d).unwrap();
        assert!(lib.header.stub && lib.header.arch64);
        assert_eq!(lib.header.platform, 2);
        assert_eq!(lib.functions.len(), 1);
        let f = &lib.functions[0];
        assert_eq!((f.name.as_str(), f.function_type, f.module_offset, f.module), ("f", 2, NO_MODULE, None));
        assert_eq!((f.air_version, f.metal_version), ((2, 6), (3, 1)));
        assert_eq!(lib.uuid, Some([0x22; 16]));
        let public = lib.section_records(Section::PublicMetadata).unwrap();
        assert_eq!(public.len(), 1);
        assert_eq!((public[0].offset, public[0].size), (0, 8));
        assert!(matches!(parse_container(&d, "x.metallib").unwrap()[..], [Item::Metallib { offset: 0, .. }]));
    }

    #[test]
    fn truncated_and_corrupted_libraries_are_errors() {
        let d = stub_library();
        for n in 0..d.len() {
            assert!(parse_metallib(&d[..n]).is_err(), "length {}", n);
        }
        for i in 0..d.len() {
            for v in [0u8, 0xff, 0x7f, 0x80] {
                let mut c = d.clone();
                c[i] = v;
                let _ = parse_metallib(&c);
                let _ = parse_container(&c, "x.metallib");
            }
        }
    }

    #[test]
    fn a_module_without_a_size_outside_a_stub_library_is_an_error() {
        let mut d = stub_library();
        d[10] = 0;
        let e = parse_metallib(&d).unwrap_err();
        assert!(e.message.contains("no MDSZ outside a stub library"), "{}", e);
    }

    #[test]
    fn unmeasured_containers_are_refused_by_name() {
        let mut fat64 = vec![0xca, 0xfe, 0xba, 0xbf];
        fat64.extend_from_slice(&[0; 28]);
        assert!(parse_container(&fat64, "x").unwrap_err().message.contains("64-bit universal file"));

        let mut gnu = b"!<arch>\n".to_vec();
        gnu.extend_from_slice(format!("{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n", "/", 0, 0, 0, 644, 0).as_bytes());
        assert!(parse_container(&gnu, "x.a").unwrap_err().message.contains("GNU archive member"));

        let mut lppm = b"LPPM".to_vec();
        for v in [2u32, 0, 0] {
            lppm.extend_from_slice(&v.to_le_bytes());
        }
        lppm.extend_from_slice(&[0; 24]);
        assert!(parse_container(&lppm, "functions").unwrap_err().message.contains("only 1.1.0 is measured"));

        assert_eq!(parse_container(b"not a shader", "x").unwrap_err().message, "no AIR container format matches");
    }
}
