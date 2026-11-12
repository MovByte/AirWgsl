//! LLVM bitcode reader: bitcode bytes to an [`ir::Module`](crate::ir::Module).
//!
//! References: "LLVM Bitcode File Format" and the reader in `llvm/lib/Bitcode/Reader` (`BitcodeReader.cpp`,
//! `MetadataLoader.cpp`) and the auto-upgrader `llvm/lib/IR/AutoUpgrade.cpp`, release 21.1.8. This is a
//! reimplementation, not a port: it reads a whole module eagerly, validates what LLVM's reader asserts or ignores,
//! and fails with an [`Error`] where LLVM would silently drop or default something. Every unknown block, record code,
//! attribute, or value encoding is an error naming it.
//!
//! The result is the module as LLVM 21 has it in memory after reading the same bytes (see [`crate::ir`] for the
//! upgrades that implies).

mod attrs;
pub mod codes;
mod constants;
mod function;
mod intrinsics;
mod llvm_intrinsics;
mod metadata;
mod tbaa;
mod triple;
mod types;
mod upgrade;

pub use intrinsics::is_known_intrinsic;
/// The LLVM release whose reader, upgrades and intrinsic tables this reader follows.
pub use llvm_intrinsics::LLVM_VERSION;

use std::collections::HashMap;
use std::fmt;

use crate::bitstream::{self, Cursor, Entry, Record};
use crate::ir::*;
use codes::*;

/// Why bitcode could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    /// Bit offset in the bitcode (from the `BC` magic), when the error is tied to a position.
    pub bit: Option<u64>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The input is not well-formed bitcode.
    Malformed,
    /// The input is well-formed but uses something this reader does not implement (named in the message).
    Unsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ErrorKind::Malformed => "malformed bitcode",
            ErrorKind::Unsupported => "unsupported bitcode",
        };
        match self.bit {
            Some(b) => write!(f, "{} at bit {}: {}", kind, b, self.message),
            None => write!(f, "{}: {}", kind, self.message),
        }
    }
}

impl std::error::Error for Error {}

impl From<bitstream::Error> for Error {
    fn from(e: bitstream::Error) -> Error {
        Error { kind: ErrorKind::Malformed, bit: Some(e.bit), message: e.message }
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// The magic of the bitcode wrapper header, `0x0B17C0DE` little-endian.
pub const WRAPPER_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
/// The magic of raw bitcode: `BC` `0xC0DE`.
pub const BITCODE_MAGIC: [u8; 4] = [b'B', b'C', 0xc0, 0xde];

/// Splits a wrapper header off `data` when it has one; returns the header and the bitcode it delimits.
///
/// The header is five little-endian `u32`s (magic, version, offset, size, cputype; `BitcodeWrapperHeader` in
/// `BitCodeEnums.h`). The bitcode is `data[offset .. offset + size]`; bytes outside it belong to the container, and
/// LLVM ignores them as this reader does. (Measured in metallib module slots: up to 16 bytes after the bitcode, zeros,
/// sometimes ending in four newline bytes.)
pub fn split_wrapper(data: &[u8]) -> Result<(Option<WrapperHeader>, &[u8])> {
    if data.len() >= 4 && data[..4] == WRAPPER_MAGIC {
        if data.len() < 20 {
            return Err(malformed(None, "truncated bitcode wrapper header"));
        }
        let word = |i: usize| u32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap());
        let header = WrapperHeader { version: word(1), offset: word(2), size: word(3), cputype: word(4) };
        let start = header.offset as usize;
        let end = start.checked_add(header.size as usize);
        match end {
            Some(end) if start >= 20 && end <= data.len() => Ok((Some(header), &data[start..end])),
            _ => Err(malformed(
                None,
                format!(
                    "wrapper header places bitcode at [{:#x}, +{:#x}), outside the {} bytes given",
                    header.offset,
                    header.size,
                    data.len()
                ),
            )),
        }
    } else {
        Ok((None, data))
    }
}

/// Reads one bitcode module, raw (`BC` `0xC0DE`) or behind a wrapper header.
pub fn parse_bitcode(data: &[u8]) -> Result<Module> {
    let (wrapper, bc) = split_wrapper(data)?;
    if bc.len() % 4 != 0 {
        return Err(malformed(None, format!("bitcode size {} is not a multiple of 4", bc.len())));
    }
    if bc.len() < 4 || bc[..4] != BITCODE_MAGIC {
        return Err(malformed(None, "no bitcode magic (BC 0xC0DE)"));
    }
    let mut reader = Reader::new(bc);
    reader.m.wrapper = wrapper;
    reader.read_file()?;
    Ok(reader.m)
}

pub(crate) fn malformed(bit: Option<u64>, message: impl Into<String>) -> Error {
    Error { kind: ErrorKind::Malformed, bit, message: message.into() }
}

pub(crate) fn unsupported(bit: Option<u64>, message: impl Into<String>) -> Error {
    Error { kind: ErrorKind::Unsupported, bit, message: message.into() }
}

/// Bit set on a [`Value::Inst`] index while it is a forward reference to value number `index & !FWD`.
pub(crate) const FWD: u32 = 1 << 31;

/// What a value number denotes while reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Global(GlobalId),
    /// An entry of `Reader::raw_consts`.
    Const(u32),
    Arg(u32),
    Inst(InstId),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VEntry {
    pub slot: Slot,
    /// Bitcode type id.
    pub bc_ty: u32,
}

/// Prefix/prologue/personality operands of a function, resolved after the constants are read.
#[derive(Clone, Copy, Debug, Default)]
struct FunctionOperands {
    function: u32,
    prologue: u64,
    prefix: u64,
    personality: u64,
}

pub(crate) struct Reader<'a> {
    pub cur: Cursor<'a>,
    pub m: Module,
    strtab: &'a [u8],
    use_strtab: bool,
    pub relative_ids: bool,
    seen_module_version: bool,
    // Types.
    pub bc_types: Vec<Option<TypeId>>,
    pub bc_contained: Vec<Vec<u32>>,
    // Attributes.
    pub attr_groups: HashMap<u64, (u32, AttrSet)>,
    pub attr_lists: Vec<Vec<(u32, AttrSet)>>,
    // Values.
    pub values: Vec<VEntry>,
    pub raw_consts: Vec<constants::RawConst>,
    section_table: Vec<String>,
    gc_table: Vec<String>,
    global_inits: Vec<(u32, u64)>,
    indirect_inits: Vec<(GlobalId, u64)>,
    function_operands: Vec<FunctionOperands>,
    /// Functions with bodies, in module order.
    functions_with_bodies: Vec<u32>,
    /// Bit positions of the `ENTER_SUBBLOCK` of each function block, in stream order.
    function_blocks: Vec<u64>,
    /// Function index -> bit position of its block, from the module-level symbol table.
    function_block_at: HashMap<u32, u64>,
    vst_offset: Option<u64>,
    seen_vst: bool,
    /// Global objects with an implicit comdat (old linkage encodings).
    implicit_comdat: Vec<GlobalId>,
    // Metadata.
    pub md: metadata::MetadataState,
    pub sync_scope_map: Vec<u32>,
    seen_type_table: bool,
    seen_attr_groups: bool,
    seen_attrs: bool,
    seen_bundle_tags: bool,
    seen_sync_scopes: bool,
    pub layout: Option<crate::ir::layout::DataLayout>,
    pub const_depth: u32,
}

impl<'a> Reader<'a> {
    fn new(bc: &'a [u8]) -> Reader<'a> {
        Reader {
            cur: Cursor::new(bc),
            m: Module::new(),
            strtab: &[],
            use_strtab: false,
            relative_ids: false,
            seen_module_version: false,
            bc_types: Vec::new(),
            bc_contained: Vec::new(),
            attr_groups: HashMap::new(),
            attr_lists: Vec::new(),
            values: Vec::new(),
            raw_consts: Vec::new(),
            section_table: Vec::new(),
            gc_table: Vec::new(),
            global_inits: Vec::new(),
            indirect_inits: Vec::new(),
            function_operands: Vec::new(),
            functions_with_bodies: Vec::new(),
            function_blocks: Vec::new(),
            function_block_at: HashMap::new(),
            vst_offset: None,
            seen_vst: false,
            implicit_comdat: Vec::new(),
            md: metadata::MetadataState::default(),
            sync_scope_map: Vec::new(),
            seen_type_table: false,
            seen_attr_groups: false,
            seen_attrs: false,
            seen_bundle_tags: false,
            seen_sync_scopes: false,
            layout: None,
            const_depth: 0,
        }
    }

    pub fn here(&self) -> Option<u64> {
        Some(self.cur.position())
    }

    pub fn malformed<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(malformed(self.here(), message))
    }

    pub fn unsupported<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(unsupported(self.here(), message))
    }

    /// Reads the top level: the identification block, one module block, and the string and symbol tables.
    fn read_file(&mut self) -> Result<()> {
        // Skip the magic.
        self.cur.reader().read(32)?;
        let mut module_at = None;
        let mut identification_at = None;
        let mut strtab: Option<&'a [u8]> = None;
        loop {
            let pos = self.cur.position();
            if self.cur.at_end() {
                break;
            }
            match self.cur.advance()? {
                Entry::EndBlock => break,
                Entry::Record(_) => return self.malformed("record at the top level of the bitcode"),
                Entry::SubBlock(id) => match id {
                    IDENTIFICATION_BLOCK_ID => {
                        if identification_at.is_some() || module_at.is_some() {
                            return self.malformed("identification block not directly before the module block");
                        }
                        identification_at = Some(pos);
                        self.cur.skip_block()?;
                    }
                    MODULE_BLOCK_ID => {
                        if module_at.is_some() {
                            return self.unsupported("more than one module in the bitcode");
                        }
                        module_at = Some(pos);
                        self.cur.skip_block()?;
                    }
                    STRTAB_BLOCK_ID => {
                        if strtab.is_some() {
                            return self.unsupported("more than one string table in the bitcode");
                        }
                        strtab = Some(self.read_blob_block(STRTAB_BLOCK_ID, STRTAB_BLOB)?);
                    }
                    SYMTAB_BLOCK_ID => {
                        // The symbol table is an index the linker reads (irsymtab); everything in it is derived
                        // from the module. Its framing is validated; its content is not read.
                        self.read_blob_block(SYMTAB_BLOCK_ID, SYMTAB_BLOB)?;
                    }
                    bitstream::BLOCKINFO_BLOCK_ID => {
                        return self.unsupported("BLOCKINFO block at the top level of the bitcode");
                    }
                    other => {
                        return self.unsupported(format!("top-level block {} ({})", other, block_name(other)));
                    }
                },
            }
        }
        let module_at = match module_at {
            Some(p) => p,
            None => return self.malformed("no module block"),
        };
        if let Some(s) = strtab {
            self.strtab = s;
        }
        if let Some(pos) = identification_at {
            self.cur.seek(pos)?;
            match self.cur.advance()? {
                Entry::SubBlock(IDENTIFICATION_BLOCK_ID) => self.read_identification()?,
                _ => return self.malformed("identification block moved"),
            }
        }
        self.cur.seek(module_at)?;
        match self.cur.advance()? {
            Entry::SubBlock(MODULE_BLOCK_ID) => self.read_module()?,
            _ => return self.malformed("module block moved"),
        }
        Ok(())
    }

    fn read_blob_block(&mut self, id: u32, code: u32) -> Result<&'a [u8]> {
        self.cur.enter_block(id)?;
        let mut blob = None;
        loop {
            match self.cur.advance()? {
                Entry::EndBlock => break,
                Entry::SubBlock(b) => return self.malformed(format!("sub-block {} inside {}", b, block_name(id))),
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    if r.code != code || r.blob.is_none() || blob.is_some() {
                        return self.malformed(format!("{} must hold exactly one blob record", block_name(id)));
                    }
                    blob = r.blob;
                }
            }
        }
        match blob {
            Some(b) => Ok(b),
            None => self.malformed(format!("{} without a blob", block_name(id))),
        }
    }

    fn read_identification(&mut self) -> Result<()> {
        self.cur.enter_block(IDENTIFICATION_BLOCK_ID)?;
        loop {
            match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block inside the identification block"),
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    match r.code {
                        IDENTIFICATION_CODE_STRING => self.m.producer = Some(self.record_string(&r.ops)?),
                        IDENTIFICATION_CODE_EPOCH => {
                            if r.ops.first() != Some(&BITCODE_CURRENT_EPOCH) {
                                return self
                                    .unsupported(format!("bitcode epoch {:?} (only epoch 0 exists)", r.ops.first()));
                            }
                        }
                        c => return self.malformed(format!("unknown identification record {}", c)),
                    }
                }
            }
        }
    }

    /// Characters of a string record (`convertToString`).
    pub fn record_bytes(&self, ops: &[u64]) -> Result<Vec<u8>> {
        ops.iter()
            .map(
                |&c| {
                    if c <= 0xff { Ok(c as u8) } else { self.malformed(format!("character {} in a string record", c)) }
                },
            )
            .collect()
    }

    pub fn record_string(&self, ops: &[u64]) -> Result<String> {
        let bytes = self.record_bytes(ops)?;
        match String::from_utf8(bytes) {
            Ok(s) => Ok(s),
            Err(_) => self.unsupported("a name that is not UTF-8"),
        }
    }

    fn strtab_name(&self, offset: u64, size: u64) -> Result<String> {
        let end = offset.checked_add(size);
        match end {
            Some(end) if end <= self.strtab.len() as u64 => {
                match std::str::from_utf8(&self.strtab[offset as usize..end as usize]) {
                    Ok(s) => Ok(s.to_string()),
                    Err(_) => self.unsupported("a name in the string table that is not UTF-8"),
                }
            }
            _ => self.malformed(format!("string table reference [{}, +{}) outside the table", offset, size)),
        }
    }

    /// Splits the strtab name off a v2 global record.
    fn record_name<'r>(&self, ops: &'r [u64]) -> Result<(String, &'r [u64])> {
        if !self.use_strtab {
            return Ok((String::new(), ops));
        }
        if ops.len() < 2 {
            return self.malformed("global record without a name");
        }
        Ok((self.strtab_name(ops[0], ops[1])?, &ops[2..]))
    }

    pub fn ty(&self, bc: u64) -> Result<TypeId> {
        match self.bc_types.get(bc as usize).copied().flatten() {
            Some(t) => Ok(t),
            None => self.malformed(format!("invalid type id {}", bc)),
        }
    }

    /// The bitcode type id contained in `bc` at `index` (a pointer's pointee is index 0).
    pub fn contained(&self, bc: u32, index: usize) -> Option<u32> {
        self.bc_contained.get(bc as usize).and_then(|v| v.get(index)).copied()
    }

    /// A bitcode type id for an IR type created while reading (`getVirtualTypeID`).
    pub fn virtual_type(&mut self, ty: TypeId, contained: Vec<u32>) -> u32 {
        // Reuse an existing id with the same IR type and first contained id, as LLVM caches them.
        for (i, t) in self.bc_types.iter().enumerate() {
            if *t == Some(ty) && self.bc_contained[i].first() == contained.first() && self.bc_contained[i] == contained
            {
                return i as u32;
            }
        }
        self.bc_types.push(Some(ty));
        self.bc_contained.push(contained);
        (self.bc_types.len() - 1) as u32
    }

    pub fn alignment(&self, exponent: u64) -> Result<Option<u64>> {
        // Bitcode stores log2(alignment) + 1, with 0 for none (`decodeMaybeAlign`); the largest is 2^32.
        if exponent > 33 {
            return self.malformed(format!("invalid alignment exponent {}", exponent));
        }
        Ok(if exponent == 0 { None } else { Some(1u64 << (exponent - 1)) })
    }

    fn read_module(&mut self) -> Result<()> {
        self.cur.enter_block(MODULE_BLOCK_ID)?;
        loop {
            let pos = self.cur.position();
            match self.cur.advance()? {
                Entry::EndBlock => break,
                Entry::SubBlock(id) => match id {
                    bitstream::BLOCKINFO_BLOCK_ID => self.cur.read_blockinfo()?,
                    PARAMATTR_BLOCK_ID => self.read_attr_block()?,
                    PARAMATTR_GROUP_BLOCK_ID => self.read_attr_group_block()?,
                    TYPE_BLOCK_ID_NEW => self.read_type_block()?,
                    VALUE_SYMTAB_BLOCK_ID => {
                        if self.seen_vst {
                            // Already read through VSTOFFSET when the first function block was met.
                            self.cur.skip_block()?;
                        } else {
                            self.read_module_vst(None)?;
                            self.seen_vst = true;
                        }
                    }
                    CONSTANTS_BLOCK_ID => {
                        self.read_constants(None)?;
                        self.resolve_global_inits()?;
                    }
                    METADATA_BLOCK_ID => self.read_metadata_block(None)?,
                    METADATA_KIND_BLOCK_ID => self.read_metadata_kinds()?,
                    FUNCTION_BLOCK_ID => {
                        self.resolve_datalayout()?;
                        if self.function_blocks.is_empty() {
                            if let Some(off) = self.vst_offset {
                                if !self.seen_vst {
                                    let back = self.cur.position();
                                    // The ENTER_SUBBLOCK was read; return to the start of the block afterwards.
                                    self.cur.seek(off)?;
                                    match self.cur.advance()? {
                                        Entry::SubBlock(VALUE_SYMTAB_BLOCK_ID) => {}
                                        _ => return self.malformed("VSTOFFSET does not point at a value symbol table"),
                                    }
                                    self.read_module_vst(Some(off))?;
                                    self.seen_vst = true;
                                    self.cur.seek(back)?;
                                }
                            }
                        }
                        // The function this block belongs to: by the symbol table's offsets, or else the next function
                        // with a body in module order.
                        let f = if self.function_block_at.is_empty() {
                            match self.functions_with_bodies.get(self.function_blocks.len()) {
                                Some(&f) => f,
                                None => return self.malformed("more function blocks than functions with bodies"),
                            }
                        } else {
                            let found: Vec<u32> =
                                self.function_block_at.iter().filter(|(_, at)| **at == pos).map(|(f, _)| *f).collect();
                            match found.as_slice() {
                                [f] => *f,
                                [] => return self.malformed("function block that no FNENTRY offset names"),
                                _ => return self.malformed("two functions share one function block"),
                            }
                        };
                        if !self.functions_with_bodies.contains(&f) {
                            return self.malformed("function block for a function declared without a body");
                        }
                        if !self.m.functions[f as usize].blocks.is_empty() {
                            return self.malformed("two function blocks for one function");
                        }
                        self.function_blocks.push(pos);
                        self.read_function_body(f)?;
                    }
                    USELIST_BLOCK_ID => self.read_use_lists(None)?,
                    OPERAND_BUNDLE_TAGS_BLOCK_ID => self.read_bundle_tags()?,
                    SYNC_SCOPE_NAMES_BLOCK_ID => self.read_sync_scopes()?,
                    other => {
                        return self.unsupported(format!("block {} ({}) in the module block", other, block_name(other)));
                    }
                },
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    self.module_record(r)?;
                }
            }
        }
        self.resolve_datalayout()?;
        self.resolve_global_inits()?;
        if !self.global_inits.is_empty() || !self.indirect_inits.is_empty() {
            return self.malformed("global initializer refers to a value that is never defined");
        }
        if self.function_operands.iter().any(|f| f.prefix != 0 || f.prologue != 0 || f.personality != 0) {
            return self.malformed("function prefix, prologue or personality refers to a value that is never defined");
        }
        if self.functions_with_bodies.len() != self.function_blocks.len() {
            return self.malformed(format!(
                "{} functions have bodies but the module holds {} function blocks",
                self.functions_with_bodies.len(),
                self.function_blocks.len()
            ));
        }
        self.finish()
    }

    /// `ResolveDataLayout`: the layout is upgraded for the target triple and parsed when something first needs it.
    fn resolve_datalayout(&mut self) -> Result<()> {
        if self.layout.is_none() {
            match triple::upgrade_datalayout(&self.m.datalayout, &self.m.triple) {
                Ok(dl) => self.m.datalayout = dl,
                Err(e) => return self.unsupported(e),
            }
            match crate::ir::layout::DataLayout::parse(&self.m.datalayout) {
                Ok(l) => self.layout = Some(l),
                Err(e) => return self.malformed(format!("data layout \"{}\": {}", self.m.datalayout, e)),
            }
        }
        Ok(())
    }

    fn module_record(&mut self, r: Record<'a>) -> Result<()> {
        match r.code {
            MODULE_CODE_VERSION => {
                let v = match r.ops.first() {
                    Some(&v) => v,
                    None => return self.malformed("empty VERSION record"),
                };
                if v > 2 {
                    return self.unsupported(format!("module version {}", v));
                }
                if self.seen_module_version {
                    return self.malformed("more than one VERSION record");
                }
                self.seen_module_version = true;
                self.relative_ids = v >= 1;
                self.use_strtab = v >= 2;
            }
            MODULE_CODE_TRIPLE => {
                if self.layout.is_some() {
                    return self.malformed("target triple after the data layout was used");
                }
                self.m.triple = self.record_string(&r.ops)?;
            }
            MODULE_CODE_DATALAYOUT => {
                // A later record replaces an earlier one until the layout is first used.
                if self.layout.is_some() {
                    return self.malformed("data layout record after the data layout was used");
                }
                self.m.datalayout = self.record_string(&r.ops)?;
            }
            MODULE_CODE_ASM => {
                let bytes = self.record_bytes(&r.ops)?;
                self.m.module_asm = bytes;
            }
            MODULE_CODE_SECTIONNAME => {
                let s = self.record_string(&r.ops)?;
                self.section_table.push(s);
            }
            MODULE_CODE_GCNAME => {
                let s = self.record_string(&r.ops)?;
                self.gc_table.push(s);
            }
            MODULE_CODE_DEPLIB => {
                return self.unsupported("DEPLIB record (dependent libraries of pre-3.x bitcode)");
            }
            MODULE_CODE_COMDAT => self.comdat_record(&r.ops)?,
            MODULE_CODE_GLOBALVAR => self.global_var_record(&r.ops)?,
            MODULE_CODE_FUNCTION => {
                self.resolve_datalayout()?;
                self.function_record(&r.ops)?;
            }
            MODULE_CODE_ALIAS | MODULE_CODE_ALIAS_OLD | MODULE_CODE_IFUNC => {
                self.indirect_symbol_record(r.code, &r.ops)?
            }
            MODULE_CODE_VSTOFFSET => {
                let v = match r.ops.first() {
                    Some(&v) if v >= 1 => v,
                    _ => return self.malformed("invalid VSTOFFSET record"),
                };
                // The offset counts 32-bit words from the start of the bitcode (the magic): one word before the
                // start of the identification or module block as the reader positions it.
                self.vst_offset = Some(v * 32);
            }
            MODULE_CODE_SOURCE_FILENAME => {
                let bytes = self.record_bytes(&r.ops)?;
                self.m.source_filename = Some(bytes);
            }
            MODULE_CODE_HASH => {
                // The module hash (ThinLTO) is derived from the module's content.
                if r.ops.len() != 5 {
                    return self.malformed("HASH record must hold five words");
                }
            }
            MODULE_CODE_METADATA_VALUES_UNUSED => {
                return self.unsupported("METADATA_VALUES_UNUSED record");
            }
            c => return self.unsupported(format!("module record code {}", c)),
        }
        Ok(())
    }

    fn comdat_record(&mut self, ops: &[u64]) -> Result<()> {
        let (mut name, rest) = self.record_name(ops)?;
        if rest.is_empty() {
            return self.malformed("invalid COMDAT record");
        }
        let kind = match rest[0] {
            1 => ComdatKind::Any,
            2 => ComdatKind::ExactMatch,
            3 => ComdatKind::Largest,
            4 => ComdatKind::NoDeduplicate,
            5 => ComdatKind::SameSize,
            k => return self.malformed(format!("unknown comdat selection kind {}", k)),
        };
        if !self.use_strtab {
            if rest.len() < 2 || rest[1] > (rest.len() - 2) as u64 {
                return self.malformed("invalid COMDAT record");
            }
            name = self.record_string(&rest[2..2 + rest[1] as usize])?;
        }
        if self.m.comdats.iter().any(|c| c.name == name) {
            return self.malformed(format!("comdat {} defined twice", name));
        }
        self.m.comdats.push(Comdat { name, kind });
        Ok(())
    }

    fn linkage(&self, raw: u64) -> Result<Linkage> {
        Ok(match raw {
            0 | 5 | 6 | 15 => Linkage::External,
            2 => Linkage::Appending,
            3 => Linkage::Internal,
            7 => Linkage::ExternalWeak,
            8 => Linkage::Common,
            9 | 13 | 14 => Linkage::Private,
            12 => Linkage::AvailableExternally,
            1 | 16 => Linkage::WeakAny,
            10 | 17 => Linkage::WeakOdr,
            4 | 18 => Linkage::LinkOnceAny,
            11 | 19 => Linkage::LinkOnceOdr,
            other => return self.malformed(format!("unknown linkage {}", other)),
        })
    }

    fn visibility(&self, raw: u64) -> Result<Visibility> {
        Ok(match raw {
            0 => Visibility::Default,
            1 => Visibility::Hidden,
            2 => Visibility::Protected,
            other => return self.malformed(format!("unknown visibility {}", other)),
        })
    }

    fn dll_storage(&self, raw: u64) -> Result<DllStorage> {
        Ok(match raw {
            0 => DllStorage::Default,
            1 => DllStorage::Import,
            2 => DllStorage::Export,
            other => return self.malformed(format!("unknown DLL storage class {}", other)),
        })
    }

    fn unnamed_addr(&self, raw: u64) -> Result<UnnamedAddr> {
        Ok(match raw {
            0 => UnnamedAddr::None,
            1 => UnnamedAddr::Global,
            2 => UnnamedAddr::Local,
            other => return self.malformed(format!("unknown unnamed_addr {}", other)),
        })
    }

    fn thread_local(&self, raw: u64) -> Result<ThreadLocal> {
        Ok(match raw {
            0 => ThreadLocal::NotThreadLocal,
            1 => ThreadLocal::GeneralDynamic,
            2 => ThreadLocal::LocalDynamic,
            3 => ThreadLocal::InitialExec,
            4 => ThreadLocal::LocalExec,
            other => return self.malformed(format!("unknown thread-local mode {}", other)),
        })
    }

    fn dso_local(&self, raw: u64) -> Result<bool> {
        match raw {
            0 => Ok(false),
            1 => Ok(true),
            other => self.malformed(format!("unknown dso_local value {}", other)),
        }
    }

    /// `upgradeDLLImportExportLinkage`: obsolete linkages 5 and 6 meant dllimport and dllexport.
    fn legacy_dll(linkage: Linkage, raw: u64) -> DllStorage {
        if linkage.is_local() {
            return DllStorage::Default;
        }
        match raw {
            5 => DllStorage::Import,
            6 => DllStorage::Export,
            _ => DllStorage::Default,
        }
    }

    /// `inferDSOLocal`.
    fn infer_dso_local(c: &mut GlobalCommon) {
        if c.linkage.is_local() || (c.visibility != Visibility::Default && c.linkage != Linkage::ExternalWeak) {
            c.dso_local = true;
        }
    }

    fn comdat_ref(&self, raw: u64) -> Result<Option<u32>> {
        if raw == 0 {
            return Ok(None);
        }
        if raw as usize > self.m.comdats.len() {
            return self.malformed(format!("invalid comdat id {}", raw));
        }
        Ok(Some(raw as u32 - 1))
    }

    fn section(&self, raw: u64) -> Result<Option<String>> {
        if raw == 0 {
            return Ok(None);
        }
        match self.section_table.get(raw as usize - 1) {
            Some(s) => Ok(Some(s.clone())),
            None => self.malformed(format!("invalid section id {}", raw)),
        }
    }

    fn has_implicit_comdat(raw: u64) -> bool {
        matches!(raw, 1 | 4 | 10 | 11)
    }

    fn global_var_record(&mut self, ops: &[u64]) -> Result<()> {
        let (name, r) = self.record_name(ops)?;
        if r.len() < 6 {
            return self.malformed("invalid GLOBALVAR record");
        }
        let mut ty_id = r[0] as u32;
        let mut ty = self.ty(r[0])?;
        let constant = r[1] & 1 != 0;
        let explicit_type = r[1] & 2 != 0;
        let addrspace;
        if explicit_type {
            addrspace = (r[1] >> 2) as u32;
        } else {
            match self.m.types.get(ty) {
                Type::Ptr(a) => addrspace = *a,
                _ => return self.malformed("global variable record with a non-pointer type"),
            }
            ty_id = match self.contained(ty_id, 0) {
                Some(t) => t,
                None => return self.malformed("missing element type for an old-style global"),
            };
            ty = self.ty(ty_id as u64)?;
        }
        let raw_linkage = r[3];
        let linkage = self.linkage(raw_linkage)?;
        let align = self.alignment(r[4])?;
        let section = self.section(r[5])?;
        let visibility = if r.len() > 6 && !linkage.is_local() { self.visibility(r[6])? } else { Visibility::Default };
        let thread_local = if r.len() > 7 { self.thread_local(r[7])? } else { ThreadLocal::NotThreadLocal };
        let unnamed_addr = if r.len() > 8 { self.unnamed_addr(r[8])? } else { UnnamedAddr::None };
        let externally_initialized = r.len() > 9 && r[9] != 0;
        let dll_storage = if r.len() > 10 {
            if linkage.is_local() { DllStorage::Default } else { self.dll_storage(r[10])? }
        } else {
            Self::legacy_dll(linkage, raw_linkage)
        };
        let gid = self.m.globals.len() as u32;
        let mut comdat = None;
        if r.len() > 11 {
            comdat = self.comdat_ref(r[11])?;
        } else if Self::has_implicit_comdat(raw_linkage) {
            self.implicit_comdat.push(GlobalId::Var(gid));
        }
        let attrs = if r.len() > 12 { self.attr_list(r[12])?.function } else { AttrSetId::EMPTY };
        let mut common = GlobalCommon {
            name,
            linkage,
            visibility,
            dll_storage,
            unnamed_addr,
            thread_local,
            dso_local: false,
            addrspace,
            partition: String::new(),
        };
        if r.len() > 13 {
            common.dso_local = self.dso_local(r[13])?;
        }
        Self::infer_dso_local(&mut common);
        if r.len() > 15 {
            common.partition = self.strtab_name(r[14], r[15])?;
        }
        let sanitizer = if r.len() > 16 { r[16] as u32 } else { 0 };
        if sanitizer > 0xf {
            return self.malformed(format!("invalid sanitizer metadata {:#x}", sanitizer));
        }
        let code_model = if r.len() > 17 && r[17] != 0 {
            Some(match r[17] {
                1 => CodeModel::Tiny,
                2 => CodeModel::Small,
                3 => CodeModel::Kernel,
                4 => CodeModel::Medium,
                5 => CodeModel::Large,
                _ => return self.malformed("invalid global variable code model"),
            })
        } else {
            None
        };
        if r.len() > 18 {
            return self.unsupported(format!("GLOBALVAR record with {} fields", r.len()));
        }
        self.m.globals.push(GlobalVariable {
            common,
            value_type: ty,
            constant,
            initializer: None,
            externally_initialized,
            align,
            section,
            comdat,
            attrs,
            code_model,
            sanitizer,
            md: Vec::new(),
            bitcode_type: Some(ty_id),
        });
        let ptr = self.m.types.ptr(addrspace);
        let bc = self.virtual_type(ptr, vec![ty_id]);
        self.values.push(VEntry { slot: Slot::Global(GlobalId::Var(gid)), bc_ty: bc });
        if r[2] != 0 {
            self.global_inits.push((gid, r[2] - 1));
        }
        Ok(())
    }

    fn function_record(&mut self, ops: &[u64]) -> Result<()> {
        let (name, r) = self.record_name(ops)?;
        if r.len() < 8 {
            return self.malformed("invalid FUNCTION record");
        }
        let mut fty_id = r[0] as u32;
        let mut fty = self.ty(r[0])?;
        if self.m.types.is_ptr(fty) {
            fty_id = match self.contained(fty_id, 0) {
                Some(t) => t,
                None => return self.malformed("missing element type for an old-style function"),
            };
            fty = self.ty(fty_id as u64)?;
        }
        let (ret, params, _) = match self.m.types.function(fty) {
            Some((r, p, v)) => (r, p.to_vec(), v),
            None => return self.malformed("function record with a non-function type"),
        };
        let _ = ret;
        let cc = r[1];
        if cc & !1023 != 0 {
            return self.malformed(format!("invalid calling convention {}", cc));
        }
        let addrspace =
            if r.len() > 16 { r[16] as u32 } else { self.layout.as_ref().map(|l| l.program_addrspace).unwrap_or(0) };
        let is_proto = r[2] != 0;
        let raw_linkage = r[3];
        let linkage = self.linkage(raw_linkage)?;
        let mut attrs = self.attr_list(r[4])?;
        let fid = self.m.functions.len() as u32;
        // Typeless byval/sret/inalloca take the parameter's pointee type.
        let param_bc: Vec<Option<u32>> = (0..params.len()).map(|i| self.contained(fty_id, i + 1)).collect();
        self.upgrade_typed_param_attrs(&mut attrs, &param_bc)?;
        let align = self.alignment(r[5])?;
        let section = self.section(r[6])?;
        let visibility = if !linkage.is_local() { self.visibility(r[7])? } else { Visibility::Default };
        let gc = if r.len() > 8 && r[8] != 0 {
            match self.gc_table.get(r[8] as usize - 1) {
                Some(g) => Some(g.clone()),
                None => return self.malformed(format!("invalid gc id {}", r[8])),
            }
        } else {
            None
        };
        let unnamed_addr = if r.len() > 9 { self.unnamed_addr(r[9])? } else { UnnamedAddr::None };
        let mut operands = FunctionOperands { function: fid, ..Default::default() };
        if r.len() > 10 {
            operands.prologue = r[10];
        }
        let dll_storage = if r.len() > 11 {
            if linkage.is_local() { DllStorage::Default } else { self.dll_storage(r[11])? }
        } else {
            Self::legacy_dll(linkage, raw_linkage)
        };
        let mut comdat = None;
        if r.len() > 12 {
            comdat = self.comdat_ref(r[12])?;
        } else if Self::has_implicit_comdat(raw_linkage) {
            self.implicit_comdat.push(GlobalId::Function(fid));
        }
        if r.len() > 13 {
            operands.prefix = r[13];
        }
        if r.len() > 14 {
            operands.personality = r[14];
        }
        let mut common = GlobalCommon {
            name,
            linkage,
            visibility,
            dll_storage,
            unnamed_addr,
            thread_local: ThreadLocal::NotThreadLocal,
            dso_local: false,
            addrspace,
            partition: String::new(),
        };
        if r.len() > 15 {
            common.dso_local = self.dso_local(r[15])?;
        }
        Self::infer_dso_local(&mut common);
        if r.len() > 18 && (r[17] + r[18]) as usize <= self.strtab.len() {
            common.partition = self.strtab_name(r[17], r[18])?;
        }
        if r.len() > 19 {
            return self.unsupported(format!("FUNCTION record with {} fields", r.len()));
        }
        if cc == 83 {
            return self.unsupported("x86_intrcc functions (their byval upgrade is not implemented)");
        }
        let args = params.iter().map(|&ty| Argument { ty, name: None }).collect();
        self.m.functions.push(Function {
            common,
            ty: fty,
            cc: cc as u32,
            attrs,
            align,
            section,
            comdat,
            gc,
            prefix: None,
            prologue: None,
            personality: None,
            args,
            blocks: Vec::new(),
            insts: Vec::new(),
            md: Vec::new(),
            bitcode_type: Some(fty_id),
        });
        let ptr = self.m.types.ptr(addrspace);
        let bc = self.virtual_type(ptr, vec![fty_id]);
        self.values.push(VEntry { slot: Slot::Global(GlobalId::Function(fid)), bc_ty: bc });
        if operands.prologue != 0 || operands.prefix != 0 || operands.personality != 0 {
            self.function_operands.push(operands);
        }
        if !is_proto {
            self.functions_with_bodies.push(fid);
        }
        Ok(())
    }

    fn indirect_symbol_record(&mut self, code: u32, ops: &[u64]) -> Result<()> {
        let (name, r) = self.record_name(ops)?;
        let new_record = code != MODULE_CODE_ALIAS_OLD;
        if r.len() < 3 + new_record as usize {
            return self.malformed("invalid alias or ifunc record");
        }
        let mut i = 0;
        let mut ty_id = r[i] as u32;
        i += 1;
        let mut ty = self.ty(ty_id as u64)?;
        let addrspace;
        if !new_record {
            match self.m.types.get(ty) {
                Type::Ptr(a) => addrspace = *a,
                _ => return self.malformed("old-style alias with a non-pointer type"),
            }
            ty_id = match self.contained(ty_id, 0) {
                Some(t) => t,
                None => return self.malformed("missing element type for an old-style alias"),
            };
            ty = self.ty(ty_id as u64)?;
        } else {
            addrspace = r[i] as u32;
            i += 1;
        }
        let value = r[i];
        i += 1;
        let raw_linkage = r[i];
        i += 1;
        let linkage = self.linkage(raw_linkage)?;
        let mut common = GlobalCommon {
            name,
            linkage,
            visibility: Visibility::Default,
            dll_storage: DllStorage::Default,
            unnamed_addr: UnnamedAddr::None,
            thread_local: ThreadLocal::NotThreadLocal,
            dso_local: false,
            addrspace,
            partition: String::new(),
        };
        if i != r.len() {
            let v = self.visibility(r[i])?;
            i += 1;
            if !linkage.is_local() {
                common.visibility = v;
            }
        }
        let is_alias = code != MODULE_CODE_IFUNC;
        if is_alias {
            if i != r.len() {
                let s = self.dll_storage(r[i])?;
                i += 1;
                if !linkage.is_local() {
                    common.dll_storage = s;
                }
            } else {
                common.dll_storage = Self::legacy_dll(linkage, raw_linkage);
            }
            if i != r.len() {
                common.thread_local = self.thread_local(r[i])?;
                i += 1;
            }
            if i != r.len() {
                common.unnamed_addr = self.unnamed_addr(r[i])?;
                i += 1;
            }
        }
        if i != r.len() {
            common.dso_local = self.dso_local(r[i])?;
            i += 1;
        }
        Self::infer_dso_local(&mut common);
        if i + 1 < r.len() {
            common.partition = self.strtab_name(r[i], r[i + 1])?;
        }
        let id = if is_alias {
            let id = GlobalId::Alias(self.m.aliases.len() as u32);
            self.m.aliases.push(Alias { common, value_type: ty, aliasee: Value::Const(ConstId(u32::MAX)) });
            id
        } else {
            let id = GlobalId::IFunc(self.m.ifuncs.len() as u32);
            self.m.ifuncs.push(IFunc { common, value_type: ty, resolver: Value::Const(ConstId(u32::MAX)) });
            id
        };
        let ptr = self.m.types.ptr(addrspace);
        let bc = self.virtual_type(ptr, vec![ty_id]);
        self.values.push(VEntry { slot: Slot::Global(id), bc_ty: bc });
        self.indirect_inits.push((id, value));
        Ok(())
    }

    /// `resolveGlobalAndIndirectSymbolInits`: sets the initializers whose values are defined by now.
    fn resolve_global_inits(&mut self) -> Result<()> {
        let inits = std::mem::take(&mut self.global_inits);
        for (g, id) in inits {
            if id as usize >= self.values.len() {
                self.global_inits.push((g, id));
                continue;
            }
            let v = self.initializer_value(id)?;
            let vt = self.m.module_value_type(v).unwrap();
            if vt != self.m.globals[g as usize].value_type {
                return self.malformed(format!(
                    "initializer of global {} has the wrong type",
                    self.m.globals[g as usize].common.name
                ));
            }
            self.m.globals[g as usize].initializer = Some(v);
        }
        let inits = std::mem::take(&mut self.indirect_inits);
        for (g, id) in inits {
            if id as usize >= self.values.len() {
                self.indirect_inits.push((g, id));
                continue;
            }
            let v = self.initializer_value(id)?;
            match g {
                GlobalId::Alias(i) => {
                    let want = self.m.types.ptr(self.m.aliases[i as usize].common.addrspace);
                    if self.m.module_value_type(v) != Some(want) {
                        return self.malformed("alias and aliasee types do not match");
                    }
                    self.m.aliases[i as usize].aliasee = v;
                }
                GlobalId::IFunc(i) => self.m.ifuncs[i as usize].resolver = v,
                _ => unreachable!(),
            }
        }
        let ops = std::mem::take(&mut self.function_operands);
        for mut f in ops {
            for (field, which) in [(f.personality, 0), (f.prefix, 1), (f.prologue, 2)] {
                if field == 0 || (field - 1) as usize >= self.values.len() {
                    continue;
                }
                let v = self.initializer_value(field - 1)?;
                let func = &mut self.m.functions[f.function as usize];
                match which {
                    0 => {
                        func.personality = Some(v);
                        f.personality = 0;
                    }
                    1 => {
                        func.prefix = Some(v);
                        f.prefix = 0;
                    }
                    _ => {
                        func.prologue = Some(v);
                        f.prologue = 0;
                    }
                }
            }
            if f.personality != 0 || f.prefix != 0 || f.prologue != 0 {
                self.function_operands.push(f);
            }
        }
        Ok(())
    }

    /// A module-level value used where only a constant is allowed (`getValueForInitializer`).
    fn initializer_value(&mut self, id: u64) -> Result<Value> {
        match self.values.get(id as usize).map(|e| e.slot) {
            Some(Slot::Global(g)) => Ok(Value::Global(g)),
            Some(Slot::Const(c)) => self.materialize_const(c, None),
            _ => self.malformed(format!("initializer refers to value {}, which is not a constant", id)),
        }
    }

    fn read_bundle_tags(&mut self) -> Result<()> {
        if self.seen_bundle_tags {
            return self.malformed("more than one operand bundle tags block");
        }
        self.seen_bundle_tags = true;
        self.cur.enter_block(OPERAND_BUNDLE_TAGS_BLOCK_ID)?;
        loop {
            match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block in the operand bundle tags block"),
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    if r.code != OPERAND_BUNDLE_TAG {
                        return self.malformed("invalid operand bundle tag record");
                    }
                    let s = self.record_string(&r.ops)?;
                    self.m.bundle_tags.push(s);
                }
            }
        }
    }

    fn read_sync_scopes(&mut self) -> Result<()> {
        if self.seen_sync_scopes {
            return self.malformed("more than one sync scope names block");
        }
        self.seen_sync_scopes = true;
        self.cur.enter_block(SYNC_SCOPE_NAMES_BLOCK_ID)?;
        loop {
            match self.cur.advance()? {
                Entry::EndBlock => {
                    if self.sync_scope_map.is_empty() {
                        return self.malformed("empty sync scope names block");
                    }
                    return Ok(());
                }
                Entry::SubBlock(_) => return self.malformed("sub-block in the sync scope names block"),
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    if r.code != SYNC_SCOPE_NAME {
                        return self.malformed("invalid sync scope record");
                    }
                    let name = self.record_string(&r.ops)?;
                    let id = match self.m.sync_scopes.iter().position(|s| *s == name) {
                        Some(i) => i as u32,
                        None => {
                            self.m.sync_scopes.push(name);
                            (self.m.sync_scopes.len() - 1) as u32
                        }
                    };
                    self.sync_scope_map.push(id);
                }
            }
        }
    }

    /// The context sync scope for a bitcode sync scope id (`getDecodedSyncScopeID`).
    pub fn sync_scope(&self, raw: u64) -> Result<u32> {
        if self.sync_scope_map.is_empty() {
            // Bitcode without the block uses the context ids (0 singlethread, 1 system) directly.
            return match raw {
                0 | 1 => Ok(raw as u32),
                _ => self.malformed(format!("sync scope {} without a sync scope names block", raw)),
            };
        }
        match self.sync_scope_map.get(raw as usize) {
            Some(&s) => Ok(s),
            None => self.malformed(format!("invalid sync scope id {}", raw)),
        }
    }

    /// The module-level value symbol table: function block offsets (v2) or global names and offsets (v1).
    fn read_module_vst(&mut self, at: Option<u64>) -> Result<()> {
        let _ = at;
        self.cur.enter_block(VALUE_SYMTAB_BLOCK_ID)?;
        loop {
            match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block in the value symbol table"),
                Entry::Record(a) => {
                    let r = self.cur.read_record(a)?;
                    match r.code {
                        VST_CODE_FNENTRY => {
                            if r.ops.len() < 2 {
                                return self.malformed("invalid FNENTRY record");
                            }
                            let slot = self.values.get(r.ops[0] as usize).map(|e| e.slot);
                            let f = match slot {
                                Some(Slot::Global(GlobalId::Function(f))) => f,
                                Some(Slot::Global(_)) if !self.use_strtab => {
                                    // Older writers emitted offsets for aliases of functions; LLVM ignores them.
                                    return self.unsupported("FNENTRY for an alias");
                                }
                                _ => return self.malformed("FNENTRY for a value that is not a function"),
                            };
                            if !self.use_strtab {
                                let name = self.record_string(&r.ops[2..])?;
                                self.name_global(GlobalId::Function(f), name)?;
                            }
                            // Offsets count 32-bit words from the bitcode magic.
                            if self.function_block_at.insert(f, r.ops[1] * 32).is_some() {
                                return self.malformed("two FNENTRY records for one function");
                            }
                        }
                        VST_CODE_ENTRY => {
                            if self.use_strtab {
                                return self.malformed("ENTRY record in a module symbol table that has a string table");
                            }
                            if r.ops.is_empty() {
                                return self.malformed("invalid ENTRY record");
                            }
                            let g = match self.values.get(r.ops[0] as usize).map(|e| e.slot) {
                                Some(Slot::Global(g)) => g,
                                _ => return self.malformed("module symbol table entry for a value that is not global"),
                            };
                            let name = self.record_string(&r.ops[1..])?;
                            self.name_global(g, name)?;
                        }
                        c => return self.malformed(format!("record {} in the module value symbol table", c)),
                    }
                }
            }
        }
    }

    fn name_global(&mut self, g: GlobalId, name: String) -> Result<()> {
        if name.contains('\0') {
            return self.malformed("value name with a NUL character");
        }
        let common = match g {
            GlobalId::Var(i) => &mut self.m.globals[i as usize].common,
            GlobalId::Function(i) => &mut self.m.functions[i as usize].common,
            GlobalId::Alias(i) => &mut self.m.aliases[i as usize].common,
            GlobalId::IFunc(i) => &mut self.m.ifuncs[i as usize].common,
        };
        common.name = name;
        Ok(())
    }

    /// After everything is read: comdats, intrinsic and attribute upgrades, metadata clean-up.
    fn finish(&mut self) -> Result<()> {
        // Implicit comdats of old linkages are named after their object (`recordValue`), on targets with comdats.
        if !self.implicit_comdat.is_empty() {
            return self.unsupported("implicit comdats of pre-3.8 linkage encodings");
        }
        upgrade::finish(self)
    }
}
