//! The LLVM bitstream container: the bit-level encoding under LLVM bitcode.
//!
//! Reference: "LLVM Bitcode File Format" (`llvm/docs/BitCodeFormat.rst`) and the reader in
//! `llvm/include/llvm/Bitstream/BitstreamReader.h` and `llvm/lib/Bitstream/Reader/BitstreamReader.cpp`, release
//! 21.1.8. A stream is a sequence of little-endian 32-bit words read as a bit string from the least significant bit
//! of each byte. It holds nested blocks; each block declares the width of its abbreviation ids, may define
//! abbreviations (record layouts), and holds records: a code and a list of unsigned operands, possibly with a blob.
//!
//! The reader here validates everything the format defines and reports every violation as an [`Error`] with the bit
//! position; it never panics on malformed input. It checks two things LLVM's reader does not: a block's `END_BLOCK`
//! must end exactly where the block's length word says it does, and a record's abbreviation must name an encoding
//! for every operand (LLVM asserts some of these instead of reporting them).

use std::fmt;
use std::rc::Rc;

/// `END_BLOCK`, `ENTER_SUBBLOCK`, `DEFINE_ABBREV` and `UNABBREV_RECORD`: the abbreviation ids every block reserves.
pub const END_BLOCK: u32 = 0;
pub const ENTER_SUBBLOCK: u32 = 1;
pub const DEFINE_ABBREV: u32 = 2;
pub const UNABBREV_RECORD: u32 = 3;
/// The first abbreviation id a block can define.
pub const FIRST_APPLICATION_ABBREV: u32 = 4;

/// `BLOCKINFO_BLOCK_ID`: the block that defines abbreviations for other blocks.
pub const BLOCKINFO_BLOCK_ID: u32 = 0;
/// `BLOCKINFO_CODE_SETBID`, `BLOCKINFO_CODE_BLOCKNAME`, `BLOCKINFO_CODE_SETRECORDNAME`.
pub const BLOCKINFO_CODE_SETBID: u32 = 1;
pub const BLOCKINFO_CODE_BLOCKNAME: u32 = 2;
pub const BLOCKINFO_CODE_SETRECORDNAME: u32 = 3;

/// Widths fixed by the format: block ids are VBR-8, code widths VBR-4, block lengths 32-bit words.
const BLOCK_ID_WIDTH: u32 = 8;
const CODE_LEN_WIDTH: u32 = 4;
const BLOCK_SIZE_WIDTH: u32 = 32;
/// The largest fixed or VBR chunk an abbreviation may declare (`BitstreamCursor::MaxChunkSize`).
pub const MAX_CHUNK_SIZE: u64 = 32;

/// A malformed stream: what is wrong and the bit offset where it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Bit offset from the start of the stream.
    pub bit: u64,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bitstream error at bit {}: {}", self.bit, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

/// Reads bits from a byte slice.
#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next bit to read.
    pos: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> BitReader<'a> {
        BitReader { data, pos: 0 }
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The position of the next bit.
    pub fn position(&self) -> u64 {
        self.pos
    }

    pub fn len_bits(&self) -> u64 {
        self.data.len() as u64 * 8
    }

    pub fn at_end(&self) -> bool {
        self.pos >= self.len_bits()
    }

    pub fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(Error { bit: self.pos, message: message.into() })
    }

    /// Moves to `bit`, which may be the end of the data but not beyond it.
    pub fn seek(&mut self, bit: u64) -> Result<()> {
        if bit > self.len_bits() {
            return self.error(format!("seek to bit {} beyond the end of the stream ({} bits)", bit, self.len_bits()));
        }
        self.pos = bit;
        Ok(())
    }

    /// Reads `width` bits (0 to 64) as an unsigned value.
    pub fn read(&mut self, width: u32) -> Result<u64> {
        if width > 64 {
            return self.error(format!("cannot read {} bits at once", width));
        }
        if width == 0 {
            return Ok(0);
        }
        let end = self.pos.checked_add(width as u64).filter(|&e| e <= self.len_bits());
        if end.is_none() {
            return self.error(format!("unexpected end of stream reading {} bits", width));
        }
        let mut value = 0u64;
        let mut done = 0u32;
        while done < width {
            let byte = self.data[(self.pos / 8) as usize] as u64;
            let shift = (self.pos % 8) as u32;
            let take = (8 - shift).min(width - done);
            let bits = (byte >> shift) & ((1u64 << take) - 1);
            value |= bits << done;
            done += take;
            self.pos += take as u64;
        }
        Ok(value)
    }

    /// Reads a variable-width integer with `width`-bit chunks whose value fits in 32 bits (`ReadVBR`).
    pub fn read_vbr32(&mut self, width: u32) -> Result<u32> {
        let start = self.pos;
        let v = self.read_vbr_bounded(width, 32)?;
        if v > u32::MAX as u64 {
            return Err(Error { bit: start, message: "VBR value does not fit in 32 bits".into() });
        }
        Ok(v as u32)
    }

    /// Reads a variable-width integer with `width`-bit chunks whose value fits in 64 bits (`ReadVBR64`).
    pub fn read_vbr64(&mut self, width: u32) -> Result<u64> {
        self.read_vbr_bounded(width, 64)
    }

    fn read_vbr_bounded(&mut self, width: u32, limit: u32) -> Result<u64> {
        if !(2..=32).contains(&width) {
            // A 1-bit chunk has no payload bits; LLVM asserts 1 <= width <= 32, and a VBR(1) chunk can only encode
            // zero or loop forever.
            return self.error(format!("invalid VBR chunk width {}", width));
        }
        let start = self.pos;
        let hi = 1u64 << (width - 1);
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let piece = self.read(width)?;
            let payload = piece & (hi - 1);
            // shift < limit here. Payload bits that would land at or above `limit` do not fit the value; LLVM drops
            // them, which no writer relies on, so they are reported.
            let room = limit - shift;
            if room < 64 && (payload >> room) != 0 {
                return Err(Error { bit: start, message: format!("VBR value does not fit in {} bits", limit) });
            }
            result |= payload << shift;
            if piece & hi == 0 {
                return Ok(result);
            }
            shift += width - 1;
            if shift >= limit {
                return Err(Error { bit: start, message: "unterminated VBR value".into() });
            }
        }
    }

    /// Skips to the next multiple of 32 bits.
    pub fn align32(&mut self) -> Result<()> {
        let aligned = (self.pos + 31) & !31;
        // Alignment may land exactly on the end of the data; data shorter than the padding is malformed.
        if aligned > self.len_bits() {
            return self.error("unexpected end of stream in 32-bit alignment padding");
        }
        self.pos = aligned;
        Ok(())
    }
}

/// One operand of an abbreviation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbbrevOp {
    /// A value fixed by the abbreviation.
    Literal(u64),
    /// A fixed-width field of 1 to 32 bits.
    Fixed(u32),
    /// A VBR field with chunks of 2 to 32 bits.
    Vbr(u32),
    /// A count (VBR-6) followed by that many elements, each in the encoding of the next operand, which is the
    /// abbreviation's last.
    Array,
    /// A 6-bit character from `[a-zA-Z0-9._]`.
    Char6,
    /// A byte count (VBR-6), 32-bit alignment, the bytes, and alignment again.
    Blob,
}

/// An abbreviation: the layout of one record form, its first operand being the record code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Abbrev {
    pub ops: Vec<AbbrevOp>,
}

/// Decodes a `Char6` value (`BitCodeAbbrevOp::DecodeChar6`).
pub fn decode_char6(v: u64) -> u8 {
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"[(v & 63) as usize]
}

/// A record: its code, operands and blob.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Record<'a> {
    pub code: u32,
    pub ops: Vec<u64>,
    /// The blob operand, when the record's abbreviation has one.
    pub blob: Option<&'a [u8]>,
    /// The abbreviation id the record was written with ([`UNABBREV_RECORD`] for none).
    pub abbrev: u32,
}

/// What [`Cursor::advance`] found next in the current block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// `ENTER_SUBBLOCK` with this block id; call [`Cursor::enter_block`] or [`Cursor::skip_block`].
    SubBlock(u32),
    /// The current block ended; the cursor is back in its parent.
    EndBlock,
    /// A record with this abbreviation id; call [`Cursor::read_record`].
    Record(u32),
}

/// Abbreviations a `BLOCKINFO` block defines for other blocks, and the names it gives.
#[derive(Clone, Debug, Default)]
pub struct BlockInfo {
    blocks: Vec<(u32, Vec<Rc<Abbrev>>)>,
}

impl BlockInfo {
    pub fn abbrevs(&self, block_id: u32) -> &[Rc<Abbrev>] {
        self.blocks.iter().find(|(id, _)| *id == block_id).map(|(_, a)| a.as_slice()).unwrap_or(&[])
    }

    fn abbrevs_mut(&mut self, block_id: u32) -> &mut Vec<Rc<Abbrev>> {
        if let Some(i) = self.blocks.iter().position(|(id, _)| *id == block_id) {
            &mut self.blocks[i].1
        } else {
            self.blocks.push((block_id, Vec::new()));
            &mut self.blocks.last_mut().unwrap().1
        }
    }
}

struct Scope {
    block_id: u32,
    code_width: u32,
    abbrevs: Vec<Rc<Abbrev>>,
    /// Bit position one past the block's last word.
    end: u64,
}

/// Walks the blocks and records of a stream.
pub struct Cursor<'a> {
    reader: BitReader<'a>,
    code_width: u32,
    abbrevs: Vec<Rc<Abbrev>>,
    /// Enclosing blocks; the innermost last. The outer level (the stream itself) is not in it.
    scope: Vec<Scope>,
    blockinfo: BlockInfo,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of `data`, at the top level of the stream (abbreviation width 2).
    pub fn new(data: &'a [u8]) -> Cursor<'a> {
        Cursor {
            reader: BitReader::new(data),
            code_width: 2,
            abbrevs: Vec::new(),
            scope: Vec::new(),
            blockinfo: BlockInfo::default(),
        }
    }

    pub fn reader(&mut self) -> &mut BitReader<'a> {
        &mut self.reader
    }

    pub fn position(&self) -> u64 {
        self.reader.position()
    }

    pub fn at_end(&self) -> bool {
        self.reader.at_end()
    }

    pub fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        self.reader.error(message)
    }

    /// The id of the innermost block the cursor is in, if any.
    pub fn block_id(&self) -> Option<u32> {
        self.scope.last().map(|s| s.block_id)
    }

    /// Nesting depth: 0 at the top level.
    pub fn depth(&self) -> usize {
        self.scope.len()
    }

    pub fn blockinfo(&self) -> &BlockInfo {
        &self.blockinfo
    }

    /// Moves to a saved position at the top level or inside the current block. The caller is responsible for the
    /// position being an entry boundary.
    pub fn seek(&mut self, bit: u64) -> Result<()> {
        self.reader.seek(bit)
    }

    /// Reads the next entry. At the top level, the end of the data is an [`Entry::EndBlock`]; inside a block it is an
    /// error.
    pub fn advance(&mut self) -> Result<Entry> {
        loop {
            if self.scope.is_empty() && self.reader.at_end() {
                return Ok(Entry::EndBlock);
            }
            if let Some(s) = self.scope.last()
                && self.reader.position() >= s.end
            {
                return self.error(format!("block {} runs past its declared length without END_BLOCK", s.block_id));
            }
            let code = self.reader.read(self.code_width)? as u32;
            match code {
                END_BLOCK => {
                    if self.scope.is_empty() {
                        return self.error("END_BLOCK at the top level of the stream");
                    }
                    self.reader.align32()?;
                    let s = self.scope.pop().unwrap();
                    if self.reader.position() != s.end {
                        return self.error(format!(
                            "block {} ends at bit {}, its length word says bit {}",
                            s.block_id,
                            self.reader.position(),
                            s.end
                        ));
                    }
                    self.code_width = s.code_width;
                    self.abbrevs = s.abbrevs;
                    return Ok(Entry::EndBlock);
                }
                ENTER_SUBBLOCK => {
                    let id = self.reader.read_vbr32(BLOCK_ID_WIDTH)?;
                    return Ok(Entry::SubBlock(id));
                }
                DEFINE_ABBREV => {
                    if self.scope.is_empty() {
                        return self.error("DEFINE_ABBREV at the top level of the stream");
                    }
                    let a = self.read_abbrev()?;
                    self.abbrevs.push(Rc::new(a));
                    // The BLOCKINFO block handles its own definitions (see read_blockinfo).
                }
                id => {
                    if self.scope.is_empty() {
                        return self.error(format!("record (abbreviation id {}) at the top level of the stream", id));
                    }
                    return Ok(Entry::Record(id));
                }
            }
        }
    }

    /// Having read [`Entry::SubBlock`]`(block_id)`, enters the block and returns its length in 32-bit words.
    pub fn enter_block(&mut self, block_id: u32) -> Result<u64> {
        let width = self.reader.read_vbr32(CODE_LEN_WIDTH)?;
        if width as u64 > MAX_CHUNK_SIZE {
            return self.error(format!("block {} declares {}-bit abbreviation ids (at most 32)", block_id, width));
        }
        if width == 0 {
            return self.error(format!("block {} declares 0-bit abbreviation ids", block_id));
        }
        self.reader.align32()?;
        let words = self.reader.read(BLOCK_SIZE_WIDTH)?;
        let end = self.reader.position() + words * 32;
        if end > self.reader.len_bits() {
            return self.error(format!("block {} of {} words runs past the end of the stream", block_id, words));
        }
        if let Some(parent) = self.scope.last()
            && end > parent.end
        {
            return self.error(format!("block {} runs past the end of its parent block {}", block_id, parent.block_id));
        }
        let mut abbrevs: Vec<Rc<Abbrev>> = self.blockinfo.abbrevs(block_id).to_vec();
        std::mem::swap(&mut abbrevs, &mut self.abbrevs);
        self.scope.push(Scope { block_id, code_width: self.code_width, abbrevs, end });
        self.code_width = width;
        Ok(words)
    }

    /// Having read [`Entry::SubBlock`], skips the block by its length word.
    pub fn skip_block(&mut self) -> Result<()> {
        self.reader.read_vbr32(CODE_LEN_WIDTH)?;
        self.reader.align32()?;
        let words = self.reader.read(BLOCK_SIZE_WIDTH)?;
        let end = self.reader.position() + words * 32;
        if end > self.reader.len_bits() {
            return self.error(format!("block of {} words runs past the end of the stream", words));
        }
        if let Some(parent) = self.scope.last()
            && end > parent.end
        {
            return self.error(format!("block runs past the end of its parent block {}", parent.block_id));
        }
        self.reader.seek(end)
    }

    fn read_abbrev(&mut self) -> Result<Abbrev> {
        let n = self.reader.read_vbr32(5)?;
        if n == 0 {
            return self.error("abbreviation with no operands");
        }
        let mut ops = Vec::new();
        let mut i = 0;
        while i < n {
            let literal = self.reader.read(1)? == 1;
            if literal {
                ops.push(AbbrevOp::Literal(self.reader.read_vbr64(8)?));
                i += 1;
                continue;
            }
            let enc = self.reader.read(3)?;
            let op = match enc {
                1 | 2 => {
                    let data = self.reader.read_vbr64(5)?;
                    if data == 0 {
                        // Fixed(0) and VBR(0) read as a literal zero.
                        AbbrevOp::Literal(0)
                    } else if data > MAX_CHUNK_SIZE {
                        return self.error(format!("abbreviation field of {} bits (at most 32)", data));
                    } else if enc == 1 {
                        AbbrevOp::Fixed(data as u32)
                    } else {
                        if data < 2 {
                            return self.error("VBR abbreviation field with 1-bit chunks");
                        }
                        AbbrevOp::Vbr(data as u32)
                    }
                }
                3 => AbbrevOp::Array,
                4 => AbbrevOp::Char6,
                5 => AbbrevOp::Blob,
                _ => return self.error(format!("invalid abbreviation operand encoding {}", enc)),
            };
            ops.push(op);
            i += 1;
        }
        // Validate the shape once, so records never meet an inconsistent abbreviation.
        match ops[0] {
            AbbrevOp::Array | AbbrevOp::Blob => return self.error("abbreviation starts with an array or a blob"),
            _ => {}
        }
        for (k, op) in ops.iter().enumerate() {
            if *op == AbbrevOp::Array {
                if k + 2 != ops.len() {
                    return self.error("array operand is not second to last in its abbreviation");
                }
                match ops[k + 1] {
                    AbbrevOp::Fixed(_) | AbbrevOp::Vbr(_) | AbbrevOp::Char6 => {}
                    _ => return self.error("array element encoding must be fixed, VBR or char6"),
                }
            }
        }
        Ok(Abbrev { ops })
    }

    /// Reads a record announced by [`Entry::Record`]`(abbrev_id)`.
    pub fn read_record(&mut self, abbrev_id: u32) -> Result<Record<'a>> {
        let limit = self.reader.len_bits();
        if abbrev_id == UNABBREV_RECORD {
            let code = self.reader.read_vbr32(6)?;
            let n = self.reader.read_vbr32(6)? as u64;
            if n >= limit {
                return self.error("record operand count is not plausible");
            }
            let mut ops = Vec::with_capacity(n.min(1 << 16) as usize);
            for _ in 0..n {
                ops.push(self.reader.read_vbr64(6)?);
            }
            return Ok(Record { code, ops, blob: None, abbrev: abbrev_id });
        }
        let index = abbrev_id - FIRST_APPLICATION_ABBREV;
        let abbrev = match self.abbrevs.get(index as usize) {
            Some(a) => a.clone(),
            None => return self.error(format!("undefined abbreviation id {}", abbrev_id)),
        };
        let code = match abbrev.ops[0] {
            AbbrevOp::Literal(v) => v,
            op => self.read_scalar(op)?,
        };
        if code > u32::MAX as u64 {
            return self.error(format!("record code {} out of range", code));
        }
        let mut ops = Vec::new();
        let mut blob = None;
        let mut i = 1;
        while i < abbrev.ops.len() {
            let op = abbrev.ops[i];
            match op {
                AbbrevOp::Literal(v) => ops.push(v),
                AbbrevOp::Fixed(_) | AbbrevOp::Vbr(_) | AbbrevOp::Char6 => ops.push(self.read_scalar(op)?),
                AbbrevOp::Array => {
                    let n = self.reader.read_vbr32(6)? as u64;
                    if n >= limit {
                        return self.error("array length is not plausible");
                    }
                    let elt = abbrev.ops[i + 1];
                    ops.reserve(n.min(1 << 16) as usize);
                    for _ in 0..n {
                        ops.push(self.read_scalar(elt)?);
                    }
                    i += 1;
                }
                AbbrevOp::Blob => {
                    let n = self.reader.read_vbr32(6)? as u64;
                    self.reader.align32()?;
                    let start = self.reader.position();
                    let end = start + n * 8;
                    let padded = start + ((n + 3) & !3) * 8;
                    if padded > limit {
                        return self.error("blob runs past the end of the stream");
                    }
                    let bytes = &self.reader.data()[(start / 8) as usize..(end / 8) as usize];
                    self.reader.seek(padded)?;
                    if blob.is_some() {
                        return self.error("abbreviation with more than one blob");
                    }
                    blob = Some(bytes);
                }
            }
            i += 1;
        }
        if let Some(s) = self.scope.last()
            && self.reader.position() > s.end
        {
            return self.error(format!("record runs past the end of block {}", s.block_id));
        }
        Ok(Record { code: code as u32, ops, blob, abbrev: abbrev_id })
    }

    fn read_scalar(&mut self, op: AbbrevOp) -> Result<u64> {
        match op {
            AbbrevOp::Literal(v) => Ok(v),
            AbbrevOp::Fixed(w) => self.reader.read(w),
            AbbrevOp::Vbr(w) => self.reader.read_vbr64(w),
            AbbrevOp::Char6 => Ok(decode_char6(self.reader.read(6)?) as u64),
            AbbrevOp::Array | AbbrevOp::Blob => self.error("array or blob where a scalar is expected"),
        }
    }

    /// Having read [`Entry::SubBlock`]`(`[`BLOCKINFO_BLOCK_ID`]`)`, reads the block and installs its abbreviations
    /// for the blocks it names. Block and record names are validated and discarded: they only label dumps.
    pub fn read_blockinfo(&mut self) -> Result<()> {
        self.enter_block(BLOCKINFO_BLOCK_ID)?;
        let mut current: Option<u32> = None;
        loop {
            // Definitions in BLOCKINFO belong to the block named by the last SETBID, not to BLOCKINFO itself, so
            // they are read here rather than by advance().
            if let Some(s) = self.scope.last()
                && self.reader.position() >= s.end
            {
                return self.error("BLOCKINFO block runs past its declared length without END_BLOCK");
            }
            let code = self.reader.read(self.code_width)? as u32;
            match code {
                END_BLOCK => {
                    self.reader.align32()?;
                    let s = self.scope.pop().unwrap();
                    if self.reader.position() != s.end {
                        return self.error("BLOCKINFO block does not end where its length word says");
                    }
                    self.code_width = s.code_width;
                    self.abbrevs = s.abbrevs;
                    return Ok(());
                }
                ENTER_SUBBLOCK => return self.error("sub-block inside BLOCKINFO"),
                DEFINE_ABBREV => {
                    let a = self.read_abbrev()?;
                    match current {
                        Some(id) => self.blockinfo.abbrevs_mut(id).push(Rc::new(a)),
                        None => return self.error("abbreviation in BLOCKINFO before SETBID"),
                    }
                }
                id => {
                    let r = self.read_record(id)?;
                    match r.code {
                        BLOCKINFO_CODE_SETBID => {
                            if r.ops.is_empty() || r.ops[0] > u32::MAX as u64 {
                                return self.error("malformed SETBID record");
                            }
                            current = Some(r.ops[0] as u32);
                        }
                        BLOCKINFO_CODE_BLOCKNAME | BLOCKINFO_CODE_SETRECORDNAME => {
                            if current.is_none() {
                                return self.error("BLOCKINFO name record before SETBID");
                            }
                            if r.code == BLOCKINFO_CODE_SETRECORDNAME && r.ops.is_empty() {
                                return self.error("malformed SETRECORDNAME record");
                            }
                        }
                        c => return self.error(format!("unknown BLOCKINFO record code {}", c)),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes bits least significant first, as the format stores them.
    struct BitWriter {
        bytes: Vec<u8>,
        bit: u64,
    }

    impl BitWriter {
        fn new() -> BitWriter {
            BitWriter { bytes: Vec::new(), bit: 0 }
        }
        fn emit(&mut self, value: u64, width: u32) {
            for i in 0..width {
                if self.bit.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                if (value >> i) & 1 == 1 {
                    *self.bytes.last_mut().unwrap() |= 1 << (self.bit % 8);
                }
                self.bit += 1;
            }
        }
        fn vbr(&mut self, mut value: u64, width: u32) {
            let hi = 1u64 << (width - 1);
            loop {
                if value < hi {
                    self.emit(value, width);
                    return;
                }
                self.emit((value & (hi - 1)) | hi, width);
                value >>= width - 1;
            }
        }
        fn align32(&mut self) {
            while !self.bit.is_multiple_of(32) {
                self.emit(0, 1);
            }
        }
    }

    #[test]
    fn reads_fixed_and_vbr_fields() {
        let mut w = BitWriter::new();
        w.emit(5, 3);
        w.vbr(1000, 6);
        w.vbr(u64::MAX, 8);
        w.emit(0x1234_5678_9abc, 48);
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.read(3).unwrap(), 5);
        assert_eq!(r.read_vbr32(6).unwrap(), 1000);
        assert_eq!(r.read_vbr64(8).unwrap(), u64::MAX);
        assert_eq!(r.read(48).unwrap(), 0x1234_5678_9abc);
    }

    #[test]
    fn vbr_overflow_is_an_error() {
        let mut w = BitWriter::new();
        w.vbr(1u64 << 40, 6);
        let mut r = BitReader::new(&w.bytes);
        assert!(r.read_vbr32(6).is_err());
        // An endless continuation is an error, not a hang.
        let bytes = [0xff; 64];
        let mut r = BitReader::new(&bytes);
        assert!(r.read_vbr64(4).is_err());
    }

    #[test]
    fn reading_past_the_end_is_an_error() {
        let bytes = [0xab];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(4).unwrap(), 0xb);
        assert!(r.read(5).is_err());
    }

    /// A block with an abbreviation using every encoding, an unabbreviated record and a nested block.
    #[test]
    fn walks_blocks_and_records() {
        let mut w = BitWriter::new();
        // Top level: abbreviation width 2. ENTER_SUBBLOCK id 8, code width 3.
        w.emit(ENTER_SUBBLOCK as u64, 2);
        w.vbr(8, 8);
        w.vbr(3, 4);
        w.align32();
        let len_at = w.bytes.len();
        w.emit(0, 32);
        let body_start = w.bit;
        // DEFINE_ABBREV: [literal 7, fixed 4, vbr 6, array of char6, ...] -> four ops then a separate blob abbrev.
        w.emit(DEFINE_ABBREV as u64, 3);
        w.vbr(5, 5);
        w.emit(1, 1);
        w.vbr(7, 8); // literal code 7
        w.emit(0, 1);
        w.emit(1, 3);
        w.vbr(4, 5); // fixed(4)
        w.emit(0, 1);
        w.emit(2, 3);
        w.vbr(6, 5); // vbr(6)
        w.emit(0, 1);
        w.emit(3, 3); // array
        w.emit(0, 1);
        w.emit(4, 3); // char6
        // Record with that abbreviation (id 4): fixed 9, vbr 300, array "a_Z".
        w.emit(4, 3);
        w.emit(9, 4);
        w.vbr(300, 6);
        w.vbr(3, 6);
        w.emit(0, 6);
        w.emit(63, 6);
        w.emit(51, 6);
        // Unabbreviated record: code 2, ops [1, 2].
        w.emit(UNABBREV_RECORD as u64, 3);
        w.vbr(2, 6);
        w.vbr(2, 6);
        w.vbr(1, 6);
        w.vbr(2, 6);
        // END_BLOCK
        w.emit(END_BLOCK as u64, 3);
        w.align32();
        let words = (w.bit - body_start) / 32;
        w.bytes[len_at..len_at + 4].copy_from_slice(&(words as u32).to_le_bytes());

        let mut c = Cursor::new(&w.bytes);
        assert_eq!(c.advance().unwrap(), Entry::SubBlock(8));
        c.enter_block(8).unwrap();
        let Entry::Record(id) = c.advance().unwrap() else { panic!() };
        let r = c.read_record(id).unwrap();
        assert_eq!(r.code, 7);
        assert_eq!(r.ops, vec![9, 300, b'a' as u64, b'_' as u64, b'Z' as u64]);
        let Entry::Record(id) = c.advance().unwrap() else { panic!() };
        let r = c.read_record(id).unwrap();
        assert_eq!((r.code, r.ops), (2, vec![1, 2]));
        assert_eq!(c.advance().unwrap(), Entry::EndBlock);
        assert_eq!(c.advance().unwrap(), Entry::EndBlock);
        assert!(c.at_end());
    }

    #[test]
    fn block_length_mismatch_is_an_error() {
        let mut w = BitWriter::new();
        w.emit(ENTER_SUBBLOCK as u64, 2);
        w.vbr(8, 8);
        w.vbr(3, 4);
        w.align32();
        w.emit(2, 32); // claims two words
        w.emit(END_BLOCK as u64, 3);
        w.align32();
        w.emit(0, 32);
        let mut c = Cursor::new(&w.bytes);
        assert_eq!(c.advance().unwrap(), Entry::SubBlock(8));
        c.enter_block(8).unwrap();
        assert!(c.advance().is_err());
    }
}
