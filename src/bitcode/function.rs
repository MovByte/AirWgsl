//! Function bodies (`parseFunctionBody` in `BitcodeReader.cpp`).
//!
//! Value numbers continue the module's: arguments, then the function's constants, then the instructions with
//! results. Operands are encoded relative to the next value number (module version 1 and later); a reference to a
//! value not yet defined carries its type and is resolved when the function ends.

use std::collections::HashMap;

use super::codes::*;
use super::constants::{self, Use};
use super::{Reader, Result, Slot, VEntry, FWD};
use crate::bitstream::Entry;
use crate::ir::*;

/// The local value names cap (`NonGlobalValueMaxNameSize`).
const LOCAL_NAME_MAX: usize = 1024;

pub(crate) struct FnState {
    f: u32,
    module_values: usize,
    module_raw_consts: usize,
    module_md: usize,
    nblocks: usize,
    cur_bb: Option<usize>,
    /// Every instruction made from a record (and upgrade temporaries), in order: METADATA_ATTACHMENT indexes it.
    inst_list: Vec<InstId>,
    /// Forward-referenced value numbers and the type their uses expect.
    pending: HashMap<u32, TypeId>,
    bundles: Vec<Bundle>,
    /// Names given so far, for LLVM's renaming on collision.
    names: HashMap<String, ()>,
    last_unique: u32,
    /// `USELIST_CODE_BB` records, in order: (block, indexes).
    block_use_orders: Vec<(usize, Vec<u64>)>,
    /// Whether a use-list block has been read; LLVM writes it after the instructions.
    seen_use_lists: bool,
}

impl<'a> Reader<'a> {
    fn next_value(&self) -> u32 {
        self.values.len() as u32
    }

    /// The value and bitcode type of a defined value number.
    fn defined_value(&mut self, valno: u32) -> Result<(Value, u32)> {
        let e = self.values[valno as usize];
        let v = match e.slot {
            Slot::Global(g) => Value::Global(g),
            Slot::Arg(a) => Value::Arg(a),
            Slot::Inst(i) => Value::Inst(i),
            Slot::Const(c) => self.materialize_const_use(c, Use::Instruction)?,
        };
        Ok((v, e.bc_ty))
    }

    fn local_type(&mut self, st: &FnState, v: Value) -> TypeId {
        self.m.value_type(st.f as usize, v)
    }

    fn decode_relative(&self, raw: u64, signed: bool) -> Result<u32> {
        let raw = if signed { constants::decode_sign_rotated(raw) } else { raw };
        if !signed && raw > u32::MAX as u64 {
            return self.malformed("value number out of range");
        }
        let raw32 = raw as u32;
        Ok(if self.relative_ids { self.next_value().wrapping_sub(raw32) } else { raw32 })
    }

    fn forward(&mut self, st: &mut FnState, valno: u32, ty: TypeId) -> Result<Value> {
        if valno >= FWD {
            return self.malformed("value number out of range");
        }
        match st.pending.get(&valno) {
            Some(&t) if t != ty => return self.malformed("forward reference used with two types"),
            _ => {
                st.pending.insert(valno, ty);
            }
        }
        Ok(Value::Inst(InstId(FWD | valno)))
    }

    /// `getValueTypePair`: a value, with its type in the record when it is a forward reference.
    fn value_type_pair(&mut self, st: &mut FnState, ops: &[u64], i: &mut usize) -> Result<(Value, TypeId, u32)> {
        if *i >= ops.len() {
            return self.malformed("truncated instruction record");
        }
        let valno = self.decode_relative(ops[*i], false)?;
        *i += 1;
        if valno < self.next_value() {
            let (v, bc) = self.defined_value(valno)?;
            let t = self.local_type(st, v);
            return Ok((v, t, bc));
        }
        if *i >= ops.len() {
            return self.malformed("forward reference without a type");
        }
        let bc = ops[*i];
        *i += 1;
        let ty = self.ty(bc)?;
        let v = self.forward(st, valno, ty)?;
        Ok((v, ty, bc as u32))
    }

    /// `getValue`/`popValue`: a value whose type the instruction implies.
    fn typed_value(&mut self, st: &mut FnState, raw: u64, ty: TypeId, signed: bool) -> Result<Value> {
        let valno = self.decode_relative(raw, signed)?;
        if self.m.types.is_metadata(ty) {
            let md = self.md_ref(valno as u64)?;
            return Ok(Value::Metadata(md));
        }
        if valno < self.next_value() {
            let (v, _) = self.defined_value(valno)?;
            if self.local_type(st, v) != ty {
                return self.malformed("operand has the wrong type");
            }
            return Ok(v);
        }
        self.forward(st, valno, ty)
    }

    fn block(&self, st: &FnState, raw: u64) -> Result<BlockId> {
        if (raw as usize) < st.nblocks {
            Ok(BlockId(raw as u32))
        } else {
            self.malformed(format!("invalid basic block {}", raw))
        }
    }

    fn fmf(raw: u64) -> FastMath {
        // UnsafeAlgebra (bit 0) is the legacy "fast": every flag.
        let mut f = 0u8;
        if raw & 1 != 0 {
            f = FastMath::ALL;
        }
        if raw & (1 << 7) != 0 {
            f |= FastMath::REASSOC;
        }
        if raw & (1 << 1) != 0 {
            f |= FastMath::NNAN;
        }
        if raw & (1 << 2) != 0 {
            f |= FastMath::NINF;
        }
        if raw & (1 << 3) != 0 {
            f |= FastMath::NSZ;
        }
        if raw & (1 << 4) != 0 {
            f |= FastMath::ARCP;
        }
        if raw & (1 << 5) != 0 {
            f |= FastMath::CONTRACT;
        }
        if raw & (1 << 6) != 0 {
            f |= FastMath::AFN;
        }
        FastMath(f)
    }

    /// Whether a phi, select or call of type `ty` is an `FPMathOperator` (`isSupportedFloatingPointType`).
    fn is_fp_math_type(&self, ty: TypeId) -> bool {
        let t = &self.m.types;
        if t.is_fp_or_fp_vector(ty) {
            return true;
        }
        match t.get(ty) {
            Type::LiteralStruct { elems, .. } => {
                !elems.is_empty() && elems.iter().all(|e| *e == elems[0]) && t.is_fp_or_fp_vector(elems[0])
            }
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

    fn push_inst(&mut self, st: &mut FnState, ty: TypeId, op: Op, bc_ty: Option<u32>, numbered: bool) -> Result<InstId> {
        let bb = match st.cur_bb {
            Some(b) => b,
            None => return self.malformed("instruction outside any basic block"),
        };
        let terminator = op.is_terminator();
        let f = &mut self.m.functions[st.f as usize];
        let id = InstId(f.insts.len() as u32);
        f.insts.push(Instruction { ty, name: None, op, md: Vec::new(), dbg: None });
        f.blocks[bb].insts.push(id);
        st.inst_list.push(id);
        if numbered && !self.m.types.is_void(ty) {
            let valno = self.next_value();
            if let Some(t) = st.pending.get(&valno) {
                if *t != ty {
                    return self.malformed("assigned value does not match the type of its forward reference");
                }
            }
            let bc = match bc_ty {
                Some(b) => b,
                None => self.virtual_type(ty, vec![]),
            };
            self.values.push(VEntry { slot: Slot::Inst(id), bc_ty: bc });
        }
        if terminator {
            st.cur_bb = if bb + 1 < st.nblocks { Some(bb + 1) } else { None };
        }
        Ok(id)
    }

    pub(super) fn read_function_body(&mut self, f: u32) -> Result<()> {
        self.cur.enter_block(FUNCTION_BLOCK_ID)?;
        let mut st = FnState {
            f,
            module_values: self.values.len(),
            module_raw_consts: self.raw_consts.len(),
            module_md: self.md.list.len(),
            nblocks: 0,
            cur_bb: None,
            inst_list: Vec::new(),
            pending: HashMap::new(),
            bundles: Vec::new(),
            names: HashMap::new(),
            last_unique: 0,
            block_use_orders: Vec::new(),
            seen_use_lists: false,
        };
        let fty_bc = self.m.functions[f as usize].bitcode_type;
        let nargs = self.m.functions[f as usize].args.len();
        for a in 0..nargs {
            let bc = match fty_bc.and_then(|b| self.contained(b, a + 1)) {
                Some(b) => b,
                None => {
                    let t = self.m.functions[f as usize].args[a].ty;
                    self.virtual_type(t, vec![])
                }
            };
            self.values.push(VEntry { slot: Slot::Arg(a as u32), bc_ty: bc });
        }
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => break,
                Entry::SubBlock(id) => {
                    match id {
                        CONSTANTS_BLOCK_ID => self.read_constants(Some(f))?,
                        VALUE_SYMTAB_BLOCK_ID => self.read_function_vst(&mut st)?,
                        METADATA_ATTACHMENT_ID => {
                            let list = st.inst_list.clone();
                            self.read_attachments(f, &list)?;
                        }
                        METADATA_BLOCK_ID => self.read_metadata_block(Some(f))?,
                        USELIST_BLOCK_ID => {
                            self.read_use_lists(Some(&mut st))?;
                            st.seen_use_lists = true;
                        }
                        other => {
                            return self.unsupported(format!("block {} ({}) in a function block", other, block_name(other)));
                        }
                    }
                    continue;
                }
                Entry::Record(a) => a,
            };
            if st.seen_use_lists {
                // The orders apply to the uses that exist when the block is read; later instructions would add uses.
                return self.unsupported("instruction records after the function's use-list block");
            }
            let r = self.cur.read_record(a)?;
            self.function_body_record(&mut st, r.code, &r.ops)?;
        }
        if !st.bundles.is_empty() {
            return self.malformed("operand bundles found with no consumer");
        }
        if st.nblocks == 0 {
            return self.malformed("function body without DECLAREBLOCKS");
        }
        if st.cur_bb.is_some() {
            return self.malformed("function body ends inside a basic block");
        }
        self.resolve_forward(&st)?;
        self.apply_block_use_orders(&st);
        // Value numbers, metadata ids and constants of the function end with it.
        self.values.truncate(st.module_values);
        self.raw_consts.truncate(st.module_raw_consts);
        self.md.list.truncate(st.module_md);
        Ok(())
    }

    /// Replaces forward references with the values they named.
    /// `parseUseLists`. Records for blocks are kept for [`Self::apply_block_use_orders`]; records for other values
    /// are checked and have no effect here: they order LLVM's in-memory use lists, which the IR does not keep and which
    /// mean nothing for the program.
    pub(crate) fn read_use_lists(&mut self, mut st: Option<&mut FnState>) -> Result<()> {
        self.cur.enter_block(USELIST_BLOCK_ID)?;
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(id) => return self.unsupported(format!("block {} ({}) in a use-list block", id, block_name(id))),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            let is_bb = match r.code {
                USELIST_CODE_DEFAULT => false,
                USELIST_CODE_BB => true,
                c => return self.unsupported(format!("use-list record {}", c)),
            };
            if r.ops.len() < 3 {
                return self.malformed("use-list record with fewer than two indexes");
            }
            let mut indexes = r.ops.clone();
            let id = indexes.pop().unwrap();
            if is_bb {
                let st = match st.as_deref_mut() {
                    Some(s) => s,
                    None => return self.unsupported("block use-list record outside a function"),
                };
                if id >= st.nblocks as u64 {
                    return self.malformed(format!("use-list record for block {} of {}", id, st.nblocks));
                }
                st.block_use_orders.push((id as usize, indexes));
            } else if id >= self.values.len() as u64 {
                return self.malformed(format!("use-list record for value {} of {}", id, self.values.len()));
            }
        }
    }

    /// Applies the block use-list records as LLVM does: a record whose index count differs from the block's number of
    /// uses is ignored; otherwise the uses are stably sorted by their indexes. A block's uses are its appearances as
    /// a successor of a terminator (`blockaddress`, the other user, is refused by this reader).
    fn apply_block_use_orders(&mut self, st: &FnState) {
        if st.block_use_orders.is_empty() {
            return;
        }
        let func = &self.m.functions[st.f as usize];
        let mut uses = vec![0usize; func.blocks.len()];
        for b in &func.blocks {
            for &i in &b.insts {
                for s in func.inst(i).op.successors() {
                    uses[s.0 as usize] += 1;
                }
            }
        }
        let func = &mut self.m.functions[st.f as usize];
        for (bb, indexes) in &st.block_use_orders {
            if indexes.len() != uses[*bb] {
                continue;
            }
            let block = &mut func.blocks[*bb];
            let mut order: Vec<u32> = block.use_order.take().unwrap_or_else(|| (0..uses[*bb] as u32).collect());
            let mut keyed: Vec<(u64, u32)> = indexes.iter().copied().zip(order.drain(..)).collect();
            keyed.sort_by_key(|k| k.0);
            block.use_order = Some(keyed.into_iter().map(|k| k.1).collect());
        }
    }

    fn resolve_forward(&mut self, st: &FnState) -> Result<()> {
        let f = st.f as usize;
        let mut map: HashMap<u32, Value> = HashMap::new();
        for &valno in st.pending.keys() {
            let e = match self.values.get(valno as usize) {
                Some(e) => *e,
                None => return self.malformed("never resolved value found in function"),
            };
            let v = match e.slot {
                Slot::Inst(i) => Value::Inst(i),
                _ => return self.malformed("forward reference to a value that is not an instruction"),
            };
            map.insert(valno, v);
        }
        let fix = |v: &mut Value| -> core::result::Result<(), ()> {
            if let Value::Inst(InstId(i)) = v {
                if *i & FWD != 0 {
                    match map.get(&(*i & !FWD)) {
                        Some(r) => *v = *r,
                        None => return Err(()),
                    }
                }
            }
            Ok(())
        };
        let mut ok = true;
        for inst in self.m.functions[f].insts.iter_mut() {
            visit_values_mut(&mut inst.op, &mut |v| {
                if fix(v).is_err() {
                    ok = false;
                }
            });
        }
        if !ok {
            return self.malformed("never resolved value found in function");
        }
        // Function-local metadata that referred forward.
        let pending = std::mem::take(&mut self.md.pending_local);
        for (md, mf, valno) in pending {
            if mf != st.f {
                self.md.pending_local.push((md, mf, valno));
                continue;
            }
            let v = match self.values.get(valno as usize).map(|e| e.slot) {
                Some(Slot::Inst(i)) => Value::Inst(i),
                _ => return self.malformed("metadata refers to a value never defined"),
            };
            if let Metadata::Local { ty, .. } = *self.m.md(md) {
                if self.m.functions[f].insts[match v {
                    Value::Inst(i) => i.0 as usize,
                    _ => unreachable!(),
                }]
                .ty
                    != ty
                {
                    return self.malformed("metadata value has the wrong type");
                }
                self.m.set_md(md, Metadata::Local { function: st.f, ty, value: v });
            }
        }
        Ok(())
    }

    fn read_function_vst(&mut self, st: &mut FnState) -> Result<()> {
        self.cur.enter_block(VALUE_SYMTAB_BLOCK_ID)?;
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => return Ok(()),
                Entry::SubBlock(_) => return self.malformed("sub-block inside a value symbol table"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            match r.code {
                VST_CODE_ENTRY => {
                    if r.ops.is_empty() {
                        return self.malformed("invalid VST entry record");
                    }
                    let name = self.record_string(&r.ops[1..])?;
                    let slot = match self.values.get(r.ops[0] as usize) {
                        Some(e) => e.slot,
                        None => return self.malformed("value symbol table entry for an undefined value"),
                    };
                    match slot {
                        Slot::Arg(a) => {
                            let n = self.local_name(st, name)?;
                            self.m.functions[st.f as usize].args[a as usize].name = n;
                        }
                        Slot::Inst(i) => {
                            let n = self.local_name(st, name)?;
                            self.m.functions[st.f as usize].insts[i.0 as usize].name = n;
                        }
                        _ => return self.malformed("function value symbol table names a value that is not local"),
                    }
                }
                VST_CODE_BBENTRY => {
                    if r.ops.is_empty() {
                        return self.malformed("invalid VST bbentry record");
                    }
                    let name = self.record_string(&r.ops[1..])?;
                    let b = self.block(st, r.ops[0])?;
                    let n = self.local_name(st, name)?;
                    self.m.functions[st.f as usize].blocks[b.0 as usize].name = n;
                }
                c => return self.malformed(format!("record {} in a function value symbol table", c)),
            }
        }
    }

    /// The name a local value gets (`Value::setName`): capped, and made unique within the function.
    fn local_name(&mut self, st: &mut FnState, mut name: String) -> Result<Option<String>> {
        if name.contains('\0') {
            return self.malformed("value name with a NUL character");
        }
        if name.is_empty() {
            return Ok(None);
        }
        if name.len() > LOCAL_NAME_MAX {
            let mut cut = LOCAL_NAME_MAX;
            while !name.is_char_boundary(cut) {
                cut -= 1;
            }
            name.truncate(cut.max(1));
        }
        if st.names.contains_key(&name) {
            let base = name.clone();
            loop {
                st.last_unique += 1;
                let candidate = format!("{}{}", base, st.last_unique);
                if !st.names.contains_key(&candidate) {
                    name = candidate;
                    break;
                }
            }
        }
        st.names.insert(name.clone(), ());
        Ok(Some(name))
    }

    fn function_body_record(&mut self, st: &mut FnState, code: u32, ops: &[u64]) -> Result<()> {
        let f = st.f;
        match code {
            FUNC_CODE_DECLAREBLOCKS => {
                if ops.is_empty() || ops[0] == 0 || st.nblocks != 0 {
                    return self.malformed("invalid DECLAREBLOCKS record");
                }
                if ops[0] > self.cur.reader().len_bits() {
                    return self.malformed("block count is not plausible");
                }
                st.nblocks = ops[0] as usize;
                self.m.functions[f as usize].blocks = vec![Block::default(); st.nblocks];
                st.cur_bb = Some(0);
            }
            FUNC_CODE_INST_UNOP => {
                let mut i = 0;
                let (v, ty, bc) = self.value_type_pair(st, ops, &mut i)?;
                if i >= ops.len() || ops[i] != 0 || !self.m.types.is_fp_or_fp_vector(ty) {
                    return self.malformed("invalid unary operator record");
                }
                i += 1;
                let fmf = if i < ops.len() { Self::fmf(ops[i]) } else { FastMath::default() };
                self.push_inst(st, ty, Op::FNeg { value: v, fmf }, Some(bc), true)?;
            }
            FUNC_CODE_INST_BINOP => {
                let mut i = 0;
                let (lhs, ty, bc) = self.value_type_pair(st, ops, &mut i)?;
                if i >= ops.len() {
                    return self.malformed("invalid binary operator record");
                }
                let rhs = self.typed_value(st, ops[i], ty, false)?;
                i += 1;
                if i >= ops.len() {
                    return self.malformed("invalid binary operator record");
                }
                let is_float = self.m.types.is_fp_or_fp_vector(ty);
                if !is_float && !self.m.types.is_int_or_int_vector(ty) {
                    return self.malformed("binary operator on a non-arithmetic type");
                }
                let op = match constants::binary_op(ops[i], is_float) {
                    Some(o) => o,
                    None => return self.malformed("invalid binary operator"),
                };
                i += 1;
                let mut flags_ = 0u8;
                let mut fmf = FastMath::default();
                if i < ops.len() {
                    let raw = ops[i];
                    match op {
                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => {
                            if raw & 2 != 0 {
                                flags_ |= flags::NSW;
                            }
                            if raw & 1 != 0 {
                                flags_ |= flags::NUW;
                            }
                        }
                        BinOp::SDiv | BinOp::UDiv | BinOp::LShr | BinOp::AShr => {
                            if raw & 1 != 0 {
                                flags_ |= flags::EXACT;
                            }
                        }
                        BinOp::Or => {
                            if raw & 1 != 0 {
                                flags_ |= flags::DISJOINT;
                            }
                        }
                        _ if op.is_float() => fmf = Self::fmf(raw),
                        _ => {}
                    }
                }
                self.push_inst(st, ty, Op::Binary { op, lhs, rhs, flags: flags_, fmf }, Some(bc), true)?;
            }
            FUNC_CODE_INST_CAST => {
                let mut i = 0;
                let (v, src, _) = self.value_type_pair(st, ops, &mut i)?;
                if i + 2 > ops.len() {
                    return self.malformed("invalid cast record");
                }
                let res_bc = ops[i];
                let dst = self.ty(res_bc)?;
                let op = match constants::cast_op(ops[i + 1]) {
                    Some(o) => o,
                    None => return self.malformed("invalid cast opcode"),
                };
                i += 2;
                let raw_flags = if i < ops.len() { Some(ops[i]) } else { None };
                // `UpgradeBitCastInst`: a bitcast between address spaces becomes ptrtoint to i64 and inttoptr.
                if op == CastOp::BitCast
                    && self.m.types.is_ptr_or_ptr_vector(src)
                    && self.m.types.is_ptr_or_ptr_vector(dst)
                    && self.m.types.pointer_addrspace(src) != self.m.types.pointer_addrspace(dst)
                {
                    let i64t = self.m.types.int(64);
                    let tmp = self.push_inst(st, i64t, Op::Cast { op: CastOp::PtrToInt, value: v, flags: 0, fmf: FastMath::default() }, None, false)?;
                    self.push_inst(st, dst, Op::Cast { op: CastOp::IntToPtr, value: Value::Inst(tmp), flags: 0, fmf: FastMath::default() }, Some(res_bc as u32), true)?;
                    return Ok(());
                }
                if !constants::cast_is_valid(&self.m.types, op, src, dst) {
                    return self.malformed(format!("invalid {} cast", op.name()));
                }
                let mut flags_ = 0u8;
                let mut fmf = FastMath::default();
                if let Some(raw) = raw_flags {
                    match op {
                        CastOp::ZExt | CastOp::UIToFP => {
                            if raw & 1 != 0 {
                                flags_ |= flags::NNEG;
                            }
                        }
                        CastOp::Trunc => {
                            if raw & 1 != 0 {
                                flags_ |= flags::NUW;
                            }
                            if raw & 2 != 0 {
                                flags_ |= flags::NSW;
                            }
                        }
                        CastOp::FPTrunc | CastOp::FPExt => fmf = Self::fmf(raw),
                        _ => {}
                    }
                }
                self.push_inst(st, dst, Op::Cast { op, value: v, flags: flags_, fmf }, Some(res_bc as u32), true)?;
            }
            FUNC_CODE_INST_GEP | FUNC_CODE_INST_GEP_OLD | FUNC_CODE_INST_INBOUNDS_GEP_OLD => {
                let mut i = 0;
                let mut gflags = 0u8;
                let mut source: Option<(TypeId, u32)> = None;
                if code == FUNC_CODE_INST_GEP {
                    if ops.len() < 2 {
                        return self.malformed("invalid GEP record");
                    }
                    gflags = constants::gep_flags(ops[0]);
                    source = Some((self.ty(ops[1])?, ops[1] as u32));
                    i = 2;
                } else if code == FUNC_CODE_INST_INBOUNDS_GEP_OLD {
                    gflags = GepFlags::INBOUNDS | GepFlags::NUSW;
                }
                let (ptr, ptr_ty, ptr_bc) = self.value_type_pair(st, ops, &mut i)?;
                let (source, mut res_bc) = match source {
                    Some(s) => s,
                    None => {
                        let mut b = self.contained(ptr_bc, 0);
                        if self.m.types.is_vector(ptr_ty) {
                            b = b.and_then(|x| self.contained(x, 0));
                        }
                        match b.and_then(|x| self.bc_types.get(x as usize).copied().flatten().map(|t| (t, x))) {
                            Some(s) => s,
                            None => return self.malformed("missing element type for an old-style GEP"),
                        }
                    }
                };
                if !self.m.types.is_ptr_or_ptr_vector(ptr_ty) {
                    return self.malformed("GEP base is not a pointer");
                }
                let mut indices = Vec::new();
                let mut index_types = Vec::new();
                while i < ops.len() {
                    let (v, t, _) = self.value_type_pair(st, ops, &mut i)?;
                    if !self.m.types.is_int_or_int_vector(t) {
                        return self.malformed("GEP index is not an integer");
                    }
                    indices.push(v);
                    index_types.push(Some(t));
                }
                // Walk the indexed types (and their bitcode ids, for typed-pointer bookkeeping).
                let mut cur = source;
                for (k, idx) in indices.iter().enumerate().skip(1) {
                    let _ = k;
                    match self.m.types.get(cur).clone() {
                        Type::Struct { .. } | Type::LiteralStruct { .. } => {
                            let field = match constants::const_index_value(&self.m, *idx) {
                                Some(fv) => fv,
                                None => return self.malformed("struct GEP index is not a constant"),
                            };
                            cur = match self.m.types.struct_elems(cur).and_then(|e| e.get(field as usize)).copied() {
                                Some(t) => t,
                                None => return self.malformed("struct GEP index out of range"),
                            };
                            res_bc = self.contained(res_bc, field as usize).unwrap_or(u32::MAX);
                        }
                        Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                            cur = elem;
                            res_bc = self.contained(res_bc, 0).unwrap_or(u32::MAX);
                        }
                        _ => return self.malformed("GEP indexes into a non-aggregate type"),
                    }
                }
                let ty = constants::gep_result_type_with(&mut self.m, ptr_ty, index_types, &indices);
                let scalar = self.m.types.scalar(ty);
                let mut bc = self.virtual_type(scalar, if res_bc == u32::MAX { vec![] } else { vec![res_bc] });
                if self.m.types.is_vector(ty) {
                    bc = self.virtual_type(ty, vec![bc]);
                }
                self.push_inst(st, ty, Op::Gep { source, ptr, indices, flags: GepFlags(gflags) }, Some(bc), true)?;
            }
            FUNC_CODE_INST_EXTRACTVAL => {
                let mut i = 0;
                let (agg, aty, abc) = self.value_type_pair(st, ops, &mut i)?;
                if i == ops.len() {
                    return self.malformed("extractvalue with no indices");
                }
                let mut cur = aty;
                let mut bc = abc;
                let mut idx = Vec::new();
                for &x in &ops[i..] {
                    if x > u32::MAX as u64 {
                        return self.malformed("extractvalue index out of range");
                    }
                    let (next, nbc) = match self.m.types.get(cur) {
                        Type::Array { len, elem } => {
                            if x >= *len {
                                return self.malformed("extractvalue array index out of range");
                            }
                            (*elem, self.contained(bc, 0))
                        }
                        Type::Struct { body: Some(e), .. } | Type::LiteralStruct { elems: e, .. } => {
                            if x as usize >= e.len() {
                                return self.malformed("extractvalue struct index out of range");
                            }
                            (e[x as usize], self.contained(bc, x as usize))
                        }
                        _ => return self.malformed("extractvalue of a non-aggregate type"),
                    };
                    cur = next;
                    bc = nbc.unwrap_or(u32::MAX);
                    idx.push(x as u32);
                }
                let bc = if bc == u32::MAX { None } else { Some(bc) };
                self.push_inst(st, cur, Op::ExtractValue { agg, indices: idx }, bc, true)?;
            }
            FUNC_CODE_INST_INSERTVAL => {
                let mut i = 0;
                let (agg, aty, abc) = self.value_type_pair(st, ops, &mut i)?;
                let (val, vty, _) = self.value_type_pair(st, ops, &mut i)?;
                if i == ops.len() {
                    return self.malformed("insertvalue with no indices");
                }
                let mut cur = aty;
                let mut idx = Vec::new();
                for &x in &ops[i..] {
                    if x > u32::MAX as u64 {
                        return self.malformed("insertvalue index out of range");
                    }
                    cur = match self.m.types.get(cur) {
                        Type::Array { len, elem } => {
                            if x >= *len {
                                return self.malformed("insertvalue array index out of range");
                            }
                            *elem
                        }
                        Type::Struct { body: Some(e), .. } | Type::LiteralStruct { elems: e, .. } => {
                            if x as usize >= e.len() {
                                return self.malformed("insertvalue struct index out of range");
                            }
                            e[x as usize]
                        }
                        _ => return self.malformed("insertvalue into a non-aggregate type"),
                    };
                    idx.push(x as u32);
                }
                if cur != vty {
                    return self.malformed("inserted value type does not match the aggregate type");
                }
                self.push_inst(st, aty, Op::InsertValue { agg, value: val, indices: idx }, Some(abc), true)?;
            }
            FUNC_CODE_INST_SELECT | FUNC_CODE_INST_VSELECT => {
                let mut i = 0;
                let (t, ty, bc) = self.value_type_pair(st, ops, &mut i)?;
                if i >= ops.len() {
                    return self.malformed("invalid select record");
                }
                let fv = self.typed_value(st, ops[i], ty, false)?;
                i += 1;
                let cond;
                if code == FUNC_CODE_INST_SELECT {
                    if i >= ops.len() {
                        return self.malformed("invalid select record");
                    }
                    let i1 = self.m.types.int(1);
                    cond = self.typed_value(st, ops[i], i1, false)?;
                    i += 1;
                } else {
                    let (c, cty, _) = self.value_type_pair(st, ops, &mut i)?;
                    let ok = match self.m.types.get(cty) {
                        Type::Vector { elem, len, .. } => {
                            self.m.types.int_bits(*elem) == Some(1) && self.m.types.vector_len(ty).map(|n| n == *len).unwrap_or(false)
                        }
                        Type::Int(1) => true,
                        _ => false,
                    };
                    if !ok {
                        return self.malformed("invalid select condition type");
                    }
                    cond = c;
                }
                let mut fmf = FastMath::default();
                if code == FUNC_CODE_INST_VSELECT && i < ops.len() && self.is_fp_math_type(ty) {
                    fmf = Self::fmf(ops[i]);
                }
                self.push_inst(st, ty, Op::Select { cond, t, f: fv, fmf }, Some(bc), true)?;
            }
            FUNC_CODE_INST_EXTRACTELT => {
                let mut i = 0;
                let (vec, vty, vbc) = self.value_type_pair(st, ops, &mut i)?;
                let (idx, ity, _) = self.value_type_pair(st, ops, &mut i)?;
                let elem = match self.m.types.get(vty) {
                    Type::Vector { elem, .. } => *elem,
                    _ => return self.malformed("extractelement of a non-vector"),
                };
                if !self.m.types.is_int(ity) {
                    return self.malformed("extractelement index is not an integer");
                }
                let bc = self.contained(vbc, 0);
                self.push_inst(st, elem, Op::ExtractElement { vec, idx }, bc, true)?;
            }
            FUNC_CODE_INST_INSERTELT => {
                let mut i = 0;
                let (vec, vty, vbc) = self.value_type_pair(st, ops, &mut i)?;
                let elem = match self.m.types.get(vty) {
                    Type::Vector { elem, .. } => *elem,
                    _ => return self.malformed("insertelement into a non-vector"),
                };
                if i >= ops.len() {
                    return self.malformed("invalid insertelement record");
                }
                let elt = self.typed_value(st, ops[i], elem, false)?;
                i += 1;
                let (idx, ity, _) = self.value_type_pair(st, ops, &mut i)?;
                if !self.m.types.is_int(ity) {
                    return self.malformed("insertelement index is not an integer");
                }
                self.push_inst(st, vty, Op::InsertElement { vec, elt, idx }, Some(vbc), true)?;
            }
            FUNC_CODE_INST_SHUFFLEVEC => {
                let mut i = 0;
                let (a, aty, abc) = self.value_type_pair(st, ops, &mut i)?;
                if i >= ops.len() {
                    return self.malformed("invalid shufflevector record");
                }
                let b = self.typed_value(st, ops[i], aty, false)?;
                i += 1;
                let (mask_v, mty, _) = self.value_type_pair(st, ops, &mut i)?;
                let (alen, elem) = match self.m.types.get(aty) {
                    Type::Vector { len, elem, scalable: false } => (*len, *elem),
                    Type::Vector { .. } => return self.unsupported("shufflevector of scalable vectors"),
                    _ => return self.malformed("shufflevector of non-vectors"),
                };
                let mlen = match self.m.types.get(mty) {
                    Type::Vector { len, elem, scalable: false } if self.m.types.int_bits(*elem) == Some(32) => *len,
                    _ => return self.malformed("shufflevector mask is not a vector of i32"),
                };
                let mask = match self.shuffle_mask(mask_v, mlen) {
                    Some(m) => m,
                    None => return self.malformed("shufflevector mask is not a constant"),
                };
                if mask.iter().any(|&e| e != -1 && (e < 0 || e as u64 >= 2 * alen as u64)) {
                    return self.malformed("shufflevector mask element out of range");
                }
                let ty = self.m.types.vector(mlen, elem);
                let ebc = self.contained(abc, 0);
                let bc = self.virtual_type(ty, ebc.into_iter().collect());
                self.push_inst(st, ty, Op::ShuffleVector { a, b, mask }, Some(bc), true)?;
            }
            FUNC_CODE_INST_CMP | FUNC_CODE_INST_CMP2 => {
                let mut i = 0;
                let (lhs, ty, _) = self.value_type_pair(st, ops, &mut i)?;
                if i >= ops.len() {
                    return self.malformed("invalid compare record");
                }
                let rhs = self.typed_value(st, ops[i], ty, false)?;
                i += 1;
                if i >= ops.len() {
                    return self.malformed("invalid compare record: operand number exceeded available operands");
                }
                let pred = ops[i];
                let is_fp = self.m.types.is_fp_or_fp_vector(ty);
                let i1 = self.m.types.int(1);
                let rty = match self.m.types.vector_len(ty) {
                    Some(n) => self.m.types.vector(n, i1),
                    None => i1,
                };
                let op = if is_fp {
                    let p = match FloatPredicate::from_raw(pred) {
                        Some(p) => p,
                        None => return self.malformed("invalid fcmp predicate"),
                    };
                    let mut fmf = FastMath::default();
                    if ops.len() > i + 1 {
                        i += 1;
                        fmf = Self::fmf(ops[i]);
                    }
                    Op::FCmp { pred: p, lhs, rhs, fmf }
                } else {
                    if !self.m.types.is_int_or_int_vector(ty) && !self.m.types.is_ptr_or_ptr_vector(ty) {
                        return self.malformed("icmp of a non-integer, non-pointer type");
                    }
                    let p = match IntPredicate::from_raw(pred) {
                        Some(p) => p,
                        None => return self.malformed("invalid icmp predicate"),
                    };
                    let mut samesign = false;
                    if ops.len() > i + 1 {
                        i += 1;
                        samesign = ops[i] & 1 != 0;
                    }
                    Op::ICmp { pred: p, lhs, rhs, samesign }
                };
                if i + 1 != ops.len() {
                    return self.malformed("invalid compare record");
                }
                self.push_inst(st, rty, op, None, true)?;
            }
            FUNC_CODE_INST_RET => {
                let void = self.m.types.void();
                let ret_ty = self.m.types.function(self.m.functions[f as usize].ty).unwrap().0;
                if ops.is_empty() {
                    if !self.m.types.is_void(ret_ty) {
                        return self.malformed("ret without a value in a function that returns a value");
                    }
                    self.push_inst(st, void, Op::Ret { value: None }, None, true)?;
                } else {
                    let mut i = 0;
                    let (v, ty, _) = self.value_type_pair(st, ops, &mut i)?;
                    if i != ops.len() {
                        return self.malformed("invalid ret record");
                    }
                    if ty != ret_ty {
                        return self.malformed("ret value has the wrong type");
                    }
                    self.push_inst(st, void, Op::Ret { value: Some(v) }, None, true)?;
                }
            }
            FUNC_CODE_INST_BR => {
                let void = self.m.types.void();
                match ops.len() {
                    1 => {
                        let dest = self.block(st, ops[0])?;
                        self.push_inst(st, void, Op::Br { dest }, None, true)?;
                    }
                    3 => {
                        let t = self.block(st, ops[0])?;
                        let fb = self.block(st, ops[1])?;
                        let i1 = self.m.types.int(1);
                        let cond = self.typed_value(st, ops[2], i1, false)?;
                        self.push_inst(st, void, Op::CondBr { cond, t, f: fb }, None, true)?;
                    }
                    _ => return self.malformed("invalid br record"),
                }
            }
            FUNC_CODE_INST_SWITCH => {
                if ops.is_empty() {
                    return self.malformed("invalid switch record");
                }
                if ops[0] >> 16 == 0x4B5 {
                    return self.unsupported("switch records with case ranges (SWITCH_INST_MAGIC)");
                }
                if ops.len() < 3 || ops.len() % 2 == 0 {
                    return self.malformed("invalid switch record");
                }
                let ty = self.ty(ops[0])?;
                if !self.m.types.is_int(ty) {
                    return self.malformed("switch on a non-integer");
                }
                let cond = self.typed_value(st, ops[1], ty, false)?;
                let default = self.block(st, ops[2])?;
                let mut cases = Vec::new();
                for pair in ops[3..].chunks(2) {
                    // Case values are absolute value numbers.
                    let c = match self.values.get(pair[0] as usize).map(|e| e.slot) {
                        Some(Slot::Const(c)) => self.materialize_const_use(c, Use::Instruction)?,
                        _ => return self.malformed("switch case is not a constant"),
                    };
                    let cid = match c {
                        Value::Const(cid) if matches!(self.m.constant(cid).kind, ConstKind::Int(_)) && self.m.constant(cid).ty == ty => cid,
                        _ => return self.malformed("switch case is not an integer constant of the condition's type"),
                    };
                    let b = self.block(st, pair[1])?;
                    cases.push((cid, b));
                }
                let void = self.m.types.void();
                self.push_inst(st, void, Op::Switch { cond, default, cases }, None, true)?;
            }
            FUNC_CODE_INST_INDIRECTBR => {
                if ops.len() < 2 {
                    return self.malformed("invalid indirectbr record");
                }
                let ty = self.ty(ops[0])?;
                let addr = self.typed_value(st, ops[1], ty, false)?;
                if !self.m.types.is_ptr(ty) {
                    return self.malformed("indirectbr address is not a pointer");
                }
                let mut dests = Vec::new();
                for &b in &ops[2..] {
                    dests.push(self.block(st, b)?);
                }
                let void = self.m.types.void();
                self.push_inst(st, void, Op::IndirectBr { addr, dests }, None, true)?;
            }
            FUNC_CODE_INST_UNREACHABLE => {
                let void = self.m.types.void();
                self.push_inst(st, void, Op::Unreachable, None, true)?;
            }
            FUNC_CODE_INST_PHI => {
                if ops.is_empty() {
                    return self.malformed("invalid phi record");
                }
                let ty = self.ty(ops[0])?;
                let nargs = (ops.len() - 1) / 2;
                let has_fmf = (ops.len() - 1) % 2 == 1;
                if has_fmf && !self.is_fp_math_type(ty) {
                    return self.malformed("invalid phi record");
                }
                let mut incoming: Vec<(Value, BlockId)> = Vec::new();
                let mut seen: HashMap<BlockId, Value> = HashMap::new();
                for k in 0..nargs {
                    let b = self.block(st, ops[k * 2 + 2])?;
                    if let Some(&v) = seen.get(&b) {
                        // A repeated predecessor takes the value already given for it.
                        incoming.push((v, b));
                        continue;
                    }
                    let v = self.typed_value(st, ops[k * 2 + 1], ty, self.relative_ids)?;
                    incoming.push((v, b));
                    seen.insert(b, v);
                }
                let fmf = if has_fmf { Self::fmf(ops[ops.len() - 1]) } else { FastMath::default() };
                let bc = ops[0] as u32;
                self.push_inst(st, ty, Op::Phi { incoming, fmf }, Some(bc), true)?;
            }
            FUNC_CODE_INST_ALLOCA => {
                if ops.len() != 4 && ops.len() != 5 {
                    return self.malformed("invalid alloca record");
                }
                let rec = ops[3];
                let inalloca = (rec >> 5) & 1 != 0;
                let explicit = (rec >> 6) & 1 != 0;
                let swifterror = (rec >> 7) & 1 != 0;
                let align_exp = (rec & 0x1f) | ((rec >> 8) & 7) << 5;
                let mut ty_bc = ops[0] as u32;
                if !explicit {
                    ty_bc = match self.contained(ty_bc, 0) {
                        Some(t) => t,
                        None => return self.malformed("missing element type for an old-style alloca"),
                    };
                }
                let allocated = self.ty(ty_bc as u64)?;
                let count_ty = self.ty(ops[1])?;
                // The count is an absolute value number.
                let count = match self.values.get(ops[2] as usize).map(|e| e.slot) {
                    Some(_) => {
                        let (v, _) = self.defined_value(ops[2] as u32)?;
                        if self.local_type(st, v) != count_ty {
                            return self.malformed("alloca count has the wrong type");
                        }
                        v
                    }
                    None => return self.malformed("invalid alloca count"),
                };
                if !self.m.types.is_int(count_ty) {
                    return self.malformed("alloca element count must have integer type");
                }
                let mut align = self.alignment(align_exp)?;
                let layout = self.layout.clone().unwrap();
                if align.is_none() {
                    if !self.m.types.is_sized(allocated) {
                        return self.malformed("alloca of unsized type");
                    }
                    align = layout.pref_align(&self.m.types, allocated);
                }
                let addrspace = if ops.len() == 5 { ops[4] as u32 } else { layout.alloca_addrspace };
                let ty = self.m.types.ptr(addrspace);
                let bc = self.virtual_type(ty, vec![ty_bc]);
                self.push_inst(
                    st,
                    ty,
                    Op::Alloca { allocated, count, align: align.unwrap(), addrspace, inalloca, swifterror },
                    Some(bc),
                    true,
                )?;
            }
            FUNC_CODE_INST_LOAD | FUNC_CODE_INST_LOADATOMIC => {
                let atomic = code == FUNC_CODE_INST_LOADATOMIC;
                let tail = if atomic { 4 } else { 2 };
                let mut i = 0;
                let (ptr, pty, pbc) = self.value_type_pair(st, ops, &mut i)?;
                if i + tail != ops.len() && i + tail + 1 != ops.len() {
                    return self.malformed("invalid load record");
                }
                if !self.m.types.is_ptr(pty) {
                    return self.malformed("load operand is not a pointer type");
                }
                let res_bc = if i + tail + 1 == ops.len() {
                    i += 1;
                    ops[i - 1] as u32
                } else {
                    match self.contained(pbc, 0) {
                        Some(b) => b,
                        None => return self.malformed("missing load type"),
                    }
                };
                let ty = self.ty(res_bc as u64)?;
                self.check_loadable(ty)?;
                let mut align = self.alignment(ops[i])?;
                let volatile = ops[i + 1] != 0;
                let mut at = None;
                if atomic {
                    let ordering = self.ordering(ops[i + 2])?;
                    if matches!(ordering, Ordering::Release | Ordering::AcqRel) {
                        return self.malformed("invalid atomic load ordering");
                    }
                    if ops[i] == 0 {
                        return self.malformed("alignment missing from atomic load");
                    }
                    at = Some(Atomic { ordering, scope: self.sync_scope(ops[i + 3])? });
                }
                if align.is_none() {
                    if !self.m.types.is_sized(ty) {
                        return self.malformed("load of unsized type");
                    }
                    align = self.layout.as_ref().unwrap().abi_align(&self.m.types, ty);
                }
                self.push_inst(st, ty, Op::Load { ptr, align: align.unwrap(), volatile, atomic: at }, Some(res_bc), true)?;
            }
            FUNC_CODE_INST_STORE | FUNC_CODE_INST_STORE_OLD | FUNC_CODE_INST_STOREATOMIC | FUNC_CODE_INST_STOREATOMIC_OLD => {
                let atomic = matches!(code, FUNC_CODE_INST_STOREATOMIC | FUNC_CODE_INST_STOREATOMIC_OLD);
                let mut i = 0;
                let (ptr, pty, pbc) = self.value_type_pair(st, ops, &mut i)?;
                if !self.m.types.is_ptr(pty) {
                    return self.malformed("store operand is not a pointer type");
                }
                let (val, vty) = if matches!(code, FUNC_CODE_INST_STORE | FUNC_CODE_INST_STOREATOMIC) {
                    let (v, t, _) = self.value_type_pair(st, ops, &mut i)?;
                    (v, t)
                } else {
                    let vbc = match self.contained(pbc, 0) {
                        Some(b) => b,
                        None => return self.malformed("missing stored value type"),
                    };
                    let vt = self.ty(vbc as u64)?;
                    if i >= ops.len() {
                        return self.malformed("invalid store record");
                    }
                    let v = self.typed_value(st, ops[i], vt, false)?;
                    i += 1;
                    (v, vt)
                };
                let tail = if atomic { 4 } else { 2 };
                if i + tail != ops.len() {
                    return self.malformed("invalid store record");
                }
                self.check_loadable(vty)?;
                let mut align = self.alignment(ops[i])?;
                let volatile = ops[i + 1] != 0;
                let mut at = None;
                if atomic {
                    let ordering = self.ordering(ops[i + 2])?;
                    if matches!(ordering, Ordering::Acquire | Ordering::AcqRel) {
                        return self.malformed("invalid atomic store ordering");
                    }
                    if ops[i] == 0 {
                        return self.malformed("alignment missing from atomic store");
                    }
                    at = Some(Atomic { ordering, scope: self.sync_scope(ops[i + 3])? });
                }
                if align.is_none() {
                    if !self.m.types.is_sized(vty) {
                        return self.malformed("store of unsized type");
                    }
                    align = self.layout.as_ref().unwrap().abi_align(&self.m.types, vty);
                }
                let void = self.m.types.void();
                self.push_inst(st, void, Op::Store { value: val, ptr, align: align.unwrap(), volatile, atomic: at }, None, true)?;
            }
            FUNC_CODE_INST_CMPXCHG => {
                let mut i = 0;
                let (ptr, pty, _) = self.value_type_pair(st, ops, &mut i)?;
                if !self.m.types.is_ptr(pty) {
                    return self.malformed("cmpxchg operand is not a pointer type");
                }
                let (cmp, cty, cbc) = self.value_type_pair(st, ops, &mut i)?;
                if i >= ops.len() {
                    return self.malformed("invalid cmpxchg record");
                }
                let new = self.typed_value(st, ops[i], cty, false)?;
                i += 1;
                if ops.len() < i + 3 || ops.len() > i + 6 {
                    return self.malformed("invalid cmpxchg record");
                }
                let volatile = ops[i] != 0;
                let success = self.ordering(ops[i + 1])?;
                if matches!(success, Ordering::Unordered) {
                    return self.malformed("invalid cmpxchg success ordering");
                }
                let scope = self.sync_scope(ops[i + 2])?;
                self.check_loadable(cty)?;
                if ops.len() < i + 5 {
                    return self.unsupported("cmpxchg records without failure ordering and weak flag");
                }
                let failure = self.ordering(ops[i + 3])?;
                if matches!(failure, Ordering::Unordered | Ordering::Release | Ordering::AcqRel) {
                    return self.malformed("invalid cmpxchg failure ordering");
                }
                let weak = ops[i + 4] != 0;
                let mut align = None;
                if ops.len() == i + 6 {
                    align = self.alignment(ops[i + 5])?;
                }
                let align = match align {
                    Some(a) => a,
                    None => match self.layout.as_ref().unwrap().store_size(&self.m.types, cty) {
                        Some(s) if s.is_power_of_two() => s,
                        _ => return self.malformed("cmpxchg of a type without a power-of-two size"),
                    },
                };
                let i1 = self.m.types.int(1);
                let ty = self.m.types.intern(Type::LiteralStruct { packed: false, elems: vec![cty, i1] });
                let i1bc = self.virtual_type(i1, vec![]);
                let bc = self.virtual_type(ty, vec![cbc, i1bc]);
                self.push_inst(st, ty, Op::CmpXchg { ptr, cmp, new, align, volatile, weak, success, failure, scope }, Some(bc), true)?;
            }
            FUNC_CODE_INST_ATOMICRMW => {
                let mut i = 0;
                let (ptr, pty, _) = self.value_type_pair(st, ops, &mut i)?;
                if !self.m.types.is_ptr(pty) {
                    return self.malformed("atomicrmw operand is not a pointer type");
                }
                let (val, vty, vbc) = self.value_type_pair(st, ops, &mut i)?;
                if ops.len() != i + 4 && ops.len() != i + 5 {
                    return self.malformed("invalid atomicrmw record");
                }
                let op = match rmw_op(ops[i]) {
                    Some(o) => o,
                    None => return self.malformed("invalid atomicrmw operation"),
                };
                let volatile = ops[i + 1] != 0;
                let ordering = self.ordering(ops[i + 2])?;
                if matches!(ordering, Ordering::Unordered) {
                    return self.malformed("invalid atomicrmw ordering");
                }
                let scope = self.sync_scope(ops[i + 3])?;
                let mut align = None;
                if ops.len() == i + 5 {
                    align = self.alignment(ops[i + 4])?;
                }
                let align = match align {
                    Some(a) => a,
                    None => match self.layout.as_ref().unwrap().store_size(&self.m.types, vty) {
                        Some(s) if s.is_power_of_two() => s,
                        _ => return self.malformed("atomicrmw of a type without a power-of-two size"),
                    },
                };
                self.push_inst(st, vty, Op::AtomicRmw { op, ptr, value: val, align, volatile, ordering, scope }, Some(vbc), true)?;
            }
            FUNC_CODE_INST_CMPXCHG_OLD | FUNC_CODE_INST_ATOMICRMW_OLD => {
                return self.unsupported("cmpxchg and atomicrmw records of LLVM before 13");
            }
            FUNC_CODE_INST_FENCE => {
                if ops.len() != 2 {
                    return self.malformed("invalid fence record");
                }
                let ordering = self.ordering(ops[0])?;
                if matches!(ordering, Ordering::Unordered | Ordering::Monotonic) {
                    return self.malformed("invalid fence ordering");
                }
                let scope = self.sync_scope(ops[1])?;
                let void = self.m.types.void();
                self.push_inst(st, void, Op::Fence { ordering, scope }, None, true)?;
            }
            FUNC_CODE_INST_CALL => self.call_record(st, ops)?,
            FUNC_CODE_INST_VAARG => {
                if ops.len() < 3 {
                    return self.malformed("invalid va_arg record");
                }
                let lty = self.ty(ops[0])?;
                let list = self.typed_value(st, ops[1], lty, false)?;
                let ty = self.ty(ops[2])?;
                self.push_inst(st, ty, Op::VaArg { list }, Some(ops[2] as u32), true)?;
            }
            FUNC_CODE_OPERAND_BUNDLE => {
                if ops.is_empty() || ops[0] as usize >= self.m.bundle_tags.len() {
                    return self.malformed("invalid operand bundle record");
                }
                let mut inputs = Vec::new();
                let mut i = 1;
                while i < ops.len() {
                    if ops[i] == OB_METADATA {
                        if i + 1 >= ops.len() {
                            return self.malformed("invalid operand bundle record");
                        }
                        let valno = self.next_value().wrapping_sub(ops[i + 1] as u32);
                        inputs.push(Value::Metadata(self.md_ref(valno as u64)?));
                        i += 2;
                    } else {
                        let (v, _, _) = self.value_type_pair(st, ops, &mut i)?;
                        inputs.push(v);
                    }
                }
                st.bundles.push(Bundle { tag: ops[0] as u32, inputs });
            }
            FUNC_CODE_INST_FREEZE => {
                let mut i = 0;
                let (v, ty, bc) = self.value_type_pair(st, ops, &mut i)?;
                if i != ops.len() {
                    return self.malformed("invalid freeze record");
                }
                self.push_inst(st, ty, Op::Freeze { value: v }, Some(bc), true)?;
            }
            FUNC_CODE_DEBUG_LOC | FUNC_CODE_DEBUG_LOC_AGAIN => {
                return self.unsupported("debug locations (DEBUG_LOC records)");
            }
            FUNC_CODE_DEBUG_RECORD_VALUE
            | FUNC_CODE_DEBUG_RECORD_DECLARE
            | FUNC_CODE_DEBUG_RECORD_ASSIGN
            | FUNC_CODE_DEBUG_RECORD_VALUE_SIMPLE
            | FUNC_CODE_DEBUG_RECORD_LABEL => return self.unsupported("debug records"),
            FUNC_CODE_BLOCKADDR_USERS => return self.unsupported("blockaddress users (BLOCKADDR_USERS)"),
            FUNC_CODE_INST_INVOKE
            | FUNC_CODE_INST_RESUME
            | FUNC_CODE_INST_LANDINGPAD
            | FUNC_CODE_INST_LANDINGPAD_OLD
            | FUNC_CODE_INST_CLEANUPRET
            | FUNC_CODE_INST_CATCHRET
            | FUNC_CODE_INST_CATCHPAD
            | FUNC_CODE_INST_CLEANUPPAD
            | FUNC_CODE_INST_CATCHSWITCH => return self.unsupported(format!("exception-handling instruction (record {})", code)),
            FUNC_CODE_INST_CALLBR => return self.unsupported("callbr"),
            c => return self.malformed(format!("unknown function record code {}", c)),
        }
        if code != FUNC_CODE_OPERAND_BUNDLE && !st.bundles.is_empty() && code != FUNC_CODE_INST_CALL {
            return self.malformed("operand bundles found with no consumer");
        }
        Ok(())
    }

    fn check_loadable(&self, ty: TypeId) -> Result<()> {
        if matches!(
            self.m.types.get(ty),
            Type::Void | Type::Label | Type::Metadata | Type::Token | Type::X86Amx | Type::Function { .. }
        ) {
            return self.malformed("cannot load or store this type");
        }
        Ok(())
    }

    fn ordering(&self, raw: u64) -> Result<Ordering> {
        Ok(match raw {
            1 => Ordering::Unordered,
            2 => Ordering::Monotonic,
            3 => Ordering::Acquire,
            4 => Ordering::Release,
            5 => Ordering::AcqRel,
            6 => Ordering::SeqCst,
            0 => return self.malformed("not-atomic ordering on an atomic operation"),
            other => return self.malformed(format!("unknown atomic ordering {}", other)),
        })
    }

    /// The lanes of a constant shuffle mask (`ShuffleVectorInst::getShuffleMask`): -1 for undef or poison lanes.
    fn shuffle_mask(&mut self, v: Value, len: u32) -> Option<Vec<i32>> {
        let c = match v {
            Value::Const(c) => c,
            _ => return None,
        };
        match self.m.constant(c).kind.clone() {
            ConstKind::Zero => Some(vec![0; len as usize]),
            ConstKind::Undef | ConstKind::Poison => Some(vec![-1; len as usize]),
            ConstKind::DataVector(e) => Some(e.iter().map(|&x| x as i32).collect()),
            ConstKind::Vector(e) => {
                let mut out = Vec::new();
                for x in e {
                    match x {
                        Value::Const(xc) => match &self.m.constant(xc).kind {
                            ConstKind::Undef | ConstKind::Poison => out.push(-1),
                            ConstKind::Int(w) => out.push(w[0] as i32),
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    fn call_record(&mut self, st: &mut FnState, ops: &[u64]) -> Result<()> {
        if ops.len() < 3 {
            return self.malformed("invalid call record");
        }
        let mut i = 0;
        let mut attrs = self.attr_list(ops[i])?;
        i += 1;
        let cc_info = ops[i];
        i += 1;
        let mut fmf = FastMath::default();
        if (cc_info >> CALL_FMF) & 1 != 0 {
            if i >= ops.len() {
                return self.malformed("invalid call record");
            }
            fmf = Self::fmf(ops[i]);
            i += 1;
            if !fmf.any() {
                return self.malformed("fast math flags indicator set for call with no FMF");
            }
        }
        let mut fty: Option<(TypeId, u32)> = None;
        if (cc_info >> CALL_EXPLICIT_TYPE) & 1 != 0 {
            if i >= ops.len() {
                return self.malformed("invalid call record");
            }
            let t = self.ty(ops[i])?;
            if !self.m.types.is_function(t) {
                return self.malformed("explicit call type is not a function type");
            }
            fty = Some((t, ops[i] as u32));
            i += 1;
        }
        let (callee, cty, cbc) = self.value_type_pair(st, ops, &mut i)?;
        if !self.m.types.is_ptr(cty) {
            return self.malformed("callee is not a pointer type");
        }
        let (fn_ty, fn_bc) = match fty {
            Some(x) => x,
            None => match self.contained(cbc, 0).and_then(|b| self.bc_types.get(b as usize).copied().flatten().map(|t| (t, b))) {
                Some((t, b)) if self.m.types.is_function(t) => (t, b),
                _ => return self.malformed("callee is not of pointer to function type"),
            },
        };
        let (ret, params, vararg) = {
            let (r, p, v) = self.m.types.function(fn_ty).unwrap();
            (r, p.to_vec(), v)
        };
        if ops.len() < params.len() + i {
            return self.malformed("insufficient operands to call");
        }
        let mut args = Vec::new();
        let mut arg_bcs: Vec<Option<u32>> = Vec::new();
        for (k, &p) in params.iter().enumerate() {
            if self.m.types.is_label(p) {
                return self.unsupported("call arguments of label type");
            }
            let v = self.typed_value(st, ops[i], p, false)?;
            i += 1;
            args.push(v);
            arg_bcs.push(self.contained(fn_bc, k + 1));
        }
        if !vararg {
            if i != ops.len() {
                return self.malformed("invalid call record");
            }
        } else {
            while i < ops.len() {
                let (v, _, bc) = self.value_type_pair(st, ops, &mut i)?;
                args.push(v);
                arg_bcs.push(Some(bc));
            }
        }
        let bundles = std::mem::take(&mut st.bundles);
        // `UpgradeOperandBundles`: clang.arc.attachedcall bundles without operands are dropped.
        let bundles: Vec<Bundle> = bundles
            .into_iter()
            .filter(|b| !(self.m.bundle_tags[b.tag as usize] == "clang.arc.attachedcall" && b.inputs.is_empty()))
            .collect();
        let cc = ((cc_info & 0x7ff) >> CALL_CCONV) as u32;
        let mut tail = TailKind::None;
        if cc_info & (1 << CALL_TAIL) != 0 {
            tail = TailKind::Tail;
        }
        if cc_info & (1 << CALL_MUSTTAIL) != 0 {
            tail = TailKind::MustTail;
        }
        if cc_info & (1 << CALL_NOTAIL) != 0 {
            tail = TailKind::NoTail;
        }
        if let Value::InlineAsm(_) = callee {
            return self.unsupported("calls to inline assembly (the elementtype upgrade for indirect constraints is not implemented)");
        }
        if let Value::Global(GlobalId::Function(cf)) = callee {
            let name = &self.m.functions[cf as usize].common.name;
            if matches!(
                name.as_str(),
                "llvm.preserve.array.access.index" | "llvm.preserve.struct.access.index"
            ) || name.starts_with("llvm.aarch64.ldaxr")
                || name.starts_with("llvm.aarch64.ldxr")
                || name.starts_with("llvm.aarch64.stlxr")
                || name.starts_with("llvm.aarch64.stxr")
                || name.starts_with("llvm.arm.ldaex")
                || name.starts_with("llvm.arm.ldrex")
                || name.starts_with("llvm.arm.stlex")
                || name.starts_with("llvm.arm.strex")
            {
                return self.unsupported(format!("calls to {} (the elementtype upgrade is not implemented)", name));
            }
        }
        // Typeless byval/sret/inalloca take the argument's pointee type (`propagateAttributeTypes`).
        let pointees: Vec<Option<u32>> = arg_bcs;
        self.upgrade_typed_param_attrs(&mut attrs, &pointees)?;
        if fmf.any() && !self.is_fp_math_type(ret) {
            return self.malformed("fast-math flags specified for a call without floating-point return type");
        }
        let res_bc = self.contained(fn_bc, 0);
        self.push_inst(st, ret, Op::Call { callee, fn_ty, args, attrs, cc, tail, fmf, bundles }, res_bc, true)?;
        Ok(())
    }
}

fn rmw_op(raw: u64) -> Option<RmwOp> {
    use RmwOp::*;
    Some(match raw {
        0 => Xchg,
        1 => Add,
        2 => Sub,
        3 => And,
        4 => Nand,
        5 => Or,
        6 => Xor,
        7 => Max,
        8 => Min,
        9 => UMax,
        10 => UMin,
        11 => FAdd,
        12 => FSub,
        13 => FMax,
        14 => FMin,
        15 => UIncWrap,
        16 => UDecWrap,
        17 => USubCond,
        18 => USubSat,
        19 => FMaximum,
        20 => FMinimum,
        _ => return None,
    })
}

/// Calls `f` on every value operand of an instruction (not blocks, not switch case constants).
pub(crate) fn visit_values_mut(op: &mut Op, f: &mut dyn FnMut(&mut Value)) {
    match op {
        Op::FNeg { value, .. } | Op::Cast { value, .. } | Op::Freeze { value } => f(value),
        Op::Binary { lhs, rhs, .. } | Op::ICmp { lhs, rhs, .. } | Op::FCmp { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Op::Select { cond, t, f: fv, .. } => {
            f(cond);
            f(t);
            f(fv);
        }
        Op::Phi { incoming, .. } => incoming.iter_mut().for_each(|(v, _)| f(v)),
        Op::Alloca { count, .. } => f(count),
        Op::Load { ptr, .. } => f(ptr),
        Op::Store { value, ptr, .. } => {
            f(value);
            f(ptr);
        }
        Op::Gep { ptr, indices, .. } => {
            f(ptr);
            indices.iter_mut().for_each(|v| f(v));
        }
        Op::ExtractElement { vec, idx } => {
            f(vec);
            f(idx);
        }
        Op::InsertElement { vec, elt, idx } => {
            f(vec);
            f(elt);
            f(idx);
        }
        Op::ShuffleVector { a, b, .. } => {
            f(a);
            f(b);
        }
        Op::ExtractValue { agg, .. } => f(agg),
        Op::InsertValue { agg, value, .. } => {
            f(agg);
            f(value);
        }
        Op::Call { callee, args, bundles, .. } => {
            f(callee);
            args.iter_mut().for_each(|v| f(v));
            for b in bundles {
                b.inputs.iter_mut().for_each(|v| f(v));
            }
        }
        Op::AtomicRmw { ptr, value, .. } => {
            f(ptr);
            f(value);
        }
        Op::CmpXchg { ptr, cmp, new, .. } => {
            f(ptr);
            f(cmp);
            f(new);
        }
        Op::VaArg { list } => f(list),
        Op::Ret { value } => {
            if let Some(v) = value {
                f(v)
            }
        }
        Op::CondBr { cond, .. } => f(cond),
        Op::Switch { cond, .. } => f(cond),
        Op::IndirectBr { addr, .. } => f(addr),
        Op::Fence { .. } | Op::Br { .. } | Op::Unreachable => {}
    }
}
