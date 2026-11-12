//! The type table (`TYPE_BLOCK_ID_NEW`; `parseTypeTableBody` in `BitcodeReader.cpp`).

use super::codes::*;
use super::{Reader, Result};
use crate::bitstream::Entry;
use crate::ir::{Type, TypeId};

impl<'a> Reader<'a> {
    /// A type id used while reading the table: defined, or a forward reference that must be an identified struct.
    fn type_or_forward(&mut self, id: u64, forward_structs: &mut Vec<bool>) -> Result<Option<TypeId>> {
        let i = id as usize;
        if i >= self.bc_types.len() {
            return Ok(None);
        }
        if let Some(t) = self.bc_types[i] {
            return Ok(Some(t));
        }
        let t = self.m.types.new_struct(String::new());
        self.bc_types[i] = Some(t);
        forward_structs[i] = true;
        Ok(Some(t))
    }

    pub(super) fn read_type_block(&mut self) -> Result<()> {
        if self.seen_type_table {
            return self.malformed("more than one type table");
        }
        self.seen_type_table = true;
        self.cur.enter_block(TYPE_BLOCK_ID_NEW)?;
        let mut num_records = 0usize;
        let mut forward_structs: Vec<bool> = Vec::new();
        let mut name: Option<String> = None;
        let mut sized = false;
        loop {
            let a = match self.cur.advance()? {
                Entry::EndBlock => {
                    if num_records != self.bc_types.len() {
                        return self.malformed(format!(
                            "type table declares {} types and defines {}",
                            self.bc_types.len(),
                            num_records
                        ));
                    }
                    if name.is_some() {
                        return self.malformed("STRUCT_NAME record without a struct");
                    }
                    return self.check_type_nesting();
                }
                Entry::SubBlock(_) => return self.malformed("sub-block inside the type table"),
                Entry::Record(a) => a,
            };
            let r = self.cur.read_record(a)?;
            let ops = &r.ops;
            let mut contained: Vec<u32> = Vec::new();
            let result: TypeId = match r.code {
                TYPE_CODE_NUMENTRY => {
                    if ops.is_empty() || sized {
                        return self.malformed("invalid NUMENTRY record");
                    }
                    // Each entry takes at least one bit of the stream.
                    if ops[0] > self.cur.reader().len_bits() {
                        return self.malformed("type count is not plausible");
                    }
                    sized = true;
                    self.bc_types = vec![None; ops[0] as usize];
                    self.bc_contained = vec![Vec::new(); ops[0] as usize];
                    forward_structs = vec![false; ops[0] as usize];
                    continue;
                }
                TYPE_CODE_VOID => self.m.types.intern(Type::Void),
                TYPE_CODE_HALF => self.m.types.intern(Type::Half),
                TYPE_CODE_BFLOAT => self.m.types.intern(Type::BFloat),
                TYPE_CODE_FLOAT => self.m.types.intern(Type::Float),
                TYPE_CODE_DOUBLE => self.m.types.intern(Type::Double),
                TYPE_CODE_X86_FP80 => self.m.types.intern(Type::X86Fp80),
                TYPE_CODE_FP128 => self.m.types.intern(Type::Fp128),
                TYPE_CODE_PPC_FP128 => self.m.types.intern(Type::PpcFp128),
                TYPE_CODE_LABEL => self.m.types.intern(Type::Label),
                TYPE_CODE_METADATA => self.m.types.intern(Type::Metadata),
                TYPE_CODE_X86_MMX => {
                    // Deprecated: decodes as <1 x i64>.
                    let i64t = self.m.types.int(64);
                    self.m.types.vector(1, i64t)
                }
                TYPE_CODE_X86_AMX => self.m.types.intern(Type::X86Amx),
                TYPE_CODE_TOKEN => self.m.types.intern(Type::Token),
                TYPE_CODE_INTEGER => {
                    let bits = match ops.first() {
                        Some(&b) if (1..=(1 << 23)).contains(&b) => b as u32,
                        _ => return self.malformed("integer type width out of range"),
                    };
                    self.m.types.int(bits)
                }
                TYPE_CODE_POINTER => {
                    if ops.is_empty() || ops.len() > 2 {
                        return self.malformed("invalid pointer type record");
                    }
                    let addrspace = if ops.len() == 2 { ops[1] } else { 0 };
                    if addrspace >= 1 << 24 {
                        return self.malformed("pointer address space out of range");
                    }
                    let pointee = match self.type_or_forward(ops[0], &mut forward_structs)? {
                        Some(t) => t,
                        None => return self.malformed("pointer to an invalid type"),
                    };
                    if matches!(
                        self.m.types.get(pointee),
                        Type::Void | Type::Label | Type::Metadata | Type::Token | Type::X86Amx
                    ) {
                        return self.malformed("invalid pointee type");
                    }
                    contained.push(ops[0] as u32);
                    self.m.types.ptr(addrspace as u32)
                }
                TYPE_CODE_OPAQUE_POINTER => {
                    if ops.len() != 1 || ops[0] >= 1 << 24 {
                        return self.malformed("invalid opaque pointer record");
                    }
                    self.m.types.ptr(ops[0] as u32)
                }
                TYPE_CODE_FUNCTION_OLD | TYPE_CODE_FUNCTION => {
                    // FUNCTION: [vararg, retty, paramty x N]; FUNCTION_OLD: [vararg, attrid, retty, paramty x N].
                    let first = if r.code == TYPE_CODE_FUNCTION { 1 } else { 2 };
                    if ops.len() < first + 1 {
                        return self.malformed("invalid function type record");
                    }
                    let ret = match self.type_or_forward(ops[first], &mut forward_structs)? {
                        Some(t) => t,
                        None => return self.malformed("function type with an invalid return type"),
                    };
                    if matches!(self.m.types.get(ret), Type::Function { .. } | Type::Label | Type::Metadata) {
                        return self.malformed("invalid function return type");
                    }
                    let mut params = Vec::new();
                    for &p in &ops[first + 1..] {
                        let t = match self.type_or_forward(p, &mut forward_structs)? {
                            Some(t) => t,
                            None => return self.malformed("function type with an invalid parameter type"),
                        };
                        if !self.m.types.is_first_class(t) {
                            return self.malformed("invalid function argument type");
                        }
                        params.push(t);
                    }
                    if ops[0] > 1 {
                        return self.malformed("invalid vararg flag");
                    }
                    contained.extend(ops[first..].iter().map(|&x| x as u32));
                    self.m.types.intern(Type::Function { ret, params, vararg: ops[0] == 1 })
                }
                TYPE_CODE_STRUCT_ANON => {
                    if ops.is_empty() || ops[0] > 1 {
                        return self.malformed("invalid anonymous struct record");
                    }
                    let mut elems = Vec::new();
                    for &e in &ops[1..] {
                        match self.type_or_forward(e, &mut forward_structs)? {
                            Some(t) => elems.push(self.check_struct_element(t)?),
                            None => return self.malformed("struct with an invalid element type"),
                        }
                    }
                    contained.extend(ops[1..].iter().map(|&x| x as u32));
                    self.m.types.intern(Type::LiteralStruct { packed: ops[0] == 1, elems })
                }
                TYPE_CODE_STRUCT_NAME => {
                    if name.is_some() {
                        return self.malformed("two STRUCT_NAME records in a row");
                    }
                    name = Some(self.record_string(ops)?);
                    continue;
                }
                TYPE_CODE_STRUCT_NAMED | TYPE_CODE_OPAQUE => {
                    if num_records >= self.bc_types.len() {
                        return self.malformed("type table holds more types than it declares");
                    }
                    let named = r.code == TYPE_CODE_STRUCT_NAMED;
                    if named && (ops.is_empty() || ops[0] > 1) {
                        return self.malformed("invalid named struct record");
                    }
                    if !named && ops.len() != 1 {
                        return self.malformed("invalid opaque type record");
                    }
                    let n = name.take().unwrap_or_default();
                    let id = match self.bc_types[num_records] {
                        Some(t) if forward_structs[num_records] => {
                            forward_structs[num_records] = false;
                            t
                        }
                        Some(_) => return self.malformed("type table entry defined twice"),
                        None => self.m.types.new_struct(String::new()),
                    };
                    let unique = self.unique_struct_name(n);
                    self.m.types.set_struct_name(id, unique);
                    if named {
                        let mut elems = Vec::new();
                        for &e in &ops[1..] {
                            match self.type_or_forward(e, &mut forward_structs)? {
                                Some(t) => elems.push(self.check_struct_element(t)?),
                                None => return self.malformed("struct with an invalid element type"),
                            }
                        }
                        if elems.contains(&id) {
                            return self.malformed("struct contains itself");
                        }
                        self.m.types.set_struct_body(id, ops[0] == 1, elems);
                        contained.extend(ops[1..].iter().map(|&x| x as u32));
                    }
                    self.bc_types[num_records] = None;
                    id
                }
                TYPE_CODE_TARGET_TYPE => {
                    if ops.is_empty() || ops[0] as usize >= ops.len() {
                        return self.malformed("invalid target extension type record");
                    }
                    let n = ops[0] as usize;
                    let mut types = Vec::new();
                    for &t in &ops[1..=n] {
                        types.push(self.ty(t)?);
                    }
                    let mut ints = Vec::new();
                    for &i in &ops[n + 1..] {
                        if i > u32::MAX as u64 {
                            return self.malformed("target extension type parameter too large");
                        }
                        ints.push(i as u32);
                    }
                    let tname = match name.take() {
                        Some(s) => s,
                        None => return self.malformed("target extension type without a name"),
                    };
                    self.m.types.intern(Type::TargetExt { name: tname, types, ints })
                }
                TYPE_CODE_ARRAY => {
                    if ops.len() < 2 {
                        return self.malformed("invalid array type record");
                    }
                    let elem = match self.type_or_forward(ops[1], &mut forward_structs)? {
                        Some(t) => t,
                        None => return self.malformed("array of an invalid type"),
                    };
                    if matches!(
                        self.m.types.get(elem),
                        Type::Void | Type::Label | Type::Metadata | Type::Function { .. } | Type::Token | Type::X86Amx
                    ) || matches!(self.m.types.get(elem), Type::Vector { scalable: true, .. })
                    {
                        return self.malformed("invalid array element type");
                    }
                    contained.push(ops[1] as u32);
                    self.m.types.intern(Type::Array { len: ops[0], elem })
                }
                TYPE_CODE_VECTOR => {
                    if ops.len() < 2 || ops.len() > 3 {
                        return self.malformed("invalid vector type record");
                    }
                    if ops[0] == 0 || ops[0] > u32::MAX as u64 {
                        return self.malformed("invalid vector length");
                    }
                    let elem = match self.type_or_forward(ops[1], &mut forward_structs)? {
                        Some(t) => t,
                        None => return self.malformed("vector of an invalid type"),
                    };
                    if !(self.m.types.is_int(elem) || self.m.types.is_float(elem) || self.m.types.is_ptr(elem)) {
                        return self.malformed("invalid vector element type");
                    }
                    let scalable = ops.len() > 2 && ops[2] != 0;
                    contained.push(ops[1] as u32);
                    self.m.types.intern(Type::Vector { len: ops[0] as u32, elem, scalable })
                }
                c => return self.malformed(format!("unknown type record code {}", c)),
            };
            if !sized || num_records >= self.bc_types.len() {
                return self.malformed("type table holds more types than it declares");
            }
            if self.bc_types[num_records].is_some() {
                return self.malformed("only named structs can be forward referenced in the type table");
            }
            if name.is_some() && !matches!(r.code, TYPE_CODE_STRUCT_NAMED | TYPE_CODE_OPAQUE | TYPE_CODE_TARGET_TYPE) {
                return self.malformed("STRUCT_NAME record not followed by a named type");
            }
            self.bc_types[num_records] = Some(result);
            self.bc_contained[num_records] = contained;
            num_records += 1;
        }
    }

    /// Rejects structs that contain themselves (`StructType::setBodyOrError`), and types nested deeper than
    /// [`MAX_TYPE_DEPTH`], a limit of this reader (every later stage walks types recursively).
    fn check_type_nesting(&self) -> Result<()> {
        let n = self.m.types.len();
        // 0 unvisited, 1 on the walk, 2 done.
        let mut state = vec![0u8; n];
        let mut depth = vec![0u32; n];
        for root in 0..n {
            if state[root] != 0 {
                continue;
            }
            let children = |t: usize| type_children(self.m.types.get(TypeId(t as u32)));
            let mut stack: Vec<(usize, Vec<TypeId>, usize)> = vec![(root, children(root), 0)];
            state[root] = 1;
            while let Some((t, kids, next)) = stack.last_mut() {
                let t = *t;
                if *next < kids.len() {
                    let c = kids[*next].0 as usize;
                    *next += 1;
                    match state[c] {
                        0 => {
                            state[c] = 1;
                            stack.push((c, children(c), 0));
                        }
                        1 => return self.malformed("recursive struct type"),
                        _ => {}
                    }
                    continue;
                }
                let d = 1 + kids.iter().map(|c| depth[c.0 as usize]).max().unwrap_or(0);
                if d > MAX_TYPE_DEPTH {
                    return self.unsupported(format!("types nested more than {} deep", MAX_TYPE_DEPTH));
                }
                depth[t] = d;
                state[t] = 2;
                stack.pop();
            }
        }
        Ok(())
    }

    fn check_struct_element(&self, t: TypeId) -> Result<TypeId> {
        if matches!(
            self.m.types.get(t),
            Type::Void | Type::Label | Type::Metadata | Type::Function { .. } | Type::Token
        ) {
            return self.malformed("invalid struct element type");
        }
        Ok(t)
    }

    /// The name LLVM gives a struct whose name may collide (`StructType::setName` appends `.N`).
    fn unique_struct_name(&mut self, name: String) -> String {
        if name.is_empty() {
            return name;
        }
        let taken = |m: &crate::ir::Module, n: &str| {
            m.types.iter().any(|(_, t)| matches!(t, Type::Struct { name, .. } if name == n))
        };
        if !taken(&self.m, &name) {
            return name;
        }
        loop {
            let candidate = format!("{}.{}", name, self.md.struct_unique_id);
            self.md.struct_unique_id += 1;
            if !taken(&self.m, &candidate) {
                return candidate;
            }
        }
    }
}

/// How deep types may nest: a limit of this reader, far above anything a shader declares.
pub(crate) const MAX_TYPE_DEPTH: u32 = 512;

fn type_children(t: &Type) -> Vec<TypeId> {
    match t {
        Type::Function { ret, params, .. } => std::iter::once(*ret).chain(params.iter().copied()).collect(),
        Type::LiteralStruct { elems, .. } => elems.clone(),
        Type::Struct { body: Some(e), .. } => e.clone(),
        Type::Array { elem, .. } | Type::Vector { elem, .. } => vec![*elem],
        Type::TargetExt { types, .. } => types.clone(),
        _ => vec![],
    }
}
