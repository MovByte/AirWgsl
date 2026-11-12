//! Types: an interned table, as LLVM 21 has them (opaque pointers).
//!
//! Every type but an identified struct is structural: the table holds one entry per distinct type, so two
//! [`TypeId`]s are equal exactly when the types are. Identified structs (named, or numbered when the name is empty)
//! are distinct entries even when their bodies are equal, as in LLVM.
//!
//! Pointers carry only their address space. Bitcode written with typed pointers (as AIR is) records pointee types;
//! the reader keeps them in [`crate::ir::Module::bitcode_types`] for the values they describe, outside this table.

use std::collections::HashMap;

/// Index of a type in a [`TypeTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub u32);

/// A type. Containers name their parts by [`TypeId`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    /// IEEE binary16.
    Half,
    /// bfloat16.
    BFloat,
    /// IEEE binary32.
    Float,
    /// IEEE binary64.
    Double,
    X86Fp80,
    Fp128,
    PpcFp128,
    Label,
    Metadata,
    Token,
    X86Amx,
    /// An integer of 1 to 2^23 bits.
    Int(u32),
    /// A pointer in an address space.
    Ptr(u32),
    Function {
        ret: TypeId,
        params: Vec<TypeId>,
        vararg: bool,
    },
    /// A literal (structurally uniqued) struct.
    LiteralStruct {
        packed: bool,
        elems: Vec<TypeId>,
    },
    /// An identified struct; `body` is `None` while opaque. Its name is empty when it is numbered.
    Struct {
        name: String,
        packed: bool,
        body: Option<Vec<TypeId>>,
    },
    Array {
        len: u64,
        elem: TypeId,
    },
    Vector {
        len: u32,
        elem: TypeId,
        scalable: bool,
    },
    TargetExt {
        name: String,
        types: Vec<TypeId>,
        ints: Vec<u32>,
    },
}

/// The floating-point formats, for constants of floating-point type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatKind {
    Half,
    BFloat,
    Float,
    Double,
    X86Fp80,
    Fp128,
    PpcFp128,
}

impl FloatKind {
    /// Width of the encoding in bits.
    pub fn bits(self) -> u32 {
        match self {
            FloatKind::Half | FloatKind::BFloat => 16,
            FloatKind::Float => 32,
            FloatKind::Double => 64,
            FloatKind::X86Fp80 => 80,
            FloatKind::Fp128 | FloatKind::PpcFp128 => 128,
        }
    }
}

/// The interned types of a module.
#[derive(Clone, Debug, Default)]
pub struct TypeTable {
    types: Vec<Type>,
    unique: HashMap<Type, TypeId>,
}

impl TypeTable {
    pub fn new() -> TypeTable {
        TypeTable::default()
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn get(&self, id: TypeId) -> &Type {
        &self.types[id.0 as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = (TypeId, &Type)> {
        self.types.iter().enumerate().map(|(i, t)| (TypeId(i as u32), t))
    }

    /// The id of a structural type, adding it if new. Identified structs go through [`TypeTable::new_struct`].
    pub fn intern(&mut self, ty: Type) -> TypeId {
        debug_assert!(!matches!(ty, Type::Struct { .. }), "identified structs are not interned");
        if let Some(&id) = self.unique.get(&ty) {
            return id;
        }
        let id = TypeId(self.types.len() as u32);
        self.types.push(ty.clone());
        self.unique.insert(ty, id);
        id
    }

    /// A new identified struct, opaque until [`TypeTable::set_struct_body`].
    pub fn new_struct(&mut self, name: String) -> TypeId {
        let id = TypeId(self.types.len() as u32);
        self.types.push(Type::Struct { name, packed: false, body: None });
        id
    }

    pub fn set_struct_body(&mut self, id: TypeId, packed_: bool, elems: Vec<TypeId>) {
        if let Type::Struct { packed, body, .. } = &mut self.types[id.0 as usize] {
            *packed = packed_;
            *body = Some(elems);
        }
    }

    pub fn set_struct_name(&mut self, id: TypeId, name_: String) {
        if let Type::Struct { name, .. } = &mut self.types[id.0 as usize] {
            *name = name_;
        }
    }

    pub fn void(&mut self) -> TypeId {
        self.intern(Type::Void)
    }

    pub fn int(&mut self, bits: u32) -> TypeId {
        self.intern(Type::Int(bits))
    }

    pub fn ptr(&mut self, addrspace: u32) -> TypeId {
        self.intern(Type::Ptr(addrspace))
    }

    pub fn vector(&mut self, len: u32, elem: TypeId) -> TypeId {
        self.intern(Type::Vector { len, elem, scalable: false })
    }

    pub fn is_void(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Void)
    }

    pub fn is_label(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Label)
    }

    pub fn is_metadata(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Metadata)
    }

    pub fn is_int(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Int(_))
    }

    pub fn int_bits(&self, t: TypeId) -> Option<u32> {
        match self.get(t) {
            Type::Int(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_ptr(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Ptr(_))
    }

    pub fn is_vector(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Vector { .. })
    }

    pub fn is_struct(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Struct { .. } | Type::LiteralStruct { .. })
    }

    pub fn is_array(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Array { .. })
    }

    pub fn is_function(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Function { .. })
    }

    pub fn float_kind(&self, t: TypeId) -> Option<FloatKind> {
        Some(match self.get(t) {
            Type::Half => FloatKind::Half,
            Type::BFloat => FloatKind::BFloat,
            Type::Float => FloatKind::Float,
            Type::Double => FloatKind::Double,
            Type::X86Fp80 => FloatKind::X86Fp80,
            Type::Fp128 => FloatKind::Fp128,
            Type::PpcFp128 => FloatKind::PpcFp128,
            _ => return None,
        })
    }

    pub fn is_float(&self, t: TypeId) -> bool {
        self.float_kind(t).is_some()
    }

    /// The element type of a vector, or the type itself.
    pub fn scalar(&self, t: TypeId) -> TypeId {
        match self.get(t) {
            Type::Vector { elem, .. } => *elem,
            _ => t,
        }
    }

    pub fn is_int_or_int_vector(&self, t: TypeId) -> bool {
        self.is_int(self.scalar(t))
    }

    pub fn is_fp_or_fp_vector(&self, t: TypeId) -> bool {
        self.is_float(self.scalar(t))
    }

    pub fn is_ptr_or_ptr_vector(&self, t: TypeId) -> bool {
        self.is_ptr(self.scalar(t))
    }

    /// Address space of a pointer or vector of pointers.
    pub fn pointer_addrspace(&self, t: TypeId) -> Option<u32> {
        match self.get(self.scalar(t)) {
            Type::Ptr(a) => Some(*a),
            _ => None,
        }
    }

    /// Size in bits of a primitive scalar (integer or floating point) or of the elements of a vector of them; 0 for
    /// anything else, as `Type::getScalarSizeInBits` (pointers give 0: their size comes from the data layout).
    pub fn scalar_bits(&self, t: TypeId) -> u32 {
        let s = self.scalar(t);
        match self.get(s) {
            Type::Int(b) => *b,
            _ => self.float_kind(s).map(|k| k.bits()).unwrap_or(0),
        }
    }

    /// Element types of a struct (literal or identified with a body).
    pub fn struct_elems(&self, t: TypeId) -> Option<&[TypeId]> {
        match self.get(t) {
            Type::LiteralStruct { elems, .. } => Some(elems),
            Type::Struct { body: Some(elems), .. } => Some(elems),
            _ => None,
        }
    }

    /// Element count of a fixed vector.
    pub fn vector_len(&self, t: TypeId) -> Option<u32> {
        match self.get(t) {
            Type::Vector { len, scalable: false, .. } => Some(*len),
            _ => None,
        }
    }

    /// Return type, parameter types and varargs flag of a function type.
    pub fn function(&self, t: TypeId) -> Option<(TypeId, &[TypeId], bool)> {
        match self.get(t) {
            Type::Function { ret, params, vararg } => Some((*ret, params, *vararg)),
            _ => None,
        }
    }

    /// The type an aggregate index selects (`extractvalue`/`insertvalue`, and GEP indices into structs and arrays).
    pub fn indexed(&self, t: TypeId, index: u64) -> Option<TypeId> {
        match self.get(t) {
            Type::Array { elem, .. } => Some(*elem),
            Type::Vector { elem, .. } => Some(*elem),
            Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => {
                elems.get(index as usize).copied()
            }
            _ => None,
        }
    }

    /// `Type::isFirstClassType`: anything but function and void.
    pub fn is_first_class(&self, t: TypeId) -> bool {
        !matches!(self.get(t), Type::Function { .. } | Type::Void)
    }

    /// `Type::isSingleValueType`: integer, floating point, pointer, vector, x86_amx and target extension types.
    pub fn is_single_value(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Int(_) | Type::Ptr(_) | Type::Vector { .. } | Type::X86Amx | Type::TargetExt { .. })
            || self.is_float(t)
    }

    /// `Type::isAggregateType`: struct or array.
    pub fn is_aggregate(&self, t: TypeId) -> bool {
        matches!(self.get(t), Type::Struct { .. } | Type::LiteralStruct { .. } | Type::Array { .. })
    }

    /// `Type::isSized`: whether the type has a size (opaque structs, functions, labels, metadata, void and tokens do
    /// not; target extension types are treated as unsized here, as their layout comes from their properties).
    pub fn is_sized(&self, t: TypeId) -> bool {
        let mut visiting = Vec::new();
        self.is_sized_inner(t, &mut visiting)
    }

    fn is_sized_inner(&self, t: TypeId, visiting: &mut Vec<TypeId>) -> bool {
        match self.get(t) {
            Type::Int(_) | Type::Ptr(_) | Type::X86Amx => true,
            Type::Half | Type::BFloat | Type::Float | Type::Double | Type::X86Fp80 | Type::Fp128 | Type::PpcFp128 => {
                true
            }
            Type::Void
            | Type::Label
            | Type::Metadata
            | Type::Token
            | Type::Function { .. }
            | Type::TargetExt { .. } => false,
            Type::Array { elem, .. } | Type::Vector { elem, .. } => self.is_sized_inner(*elem, visiting),
            Type::LiteralStruct { elems, .. } => elems.iter().all(|e| self.is_sized_inner(*e, visiting)),
            Type::Struct { body, .. } => match body {
                None => false,
                Some(elems) => {
                    if visiting.contains(&t) {
                        return false;
                    }
                    visiting.push(t);
                    let r = elems.iter().all(|e| self.is_sized_inner(*e, visiting));
                    visiting.pop();
                    r
                }
            },
        }
    }
}
