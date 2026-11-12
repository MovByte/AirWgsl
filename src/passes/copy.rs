//! Copying module contents into another module with every index remapped: types, attribute sets, constants, globals,
//! function bodies and metadata. Linking uses it to bring definitions from imported modules in; extraction uses it
//! to build the canonical module from the entry point and what it reaches.
//!
//! Copies are lazy: copying a function copies what its body refers to. Identified struct types keep their names
//! (a clash in the destination gets a numeric suffix, as LLVM's type mapper does). A reference from metadata to a
//! global that the copy does not keep becomes a null operand, which is what LLVM's metadata becomes when the global
//! is deleted.

use std::collections::HashMap;

use crate::ir::*;

/// Which globals a copy brings along when something refers to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    /// Everything referred to is copied (with function bodies).
    All,
    /// References from metadata to globals not yet copied become null; references from code and constants copy.
    CodeOnly,
}

pub struct Copier<'s> {
    pub src: &'s Module,
    pub reach: Reach,
    /// Functions first reached from metadata are copied as declarations (with external linkage, as a declaration
    /// must have): metadata names them, the copied code does not call them.
    pub metadata_functions_as_declarations: bool,
    from_metadata: bool,
    types: HashMap<TypeId, TypeId>,
    consts: HashMap<ConstId, ConstId>,
    pub globals: HashMap<GlobalId, GlobalId>,
    mds: HashMap<MdId, MdId>,
    attrs: HashMap<AttrSetId, AttrSetId>,
    bitcode_types: HashMap<u32, u32>,
    inst_maps: HashMap<u32, HashMap<InstId, InstId>>,
    /// Functions whose bodies are still to be copied.
    pending: Vec<(u32, u32)>,
}

impl<'s> Copier<'s> {
    pub fn new(src: &'s Module, reach: Reach) -> Copier<'s> {
        Copier {
            src,
            reach,
            metadata_functions_as_declarations: false,
            from_metadata: false,
            types: HashMap::new(),
            consts: HashMap::new(),
            globals: HashMap::new(),
            mds: HashMap::new(),
            attrs: HashMap::new(),
            bitcode_types: HashMap::new(),
            inst_maps: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// Maps source global `g` to an existing destination global (linking resolves declarations this way).
    pub fn bind(&mut self, g: GlobalId, to: GlobalId) {
        self.globals.insert(g, to);
    }

    // Types -----------------------------------------------------------------------------------------------------------

    pub fn ty(&mut self, dst: &mut Module, t: TypeId) -> TypeId {
        if let Some(&d) = self.types.get(&t) {
            return d;
        }
        let d = match self.src.types.get(t).clone() {
            Type::Struct { name, packed, body } => {
                let name = unique_struct_name(dst, &name);
                let id = dst.types.new_struct(name);
                self.types.insert(t, id);
                if let Some(elems) = body {
                    let e: Vec<TypeId> = elems.iter().map(|&x| self.ty(dst, x)).collect();
                    dst.types.set_struct_body(id, packed, e);
                }
                return id;
            }
            Type::Function { ret, params, vararg } => {
                let ret = self.ty(dst, ret);
                let params = params.iter().map(|&p| self.ty(dst, p)).collect();
                dst.types.intern(Type::Function { ret, params, vararg })
            }
            Type::LiteralStruct { packed, elems } => {
                let elems = elems.iter().map(|&e| self.ty(dst, e)).collect();
                dst.types.intern(Type::LiteralStruct { packed, elems })
            }
            Type::Array { len, elem } => {
                let elem = self.ty(dst, elem);
                dst.types.intern(Type::Array { len, elem })
            }
            Type::Vector { len, elem, scalable } => {
                let elem = self.ty(dst, elem);
                dst.types.intern(Type::Vector { len, elem, scalable })
            }
            Type::TargetExt { name, types, ints } => {
                let types = types.iter().map(|&x| self.ty(dst, x)).collect();
                dst.types.intern(Type::TargetExt { name, types, ints })
            }
            other => dst.types.intern(other),
        };
        self.types.insert(t, d);
        d
    }

    fn bitcode_type(&mut self, dst: &mut Module, i: u32) -> u32 {
        if let Some(&d) = self.bitcode_types.get(&i) {
            return d;
        }
        let Some(bt) = self.src.bitcode_types.get(i as usize).cloned() else { return i };
        let id = dst.bitcode_types.len() as u32;
        dst.bitcode_types.push(BitcodeType { ty: TypeId(0), contained: Vec::new() });
        self.bitcode_types.insert(i, id);
        let ty = self.ty(dst, bt.ty);
        let contained = bt.contained.iter().map(|&c| self.bitcode_type(dst, c)).collect();
        dst.bitcode_types[id as usize] = BitcodeType { ty, contained };
        id
    }

    pub fn attr_set(&mut self, dst: &mut Module, a: AttrSetId) -> AttrSetId {
        if let Some(&d) = self.attrs.get(&a) {
            return d;
        }
        let mut set = AttrSet::new();
        for at in self.src.attrs(a).clone().iter() {
            set.add(match at {
                Attribute::Type(k, Some(t)) => Attribute::Type(*k, Some(self.ty(dst, *t))),
                other => other.clone(),
            });
        }
        let d = dst.attr_set(set);
        self.attrs.insert(a, d);
        d
    }

    pub fn attr_list(&mut self, dst: &mut Module, l: &AttrList) -> AttrList {
        AttrList {
            function: self.attr_set(dst, l.function),
            ret: self.attr_set(dst, l.ret),
            params: l.params.iter().map(|&p| self.attr_set(dst, p)).collect(),
        }
    }

    // Values ----------------------------------------------------------------------------------------------------------

    /// A module-level value (local values are unchanged: function bodies keep their numbering).
    pub fn value(&mut self, dst: &mut Module, v: Value) -> Value {
        match v {
            Value::Const(c) => Value::Const(self.constant(dst, c)),
            Value::Global(g) => Value::Global(self.global(dst, g)),
            Value::Metadata(md) => {
                dst.types.intern(Type::Metadata);
                // Metadata a call passes is part of the code: it is copied whole.
                let saved = self.reach;
                self.reach = Reach::All;
                let d = self.md(dst, md);
                self.reach = saved;
                Value::Metadata(d.expect("metadata copied whole"))
            }
            Value::InlineAsm(i) => {
                let mut a = self.src.inline_asms[i as usize].clone();
                a.fn_ty = self.ty(dst, a.fn_ty);
                dst.inline_asms.push(a);
                Value::InlineAsm(dst.inline_asms.len() as u32 - 1)
            }
            local => local,
        }
    }

    pub fn constant(&mut self, dst: &mut Module, c: ConstId) -> ConstId {
        if let Some(&d) = self.consts.get(&c) {
            return d;
        }
        let k = self.src.constant(c).clone();
        let ty = self.ty(dst, k.ty);
        let vals = |this: &mut Self, dst: &mut Module, vs: &[Value]| -> Vec<Value> {
            vs.iter().map(|&v| this.value(dst, v)).collect()
        };
        let kind = match k.kind {
            ConstKind::Struct(v) => ConstKind::Struct(vals(self, dst, &v)),
            ConstKind::Array(v) => ConstKind::Array(vals(self, dst, &v)),
            ConstKind::Vector(v) => ConstKind::Vector(vals(self, dst, &v)),
            ConstKind::Expr(e) => ConstKind::Expr(match e {
                ConstExpr::Cast { op, value } => ConstExpr::Cast { op, value: self.value(dst, value) },
                ConstExpr::Binary { op, lhs, rhs, flags } => {
                    ConstExpr::Binary { op, lhs: self.value(dst, lhs), rhs: self.value(dst, rhs), flags }
                }
                ConstExpr::Gep { source, flags, inrange, ptr, indices } => ConstExpr::Gep {
                    source: self.ty(dst, source),
                    flags,
                    inrange,
                    ptr: self.value(dst, ptr),
                    indices: vals(self, dst, &indices),
                },
                ConstExpr::ExtractElement { vec, idx } => {
                    ConstExpr::ExtractElement { vec: self.value(dst, vec), idx: self.value(dst, idx) }
                }
                ConstExpr::InsertElement { vec, elt, idx } => ConstExpr::InsertElement {
                    vec: self.value(dst, vec),
                    elt: self.value(dst, elt),
                    idx: self.value(dst, idx),
                },
                ConstExpr::ShuffleVector { a, b, mask } => {
                    ConstExpr::ShuffleVector { a: self.value(dst, a), b: self.value(dst, b), mask }
                }
            }),
            ConstKind::BlockAddress { function, block } => {
                let GlobalId::Function(f) = self.global(dst, GlobalId::Function(function)) else { unreachable!() };
                ConstKind::BlockAddress { function: f, block }
            }
            ConstKind::DsoLocalEquivalent(g) => ConstKind::DsoLocalEquivalent(self.global(dst, g)),
            ConstKind::NoCfi(g) => ConstKind::NoCfi(self.global(dst, g)),
            ConstKind::PtrAuth { ptr, key, disc, addr_disc } => ConstKind::PtrAuth {
                ptr: self.value(dst, ptr),
                key: self.value(dst, key),
                disc: self.value(dst, disc),
                addr_disc: self.value(dst, addr_disc),
            },
            other => other,
        };
        let d = dst.intern_const(Constant { ty, kind });
        self.consts.insert(c, d);
        d
    }

    pub fn global(&mut self, dst: &mut Module, g: GlobalId) -> GlobalId {
        if let Some(&d) = self.globals.get(&g) {
            return d;
        }
        let src = self.src;
        // A global is a value of pointer type: the type table has that type, as LLVM's context always does.
        dst.types.ptr(src.global(g).addrspace);
        match g {
            GlobalId::Var(i) => {
                let gv = &src.globals[i as usize];
                let mut common = gv.common.clone();
                common.name = unique_global_name(dst, &common.name, common.linkage);
                let id = GlobalId::Var(dst.globals.len() as u32);
                dst.globals.push(GlobalVariable {
                    common,
                    value_type: TypeId(0),
                    constant: gv.constant,
                    initializer: None,
                    externally_initialized: gv.externally_initialized,
                    align: gv.align,
                    section: gv.section.clone(),
                    comdat: None,
                    attrs: AttrSetId::EMPTY,
                    code_model: gv.code_model,
                    sanitizer: gv.sanitizer,
                    md: Vec::new(),
                    bitcode_type: None,
                });
                self.globals.insert(g, id);
                let k = id_index(id);
                let vt = self.ty(dst, gv.value_type);
                let init = gv.initializer.map(|v| self.value(dst, v));
                let comdat = gv.comdat.map(|c| self.comdat(dst, c));
                let attrs = self.attr_set(dst, gv.attrs);
                let md = self.attachments(dst, &gv.md);
                let bt = gv.bitcode_type.map(|b| self.bitcode_type(dst, b));
                let d = &mut dst.globals[k];
                d.value_type = vt;
                d.initializer = init;
                d.comdat = comdat;
                d.attrs = attrs;
                d.md = md;
                d.bitcode_type = bt;
                id
            }
            GlobalId::Function(i) => {
                let f = &src.functions[i as usize];
                let mut common = f.common.clone();
                common.name = unique_global_name(dst, &common.name, common.linkage);
                let id = dst.functions.len() as u32;
                dst.functions.push(Function {
                    common,
                    ty: TypeId(0),
                    cc: f.cc,
                    attrs: AttrList::default(),
                    align: f.align,
                    section: f.section.clone(),
                    comdat: None,
                    gc: f.gc.clone(),
                    prefix: None,
                    prologue: None,
                    personality: None,
                    args: Vec::new(),
                    blocks: Vec::new(),
                    insts: Vec::new(),
                    md: Vec::new(),
                    bitcode_type: None,
                });
                self.globals.insert(g, GlobalId::Function(id));
                self.function_header(dst, i, id);
                if self.from_metadata && self.metadata_functions_as_declarations {
                    let df = &mut dst.functions[id as usize];
                    df.common.linkage = Linkage::External;
                    df.common.visibility = Visibility::Default;
                    df.section = None;
                    df.comdat = None;
                } else if !f.is_declaration() {
                    self.pending.push((i, id));
                }
                GlobalId::Function(id)
            }
            GlobalId::Alias(i) => {
                let a = &src.aliases[i as usize];
                let mut common = a.common.clone();
                common.name = unique_global_name(dst, &common.name, common.linkage);
                let id = dst.aliases.len() as u32;
                dst.aliases.push(Alias { common, value_type: TypeId(0), aliasee: Value::Arg(0) });
                self.globals.insert(g, GlobalId::Alias(id));
                let vt = self.ty(dst, a.value_type);
                let al = self.value(dst, a.aliasee);
                dst.aliases[id as usize].value_type = vt;
                dst.aliases[id as usize].aliasee = al;
                GlobalId::Alias(id)
            }
            GlobalId::IFunc(i) => {
                let a = &src.ifuncs[i as usize];
                let mut common = a.common.clone();
                common.name = unique_global_name(dst, &common.name, common.linkage);
                let id = dst.ifuncs.len() as u32;
                dst.ifuncs.push(IFunc { common, value_type: TypeId(0), resolver: Value::Arg(0) });
                self.globals.insert(g, GlobalId::IFunc(id));
                let vt = self.ty(dst, a.value_type);
                let r = self.value(dst, a.resolver);
                dst.ifuncs[id as usize].value_type = vt;
                dst.ifuncs[id as usize].resolver = r;
                GlobalId::IFunc(id)
            }
        }
    }

    fn comdat(&mut self, dst: &mut Module, c: u32) -> u32 {
        let cd = self.src.comdats[c as usize].clone();
        if let Some(i) = dst.comdats.iter().position(|x| x.name == cd.name) {
            return i as u32;
        }
        dst.comdats.push(cd);
        dst.comdats.len() as u32 - 1
    }

    /// Copies a function's signature and attributes into destination function `d`.
    pub fn function_header(&mut self, dst: &mut Module, s: u32, d: u32) {
        let f = &self.src.functions[s as usize];
        let ty = self.ty(dst, f.ty);
        let attrs = self.attr_list(dst, &f.attrs);
        let args: Vec<Argument> =
            f.args.iter().map(|a| Argument { ty: self.ty(dst, a.ty), name: a.name.clone() }).collect();
        let comdat = f.comdat.map(|c| self.comdat(dst, c));
        let prefix = f.prefix.map(|v| self.value(dst, v));
        let prologue = f.prologue.map(|v| self.value(dst, v));
        let personality = f.personality.map(|v| self.value(dst, v));
        let md = self.attachments(dst, &f.md);
        let bt = f.bitcode_type.map(|b| self.bitcode_type(dst, b));
        let df = &mut dst.functions[d as usize];
        df.ty = ty;
        df.attrs = attrs;
        df.args = args;
        df.comdat = comdat;
        df.prefix = prefix;
        df.prologue = prologue;
        df.personality = personality;
        df.md = md;
        df.bitcode_type = bt;
    }

    /// The new numbers of function `s`'s live instructions: their order in its blocks. Copies leave out the arena
    /// slots of removed instructions.
    fn inst_map(&mut self, s: u32) -> &HashMap<InstId, InstId> {
        let src = self.src;
        self.inst_maps.entry(s).or_insert_with(|| {
            let f = &src.functions[s as usize];
            let mut map = HashMap::new();
            for b in &f.blocks {
                for &i in &b.insts {
                    let n = InstId(map.len() as u32);
                    map.insert(i, n);
                }
            }
            map
        })
    }

    /// Copies the body of source function `s` into destination function `d` (live instructions only).
    pub fn function_body(&mut self, dst: &mut Module, s: u32, d: u32) {
        let map = self.inst_map(s).clone();
        let f = &self.src.functions[s as usize];
        let mut order: Vec<InstId> = f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
        order.sort_by_key(|i| map[i].0);
        let mut insts = Vec::with_capacity(order.len());
        for i in order {
            let inst = f.inst(i);
            let ty = self.ty(dst, inst.ty);
            let mut op = self.op(dst, &inst.op, s, d);
            for v in super::util::operands_mut(&mut op) {
                if let Value::Inst(x) = v {
                    *x = *map.get(x).expect("an operand that is not a live instruction");
                }
            }
            let md = self.attachments(dst, &inst.md);
            insts.push(Instruction { ty, name: inst.name.clone(), op, md, dbg: None });
        }
        let blocks = f
            .blocks
            .iter()
            .map(|b| Block {
                name: b.name.clone(),
                insts: b.insts.iter().map(|i| map[i]).collect(),
                use_order: b.use_order.clone(),
            })
            .collect();
        let df = &mut dst.functions[d as usize];
        df.blocks = blocks;
        df.insts = insts;
    }

    /// Copies the bodies of every function copied so far (and of those they bring in).
    pub fn finish(&mut self, dst: &mut Module) {
        while let Some((s, d)) = self.pending.pop() {
            self.function_body(dst, s, d);
        }
    }

    fn op(&mut self, dst: &mut Module, op: &Op, _sf: u32, _df: u32) -> Op {
        let saved = self.from_metadata;
        self.from_metadata = false;
        let op = self.op_inner(dst, op);
        self.from_metadata = saved;
        op
    }

    fn op_inner(&mut self, dst: &mut Module, op: &Op) -> Op {
        let mut op = op.clone();
        match &mut op {
            Op::Alloca { allocated, .. } => *allocated = self.ty(dst, *allocated),
            Op::Gep { source, .. } => *source = self.ty(dst, *source),
            Op::Call { fn_ty, attrs, bundles, .. } => {
                *fn_ty = self.ty(dst, *fn_ty);
                *attrs = self.attr_list(dst, attrs);
                for b in bundles.iter_mut() {
                    let tag = self.src.bundle_tags[b.tag as usize].clone();
                    b.tag = intern_name(&mut dst.bundle_tags, &tag);
                }
            }
            Op::Switch { cases, .. } => {
                for (c, _) in cases.iter_mut() {
                    *c = self.constant(dst, *c);
                }
            }
            _ => {}
        }
        match &mut op {
            Op::Load { atomic: Some(a), .. } | Op::Store { atomic: Some(a), .. } => {
                a.scope = self.scope(dst, a.scope);
            }
            Op::AtomicRmw { scope, .. } | Op::CmpXchg { scope, .. } | Op::Fence { scope, .. } => {
                *scope = self.scope(dst, *scope);
            }
            _ => {}
        }
        for v in super::util::operands_mut(&mut op) {
            *v = self.value(dst, *v);
        }
        op
    }

    fn scope(&mut self, dst: &mut Module, s: u32) -> u32 {
        let name = self.src.sync_scopes[s as usize].clone();
        intern_name(&mut dst.sync_scopes, &name)
    }

    pub fn attachments(&mut self, dst: &mut Module, md: &[(u32, MdId)]) -> Vec<(u32, MdId)> {
        let mut out = Vec::new();
        for &(k, m) in md {
            let name = self.src.md_kinds[k as usize].clone();
            let kd = intern_name(&mut dst.md_kinds, &name);
            if let Some(mm) = self.md(dst, m) {
                out.push((kd, mm));
            }
        }
        out.sort_by_key(|x| x.0);
        out
    }

    // Metadata --------------------------------------------------------------------------------------------------------

    /// Copies metadata; `None` when it is a reference to a global the copy does not keep.
    pub fn md(&mut self, dst: &mut Module, m: MdId) -> Option<MdId> {
        if let Some(&d) = self.mds.get(&m) {
            return Some(d);
        }
        let d = match self.src.md(m).clone() {
            Metadata::String(s) => dst.intern_md(Metadata::String(s)),
            Metadata::Value { ty, value } => {
                if let (Reach::CodeOnly, Value::Global(g)) = (self.reach, value)
                    && !self.globals.contains_key(&g)
                {
                    return None;
                }
                let ty = self.ty(dst, ty);
                let saved = self.from_metadata;
                self.from_metadata = true;
                let value = self.value(dst, value);
                self.from_metadata = saved;
                dst.intern_md(Metadata::Value { ty, value })
            }
            Metadata::Local { function, ty, value } => {
                let GlobalId::Function(f) = self.global(dst, GlobalId::Function(function)) else { unreachable!() };
                let ty = self.ty(dst, ty);
                let value = match value {
                    Value::Inst(i) => match self.inst_map(function).get(&i) {
                        Some(&n) => Value::Inst(n),
                        None => return None,
                    },
                    other => other,
                };
                dst.intern_md(Metadata::Local { function: f, ty, value })
            }
            Metadata::Tuple { distinct: true, ops } => {
                let id = dst.push_md(Metadata::Tuple { distinct: true, ops: Vec::new() });
                self.mds.insert(m, id);
                let ops = ops.iter().map(|o| o.and_then(|o| self.md(dst, o))).collect();
                dst.set_md(id, Metadata::Tuple { distinct: true, ops });
                return Some(id);
            }
            Metadata::Tuple { distinct: false, ops } => {
                let ops = ops.iter().map(|o| o.and_then(|o| self.md(dst, o))).collect();
                dst.intern_md(Metadata::Tuple { distinct: false, ops })
            }
            Metadata::ArgList(items) => {
                let items = items.iter().filter_map(|&i| self.md(dst, i)).collect();
                dst.intern_md(Metadata::ArgList(items))
            }
        };
        self.mds.insert(m, d);
        Some(d)
    }
}

fn id_index(g: GlobalId) -> usize {
    match g {
        GlobalId::Var(i) | GlobalId::Function(i) | GlobalId::Alias(i) | GlobalId::IFunc(i) => i as usize,
    }
}

fn intern_name(list: &mut Vec<String>, name: &str) -> u32 {
    match list.iter().position(|x| x == name) {
        Some(i) => i as u32,
        None => {
            list.push(name.to_string());
            list.len() as u32 - 1
        }
    }
}

fn unique_struct_name(dst: &Module, name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let taken = |n: &str| dst.types.iter().any(|(_, t)| matches!(t, Type::Struct { name, .. } if name == n));
    if !taken(name) {
        return name.to_string();
    }
    (0..).map(|k| format!("{}.{}", name, k)).find(|n| !taken(n)).unwrap()
}

/// A name for a copied global: its own, unless a local (internal or private) global would clash, which gets a
/// numeric suffix. External names are kept as they are: linking resolves them before copying.
fn unique_global_name(dst: &Module, name: &str, linkage: Linkage) -> String {
    if name.is_empty() {
        return String::new();
    }
    let taken = |n: &str| {
        dst.globals.iter().any(|g| g.common.name == n)
            || dst.functions.iter().any(|f| f.common.name == n)
            || dst.aliases.iter().any(|a| a.common.name == n)
            || dst.ifuncs.iter().any(|a| a.common.name == n)
    };
    if !linkage.is_local() || !taken(name) {
        return name.to_string();
    }
    (1..).map(|k| format!("{}.{}", name, k)).find(|n| !taken(n)).unwrap()
}

/// A fresh module with the target description of `src` and nothing else.
pub fn empty_like(src: &Module) -> Module {
    let mut m = Module::new();
    m.producer = src.producer.clone();
    m.wrapper = src.wrapper;
    m.source_filename = src.source_filename.clone();
    m.triple = src.triple.clone();
    m.datalayout = src.datalayout.clone();
    m.module_asm = src.module_asm.clone();
    m
}
