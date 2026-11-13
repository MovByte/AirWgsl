//! The WGSL back end: [`compile`] for vertex, fragment and compute pipelines without emulation passes.
//!
//! # Input
//!
//! The lowering works on a *canonical function*: an entry point whose module holds no call of a defined function,
//! no unresolved function constant and no static initializer, which [`crate::passes::canonicalize_air`] produces
//! (plan 5.1 stage 5): [`compile`] canonicalizes each stage module, linked with the pipeline's modules that are not
//! stages, with the stage's function constant values ([`canonical_input`]), and lowers the canonical module. The
//! entry point comes typed from the AIR semantic layer ([`crate::air`]) decoding the canonical module, and the
//! reflection from [`crate::reflect`] with the values canonicalization determined
//! ([`crate::passes::reflection::canonical_inputs`]: predicate values, binding slots computed from constants, and the
//! arguments the canonical function does not use). `check_canonical` still verifies the form before lowering.
//!
//! # Output
//!
//! One WGSL module per stage, entry point `aw_main`. Identifiers the back end generates start with `aw_` (values `vN`
//! and temporaries `tN` are local); the numerics library's start with `nm_`. Each stage's module declares the uniform
//! block, the stage's buffers, its threadgroup, private and constant memory, the structs its SSA aggregates use, the
//! library items it calls, and the entry function: function-scope declarations, the arguments' bindings, then the
//! structured body.
//!
//! Derivatives: MSL leaves derivatives undefined in non-uniform control flow (MSL 6.11.1.1) and WGSL returns an
//! indeterminate value there (WGSL 17.6.1); in uniform control flow both compute the derivative. WGSL's static
//! analysis cannot prove uniformity in every case where it holds at run time (branches on uniform buffer values, the
//! jump flags of [`crate::cfg`]), so a module with derivatives disables the `derivative_uniformity` diagnostic
//! (plan 5.3), which changes no result.

// The refusal is the compiler's only failure value (api::compile) and is returned by value throughout.
#![allow(clippy::result_large_err)]

pub mod emit;
pub mod hooks;
pub mod io;
pub mod tracked;
pub mod types;

use crate::air;
use crate::api::*;
use crate::cfg::{self, Term};
use crate::ir::layout::DataLayout;
use crate::ir::{GlobalId, Module, Op, Value};
use crate::layout::Layout;
use crate::mem::{Memory, RegionKind};
use crate::num::{self, Rounding};
use hooks::Rules;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

type R<T> = Result<T, Refusal>;

/// The entry function's name in every generated module.
pub const ENTRY_POINT: &str = "aw_main";

/// Compile with the registered rules ([`hooks::registered`]) and the exact rounding policy.
pub fn compile(stage_modules: &[AirModule], desc: &PipelineDesc, host: &HostCaps) -> R<Compiled> {
    compile_with(stage_modules, desc, host, &hooks::registered(), &Rounding::SoftFloat)
}

/// A module and its entry, read and checked.
pub struct Input {
    /// The decoded module (its IR in `module.ir`).
    pub module: air::AirModule,
    pub entry: air::EntryPoint,
    /// The entry's stage.
    pub stage: MetalStage,
    pub dl: DataLayout,
    /// What reflection needs from canonicalization; for a module taken as it was read ([`input_of`]), only the
    /// arguments its function does not use.
    pub reflection: crate::reflect::Inputs,
}

fn unreadable(detail: String) -> Refusal {
    tracked::refuse("module-unreadable", "module", None, detail)
}

/// The Metal stage of an entry point lowered as a pipeline stage; other entry points are linked into stages.
fn metal_stage(e: &air::EntryPoint) -> R<MetalStage> {
    let loc = Some(Location { function: e.name.clone(), instruction: None });
    Ok(match e.stage {
        air::Stage::Vertex => MetalStage::Vertex,
        air::Stage::Fragment => MetalStage::Fragment,
        air::Stage::Kernel => MetalStage::Kernel,
        air::Stage::Visible => return Err(tracked::refuse_family("visible-stitchable", e.name.clone(), loc)),
        air::Stage::Intersection => return Err(tracked::refuse_family("ray-tracing", e.name.clone(), loc)),
        air::Stage::Mesh | air::Stage::Object => {
            return Err(tracked::refuse_family("mesh-object", e.name.clone(), loc));
        }
    })
}

/// Check a decoded module's entry for `function` and make it an input.
pub fn input_of(module: air::AirModule, function: &str, function_type: u64) -> R<Input> {
    let entry = module.entry_point(function).cloned().ok_or_else(|| {
        tracked::refuse(
            "entry-metadata",
            function.to_string(),
            None,
            "the function is not an entry point of the module",
        )
    })?;
    if function_type != entry.stage.function_type() {
        return Err(tracked::refuse(
            "entry-metadata",
            function.to_string(),
            None,
            "the function type does not match its record",
        ));
    }
    let stage = metal_stage(&entry)?;
    let dl = DataLayout::parse(&module.ir.datalayout).map_err(unreadable)?;
    // A module taken as it is read determines no function constant: only the arguments its function does not use.
    let used = used_args(&module.ir, entry.function as usize);
    let reflection = crate::reflect::Inputs {
        unused_arguments: used.iter().enumerate().filter(|(_, u)| !**u).map(|(k, _)| k as u32).collect(),
        ..Default::default()
    };
    Ok(Input { module, entry, stage, dl, reflection })
}

/// Canonicalize entry `function` of `stage` (linked with `libraries`, specialized with `constants`) and make the
/// canonical module, decoded by the AIR semantic layer, an input; its reflection inputs come from the
/// canonicalization.
pub fn canonical_input(
    stage: &air::AirModule,
    libraries: &[air::AirModule],
    function: &str,
    function_type: u64,
    constants: &[FunctionConstant],
) -> R<Input> {
    let entry = stage.entry_point(function).cloned().ok_or_else(|| {
        tracked::refuse(
            "entry-metadata",
            function.to_string(),
            None,
            "the function is not an entry point of the module",
        )
    })?;
    metal_stage(&entry)?;
    let modules: Vec<air::AirModule> = std::iter::once(stage.clone()).chain(libraries.iter().cloned()).collect();
    let c = crate::passes::canonicalize_air(&modules, &entry, constants)?;
    let reflection = crate::passes::reflection::canonical_inputs(&c);
    let decoded = air::AirModule::decode(c.module).map_err(tracked::name_item)?;
    let mut inp = input_of(decoded, function, function_type)?;
    resolve_slots(&mut inp.entry, &reflection.slots);
    check_locations(&inp.module.ir, &inp.entry, &reflection)?;
    inp.reflection = reflection;
    Ok(inp)
}

/// Refuse arguments of one binding kind whose locations overlap once function constants are set. Metal refuses such
/// a specialization: `newFunctionWithName:constantValues:` fails with `MTLLibraryErrorCompileFailure` ("texture
/// argument b has invalid location (<0,1>)"; measured on an Apple M4, macOS 27.0.1, for textures and buffers whose
/// index a function constant gives). Arguments a false predicate removes take no location.
fn check_locations(m: &Module, entry: &air::EntryPoint, inputs: &crate::reflect::Inputs) -> R<()> {
    use air::{Role, Slot};
    let f = &m.functions[entry.function as usize];
    let mut taken: Vec<(&'static str, u32, u32, String)> = Vec::new();
    for a in &entry.arguments {
        if let Some(p) = &a.predicate
            && inputs.predicates.get(&p.global) != Some(&true)
        {
            continue;
        }
        let space = |b: &air::Buffer| {
            b.address_space.or_else(|| f.args.get(a.position as usize).and_then(|x| m.types.pointer_addrspace(x.ty)))
        };
        let (kind, loc) = match &a.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) if space(b) == Some(3) => ("threadgroup", b.location),
            Role::Buffer(b) | Role::IndirectBuffer(b) => ("buffer", b.location),
            Role::InstanceAccelerationStructure(l, _)
            | Role::PrimitiveAccelerationStructure(l, _)
            | Role::IntersectionFunctionTable(l, _)
            | Role::VisibleFunctionTable(l, _) => ("buffer", *l),
            Role::Texture { location, .. } => ("texture", *location),
            Role::Sampler(l) => ("sampler", *l),
            _ => continue,
        };
        let (Slot::Value(lo), Slot::Value(n)) = (loc.index, loc.count) else { continue };
        let name = a.name.clone().unwrap_or_else(|| format!("argument {}", a.position));
        let hi = lo.saturating_add(n.max(1));
        if let Some((_, _, _, other)) = taken.iter().find(|(k, l, h, _)| *k == kind && lo < *h && *l < hi) {
            return Err(tracked::refuse(
                "binding-location",
                name.clone(),
                Some(Location { function: f.common.name.clone(), instruction: None }),
                format!("{} argument {} has location [{}, {}), which overlaps {}'s", kind, name, lo, hi, other),
            ));
        }
        taken.push((kind, lo, hi, name));
    }
    Ok(())
}

/// Replace the binding indices and counts the entry's records take from globals by the values canonicalization left
/// in those globals, so that every lowering (and reflection) reads them as numbers. A global without a value stays,
/// for the lowering that needs it to refuse.
fn resolve_slots(entry: &mut air::EntryPoint, slots: &BTreeMap<u32, u32>) {
    use air::{OutputRole, Role, Slot};
    let fix = |s: &mut Slot| {
        if let Slot::Constant(g) = *s
            && let Some(&v) = slots.get(&g)
        {
            *s = Slot::Value(v);
        }
    };
    let loc = |l: &mut air::Location| {
        fix(&mut l.index);
        fix(&mut l.count);
    };
    for a in &mut entry.arguments {
        match &mut a.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) => loc(&mut b.location),
            Role::IndirectConstant(l)
            | Role::Texture { location: l, .. }
            | Role::Sampler(l)
            | Role::InstanceAccelerationStructure(l, _)
            | Role::PrimitiveAccelerationStructure(l, _)
            | Role::IntersectionFunctionTable(l, _)
            | Role::VisibleFunctionTable(l, _)
            | Role::CommandBuffer(l)
            | Role::ComputePipelineState(l)
            | Role::RenderPipelineState(l)
            | Role::VertexInput(l)
            | Role::PatchInput(l)
            | Role::StageIn(Some(l)) => loc(l),
            Role::Imageblock(i) => {
                if let Some(l) = &mut i.location {
                    loc(l);
                }
            }
            Role::RenderTarget { index, .. } => fix(index),
            _ => {}
        }
    }
    for o in &mut entry.outputs {
        if let OutputRole::RenderTarget { index, .. } = &mut o.role {
            fix(index);
        }
    }
}

/// Read and decode a pipeline module.
fn decode(m: &AirModule) -> R<air::AirModule> {
    let ir = crate::parse_bitcode(&m.bitcode).map_err(|e| unreadable(e.to_string()))?;
    air::AirModule::decode(ir).map_err(tracked::name_item)
}

/// Read a stage's module, canonicalize its entry with the pipeline's other modules (`libraries`) and its function
/// constant values, and make it an input.
pub fn input(m: &AirModule, libraries: &[air::AirModule]) -> R<Input> {
    let ep = m.entry.as_ref().ok_or_else(|| {
        tracked::refuse("non-canonical-input", "module", None, "a linked module is not a stage (it is linked into one)")
    })?;
    let module = decode(m)?;
    canonical_input(&module, libraries, &ep.function, ep.function_type, &ep.constants)
}

/// Refuse a function that is not in canonical form.
fn check_canonical(m: &Module, fi: usize, function_constants: &[air::FunctionConstant]) -> R<()> {
    let f = &m.functions[fi];
    let loc = || Some(Location { function: f.common.name.clone(), instruction: None });
    let constants: BTreeSet<u32> = function_constants.iter().map(|c| c.global).collect();
    // A constructor list with entries runs static initializers before the entry point; an empty one (which the front
    // end emits for every module of some libraries) does nothing.
    if m.globals.iter().any(|g| {
        g.common.name == "llvm.global_ctors"
            && !matches!(m.types.get(g.value_type), crate::ir::Type::Array { len: 0, .. })
    }) {
        return Err(tracked::refuse_family("static-init", "llvm.global_ctors", loc()));
    }
    for inst in &f.insts {
        if let Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } = &inst.op
            && !m.functions[*c as usize].is_declaration()
        {
            return Err(tracked::refuse(
                "non-canonical-input",
                m.functions[*c as usize].common.name.clone(),
                loc(),
                "a call of a defined function (inlining is canonicalization's)",
            ));
        }
        for v in inst.op.operands() {
            if let Value::Global(GlobalId::Var(g)) = v {
                let gv = &m.globals[g as usize];
                if constants.contains(&g)
                    || gv.section.as_deref() == Some("air.fc_initializer")
                    || gv.externally_initialized
                {
                    return Err(tracked::refuse_family("function-constants", gv.common.name.clone(), loc()));
                }
            }
        }
    }
    Ok(())
}

/// The control-flow terminators of a function's blocks.
fn terms(m: &Module, fi: usize) -> R<Vec<Term>> {
    let f = &m.functions[fi];
    let mut out = Vec::with_capacity(f.blocks.len());
    for b in &f.blocks {
        let last = *b.insts.last().ok_or_else(|| unreadable("an empty block".into()))?;
        out.push(match &f.inst(last).op {
            Op::Ret { .. } => Term::Return,
            Op::Unreachable => Term::Unreachable,
            Op::Br { dest } => Term::Br(dest.0 as usize),
            Op::CondBr { t, f, .. } => Term::CondBr(t.0 as usize, f.0 as usize),
            Op::Switch { default, cases, .. } => {
                Term::Switch(default.0 as usize, cases.iter().map(|(_, b)| b.0 as usize).collect())
            }
            _ => {
                return Err(tracked::refuse(
                    "llvm-instruction",
                    "indirectbr",
                    Some(Location { function: f.common.name.clone(), instruction: None }),
                    "a terminator AIR does not use",
                ));
            }
        });
    }
    Ok(out)
}

/// One lowered stage.
pub struct LoweredStage {
    pub stage: MetalStage,
    pub source: String,
    pub requirements: HostRequirements,
    pub workgroup_bytes: u64,
}

struct Lib {
    items: Vec<(String, String, Vec<usize>)>,
}

fn aw_lib() -> &'static Lib {
    static L: OnceLock<Lib> = OnceLock::new();
    L.get_or_init(|| {
        let src = include_str!("lib.wgsl");
        let lines: Vec<&str> = src.lines().collect();
        let starts: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("fn aw_")).collect();
        let mut items: Vec<(String, String, Vec<usize>)> = Vec::new();
        for (k, &s) in starts.iter().enumerate() {
            let mut first = s;
            while first > 0 && lines[first - 1].starts_with("//") {
                first -= 1;
            }
            let mut end = starts.get(k + 1).copied().unwrap_or(lines.len());
            while end > s + 1 && (lines[end - 1].starts_with("//") || lines[end - 1].trim().is_empty()) {
                end -= 1;
            }
            let name = lines[s][3..].split('(').next().unwrap().to_string();
            items.push((name, lines[first..end].join("\n") + "\n", Vec::new()));
        }
        let names: Vec<String> = items.iter().map(|i| i.0.clone()).collect();
        for it in items.iter_mut() {
            let body = it.1.clone();
            it.2 = names
                .iter()
                .enumerate()
                .filter(|(_, n)| **n != it.0 && body.contains(&format!("{}(", n)))
                .map(|(i, _)| i)
                .collect();
        }
        Lib { items }
    })
}

fn lib_text(wanted: &BTreeSet<&'static str>) -> String {
    let lib = aw_lib();
    let mut need = vec![false; lib.items.len()];
    let mut stack: Vec<usize> =
        lib.items.iter().enumerate().filter(|(_, i)| wanted.contains(i.0.as_str())).map(|(k, _)| k).collect();
    while let Some(k) = stack.pop() {
        if !need[k] {
            need[k] = true;
            stack.extend(lib.items[k].2.iter().copied());
        }
    }
    lib.items.iter().zip(need).filter(|(_, n)| *n).map(|(i, _)| i.1.clone() + "\n").collect()
}

/// The arguments some instruction uses.
fn used_args(m: &Module, fi: usize) -> Vec<bool> {
    let f = &m.functions[fi];
    let mut used = vec![false; f.args.len()];
    for inst in &f.insts {
        for v in inst.op.operands() {
            if let Value::Arg(a) = v {
                used[a as usize] = true;
            }
        }
    }
    used
}

fn words_decl(words: u64, atomic: bool) -> String {
    format!("array<{}, {}>", if atomic { "atomic<u32>" } else { "u32" }, words.max(1))
}

/// Lower one stage of a pipeline.
#[allow(clippy::too_many_arguments)]
pub fn lower_stage(
    inp: &Input,
    desc: &PipelineDesc,
    host: &HostCaps,
    rules: &Rules,
    rounding: &Rounding,
    varyings: &io::Varyings,
    layout: &mut Layout,
) -> R<LoweredStage> {
    let m = &inp.module.ir;
    let fi = inp.entry.function as usize;
    let stage = inp.stage;
    let fname = m.functions[fi].common.name.clone();
    let loc = || Some(Location { function: fname.clone(), instruction: None });
    check_canonical(m, fi, &inp.module.function_constants)?;
    let terms = terms(m, fi)?;
    let (analysis, structured) = cfg::structure(&terms).map_err(|e| match e {
        cfg::Error::Irreducible { from, to } => tracked::refuse(
            "irreducible-control-flow",
            "br",
            loc(),
            format!("the edge from block {} to block {} enters a loop other than through its header", from, to),
        ),
        cfg::Error::LoopWithoutExit { header } => tracked::refuse(
            "loop-without-exit",
            "loop",
            loc(),
            format!("the loop headed by block {} has no exit", header),
        ),
    })?;
    let bindings = desc.bindings.iter().find(|b| b.stage == stage);
    let used = used_args(m, fi);
    let icx = io::Ctx {
        m,
        entry: &inp.entry,
        stage,
        desc,
        host,
        bindings,
        varyings,
        used: &used,
        values: &inp.reflection,
        rules,
    };
    let iface = io::lower(&icx, layout)?;
    let shared_device = stage != MetalStage::Vertex;
    let mem = Memory::analyze(m, fi, &inp.dl, &iface.regions, shared_device)?;
    let target = num::Target { rounding, host };
    let mut e = emit::FnEmit::new(m, fi, &inp.dl, mem, &analysis, rules, target, &iface, stage);
    e.records = inp.entry.arguments.clone();
    e.bindings = bindings.cloned();
    e.desc = Some(desc);
    e.prepare()?;
    if iface.early_fragment_tests {
        let writes =
            e.mem.regions.iter().any(|r| (r.written || r.atomic) && matches!(r.kind, RegionKind::Buffer { .. }));
        crate::stage::fragment::early_fragment_tests(m, fi, &inp.entry, desc, writes)?;
    }
    for r in &e.mem.regions {
        if let RegionKind::Buffer { .. } = r.kind {
            layout.note_access(&r.name, r.written, r.atomic);
        }
    }
    let (body, behavior) = e.body(&structured)?;
    for (name, wgsl_type, resource, source) in std::mem::take(&mut e.resources) {
        layout.add_resource(crate::layout::ResourceVar { stage, name, wgsl_type, resource, source, binding: 0 });
    }
    for p in &iface.prelude {
        e.prelude.require(p).map_err(|x| tracked::refuse("unregistered-lowering", *p, None, x.0))?;
    }
    // Module text.
    let mut out = String::new();
    let mut enables = e.enables.clone();
    enables.extend(iface.enables.iter().cloned());
    for en in &enables {
        out.push_str(&format!("enable {};\n", en));
    }
    if e.derivatives {
        out.push_str("diagnostic(off, derivative_uniformity);\n");
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(&e.types.declarations());
    out.push_str(&iface.ret_decl);
    let mut workgroup_bytes = 0;
    let mut module_vars = String::new();
    let mut fn_vars = Vec::new();
    for r in &e.mem.regions {
        match &r.kind {
            RegionKind::ThreadgroupArg { .. } | RegionKind::ThreadgroupGlobal { .. } => {
                let w = r.words.unwrap_or(1);
                workgroup_bytes += 4 * w;
                module_vars.push_str(&format!("var<workgroup> {}: {};\n", r.name, words_decl(w, r.atomic)));
            }
            RegionKind::PrivateGlobal { .. } | RegionKind::ConstantGlobal { .. } => {
                let w = r.words.unwrap_or(1);
                let init = match &r.init {
                    Some(bytes) => {
                        let words: Vec<String> = (0..w as usize)
                            .map(|k| {
                                let mut v = 0u32;
                                for b in 0..4 {
                                    v |= (*bytes.get(4 * k + b).unwrap_or(&0) as u32) << (8 * b);
                                }
                                format!("{}u", v)
                            })
                            .collect();
                        format!(" = array<u32, {}>({})", w.max(1), words.join(", "))
                    }
                    None => String::new(),
                };
                module_vars.push_str(&format!("var<private> {}: {}{};\n", r.name, words_decl(w, false), init));
            }
            RegionKind::Alloca { .. } => {
                fn_vars.push(format!("var {}: {};", r.name, words_decl(r.words.unwrap_or(1), false)))
            }
            _ => {}
        }
    }
    // The layout's declarations are added by the caller once every stage is lowered (binding numbers are
    // pipeline-wide); a marker line keeps their place.
    out.push_str("//@aw:bindings\n");
    out.push_str(&module_vars);
    for t in e.module_items.values().chain(iface.module_items.values()) {
        out.push_str(t);
        out.push('\n');
    }
    out.push('\n');
    let prelude = e.prelude.emit().map_err(|x| tracked::refuse("unregistered-lowering", "prelude", None, x.0))?;
    out.push_str(&prelude);
    out.push_str(&lib_text(&e.lib));
    // Entry function.
    let attr = match stage {
        MetalStage::Vertex => "@vertex".to_string(),
        MetalStage::Fragment => "@fragment".to_string(),
        _ => {
            let w = iface.workgroup_size.unwrap();
            format!("@compute @workgroup_size({}, {}, {})", w[0], w[1], w[2])
        }
    };
    let ret = match &iface.ret_type {
        Some(t) => format!(" -> {}", t),
        None => String::new(),
    };
    out.push_str(&format!("{}\nfn {}({}){} {{\n", attr, ENTRY_POINT, iface.params.join(", "), ret));
    for d in fn_vars.iter().chain(e.decls.iter()) {
        out.push_str(&format!("    {}\n", d));
    }
    for p in &iface.prologue {
        out.push_str(&format!("    {}\n", p));
    }
    for l in &body {
        out.push_str(l);
        out.push('\n');
    }
    if let Some(t) = &iface.ret_type
        && (behavior.next || !body.last().is_some_and(|l| l.trim_start().starts_with("return")))
    {
        // Every path of the body returns (the structuring keeps the graph's paths, and each ends in `ret` or
        // `unreachable`); WGSL's behavior analysis does not follow the jump flags, and naga does not apply the loop
        // rule of WGSL 9.7.1, so the statically unreachable return a function with a result needs is stated.
        out.push_str(&format!("    return {}();\n", t));
    }
    out.push_str("}\n");
    let mut requirements = e.requirements.clone();
    requirements.features.extend(iface.features.iter().cloned());
    if stage == MetalStage::Kernel {
        let w = iface.workgroup_size.unwrap();
        let lim = &mut requirements.limits;
        lim.insert("maxComputeWorkgroupSizeX".into(), w[0]);
        lim.insert("maxComputeWorkgroupSizeY".into(), w[1]);
        lim.insert("maxComputeWorkgroupSizeZ".into(), w[2]);
        lim.insert("maxComputeInvocationsPerWorkgroup".into(), w[0] * w[1] * w[2]);
        if workgroup_bytes > 0 {
            lim.insert("maxComputeWorkgroupStorageSize".into(), workgroup_bytes);
        }
    }
    Ok(LoweredStage { stage, source: out, requirements, workgroup_bytes })
}

/// Replace a stage module's binding marker with the layout's declarations.
fn finish_source(s: &LoweredStage, layout: &Layout) -> String {
    s.source.replace("//@aw:bindings\n", &layout.declarations(s.stage))
}

fn shader_stage(s: MetalStage) -> ShaderStage {
    match s {
        MetalStage::Vertex => ShaderStage::Vertex,
        MetalStage::Fragment => ShaderStage::Fragment,
        _ => ShaderStage::Compute,
    }
}

/// Compile with an explicit rule set and rounding policy.
pub fn compile_with(
    stage_modules: &[AirModule],
    desc: &PipelineDesc,
    host: &HostCaps,
    rules: &Rules,
    rounding: &Rounding,
) -> R<Compiled> {
    // Modules without an entry are linked into the stages: imported libraries and linked functions.
    let mut libraries = Vec::new();
    for m in stage_modules.iter().filter(|m| m.entry.is_none()) {
        libraries.push(decode(m)?);
    }
    let mut inputs = Vec::new();
    for m in stage_modules.iter().filter(|m| m.entry.is_some()) {
        inputs.push(input(m, &libraries)?);
    }
    let by = |s: MetalStage| inputs.iter().filter(|i| i.stage == s).collect::<Vec<_>>();
    let order: Vec<&Input> = match &desc.kind {
        PipelineKind::Compute(_) => {
            let k = by(MetalStage::Kernel);
            if k.len() != 1 || inputs.len() != 1 {
                return Err(tracked::refuse(
                    "pipeline-kind",
                    "compute",
                    None,
                    "a compute pipeline needs one kernel function",
                ));
            }
            k
        }
        PipelineKind::Render(r) => {
            if r.max_vertex_amplification_count > 1 {
                return Err(tracked::refuse("pipeline-kind", "vertex amplification", None, "vertex amplification"));
            }
            let (v, f) = (by(MetalStage::Vertex), by(MetalStage::Fragment));
            if v.len() != 1 || f.len() > 1 || inputs.len() != v.len() + f.len() {
                return Err(tracked::refuse(
                    "pipeline-kind",
                    "render",
                    None,
                    "a render pipeline needs one vertex function and at most one fragment function",
                ));
            }
            v.into_iter().chain(f).collect()
        }
        PipelineKind::Tile(_) => return Err(tracked::refuse_family("imageblock-tile", "tile pipeline", None)),
    };
    let vin = order.iter().find(|i| i.stage == MetalStage::Vertex).map(|i| (&i.module.ir, &i.entry));
    let fin = order.iter().find(|i| i.stage == MetalStage::Fragment).map(|i| (&i.module.ir, &i.entry));
    let varyings = if vin.is_some() || fin.is_some() { io::varyings(vin, fin)? } else { Default::default() };
    let mut layout = Layout::new();
    let mut stages = Vec::new();
    for inp in &order {
        stages.push(lower_stage(inp, desc, host, rules, rounding, &varyings, &mut layout)?);
    }
    layout.finish(host, &desc.bindings)?;
    let mut requirements = HostRequirements::default();
    for s in &stages {
        requirements.features.extend(s.requirements.features.iter().cloned());
        requirements.wgsl_language_features.extend(s.requirements.wgsl_language_features.iter().cloned());
        for (k, v) in &s.requirements.limits {
            let e = requirements.limits.entry(k.clone()).or_insert(0);
            *e = (*e).max(*v);
        }
    }
    let mut per_stage: BTreeMap<MetalStage, u64> = BTreeMap::new();
    for b in layout.buffers() {
        *per_stage.entry(b.stage).or_default() += 1;
    }
    if let Some(n) = per_stage.values().max() {
        requirements.limits.insert("maxStorageBuffersPerShaderStage".into(), *n);
    }
    let groups = layout.binding_layout().groups.len() as u64;
    if groups > 0 {
        requirements.limits.insert("maxBindGroups".into(), groups);
    }
    // WebGPU 10.3.1 validating inter-stage interfaces: the fragment stage's front_facing, sample_index and sample_mask
    // inputs count as inter-stage variables, and a point-list pipeline has one variable fewer for vertex outputs.
    let fragment_builtins = stages
        .iter()
        .filter(|s| s.stage == MetalStage::Fragment)
        .filter_map(|s| s.source.lines().find(|l| l.starts_with(&format!("fn {}(", ENTRY_POINT))))
        .map(|l| {
            ["@builtin(front_facing)", "@builtin(sample_index)", "@builtin(sample_mask)"]
                .iter()
                .filter(|b| l.contains(*b))
                .count() as u64
        })
        .sum::<u64>();
    let points = match &desc.kind {
        PipelineKind::Render(r) => r.input_primitive_topology == 1 || r.primitive_type == Some(0),
        _ => false,
    };
    let inter_stage = (varyings.len() as u64 + fragment_builtins).max(varyings.len() as u64 + points as u64);
    if inter_stage > 0 {
        requirements.limits.insert("maxInterStageShaderVariables".into(), inter_stage);
    }
    if layout.uniform_size() > 0 {
        requirements.limits.insert("maxUniformBufferBindingSize".into(), layout.uniform_size());
    }
    let mut wgsl = Vec::new();
    for (s, inp) in stages.iter().zip(&order) {
        let source = finish_source(s, &layout);
        rules.check_stage(s.stage, &inp.module.ir.functions[inp.entry.function as usize].common.name, &source)?;
        wgsl.push(Stage { stage: shader_stage(s.stage), entry_point: ENTRY_POINT.into(), source });
    }
    // Reflection from the AIR metadata (package S2) with what canonicalization determined: predicate values, binding
    // slots computed from function constants, and the arguments the canonical function does not use.
    let mut stages_reflection = Vec::new();
    for i in &order {
        let inputs = &i.reflection;
        let stage_input = match (&desc.kind, i.stage) {
            (PipelineKind::Render(r), MetalStage::Vertex) => match &r.vertex_descriptor {
                Some(vd) => crate::reflect::StageInput::Vertex(vd),
                None => crate::reflect::StageInput::None,
            },
            (PipelineKind::Compute(c), MetalStage::Kernel) => match &c.stage_input {
                Some(si) => crate::reflect::StageInput::Compute(si),
                None => crate::reflect::StageInput::None,
            },
            _ => crate::reflect::StageInput::None,
        };
        stages_reflection
            .push(crate::reflect::stage(&i.module, &i.entry, inputs, stage_input).map_err(tracked::name_item)?);
    }
    let reflection = Reflection { stages: stages_reflection };
    let layout = layout.binding_layout();
    // Framebuffer fetch (package S6c): the pass plan, with the overlap prover of the vertex function when it qualifies.
    let plan = crate::fetch::plan(&layout, || {
        let v = order.iter().find(|i| i.stage == MetalStage::Vertex)?;
        let key = crate::fetch::overlap::key(stage_modules);
        crate::fetch::overlap::build(&v.module.ir, &v.entry, rounding, host, key).ok()
    });
    // Tessellation (package S6e): the vertex function is a post-tessellation function exactly when the pipeline has
    // tessellation state (the guest sets the state from the function's patch record).
    let patch = order.iter().find(|i| i.stage == MetalStage::Vertex).and_then(|v| v.entry.patch);
    let tessellated = matches!(&desc.kind, PipelineKind::Render(r) if r.tessellation.is_some());
    let plan = match (patch, tessellated, plan) {
        (None, false, plan) => plan,
        (Some(p), true, PassPlan::Direct) => PassPlan::Tessellation { quad: p.kind == crate::air::PatchKind::Quad },
        (Some(_), true, _) => {
            return Err(tracked::refuse(
                "post-tessellation-input",
                "tessellation",
                None,
                "a tessellation pipeline whose fragment function reads color attachments: the per-primitive passes of \
                 framebuffer fetch over tessellated draws are not implemented",
            ));
        }
        (None, true, _) => {
            return Err(tracked::refuse(
                "pipeline-kind",
                "tessellation",
                None,
                "tessellation state with a vertex function that is not a post-tessellation function",
            ));
        }
        (Some(_), false, _) => {
            return Err(tracked::refuse(
                "pipeline-kind",
                "air.patch",
                None,
                "a post-tessellation vertex function in a pipeline without tessellation state",
            ));
        }
    };
    Ok(Compiled { wgsl, layout, passes: vec![plan], reflection, requirements })
}

/// Lower one entry point on its own (for corpus checks): a kernel as a compute pipeline with `threads` threads per
/// threadgroup, a vertex or fragment function as a render stage whose varyings are its own, `[[stage_in]]` attributes
/// described as read in their own types, and the color attachments a fragment function reads given formats of their
/// inputs' classes ([`crate::fetch::corpus_attachments`]).
pub fn compile_entry(
    inp: &Input,
    host: &HostCaps,
    rules: &Rules,
    threads: [u64; 3],
    tg_lengths: &[u64],
) -> R<(String, HostRequirements)> {
    // `[[stage_in]]` attributes are read in their own types from one buffer (crate::stage::fetch::natural_layout); a
    // post-tessellation function gets tessellation state and its patch attributes' layout (crate::tess::stage).
    let natural = crate::stage::fetch::natural_layout(&inp.module.ir, &inp.entry);
    let tessellated = crate::tess::stage::corpus_state(&inp.module.ir, &inp.entry);
    let desc = match inp.stage {
        MetalStage::Kernel => PipelineDesc {
            kind: PipelineKind::Compute(ComputeState {
                threads_per_threadgroup: threads,
                non_uniform_grid: false,
                threadgroup_memory_lengths: tg_lengths.to_vec(),
                stage_input: natural.map(|(buffer, stride, attrs)| {
                    let mut layouts = vec![(0, 0, 0); buffer as usize + 1];
                    layouts[buffer as usize] = (stride, 5, 1);
                    let mut attributes = Vec::new();
                    for (index, format, offset) in attrs {
                        attributes.resize(attributes.len().max(index as usize + 1), (0, 0, 0));
                        attributes[index as usize] = (format, offset, buffer);
                    }
                    StageInputDescriptor { layouts, attributes, index_type: 0, index_buffer_index: 0 }
                }),
            }),
            bindings: Vec::new(),
        },
        _ => PipelineDesc {
            kind: PipelineKind::Render(RenderState {
                vertex_descriptor: tessellated.as_ref().and_then(|t| t.1.clone()).or_else(|| {
                    natural.map(|(buffer, stride, attrs)| {
                        let mut layouts =
                            vec![VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 }; buffer as usize + 1];
                        layouts[buffer as usize].stride = stride;
                        let mut attributes = Vec::new();
                        for (index, format, offset) in attrs {
                            attributes.resize(
                                attributes.len().max(index as usize + 1),
                                VertexAttribute { format: 0, offset: 0, buffer_index: 0 },
                            );
                            attributes[index as usize] = VertexAttribute { format, offset, buffer_index: buffer };
                        }
                        VertexDescriptor { layouts, attributes }
                    })
                }),
                // The attachments a fragment function reads, in formats of their inputs' classes (package S6c).
                color_attachments: if inp.stage == MetalStage::Fragment {
                    crate::fetch::corpus_attachments(&inp.module.ir, &inp.entry)
                } else {
                    Vec::new()
                },
                depth_attachment_pixel_format: 0,
                stencil_attachment_pixel_format: 0,
                raster_sample_count: 1,
                alpha_to_coverage_enabled: false,
                alpha_to_one_enabled: false,
                rasterization_enabled: true,
                input_primitive_topology: 3,
                primitive_type: None,
                tessellation: tessellated.as_ref().map(|t| t.0),
                max_vertex_amplification_count: 1,
            }),
            bindings: Vec::new(),
        },
    };
    let pair = Some((&inp.module.ir, &inp.entry));
    let varyings = match inp.stage {
        MetalStage::Vertex => io::varyings(pair, None)?,
        MetalStage::Fragment => io::varyings(None, pair)?,
        _ => Default::default(),
    };
    let mut layout = Layout::new();
    let s = lower_stage(inp, &desc, host, rules, &Rounding::SoftFloat, &varyings, &mut layout)?;
    layout.finish(host, &[])?;
    let source = finish_source(&s, &layout);
    rules.check_stage(s.stage, &inp.module.ir.functions[inp.entry.function as usize].common.name, &source)?;
    Ok((source, s.requirements))
}
