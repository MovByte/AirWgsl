//! The tracked refusal list: every refusal this compiler returns names one item of it.
//!
//! An item is one of:
//!
//! - a row of WebMetal's public refusal table (`tables/refusals.md`): Metal behavior with no exact WebGPU form on
//!   some or all hosts;
//! - a family of `tables/intrinsics.toml`, for an `air.*` function or metadata name: its decision says whether it is
//!   refused on the device, refused for having no WGSL form, or lowered by a named package that has not registered its
//!   rules yet;
//! - a row of `tables/numerics.toml` whose decision is a refusal (the numerics lowering names it);
//! - an item of [`ITEMS`] below: constructs of the IR itself that the structuring, the memory model or the emitter
//!   refuse, each with the plan section that decides it, and the constructs of other packages' paths (canonical form,
//!   stage interface, argument buffers) that reach this compiler before those packages land.
//!
//! The item is the refusal's `reason` prefix `[namespace:id]`; [`item_of`] reads it back.

use crate::api::{Location, Refusal, RefusalKind};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Where a tracked item is listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Namespace {
    /// A row id of WebMetal `tables/refusals.md`.
    Refusals,
    /// A family id of `tables/intrinsics.toml`.
    Family,
    /// A row id of `tables/numerics.toml`.
    Numerics,
    /// An item of [`ITEMS`].
    Compiler,
}

impl Namespace {
    pub fn prefix(self) -> &'static str {
        match self {
            Namespace::Refusals => "refusal",
            Namespace::Family => "family",
            Namespace::Numerics => "numerics",
            Namespace::Compiler => "compiler",
        }
    }
}

/// One item of the compiler's own list.
#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub id: &'static str,
    /// The package that lowers the construct once it lands, when one does.
    pub owner: Option<&'static str>,
    pub kind: RefusalKind,
    /// What is refused and why.
    pub summary: &'static str,
    /// The plan section or specification clause that decides it.
    pub reference: &'static str,
}

/// The compiler's own tracked items.
pub const ITEMS: &[Item] = &[
    Item {
        id: "irreducible-control-flow",
        owner: None,
        kind: RefusalKind::NoExactLowering,
        summary: "a control-flow graph with a retreating edge whose target does not dominate its source",
        reference: "plan 5.6",
    },
    Item {
        id: "loop-without-exit",
        owner: None,
        kind: RefusalKind::NoExactLowering,
        summary: "a loop with no exit edge: its WGSL behavior is empty, which WGSL rejects, and it is not rewritten",
        reference: "plan 5.6; WGSL 9.7.1",
    },
    Item {
        id: "unbounded-pointer-provenance",
        owner: None,
        kind: RefusalKind::NoExactLowering,
        summary: "a pointer made from an integer whose region cannot be bounded to a finite set of regions",
        reference: "plan 5.4",
    },
    Item {
        id: "pointer-in-memory",
        owner: Some("S6d"),
        kind: RefusalKind::NotImplemented,
        summary: "a pointer loaded from or stored to memory: only argument-buffer handle slots carry pointers in memory",
        reference: "plan 5.4, 5.5",
    },
    Item {
        id: "pointer-aggregate",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "a vector of pointers, or a pointer inside an aggregate value",
        reference: "plan 5.4",
    },
    Item {
        id: "non-canonical-input",
        owner: Some("S3"),
        kind: RefusalKind::NotImplemented,
        summary: "a call of a defined function, or another form canonicalization removes before lowering",
        reference: "plan 5.1 stage 5",
    },
    Item {
        id: "dynamic-alloca",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "an alloca outside the entry block or with a count that is not a constant",
        reference: "plan 5.4",
    },
    Item {
        id: "integer-width",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "an integer type other than i1, i8, i16, i32 and i64",
        reference: "plan 5.2",
    },
    Item {
        id: "vector-shape",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "a vector of other than 2, 3 or 4 lanes, or a vector of i1 in memory",
        reference: "plan 5.2",
    },
    Item {
        id: "llvm-instruction",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "an LLVM instruction AIR does not use for MSL (va_arg, indirectbr, fence, LLVM atomic instructions)",
        reference: "plan 5.1 stage 6",
    },
    Item {
        id: "llvm-intrinsic",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "an llvm.* intrinsic outside the numerics families and the memory intrinsics",
        reference: "plan 5.7",
    },
    Item {
        id: "foreign-callee",
        owner: Some("S3"),
        kind: RefusalKind::NotImplemented,
        summary: "a call of a declared function that is neither air.* nor llvm.* (a stitching hook or a visible function \
                  reference that linking resolves)",
        reference: "plan 5.1 stage 5",
    },
    Item {
        id: "runtime-library-function",
        owner: Some("S5"),
        kind: RefusalKind::NotImplemented,
        summary: "a call of a Metal runtime-library helper (___metal_*: ilogb, frexp, fract and others) that real Metal \
                  links at pipeline creation",
        reference: "plan 5.7",
    },
    Item {
        id: "constant-store",
        owner: None,
        kind: RefusalKind::RejectedByDevice,
        summary: "a store into constant address space memory, which MSL makes read-only",
        reference: "MSL 4.2",
    },
    Item {
        id: "unregistered-lowering",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "an air.* function whose family the numerics, memory or registered rules do not lower",
        reference: "plan 5.7",
    },
    Item {
        id: "stage-builtin",
        owner: Some("S6b"),
        kind: RefusalKind::NotImplemented,
        summary: "a stage input or output attribute without a direct WGSL built-in (needs the stage-interface lowering)",
        reference: "plan 5.7, 5.10",
    },
    Item {
        id: "vertex-input",
        owner: Some("S6b"),
        kind: RefusalKind::NotImplemented,
        summary: "a [[stage_in]] vertex attribute: vertex pulling or native vertex input",
        reference: "plan 5.10",
    },
    Item {
        id: "varying-mismatch",
        owner: Some("S6b"),
        kind: RefusalKind::NotImplemented,
        summary: "a fragment input whose type or interpolation differs from the vertex output of the same name",
        reference: "plan 5.10",
    },
    Item {
        id: "non-uniform-dispatch",
        owner: Some("S6b"),
        kind: RefusalKind::NotImplemented,
        summary: "a dispatch whose grid is not a multiple of the threadgroup size (dispatchThreads:)",
        reference: "plan 5.7; S6b non-uniform dispatch guard",
    },
    Item {
        id: "framebuffer-fetch",
        owner: Some("S6c"),
        kind: RefusalKind::NotImplemented,
        summary: "a fragment function reading a color attachment (air.render_target input)",
        reference: "plan 5.9",
    },
    Item {
        id: "resource-argument",
        owner: Some("S6a"),
        kind: RefusalKind::NotImplemented,
        summary: "a texture or sampler argument",
        reference: "plan 5.5",
    },
    Item {
        id: "argument-buffer",
        owner: Some("S6d"),
        kind: RefusalKind::NotImplemented,
        summary: "an argument buffer (air.indirect_buffer)",
        reference: "plan 5.5",
    },
    Item {
        id: "entry-metadata",
        owner: Some("S2"),
        kind: RefusalKind::NotImplemented,
        summary: "an entry-point or argument record the AIR semantic layer does not decode, or one that does not match \
                  the function",
        reference: "plan 5.1 stage 4",
    },
    Item {
        id: "pipeline-kind",
        owner: None,
        kind: RefusalKind::NotImplemented,
        summary: "a tile pipeline, or a render pipeline with tessellation, vertex amplification or no vertex function",
        reference: "plan 5.7",
    },
    Item {
        id: "host-limit",
        owner: None,
        kind: RefusalKind::HostUnsupported,
        summary: "the pipeline needs more of a per-stage resource than the host's limit allows, and the exact \
                  coalescing of plan 5.5 does not apply",
        reference: "plan 5.5",
    },
    Item {
        id: "vertex-buffer-written",
        owner: None,
        kind: RefusalKind::NoExactLowering,
        summary: "a buffer the shader writes that the draw also binds as a vertex or index buffer: no WebGPU usage \
                  scope allows both",
        reference: "plan 5.5",
    },
    Item {
        id: "threadgroup-size",
        owner: None,
        kind: RefusalKind::RejectedByDevice,
        summary: "a dispatch with more threads per threadgroup than the kernel's [[max_total_threads_per_threadgroup]]",
        reference: "MSL 5.1.3",
    },
    Item {
        id: "module-unreadable",
        owner: Some("S1"),
        kind: RefusalKind::RejectedByDevice,
        summary: "a module the bitcode reader cannot read, or whose entry function or data layout is missing",
        reference: "plan 5.1 stages 2 and 3",
    },
    Item {
        id: "barrier-uniformity",
        owner: None,
        kind: RefusalKind::NoExactLowering,
        summary: "a threadgroup or SIMD-group barrier that WGSL's uniformity analysis cannot prove to be reached in \
                  uniform control flow (WGSL rejects such a shader; on Apple GPUs threads that have ended stop taking \
                  part in barriers)",
        reference: "WGSL 15.2, 17.11; MSL 6.10.1",
    },
    Item {
        id: "barrier-stage",
        owner: None,
        kind: RefusalKind::NoExactLowering,
        summary: "a threadgroup or SIMD-group barrier outside a kernel: WGSL has synchronization functions in compute \
                  shaders only",
        reference: "WGSL 17.11; MSL 6.10",
    },
    Item {
        id: "uniformity-text",
        owner: Some("S6d"),
        kind: RefusalKind::NotImplemented,
        summary: "a generated module with a barrier whose text the uniformity analysis does not read (a pointer \
                  declaration or another construct it does not model)",
        reference: "WGSL 15.2",
    },
    Item {
        id: "sync-operand",
        owner: None,
        kind: RefusalKind::RejectedByDevice,
        summary: "an atomic or barrier operand the target does not accept: a memory order other than relaxed (D1: \
                  supportsMemoryOrderAtomics = false), or a scope, memory flag or operand count MSL 3.1's front end \
                  does not produce",
        reference: "MSL 6.10.1, 6.16; D1 device profile",
    },
    Item {
        id: "icb-layout",
        owner: Some("C0"),
        kind: RefusalKind::NotImplemented,
        summary: "an indirect command buffer encoded from a shader, or a command buffer or pipeline state passed to \
                  one: the wire schema does not define the indirect command buffer's memory layout the commands are \
                  written in",
        reference: "plan 5.7 (indirect command buffers); MSL 6.17",
    },
    Item {
        id: "threadgroup-memory-length",
        owner: None,
        kind: RefusalKind::RejectedByDevice,
        summary: "a threadgroup buffer argument without a setThreadgroupMemoryLength:atIndex: length",
        reference: "plan 5.4",
    },
];

/// The compiler item `id`. Panics on an unknown id: callers name items of [`ITEMS`] literally.
pub fn compiler_item(id: &str) -> &'static Item {
    ITEMS.iter().find(|i| i.id == id).unwrap_or_else(|| panic!("tracked item {} is not in the list", id))
}

/// A refusal naming a compiler item.
pub fn refuse(
    id: &str,
    construct: impl Into<String>,
    location: Option<Location>,
    detail: impl Into<String>,
) -> Refusal {
    let item = compiler_item(id);
    Refusal {
        kind: item.kind,
        construct: construct.into(),
        location,
        reason: format!("[compiler:{}] {}", id, detail.into()),
        reference: Some(item.reference.to_string()),
    }
}

/// A refusal naming a row of WebMetal's refusal table.
pub fn refuse_row(
    row: &str,
    kind: RefusalKind,
    construct: impl Into<String>,
    location: Option<Location>,
    detail: impl Into<String>,
    reference: &str,
) -> Refusal {
    Refusal {
        kind,
        construct: construct.into(),
        location,
        reason: format!("[refusal:{}] {}", row, detail.into()),
        reference: Some(reference.to_string()),
    }
}

/// A refusal naming an intrinsics.toml family; the kind follows the family's decision.
pub fn refuse_family(family: &str, construct: impl Into<String>, location: Option<Location>) -> Refusal {
    let info = family_info(family);
    let (kind, why) = match info.map(|f| f.decision.as_str()) {
        Some("device-unsupported") => {
            (RefusalKind::RejectedByDevice, "the target device does not support it".to_string())
        }
        Some("not-from-msl") => (RefusalKind::RejectedByDevice, "the MSL front end does not produce it".to_string()),
        Some("refusal") => (RefusalKind::NoExactLowering, "the family has no exact WGSL lowering".to_string()),
        Some(d) => (
            RefusalKind::NotImplemented,
            match info.and_then(|f| f.owner.clone()) {
                Some(o) => format!("lowered by package {} (decision {}), whose rules are not registered", o, d),
                None => format!("decision {}: no lowering is registered", d),
            },
        ),
        None => (RefusalKind::NotImplemented, "not a family of intrinsics.toml".to_string()),
    };
    Refusal {
        kind,
        construct: construct.into(),
        location,
        reason: format!("[family:{}] {}", family, why),
        reference: None,
    }
}

/// A refusal from another layer (the AIR semantic layer's closed grammar) that names no tracked item: named by the
/// intrinsics.toml family of its construct when it is an `air.*` name, otherwise as an entry-point record outside the
/// decoded grammar (package S2's).
pub fn name_item(mut r: Refusal) -> Refusal {
    if item_of(&r).is_some() {
        return r;
    }
    r.reason = match family_of(&r.construct) {
        Some(f) => format!("[family:{}] {}", f, r.reason),
        None => format!("[compiler:entry-metadata] {}", r.reason),
    };
    r
}

/// The tracked item a refusal names: (namespace, id).
pub fn item_of(r: &Refusal) -> Option<(Namespace, String)> {
    let rest = r.reason.strip_prefix('[')?;
    let end = rest.find(']')?;
    let (ns, id) = rest[..end].split_once(':')?;
    let ns = match ns {
        "refusal" => Namespace::Refusals,
        "family" => Namespace::Family,
        "numerics" => Namespace::Numerics,
        "compiler" => Namespace::Compiler,
        _ => return None,
    };
    Some((ns, id.to_string()))
}

/// A family of intrinsics.toml.
#[derive(Clone, Debug)]
pub struct FamilyInfo {
    pub id: String,
    pub decision: String,
    pub owner: Option<String>,
}

struct Intrinsics {
    families: HashMap<String, FamilyInfo>,
    /// `[[function]]` names to their family.
    functions: HashMap<String, String>,
    /// `[[name]]` rows (metadata and other names) to their family.
    names: HashMap<String, String>,
}

const INTRINSICS_TOML: &str = include_str!("../../tables/intrinsics.toml");

fn unquote(v: &str) -> String {
    v.trim().trim_matches('"').to_string()
}

fn intrinsics() -> &'static Intrinsics {
    static T: OnceLock<Intrinsics> = OnceLock::new();
    T.get_or_init(|| {
        // The rows used here are `key = "value"` lines inside `[[family]]`, `[[function]]` and `[[name]]` tables.
        let mut families = HashMap::new();
        let mut functions = HashMap::new();
        let mut names = HashMap::new();
        let mut section = "";
        let mut fields: HashMap<&str, String> = HashMap::new();
        let mut flush = |section: &str, fields: &mut HashMap<&str, String>| {
            match section {
                "family" => {
                    if let (Some(id), Some(d)) = (fields.get("id"), fields.get("decision")) {
                        families.insert(
                            id.clone(),
                            FamilyInfo { id: id.clone(), decision: d.clone(), owner: fields.get("owner").cloned() },
                        );
                    }
                }
                "function" => {
                    if let (Some(n), Some(f)) = (fields.get("name"), fields.get("family")) {
                        functions.insert(n.clone(), f.clone());
                    }
                }
                "name" => {
                    if let (Some(n), Some(f)) = (fields.get("name"), fields.get("family")) {
                        names.insert(n.clone(), f.clone());
                    }
                }
                _ => {}
            }
            fields.clear();
        };
        for line in INTRINSICS_TOML.lines() {
            if line.starts_with('[') {
                flush(section, &mut fields);
                section = match line.trim() {
                    "[[family]]" => "family",
                    "[[function]]" => "function",
                    "[[name]]" => "name",
                    _ => "",
                };
                continue;
            }
            if section.is_empty() {
                continue;
            }
            if let Some((k, v)) = line.split_once(" = ") {
                let k = k.trim();
                if matches!(k, "id" | "decision" | "owner" | "name" | "family") && v.trim_start().starts_with('"') {
                    fields.insert(k, unquote(v));
                }
            }
        }
        flush(section, &mut fields);
        Intrinsics { families, functions, names }
    })
}

pub fn family_info(id: &str) -> Option<&'static FamilyInfo> {
    intrinsics().families.get(id)
}

/// The intrinsics.toml family of an `air.*` function name: the longest `[[function]]` name that is the name or is
/// followed in it by a dot (the target's lookup), else a `[[name]]` row.
pub fn family_of(name: &str) -> Option<&'static str> {
    let t = intrinsics();
    let mut candidate = name;
    loop {
        if let Some(f) = t.functions.get(candidate) {
            return Some(f.as_str());
        }
        match candidate.rfind('.') {
            Some(i) if i > 0 => candidate = &candidate[..i],
            _ => break,
        }
    }
    t.names.get(name).map(|s| s.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_intrinsics_table_is_read() {
        let t = intrinsics();
        assert!(t.families.len() >= 50, "{} families", t.families.len());
        assert!(t.functions.len() >= 10000, "{} functions", t.functions.len());
        assert_eq!(family_of("air.sample_texture_2d.v4f32"), Some("texture-sample"));
        assert_eq!(family_of("air.wg.barrier"), Some("threadgroup-barrier"));
        assert_eq!(family_info("texture-sample").and_then(|f| f.owner.as_deref()), Some("S6a"));
        assert_eq!(family_of("air.buffer"), Some("metadata-arguments"));
    }

    #[test]
    fn items_are_unique_and_named_back() {
        for (i, a) in ITEMS.iter().enumerate() {
            assert!(ITEMS[i + 1..].iter().all(|b| b.id != a.id), "{} twice", a.id);
            let r = refuse(a.id, "x", None, "detail");
            assert_eq!(item_of(&r), Some((Namespace::Compiler, a.id.to_string())));
        }
        let r = refuse_family("texture-sample", "air.sample_texture_2d.v4f32", None);
        assert_eq!(r.kind, RefusalKind::NotImplemented);
        assert_eq!(item_of(&r), Some((Namespace::Family, "texture-sample".into())));
        assert_eq!(refuse_family("ray-tracing", "air.intersect", None).kind, RefusalKind::RejectedByDevice);
    }
}
