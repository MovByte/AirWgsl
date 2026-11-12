//! The stage interface lowering (package S6b, plan 5.7 and 5.10): what lies between the stages and the fixed-function
//! pipeline beyond the direct mappings of [`crate::wgsl::io`].
//!
//! - [`fetch`]: `[[stage_in]]` vertex attributes and kernel stage-in attributes, pulled from their buffers with
//!   Metal's format conversions and step functions.
//! - [`fragment`]: color outputs against the pipeline's attachment formats, dual-source outputs, the coverage mask,
//!   `discard_fragment`, the sample count and positions, and early fragment tests.
//! - [`builtins`]: built-in values that need more than a direct mapping (the stage-in region, the linear thread index
//!   of a 3D threadgroup, thread positions in dimensions of threadgroup size 1, point size).
//!
//! The lowerings register as a [`StageRule`] (asked before the direct mappings) and a
//! [`CallRule`](crate::wgsl::hooks::CallRule) ([`register`]). What has no exact lowering is refused with its tracked
//! item ([`crate::wgsl::tracked`]).

// The refusal is the compiler's only failure value (api::compile) and is returned by value throughout.
#![allow(clippy::result_large_err)]

pub mod builtins;
pub mod fetch;
pub mod fragment;

use crate::air::{Argument, Output};
use crate::api::{Location, MetalStage, Refusal};
use crate::ir::{Module, TypeId};
use crate::wgsl::hooks::{OutputField, Rules, StageCx, StageRule};
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Add the package's rules to a rule set.
pub fn register(rules: Rules) -> Rules {
    rules.with_stage(Box::new(Stage)).with(Box::new(fragment::FragmentCalls))
}

pub(crate) fn loc(m: &Module, fi: usize) -> Option<Location> {
    Some(Location { function: m.functions[fi].common.name.clone(), instruction: None })
}

/// The functions of `fetch.wgsl`: (name, text, indices of the functions it calls).
fn library() -> &'static Vec<(String, String, Vec<usize>)> {
    static L: OnceLock<Vec<(String, String, Vec<usize>)>> = OnceLock::new();
    L.get_or_init(|| {
        let src = include_str!("fetch.wgsl");
        let lines: Vec<&str> = src.lines().collect();
        let starts: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("fn aw_vf_")).collect();
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
        items
    })
}

/// The library items `name` needs (itself and what it calls), as (key, text).
fn library_items(name: &str) -> Vec<(String, String)> {
    let lib = library();
    let mut seen = BTreeSet::new();
    let mut stack: Vec<usize> = lib.iter().position(|i| i.0 == name).into_iter().collect();
    assert!(!stack.is_empty(), "{} is not in fetch.wgsl", name);
    while let Some(k) = stack.pop() {
        if seen.insert(k) {
            stack.extend(lib[k].2.iter().copied());
        }
    }
    seen.into_iter().map(|k| (lib[k].0.clone(), lib[k].1.clone())).collect()
}

/// Declare a `fetch.wgsl` function (and what it calls) in the stage's module.
pub(crate) fn lib(cx: &mut dyn StageCx, name: &str) {
    for (k, t) in library_items(name) {
        cx.module_item(&k, t);
    }
}

/// The package's stage rule.
struct Stage;

impl StageRule for Stage {
    fn handles_argument(&self, stage: MetalStage, kind: &str) -> bool {
        matches!(
            (stage, kind),
            (MetalStage::Vertex, "air.vertex_input")
                | (
                    MetalStage::Kernel,
                    "air.stage_in"
                        | "air.stage_in_grid_origin"
                        | "air.stage_in_grid_size"
                        | "air.thread_index_in_threadgroup"
                        | "air.thread_position_in_grid"
                        | "air.thread_position_in_threadgroup"
                )
                | (MetalStage::Fragment, "air.sample_mask_in" | "air.point_coord")
        )
    }

    fn argument(&self, cx: &mut dyn StageCx, arg: u32, record: &Argument) -> Result<String, Refusal> {
        match crate::wgsl::io::role_key(&record.role) {
            "air.vertex_input" => fetch::vertex_input(cx, arg, record),
            "air.stage_in" => fetch::stage_in(cx, arg, record),
            "air.stage_in_grid_origin" | "air.stage_in_grid_size" => builtins::stage_in_region(cx, arg, record),
            "air.thread_index_in_threadgroup" => builtins::thread_index(cx, arg),
            "air.thread_position_in_grid" | "air.thread_position_in_threadgroup" => {
                builtins::kernel_position(cx, arg, record)
            }
            "air.sample_mask_in" => fragment::sample_mask_in(cx, arg),
            "air.point_coord" => Err(builtins::point_coord(cx)),
            other => unreachable!("{} is not claimed", other),
        }
    }

    fn handles_output(&self, stage: MetalStage, kind: &str) -> bool {
        matches!(
            (stage, kind),
            (MetalStage::Vertex, "air.point_size") | (MetalStage::Fragment, "air.render_target" | "air.sample_mask")
        )
    }

    fn output(
        &self,
        cx: &mut dyn StageCx,
        index: usize,
        record: &Output,
        ty: TypeId,
    ) -> Result<Option<OutputField>, Refusal> {
        match crate::wgsl::io::output_key(&record.role) {
            "air.point_size" => builtins::point_size(cx, index, ty),
            "air.render_target" => fragment::color_output(cx, record, ty),
            "air.sample_mask" => fragment::sample_mask_out(cx, ty),
            other => unreachable!("{} is not claimed", other),
        }
    }
}

/// A `vec3<u32>` value (`w`) as the carrier of IR type `t` (`uint`, `ushort`, 1 to 3 lanes); `None` for another type.
pub(crate) fn from_vec3(m: &Module, w: &[String; 3], t: TypeId) -> Option<String> {
    let (n, e) = match m.types.get(t) {
        crate::ir::Type::Vector { len, elem, .. } => (*len, *elem),
        _ => (1, t),
    };
    let bits = m.types.int_bits(e)?;
    if n > 3 || !(bits == 32 || bits == 16) {
        return None;
    }
    let lanes: Vec<String> =
        w[..n as usize].iter().map(|x| if bits == 16 { format!("({} & 0xffffu)", x) } else { x.clone() }).collect();
    Some(if n == 1 { lanes[0].clone() } else { format!("vec{}<u32>({})", n, lanes.join(", ")) })
}
