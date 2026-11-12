//! The closed key grammar of AIR metadata records.
//!
//! A record of an entry point's outputs or arguments, of an argument buffer's members, or of a struct's members is a
//! sequence of keys (`!"air.*"` strings), each followed by the operands this table gives it. A key that is not in the
//! table for the record's context is refused with its name; nothing in a record is skipped.
//!
//! The keys and their operands were measured on the system corpus (the metallibs of iPadOS 17.7.11, 21H461) and the
//! generated corpus (MSL compiled for `air64-apple-ios17.7` at every language version the target accepts), and every
//! key here is an `air.*` name of the target's compiler (`tables/intrinsics.toml`, rows of kind `metadata`); the test
//! below checks both. Names the target's compiler holds that no module of the corpora uses (rows of kind `string`)
//! are not in the table: their record layout is unmeasured, so a module that uses one is refused as not implemented.

use super::md::{Cursor, Operand};
use crate::api::{Refusal, RefusalKind};

/// The kind of record a key appears in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Context {
    /// An entry point's argument, or a resource member of an argument buffer.
    Argument,
    /// An entry point's output (its return value or a member of it).
    Output,
    /// The keys after a struct member's offset, size, array length, type name and name.
    Member,
}

/// The operands that follow a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    /// No operand: the key is a flag.
    None,
    /// One `i32`.
    I32,
    /// One string.
    Str,
    /// One tuple.
    Node,
    /// One `i32` or one tuple.
    I32OrNode,
    /// Two operands, each an `i32` or a function-constant global (`air.location_index`).
    Slots2,
    /// One operand, an `i32` or a function-constant global.
    Slot,
    /// An `i32` or a function-constant global, then an `i32` (a fragment output's `air.render_target`).
    SlotI32,
    /// One or more tuples (`air.patch_control_point_input`).
    Nodes,
    /// An `i32`, then a string (`air.mesh_vertex_data`, `air.mesh_primitive_data`).
    I32Str,
}

const ARG: u8 = 1;
const OUT: u8 = 2;
const MEM: u8 = 4;

/// A key, the contexts it is valid in (bit mask of [`Context`]), and its operands there.
struct Key {
    name: &'static str,
    contexts: u8,
    arity: Arity,
}

const fn k(name: &'static str, contexts: u8, arity: Arity) -> Key {
    Key { name, contexts, arity }
}

/// The table. Where a key's operands differ by context, it has one row per context.
const KEYS: &[Key] = &[
    // Descriptions every argument or output may carry.
    k("air.arg_name", ARG | OUT, Arity::Str),
    k("air.arg_type_name", ARG | OUT, Arity::Str),
    k("air.arg_type_size", ARG, Arity::I32),
    k("air.arg_type_align_size", ARG, Arity::I32),
    k("air.arg_unused", ARG, Arity::None),
    k("air.function_constant", ARG | OUT, Arity::Node),
    // Resources.
    k("air.buffer", ARG, Arity::None),
    k("air.indirect_buffer", ARG, Arity::None),
    k("air.indirect_constant", ARG, Arity::None),
    k("air.texture", ARG, Arity::None),
    k("air.sampler", ARG, Arity::None),
    k("air.buffer_size", ARG, Arity::I32),
    k("air.location_index", ARG, Arity::Slots2),
    k("air.address_space", ARG, Arity::I32),
    k("air.read", ARG, Arity::None),
    k("air.write", ARG, Arity::None),
    k("air.read_write", ARG, Arity::None),
    k("air.sample", ARG, Arity::None),
    k("air.struct_type_info", ARG, Arity::Node),
    k("air.instance_acceleration_structure", ARG, Arity::None),
    k("air.primitive_acceleration_structure", ARG, Arity::None),
    k("air.intersection_function_table", ARG, Arity::None),
    k("air.visible_function_table", ARG, Arity::None),
    k("air.command_buffer", ARG, Arity::None),
    k("air.compute_pipeline_state", ARG, Arity::None),
    k("air.render_pipeline_state", ARG, Arity::None),
    k("air.imageblock", ARG, Arity::Str),
    k("air.imageblock_data_size", ARG, Arity::I32),
    // Stage inputs and outputs.
    k("air.vertex_input", ARG, Arity::None),
    k("air.stage_in", ARG, Arity::None),
    k("air.patch_input", ARG, Arity::None),
    k("air.patch_control_point_input", ARG, Arity::Nodes),
    k("air.fragment_input", ARG, Arity::Str),
    k("air.vertex_output", OUT, Arity::Str),
    k("air.render_target", ARG, Arity::Slot),
    k("air.render_target", OUT, Arity::SlotI32),
    k("air.raster_order_group", ARG | OUT, Arity::I32),
    k("air.interpolation_function", ARG, Arity::None),
    k("air.center", ARG, Arity::None),
    k("air.centroid", ARG, Arity::None),
    k("air.flat", ARG, Arity::None),
    k("air.perspective", ARG, Arity::None),
    k("air.no_perspective", ARG, Arity::None),
    k("air.shared", OUT, Arity::None),
    k("air.invariant", OUT, Arity::None),
    k("air.clip_distance", OUT, Arity::None),
    k("air.clip_distance_array_size", OUT, Arity::I32),
    k("air.depth", OUT, Arity::None),
    k("air.depth_qualifier", OUT, Arity::None),
    k("air.any", OUT, Arity::None),
    k("air.greater", OUT, Arity::None),
    k("air.less", OUT, Arity::None),
    k("air.payload", ARG, Arity::None),
    k("air.mesh", ARG, Arity::Node),
    k("air.mesh_grid_properties", ARG, Arity::None),
    k("air.visible_input", ARG, Arity::None),
    k("air.visible_output", OUT, Arity::None),
    k("air.mesh_vertex_data", OUT, Arity::I32Str),
    k("air.mesh_primitive_data", OUT, Arity::I32Str),
    // Builtin outputs.
    k("air.position", ARG | OUT, Arity::None),
    k("air.point_size", OUT, Arity::None),
    k("air.render_target_array_index", ARG | OUT, Arity::None),
    k("air.viewport_array_index", ARG | OUT, Arity::None),
    k("air.sample_mask", OUT, Arity::None),
    k("air.stencil", OUT, Arity::None),
    k("air.accept_intersection", OUT, Arity::None),
    k("air.distance", ARG | OUT, Arity::None),
    // Builtin inputs.
    k("air.vertex_id", ARG, Arity::None),
    k("air.instance_id", ARG, Arity::None),
    k("air.base_vertex", ARG, Arity::None),
    k("air.base_instance", ARG, Arity::None),
    k("air.amplification_id", ARG, Arity::None),
    k("air.amplification_count", ARG, Arity::None),
    k("air.front_facing", ARG, Arity::None),
    k("air.point_coord", ARG, Arity::None),
    k("air.primitive_id", ARG, Arity::None),
    k("air.barycentric_coord", ARG, Arity::None),
    k("air.sample_id", ARG, Arity::None),
    k("air.sample_mask_in", ARG, Arity::None),
    k("air.patch_id", ARG, Arity::None),
    k("air.position_in_patch", ARG, Arity::None),
    k("air.thread_position_in_grid", ARG, Arity::None),
    k("air.thread_position_in_threadgroup", ARG, Arity::None),
    k("air.thread_index_in_threadgroup", ARG, Arity::None),
    k("air.threadgroup_position_in_grid", ARG, Arity::None),
    k("air.threadgroups_per_grid", ARG, Arity::None),
    k("air.threads_per_grid", ARG, Arity::None),
    k("air.threads_per_threadgroup", ARG, Arity::None),
    k("air.dispatch_threads_per_threadgroup", ARG, Arity::None),
    k("air.thread_index_in_simdgroup", ARG, Arity::None),
    k("air.threads_per_simdgroup", ARG, Arity::None),
    k("air.simdgroup_index_in_threadgroup", ARG, Arity::None),
    k("air.simdgroups_per_threadgroup", ARG, Arity::None),
    k("air.dispatch_simdgroups_per_threadgroup", ARG, Arity::None),
    k("air.thread_index_in_quadgroup", ARG, Arity::None),
    k("air.quadgroup_index_in_threadgroup", ARG, Arity::None),
    k("air.quadgroups_per_threadgroup", ARG, Arity::None),
    k("air.dispatch_quadgroups_per_threadgroup", ARG, Arity::None),
    k("air.stage_in_grid_origin", ARG, Arity::None),
    k("air.stage_in_grid_size", ARG, Arity::None),
    k("air.origin", ARG, Arity::None),
    k("air.direction", ARG, Arity::None),
    k("air.min_distance", ARG, Arity::None),
    k("air.max_distance", ARG, Arity::None),
    k("air.geometry_id", ARG, Arity::None),
    k("air.object_to_world_transform", ARG, Arity::None),
    k("air.geometry_intersection_function_table_offset", ARG, Arity::None),
    // Struct members.
    k("air.indirect_argument", MEM, Arity::I32OrNode),
    k("air.render_target", MEM, Arity::I32),
];

fn lookup(name: &[u8], context: Context) -> Option<&'static Key> {
    let bit = match context {
        Context::Argument => ARG,
        Context::Output => OUT,
        Context::Member => MEM,
    };
    KEYS.iter().find(|k| k.contexts & bit != 0 && k.name.as_bytes() == name)
}

/// Whether `name` is a key of any context.
pub fn is_key(name: &[u8]) -> bool {
    KEYS.iter().any(|k| k.name.as_bytes() == name)
}

/// One key of a record and its operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item<'m> {
    pub key: &'static str,
    pub ops: Vec<Operand<'m>>,
}

/// Every key name of the table, once each.
pub fn key_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = KEYS.iter().map(|k| k.name).collect();
    v.sort_unstable();
    v.dedup();
    v
}

fn slot(op: Option<Operand<'_>>) -> bool {
    matches!(op, Some(Operand::Int { bits: 32, .. }) | Some(Operand::Global(_)))
}

/// Reads one key and its operands at the cursor.
pub fn item<'m>(c: &mut Cursor<'m>, context: Context) -> Result<Item<'m>, Refusal> {
    let Some(name) = c.peek_str() else {
        return Err(c.error(c.construct(), "a key"));
    };
    let Some(key) = lookup(name, context) else {
        let text = String::from_utf8_lossy(name).into_owned();
        let known = is_key(name);
        let reason = if known {
            format!("{} {}: the key is not valid in a record of this kind ({:?})", c.what, c.render(), context)
        } else if name.starts_with(b"air.") {
            format!("{} {}: the key is not in the measured AIR metadata grammar", c.what, c.render())
        } else {
            format!("{} {}: a string that is not a key stands where the AIR grammar has a key", c.what, c.render())
        };
        return Err(super::md::refusal(RefusalKind::NotImplemented, text, c.function.as_deref(), reason));
    };
    c.advance();
    let mut ops = Vec::new();
    match key.arity {
        Arity::None => {}
        Arity::I32 => {
            c.i32(key.name)?;
            ops.push(op_before(c));
        }
        Arity::Str => {
            c.bytes(key.name)?;
            ops.push(op_before(c));
        }
        Arity::Node => {
            c.node_id(key.name)?;
            ops.push(op_before(c));
        }
        Arity::I32OrNode => match c.peek() {
            Some(Operand::Int { bits: 32, .. }) | Some(Operand::Node(_)) => ops.push(c.advance().unwrap()),
            _ => return Err(c.error(key.name, "an i32 or a metadata tuple")),
        },
        Arity::Slot => {
            if !slot(c.peek()) {
                return Err(c.error(key.name, "an i32 or a function constant"));
            }
            ops.push(c.advance().unwrap());
        }
        Arity::Slots2 => {
            for _ in 0..2 {
                if !slot(c.peek()) {
                    return Err(c.error(key.name, "an i32 or a function constant"));
                }
                ops.push(c.advance().unwrap());
            }
        }
        Arity::SlotI32 => {
            if !slot(c.peek()) {
                return Err(c.error(key.name, "an i32 or a function constant"));
            }
            ops.push(c.advance().unwrap());
            c.i32(key.name)?;
            ops.push(op_before(c));
        }
        Arity::I32Str => {
            c.i32(key.name)?;
            ops.push(op_before(c));
            c.bytes(key.name)?;
            ops.push(op_before(c));
        }
        Arity::Nodes => {
            c.node_id(key.name)?;
            ops.push(op_before(c));
            while let Some(Operand::Node(_)) = c.peek() {
                ops.push(c.advance().unwrap());
            }
        }
    }
    Ok(Item { key: key.name, ops })
}

fn op_before<'m>(c: &Cursor<'m>) -> Operand<'m> {
    c.op_at(c.position() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key of the table is an `air.*` name of the target's compiler that the corpora use (a row of kind
    /// `metadata` in `tables/intrinsics.toml`), or one of the target compiler's strings (kind `string`, no corpus
    /// module has it) whose writer in the compiler's code establishes its form.
    #[test]
    fn keys_are_names_of_the_target_compiler_seen_in_the_corpora() {
        // Keys of the target compiler's code (libGPUCompilerImpl, 21H461) that no corpus module has: the writer that
        // establishes each one's form.
        let from_the_compiler = [
            // `AIRMetadataPrunerPass::run` (0x1c4396ce4) and `upgradeAIRModule` (0x1c43b2808) write it in place of
            // `air.compile.framebuffer_fetch_enable` when framebuffer fetch is off.
            "air.compile.framebuffer_fetch_disable",
            // `AIRenameQualifiedFunctionSignature::getImpl` (0x1c4362958).
            "air.rename_qualified_function_signature",
        ];
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tables/intrinsics.toml")).unwrap();
        let table: toml::Value = toml::from_str(&text).unwrap();
        let rows = table["name"].as_array().unwrap();
        let module_keys = [
            "air.vertex",
            "air.fragment",
            "air.kernel",
            "air.visible",
            "air.intersection",
            "air.mesh",
            "air.object",
            "air.version",
            "air.language_version",
            "air.source_file_name",
            "air.compile_options",
            "air.compile.fast_math_enable",
            "air.compile.fast_math_disable",
            "air.compile.denorms_disable",
            "air.compile.framebuffer_fetch_enable",
            "air.compile.framebuffer_fetch_disable",
            "air.compile.native_double_disable",
            "air.function_constants",
            "air.sampler_states",
            "air.sampler_state",
            "air.visible_function_references",
            "air.visible_function_reference",
            "air.external_initializations",
            "air.externally_initialized",
            "air.signature",
            "air.ci",
            "air.struct_type_info",
            "air.indirect_argument",
            "air.patch",
            "air.patch_control_point",
            "air.max_work_group_size",
            "air.bounding_box",
            "air.triangle",
            "air.instancing",
            "air.multi_level_instancing",
            "air.triangle_data",
            "air.world_space_data",
            "air.max_device_buffers",
            "air.max_constant_buffers",
            "air.max_threadgroup_buffers",
            "air.max_textures",
            "air.max_read_write_textures",
            "air.max_samplers",
        ];
        let all = key_names().into_iter().chain(super::super::desc::KEYS.iter().copied()).chain(module_keys);
        for key in all {
            let row = rows.iter().find(|r| r["name"].as_str() == Some(key));
            let row = row.unwrap_or_else(|| panic!("{} is not an air.* name of the target's compiler", key));
            let kind = if from_the_compiler.contains(&key) { "string" } else { "metadata" };
            assert_eq!(row["kind"].as_str(), Some(kind), "{} is not seen in the corpora", key);
        }
    }
}
