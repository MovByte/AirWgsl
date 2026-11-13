//! Tessellation (plan 5.7): Metal's fixed-function tessellator, its factor buffers, and post-tessellation vertex
//! functions.
//!
//! WebGPU has no tessellation stage. A patch draw becomes compute work that writes the generated domain points and the
//! triangles connecting them, in Metal's order, followed by an indexed indirect draw of those triangles whose vertex
//! stage is the post-tessellation vertex function.
//!
//! [`hw`] generates one patch from its final factors; [`factors`] turns the half-precision factors of a factor buffer
//! into those final factors as Metal does; [`kernel`] is the same computation as WebGPU compute work, which the
//! executor runs for a patch draw.

pub mod factors;
pub mod hw;
pub mod kernel;

pub use factors::{PartitionMode, PatchType, Settings};

/// Tessellate one patch from the factors Metal reads for it from the factor buffer: `MTLTriangleTessellationFactorsHalf`
/// (three edge factors and the inside factor) or `MTLQuadTessellationFactorsHalf` (four edge factors and two inside
/// factors), as binary16 bit patterns. `scale` is the encoder's `setTessellationFactorScale:` value.
pub fn tessellate(settings: &Settings, patch: PatchType, halves: &[u16], scale: f32) -> hw::Patch {
    let f = factors::process(settings, patch, halves, scale);
    let Some(f) = f else { return hw::Patch::default() };
    match patch {
        PatchType::Triangle => {
            hw::triangle(f.partitioning, settings.clockwise, [f.edges[0], f.edges[1], f.edges[2]], f.inside[0])
        }
        PatchType::Quad => hw::quad(f.partitioning, settings.clockwise, f.edges, f.inside),
    }
}

/// The sizes [`tessellate`] produces.
pub fn counts(settings: &Settings, patch: PatchType, halves: &[u16], scale: f32) -> hw::Counts {
    let Some(f) = factors::process(settings, patch, halves, scale) else { return hw::Counts::default() };
    match patch {
        PatchType::Triangle => hw::triangle_counts(f.partitioning, [f.edges[0], f.edges[1], f.edges[2]], f.inside[0]),
        PatchType::Quad => hw::quad_counts(f.partitioning, f.edges, f.inside),
    }
}
