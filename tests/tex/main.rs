//! Differential tests of the texture and sampler lowering: compute functions that read, write, sample and gather
//! textures run on real Metal, on wgpu and on Dawn. Results that MSL defines exactly (texel selection by address modes
//! and coordinates, out-of-bounds defaults, format conversions, write rounding, size queries) must be byte-identical.
//! Samples are taken at texel centres with nearest filtering, where no filter weight is involved. `AW_DAWN_NODE`
//! names Dawn's `dawn.node`; without it the Dawn runs are reported as not done.

mod cases;
mod harness;
