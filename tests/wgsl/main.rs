//! Differential tests of the WGSL back end: kernels in the exact subset (integer arithmetic, control flow, memory
//! moves, conversions MSL defines exactly) run on real Metal, on wgpu and on Dawn, with byte-identical results
//! required. `AW_DAWN_NODE` names Dawn's `dawn.node`; without it the Dawn runs are reported as not done.

mod harness;
mod hooks;
mod kernels;
mod layouts;
mod render;
