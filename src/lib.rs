//! AirWgsl: translates Metal shaders, as AIR bitcode, to WGSL for WebGPU.
//!
//! [`api`] is the interface the executor uses: [`compile`] takes a pipeline's AIR modules and the state that changes
//! the generated code, and returns WGSL, the binding layout, the pass structure, reflection and the WebGPU features
//! and limits the result needs, or a [`Refusal`].

pub mod air;
pub mod api;
pub mod argbuf;
pub mod bitcode;
pub mod bitstream;
pub mod cfg;
pub mod ftable;
pub mod icb;
pub mod ir;
pub mod layout;
pub mod mem;
pub mod mtlb;
pub mod num;
pub mod reflect;
pub mod sync;
pub mod wgsl;

pub use api::*;
pub use bitcode::parse_bitcode;
pub use mtlb::parse_metallib;
