//! AirWgsl: translates Metal shaders, as AIR bitcode, to WGSL for WebGPU.
//!
//! [`api`] is the interface the executor uses: [`compile`] takes a pipeline's AIR modules and the state that changes
//! the generated code, and returns WGSL, the binding layout, the pass structure, reflection and the WebGPU features
//! and limits the result needs, or a [`Refusal`].

pub mod api;

pub use api::*;
