//! Tests of the stage interface lowering (package S6b): the V1 reference harness's stage-interface cases replayed on
//! wgpu and Dawn against the goldens recorded on real Metal, attribute fetch conversions compared with real Metal on
//! this Mac for every code of the 8-, 10- and 16-bit formats, and the refusals for what has no exact lowering.
//! `AW_DAWN_NODE` names Dawn's `dawn.node`; without it the Dawn runs are reported as not done. `AW_V1_REFERENCE` names
//! WebMetal's `tests/reference` when it is not next to this crate in the UniXNU tree.

// Pipelines are compiled once per case, so returning the refusal by value costs nothing that matters.
#![allow(clippy::result_large_err)]

mod fetch;
mod harness;
mod rules;
mod v1;
