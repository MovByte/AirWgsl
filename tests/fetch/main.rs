//! Tests of framebuffer fetch (package S6c): WebMetal's V1 `render.framebuffer-fetch` case replayed against its golden,
//! differential cases against this Mac's Metal (overlapping and disjoint primitives, strips, blending with fetched
//! values, multisampled targets, every type a color input can read from each format class), the overlap prover's
//! groups against the one-primitive-per-pass baseline, and the refusals. Every scene runs on wgpu and, when
//! `AW_DAWN_NODE` names Dawn's `dawn.node`, on Dawn. `AW_V1_REFERENCE` names WebMetal's `tests/reference` when it is
//! not next to this crate in the UniXNU tree.

// Pipelines are compiled once per case, so returning the refusal by value costs nothing that matters.
#![allow(clippy::result_large_err)]

#[path = "../stage/harness.rs"]
#[allow(dead_code)]
mod stage;

mod cases;
mod harness;
