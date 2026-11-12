//! Numerics tests: the WGSL library run on a real WebGPU implementation (wgpu, Metal backend) and compared bit for
//! bit, or against the bounds of the numerics table, with CPU references written independently of it.
//!
//! The quick tests run by default. The exhaustive sweeps (every binary32 input of a unary operation, every pair of
//! binary16 inputs of a binary operation, long random runs) are `#[ignore]`d and run with
//! `cargo test --release --test num -- --ignored`.

mod common;
mod gpu;
mod inputs;
mod lower;
mod math;
mod metal_special;
mod native;
mod pack;
mod reference;
mod selftest;
mod softfloat;

use airwgsl::num::prelude::Prelude;

/// The library items `names` need, as WGSL text.
pub fn prelude(names: &[&str]) -> String {
    let mut p = Prelude::new();
    for n in names {
        p.require(n).unwrap_or_else(|e| panic!("{:?}", e));
    }
    p.emit().unwrap()
}
