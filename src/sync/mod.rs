//! Atomics and barriers (plan 5.8 and the barrier row of 5.7): the lowering rules of the `atomic-int32`,
//! `threadgroup-barrier` and `simdgroup-barrier` families, and WGSL's uniformity analysis, which decides where a
//! barrier has an exact WGSL form.

pub mod atomic;
pub mod barrier;
pub mod uniform;

use crate::wgsl::hooks::CallRule;

/// The rules of this package.
pub fn rules() -> Vec<Box<dyn CallRule>> {
    vec![Box::new(atomic::Atomics), Box::new(barrier::Barriers)]
}
