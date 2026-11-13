//! Tests of the tessellator (package S6e): the patch draws of WebMetal's V1 `tessellation.*` cases replayed through
//! [`airwgsl::tess`] and compared with the goldens recorded on real Metal (Apple M4): the domain points bit for bit
//! and the triangles in rasterization order, vertex order included. `AW_V1_REFERENCE` names WebMetal's
//! `tests/reference` when it is not next to this crate in the UniXNU tree.

#[path = "../stage/harness.rs"]
#[allow(dead_code)]
mod stage;

mod draw;
mod kernel;

use draw::{Draw, Expected};

/// The tessellation cases of the V1 harness.
const CASES: &[&str] = &[
    "tessellation.quad",
    "tessellation.triangle",
    "tessellation.factors",
    "tessellation.draws",
    "tessellation.misuse",
];

#[test]
fn tessellator_equals_metal_over_the_v1_sweeps() {
    let Some(reference) = stage::reference() else {
        eprintln!("not run: WebMetal's tests/reference is not next to this crate and AW_V1_REFERENCE is unset");
        return;
    };
    let (mut exact, mut batched_equal_sets, mut total) = (0, 0, 0);
    for case in CASES {
        for (name, draw, expected) in draw::load(&reference, case) {
            total += 1;
            let got = draw.run();
            assert_eq!(got.points, expected.points, "{case} {name}: points");
            if got.batched {
                // A blade wider than the M4 emits whole: its order depends on the patches before it (hw::Counts); the
                // triangles must still be Metal's, vertex order included.
                let mut a = got.triangles.clone();
                let mut b = expected.triangles.clone();
                a.sort();
                b.sort();
                assert_eq!(a, b, "{case} {name}: triangles");
                batched_equal_sets += 1;
            } else {
                if got.triangles != expected.triangles {
                    let first = got.triangles.iter().zip(&expected.triangles).position(|(a, b)| a != b);
                    panic!(
                        "{case} {name}: {} triangles, Metal {}; first difference at {:?}",
                        got.triangles.len(),
                        expected.triangles.len(),
                        first
                    );
                }
                exact += 1;
            }
        }
    }
    eprintln!(
        "{total} draws: {exact} equal Metal's points and triangle order exactly; {batched_equal_sets} with blades the \
         M4 batches equal its points and triangles as a set"
    );
    assert!(exact > 0);
}

/// The draws of the cases are the ones the comparison above found, so a golden change shows up as a count change.
#[test]
fn v1_sweeps_are_complete() {
    let Some(reference) = stage::reference() else { return };
    let counts: Vec<usize> = CASES.iter().map(|c| draw::load(&reference, c).len()).collect();
    assert_eq!(counts, vec![120, 120, 35, 8, 4]);
    let _ = Expected::default();
    let _ = Draw::default();
}
