//! The patch draws of the V1 `tessellation.*` cases: their inputs, their recorded outputs, and the same draw replayed
//! through the CPU tessellator.

use super::stage::{GOLDEN_DEVICE, Json, parse_json};
use airwgsl::tess::{self, PartitionMode, PatchType, Settings, hw};
use std::path::Path;

/// One patch draw, as the case's `.input` record holds it.
#[derive(Clone, Debug, Default)]
pub struct Draw {
    pub quad: bool,
    pub mode: u32,
    pub max_factor: u32,
    /// `MTLWinding`.
    pub winding: u32,
    pub scale_enabled: bool,
    pub scale: f32,
    /// `MTLTessellationFactorStepFunction`.
    pub step: u32,
    pub offset: u32,
    pub instance_stride: u32,
    pub patch_start: u32,
    pub patch_count: u32,
    pub instance_count: u32,
    pub base_instance: u32,
    /// The factor buffer's binary16 values.
    pub factors: Vec<u16>,
    pub patch_indices: Vec<u32>,
}

/// A draw's outputs: points sorted by (instance, patch, u, v, w) and the non-degenerate triangles in order, as point
/// indices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expected {
    pub points: Vec<[u32; 5]>,
    pub triangles: Vec<[u32; 3]>,
}

/// The CPU tessellator's outputs for a draw, in the recorded form.
pub struct Got {
    pub points: Vec<[u32; 5]>,
    pub triangles: Vec<[u32; 3]>,
    /// Some patch has an inside blade the M4 may emit in batches.
    pub batched: bool,
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn hex(r: &Json) -> Vec<u8> {
    let h = r.get("hex").and_then(|h| h.str()).unwrap_or("");
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// Every draw of a case: (name, draw, expected outputs).
pub fn load(reference: &Path, case: &str) -> Vec<(String, Draw, Expected)> {
    let p = reference.join("goldens").join(GOLDEN_DEVICE).join("release").join(format!("{case}.json"));
    let j = parse_json(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())));
    assert_eq!(j.get("status").and_then(|s| s.str()), Some("ok"), "{case} did not run to its end");
    let records = j.get("records").unwrap().arr();
    let named = |n: &str| records.iter().find(|r| r.get("name").and_then(|x| x.str()) == Some(n));
    let mut out = Vec::new();
    for r in records {
        let Some(name) = r.get("name").and_then(|n| n.str()).and_then(|n| n.strip_suffix(".input")) else { continue };
        let w = words(&hex(r));
        let nf = w[13] as usize;
        let ni = w[14] as usize;
        let d = Draw {
            quad: w[0] != 0,
            mode: w[1],
            max_factor: w[2],
            winding: w[3],
            scale_enabled: w[4] != 0,
            scale: f32::from_bits(w[5]),
            step: w[6],
            offset: w[7],
            instance_stride: w[8],
            patch_start: w[9],
            patch_count: w[10],
            instance_count: w[11],
            base_instance: w[12],
            factors: w[15..15 + nf].iter().map(|&h| h as u16).collect(),
            patch_indices: w[15 + nf..15 + nf + ni].to_vec(),
        };
        let pos = words(&hex(named(&format!("{name}.points")).expect("points")));
        let patches = named(&format!("{name}.point-patches")).map(|r| words(&hex(r)));
        let points = (0..pos.len() / 3)
            .map(|i| {
                let (inst, patch) = match &patches {
                    Some(p) => (p[2 * i], p[2 * i + 1]),
                    None => {
                        (d.base_instance, d.patch_indices.get(d.patch_start as usize).copied().unwrap_or(d.patch_start))
                    }
                };
                [inst, patch, pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]
            })
            .collect();
        let t = hex(named(&format!("{name}.triangles")).expect("triangles"));
        let t: Vec<u32> = t.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect();
        let triangles = t.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        out.push((name.to_string(), d, Expected { points, triangles }));
    }
    out
}

impl Draw {
    pub fn settings(&self) -> Settings {
        Settings {
            partition: PartitionMode::from_metal(self.mode as u64).unwrap(),
            max_factor: self.max_factor,
            scale_enabled: self.scale_enabled,
            clockwise: self.winding == 0,
        }
    }

    pub fn patch_type(&self) -> PatchType {
        if self.quad { PatchType::Quad } else { PatchType::Triangle }
    }

    /// The factors of the patch at `draw_patch` of instance `instance_id` (which includes the base instance), as the
    /// step function addresses them (measured, V1 `tessellation.draws`).
    pub fn factors_of(&self, draw_patch: u32, instance_id: u32) -> Vec<u16> {
        let size = self.patch_type().factors_size() as u32;
        let mut at = self.offset;
        if self.step == 1 || self.step == 3 {
            at += draw_patch * size;
        }
        if self.step == 2 || self.step == 3 {
            at += instance_id * self.instance_stride;
        }
        let first = (at / 2) as usize;
        self.factors[first..first + size as usize / 2].to_vec()
    }

    /// The patch identifier of the patch at `draw_patch`: the patch index buffer's element there, else itself.
    pub fn patch_id(&self, draw_patch: u32) -> u32 {
        if self.patch_indices.is_empty() { draw_patch } else { self.patch_indices[draw_patch as usize] }
    }

    /// Replay the draw: instances in order, each instance's patches in order.
    pub fn run(&self) -> Got {
        let settings = self.settings();
        let mut raw: Vec<[u32; 5]> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let mut batched = false;
        for i in 0..self.instance_count {
            let instance_id = self.base_instance + i;
            for dp in self.patch_start..self.patch_start + self.patch_count {
                let halves = self.factors_of(dp, instance_id);
                let scale = if self.scale_enabled { self.scale } else { 1.0 };
                let counts = tess::counts(&settings, self.patch_type(), &halves, scale);
                batched |= counts.batched;
                let patch = tess::tessellate(&settings, self.patch_type(), &halves, scale);
                assert_eq!(
                    (counts.points as usize, counts.indices as usize),
                    (patch.points.len(), patch.indices.len())
                );
                let base = raw.len();
                for p in &patch.points {
                    let (u, v) = (hw::fixed_to_float(p[0]), hw::fixed_to_float(p[1]));
                    let w = if self.quad { 0 } else { hw::fixed_to_float(hw::FXP_ONE - p[0] - p[1]).to_bits() };
                    raw.push([instance_id, self.patch_id(dp), u.to_bits(), v.to_bits(), w]);
                }
                for t in patch.indices.chunks(3) {
                    tris.push([base + t[0] as usize, base + t[1] as usize, base + t[2] as usize]);
                }
            }
        }
        let mut points = raw.clone();
        points.sort();
        points.dedup();
        let index = |k: usize| points.binary_search(&raw[k]).unwrap() as u32;
        let triangles = tris
            .iter()
            .map(|t| [index(t[0]), index(t[1]), index(t[2])])
            // A triangle with two points at one position covers no pixel: real Metal's is not recorded.
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
            .collect();
        Got { points, triangles, batched }
    }
}
