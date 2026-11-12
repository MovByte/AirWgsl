//! The memory-model row of plan 10.3: random struct, array, packed and union layouts generated in MSL, written from
//! the CPU with known bytes, read in shaders, and written back, byte-exact against real Metal.
//!
//! Each case declares random structs (scalars of every width, vectors, packed vectors, arrays, nested structs and
//! unions) and a kernel that, per thread, (1) reads every scalar leaf of its input element and writes its bits to a
//! word array (loads at every offset and alignment the layout produces), (2) copies the element field by field in
//! reverse order into an output element (stores of every field type, neighbors written by other threads), and (3)
//! copies one whole element (the front end's memcpy). Output buffers start as a fill pattern, so bytes no store may
//! touch (padding, neighbors) are compared too.

use super::harness::{bytes, differential};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A scalar: MSL name, size in bytes, how its bits become a uint (one or two words).
const SCALARS: &[(&str, u32)] = &[
    ("char", 1),
    ("uchar", 1),
    ("short", 2),
    ("ushort", 2),
    ("int", 4),
    ("uint", 4),
    ("long", 8),
    ("ulong", 8),
    ("half", 2),
    ("float", 4),
];

#[derive(Clone)]
enum Ty {
    Scalar(&'static str, u32),
    /// `name`, scalar, lanes, packed.
    Vector(&'static str, u32, u32, bool),
    Array(Box<Ty>, u32),
    Struct(usize),
    Union(usize),
}

struct Gen {
    rng: Rng,
    decls: Vec<String>,
    /// Members of each declared struct or union.
    bodies: Vec<Vec<(String, Ty)>>,
}

impl Gen {
    fn scalar(&mut self) -> Ty {
        let (n, s) = SCALARS[self.rng.below(SCALARS.len())];
        Ty::Scalar(n, s)
    }

    fn ty(&mut self, depth: u32) -> Ty {
        match self.rng.below(if depth > 1 { 6 } else { 9 }) {
            0..=2 => self.scalar(),
            3 | 4 => {
                // Vectors of 8, 16 and 32-bit scalars (no long vectors in packed form on every version).
                let (n, s) = loop {
                    let (n, s) = SCALARS[self.rng.below(SCALARS.len())];
                    if s <= 4 {
                        break (n, s);
                    }
                };
                let lanes = 2 + self.rng.below(3) as u32;
                Ty::Vector(n, s, lanes, self.rng.below(2) == 0)
            }
            5 => {
                let e = if self.rng.below(2) == 0 { self.scalar() } else { self.ty(depth + 1) };
                Ty::Array(Box::new(e), 1 + self.rng.below(4) as u32)
            }
            6 | 7 => self.record(depth + 1, false),
            _ => self.record(depth + 1, true),
        }
    }

    fn record(&mut self, depth: u32, union: bool) -> Ty {
        let n = 1 + self.rng.below(if union { 3 } else { 6 });
        let mut members = Vec::new();
        for k in 0..n {
            let t = if union {
                // Union members: scalars and packed vectors, whose bytes all compare as written.
                match self.rng.below(2) {
                    0 => self.scalar(),
                    _ => Ty::Vector("uchar", 1, 4, true),
                }
            } else {
                self.ty(depth)
            };
            members.push((format!("m{}", k), t));
        }
        let id = self.bodies.len();
        self.bodies.push(members.clone());
        let mut d = format!("{} R{} {{\n", if union { "union" } else { "struct" }, id);
        for (name, t) in &members {
            let (base, suffix) = self.spell(t);
            d.push_str(&format!("    {} {}{};\n", base, name, suffix));
        }
        d.push_str("};\n");
        self.decls.push(d);
        if union { Ty::Union(id) } else { Ty::Struct(id) }
    }

    fn spell(&self, t: &Ty) -> (String, String) {
        match t {
            Ty::Scalar(n, _) => (n.to_string(), String::new()),
            Ty::Vector(n, _, l, packed) => {
                (format!("{}{}{}", if *packed { "packed_" } else { "" }, n, l), String::new())
            }
            Ty::Array(e, len) => {
                let (b, s) = self.spell(e);
                (b, format!("[{}]{}", len, s))
            }
            Ty::Struct(i) | Ty::Union(i) => (format!("R{}", i), String::new()),
        }
    }

    /// Statements writing the bits of every scalar leaf of `expr` (of type `t`) to `w[k++]`. Unions are read
    /// through their first member only (the bytes another member would read are compared by the copies).
    fn read_leaves(&self, t: &Ty, expr: &str, out: &mut String) {
        match t {
            Ty::Scalar(n, s) => out.push_str(&leaf(n, *s, expr)),
            Ty::Vector(n, s, l, packed) => {
                for i in 0..*l {
                    // Packed vector lanes are read through an unpacked copy.
                    let e = if *packed { format!("{}{}({})[{}]", n, l, expr, i) } else { format!("{}[{}]", expr, i) };
                    out.push_str(&leaf(n, *s, &e));
                }
            }
            Ty::Array(e, len) => {
                for i in 0..*len {
                    self.read_leaves(e, &format!("{}[{}]", expr, i), out);
                }
            }
            Ty::Struct(id) => {
                for (m, mt) in &self.bodies[*id] {
                    self.read_leaves(mt, &format!("{}.{}", expr, m), out);
                }
            }
            Ty::Union(id) => {
                let (m, mt) = &self.bodies[*id][0];
                self.read_leaves(mt, &format!("{}.{}", expr, m), out);
            }
        }
    }

    /// Field-by-field copy, members in reverse order.
    fn copy(&self, t: &Ty, dst: &str, src: &str, out: &mut String) {
        match t {
            Ty::Scalar(..) | Ty::Vector(..) | Ty::Union(..) => out.push_str(&format!("    {} = {};\n", dst, src)),
            Ty::Array(e, len) => {
                for i in (0..*len).rev() {
                    self.copy(e, &format!("{}[{}]", dst, i), &format!("{}[{}]", src, i), out);
                }
            }
            Ty::Struct(id) => {
                for (m, mt) in self.bodies[*id].iter().rev() {
                    self.copy(mt, &format!("{}.{}", dst, m), &format!("{}.{}", src, m), out);
                }
            }
        }
    }
}

fn leaf(n: &str, s: u32, e: &str) -> String {
    match (n, s) {
        ("half", _) => format!("    w[k++] = uint(as_type<ushort>({}));\n", e),
        ("float", _) => format!("    w[k++] = as_type<uint>({});\n", e),
        (_, 8) => format!("    w[k++] = uint(ulong({e})); w[k++] = uint(ulong({e}) >> 32);\n", e = e),
        ("char" | "short" | "int", _) => format!("    w[k++] = uint(int({}));\n", e),
        _ => format!("    w[k++] = uint({});\n", e),
    }
}

fn count_leaves(g: &Gen, t: &Ty) -> usize {
    match t {
        Ty::Scalar(_, 8) => 2,
        Ty::Scalar(..) => 1,
        Ty::Vector(_, _, l, _) => *l as usize,
        Ty::Array(e, len) => count_leaves(g, e) * *len as usize,
        Ty::Struct(id) => g.bodies[*id].iter().map(|(_, t)| count_leaves(g, t)).sum(),
        Ty::Union(id) => count_leaves(g, &g.bodies[*id][0].1),
    }
}

/// A case: MSL source, element size bound (bytes), leaves per element.
fn case(seed: u64) -> (String, usize) {
    let mut g = Gen { rng: Rng(seed), decls: Vec::new(), bodies: Vec::new() };
    let top = g.record(0, false);
    let leaves = count_leaves(&g, &top);
    let mut reads = String::new();
    g.read_leaves(&top, "in[i]", &mut reads);
    let mut copies = String::new();
    g.copy(&top, "out[i]", "in[i]", &mut copies);
    let Ty::Struct(id) = top else { unreachable!() };
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\n{}\n\
         kernel void layout(device const R{id}* in [[buffer(0)]], device R{id}* out [[buffer(1)]],\n\
                            device uint* words [[buffer(2)]], device R{id}* whole [[buffer(3)]],\n\
                            uint i [[thread_position_in_grid]]) {{\n\
             device uint* w = words + i * {leaves};\n\
             uint k = 0;\n{reads}{copies}\
             if (i % 3u == 0u) {{\n        whole[i] = in[i];\n    }}\n\
         }}\n",
        g.decls.join("\n"),
        id = id,
        leaves = leaves,
        reads = reads,
        copies = copies
    );
    (src, leaves)
}

#[test]
fn random_layouts_are_byte_exact() {
    let cases: u64 = std::env::var("AW_LAYOUT_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let mut dawn = 0;
    for seed in 0..cases {
        let (src, leaves) = case(0x1a70_0000 + seed);
        let n = 128;
        // Element size bound: every leaf is at most 8 bytes, padded at most to 16; the buffers are over-sized and the
        // tail is compared as untouched fill.
        let elem_bound = 64 * leaves.max(1);
        let input = bytes(seed * 7 + 1, n * elem_bound);
        let out = vec![0xa5u8; n * elem_bound];
        let words = vec![0x5au8; n * leaves * 4];
        let whole = vec![0x3cu8; n * elem_bound];
        let r = differential(
            &format!("layout{}", seed),
            &src,
            "layout",
            [32, 1, 1],
            [4, 1, 1],
            &[],
            &[input, out, words, whole],
        );
        dawn += r.dawn as usize;
    }
    eprintln!("{} random layouts byte-exact on Metal and wgpu, {} also on Dawn", cases, dawn);
}
