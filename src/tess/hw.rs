//! The fixed-function tessellator of one patch: domain points in 16.16 fixed point and the triangles that connect
//! them, in the order real Metal rasterizes them.
//!
//! The points, the factor processing and the stitching of the outer edges to the outermost inside ring are Microsoft's
//! D3D11 reference tessellator (`CHWTessellator`, May 7, 2012), adapted from the copy in Mesa
//! (`src/gallium/auxiliary/tessellator/tessellator.cpp`, MIT licence, credited in `NOTICE`): the same fixed-point
//! arithmetic, the same point placement and the same triangles. Its factor clamps to D3D11's ranges are kept:
//! [`super::factors`] applies Metal's ranges first, which lie inside D3D11's, so they leave Metal's factors unchanged.
//! The isoline domain is left out: Metal has triangle and quad patches only.
//!
//! Where Metal's order differs from the reference's, Metal's is followed, as measured on an Apple M4 (macOS 27.0.1;
//! WebMetal V1 cases `tessellation.*`): every point position and every triangle equal the reference's, and
//!
//! - the transition triangles come in the reference's order, but the middle triangle of an edge whose two rows have
//!   different parities, and the second of the two middle triangles when both are odd, start at their first outer
//!   point (the reference starts the first at its inner point), as does the second triangle of a quad patch whose
//!   factors are all 1; the triangle of a triangle patch whose factors are all 1, and an odd inside factor's center
//!   triangle when no ring lies inside the transition, start at the point of edge 1 (W);
//! - the inside of the transition is not emitted ring by ring but blade by blade: a quad patch's inside grid in four
//!   blades around its center, a triangle patch's in three, each blade's quads in rows away from the center, each quad
//!   as two triangles from its far corner ([`QuadBlade`], [`TriGrid`]).
//!
//! A blade spanning more than [`WHOLE_BLADE_POINTS`] grid points the M4 emits in tiles whose sizes depend on what the
//! draw emitted before it, including earlier patches; this order is not reproduced, and [`Counts::batched`] marks the
//! patches that have such a blade. Metal's documentation leaves the subdivision implementation-dependent ("subdivision
//! is performed in an implementation-dependent manner", Metal Programming Guide, Tessellation): the A10X's tessellator
//! is another design (its driver encodes patch draws with the `bucketed-tessellation-*` driver shaders of
//! `AGXMetalA10.bundle/ds.g9g`) whose order the M4's stands in for.

/// A 16.16 fixed-point number (the reference's `FXP`, an `unsigned int`).
pub type Fxp = u32;

const FXP_FRACTION_BITS: u32 = 16;
const FXP_FRACTION_MASK: Fxp = 0x0000_ffff;
const FXP_INTEGER_MASK: Fxp = 0x7fff_0000;
/// 1.0.
pub const FXP_ONE: Fxp = 1 << FXP_FRACTION_BITS;
const FXP_ONE_THIRD: Fxp = 0x0000_5555;
const FXP_TWO_THIRDS: Fxp = 0x0000_aaaa;
const FXP_ONE_HALF: Fxp = 0x0000_8000;

const MIN_ODD_TESSELLATION_FACTOR: f32 = 1.0;
const MAX_ODD_TESSELLATION_FACTOR: f32 = 63.0;
const MIN_EVEN_TESSELLATION_FACTOR: f32 = 2.0;
const MAX_EVEN_TESSELLATION_FACTOR: f32 = 64.0;
/// The largest factor the tables below cover.
pub const MAX_TESSELLATION_FACTOR: u32 = 64;

/// 2^-16, the smallest positive fixed-point fraction.
const EPSILON: f32 = 1.0 / 65536.0;
const MIN_ODD_TESSFACTOR_PLUS_HALF_EPSILON: f32 = MIN_ODD_TESSELLATION_FACTOR + EPSILON / 2.0;

/// 1/n in 16.16 fixed point for n in 1..=64 (entry 0 is unused).
const FIXED_RECIPROCAL: [Fxp; MAX_TESSELLATION_FACTOR as usize + 1] = [
    0xffff_ffff,
    0x10000,
    0x8000,
    0x5555,
    0x4000,
    0x3333,
    0x2aab,
    0x2492,
    0x2000,
    0x1c72,
    0x199a,
    0x1746,
    0x1555,
    0x13b1,
    0x1249,
    0x1111,
    0x1000,
    0xf0f,
    0xe39,
    0xd79,
    0xccd,
    0xc31,
    0xba3,
    0xb21,
    0xaab,
    0xa3d,
    0x9d9,
    0x97b,
    0x925,
    0x8d4,
    0x889,
    0x842,
    0x800,
    0x7c2,
    0x788,
    0x750,
    0x71c,
    0x6eb,
    0x6bd,
    0x690,
    0x666,
    0x63e,
    0x618,
    0x5f4,
    0x5d1,
    0x5b0,
    0x591,
    0x572,
    0x555,
    0x539,
    0x51f,
    0x505,
    0x4ec,
    0x4d5,
    0x4be,
    0x4a8,
    0x492,
    0x47e,
    0x46a,
    0x457,
    0x444,
    0x432,
    0x421,
    0x410,
    0x400,
];

/// The partitioning the tessellator itself distinguishes. Metal's power-of-two mode reaches it as integer
/// partitioning after its factors are rounded to powers of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partitioning {
    Integer,
    FractionalOdd,
    FractionalEven,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parity {
    Even,
    Odd,
}

/// One generated patch: points as (u, v) in 16.16 fixed point (`w = 1 - u - v` on a triangle) and three indices into
/// `points` per triangle, in output order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Patch {
    pub points: Vec<[Fxp; 2]>,
    pub indices: Vec<u32>,
}

/// The reference's `fixedToFloat`. Exact for the values the tessellator generates, which are at most 1.0 and have at
/// most 17 significant bits.
pub fn fixed_to_float(x: Fxp) -> f32 {
    (x >> FXP_FRACTION_BITS) as f32 + (x & FXP_FRACTION_MASK) as f32 / (1u32 << FXP_FRACTION_BITS) as f32
}

/// The reference's `floatToIDotF<15, 16, false>`: round to nearest even, NaN and values below 2^-17 to 0, values at
/// or above 2^15 to all ones.
pub fn float_to_fixed(input: f32) -> Fxp {
    const I_BITS: i32 = 15;
    const F_BITS: i32 = 16;
    const MANTISSA_BITS: i32 = 23;
    const EXPONENT_BIAS: i32 = 127;
    const HIDDEN_BIT: i32 = 1 << MANTISSA_BITS;
    const MANTISSA_MASK: i32 = HIDDEN_BIT - 1;
    const EXPONENT_MASK: i32 = 0xff << MANTISSA_BITS;
    const SIGN_BIT: i32 = i32::MIN;
    // Unsigned: the most negative result is 0 and the most positive all ones; the largest representable float is
    // 2^15 (the subtraction of the reference's iShift does not apply, iShift being negative).
    let min_result: i32 = 0;
    let max_result: i32 = !min_result;
    let max_pos_value_float: i32 = (EXPONENT_BIAS + I_BITS) << MANTISSA_BITS;
    let max_neg_value_float: i32 = 0;

    let mut output = input.to_bits() as i32;
    let unbiased_exponent = ((output & EXPONENT_MASK) >> MANTISSA_BITS) - EXPONENT_BIAS;
    let is_negative = output & SIGN_BIT;
    if unbiased_exponent == EXPONENT_BIAS + 1 && (output & MANTISSA_MASK) != 0 {
        output = 0;
    } else if is_negative == 0 && output >= max_pos_value_float {
        output = max_result;
    } else if is_negative != 0 && (output & !SIGN_BIT) >= max_neg_value_float {
        output = min_result;
    } else if unbiased_exponent < -F_BITS - 1 {
        output = 0;
    } else {
        output = (output & MANTISSA_MASK) | HIDDEN_BIT;
        let extra_bits = MANTISSA_BITS - F_BITS - unbiased_exponent;
        if extra_bits >= 0 {
            if is_negative != 0 {
                output = (!output).wrapping_add(1);
            }
            let lsb = 1i32 << extra_bits;
            let extra_bits_mask = lsb - 1;
            let half = lsb >> 1;
            if (output & lsb) != 0 || (output & extra_bits_mask) > half {
                output += half;
            }
            output >>= extra_bits;
        } else {
            output <<= -extra_bits;
            if is_negative != 0 {
                output = (!output).wrapping_add(1);
            }
        }
    }
    output as Fxp
}

fn is_even(input: f32) -> bool {
    (input as i32) & 1 == 0
}

fn fxp_ceil(input: Fxp) -> Fxp {
    if input & FXP_FRACTION_MASK != 0 { (input & FXP_INTEGER_MASK).wrapping_add(FXP_ONE) } else { input }
}

fn fxp_floor(input: Fxp) -> Fxp {
    input & FXP_INTEGER_MASK
}

fn is_nan(a: f32) -> bool {
    a.is_nan()
}

/// Flush a denormal to a zero of its sign.
fn flush(a: f32) -> f32 {
    let bits = a.to_bits();
    if bits & 0x7fff_ffff < 0x0080_0000 { f32::from_bits(bits & 0x8000_0000) } else { a }
}

/// IEEE 754R min of the reference (`tess_fmin`): a NaN `b` returns `a`.
fn fmin(a: f32, b: f32) -> f32 {
    let (fa, fb) = (flush(a), flush(b));
    if is_nan(fb) {
        a
    } else if fa == 0.0 && fb == 0.0 {
        if fa.to_bits() & 0x8000_0000 != 0 { a } else { b }
    } else if fa < fb {
        a
    } else {
        b
    }
}

/// IEEE 754R max of the reference (`tess_fmax`): a NaN `b` returns `a`.
fn fmax(a: f32, b: f32) -> f32 {
    let (fa, fb) = (flush(a), flush(b));
    if is_nan(fb) {
        a
    } else if fa == 0.0 && fb == 0.0 {
        if fb.to_bits() & 0x8000_0000 != 0 { a } else { b }
    } else if fa >= fb {
        a
    } else {
        b
    }
}

fn remove_msb(val: i32) -> i32 {
    let mut check: i32 = if val <= 0x0000_ffff {
        if val <= 0xff { 0x80 } else { 0x8000 }
    } else if val <= 0x00ff_ffff {
        0x0080_0000
    } else {
        i32::MIN
    };
    for _ in 0..8 {
        if val & check != 0 {
            return val & !check;
        }
        check = ((check as u32) >> 1) as i32;
    }
    0
}

/// The per-factor context `PlacePointIn1D` uses (`TESS_FACTOR_CONTEXT`).
#[derive(Clone, Copy, Debug, Default)]
struct FactorContext {
    inv_num_segments_on_floor: Fxp,
    inv_num_segments_on_ceil: Fxp,
    half_fraction: Fxp,
    num_half_points: i32,
    split_point_on_floor_half: i32,
}

/// The index remapping of `m_IndexPatchContext`.
#[derive(Clone, Copy, Debug, Default)]
struct IndexPatch {
    inside_delta: i32,
    inside_bad: i32,
    inside_replacement: i32,
    outside_patch_base: i32,
    outside_delta: i32,
    outside_bad: i32,
    outside_replacement: i32,
}

/// The index remapping of `m_IndexPatchContext2`.
#[derive(Clone, Copy, Debug, Default)]
struct IndexPatch2 {
    base_to_invert: i32,
    inversion_end: i32,
    corner_bad: i32,
    corner_replacement: i32,
}

/// The tessellator's state for one patch.
struct Tessellator {
    original_partitioning: Partitioning,
    original_parity: Parity,
    parity: Parity,
    clockwise: bool,
    points: Vec<[Fxp; 2]>,
    indices: Vec<u32>,
    num_points: i32,
    num_indices: i32,
    patched: bool,
    patched2: bool,
    patch: IndexPatch,
    patch2: IndexPatch2,
}

const U: usize = 0;
const V: usize = 1;
const QUAD_EDGES: usize = 4;
const TRI_EDGES: usize = 3;

#[derive(Default)]
struct QuadFactors {
    culled: bool,
    minimum: bool,
    outside: [Fxp; 4],
    inside: [Fxp; 2],
    outside_parity: [Option<Parity>; 4],
    inside_parity: [Option<Parity>; 2],
    outside_ctx: [FactorContext; 4],
    inside_ctx: [FactorContext; 2],
    num_points_outside: [i32; 4],
    num_points_inside: [i32; 2],
    inside_edge_point_base: i32,
}

#[derive(Default)]
struct TriFactors {
    culled: bool,
    minimum: bool,
    outside: [Fxp; 3],
    inside: Fxp,
    outside_parity: [Option<Parity>; 3],
    inside_parity: Option<Parity>,
    outside_ctx: [FactorContext; 3],
    inside_ctx: FactorContext,
    num_points_outside: [i32; 3],
    num_points_inside: i32,
    inside_edge_point_base: i32,
}

fn par(p: Option<Parity>) -> Parity {
    p.expect("parity set before use")
}

impl Tessellator {
    fn new(partitioning: Partitioning, clockwise: bool) -> Tessellator {
        let parity = match partitioning {
            Partitioning::FractionalOdd => Parity::Odd,
            // The reference leaves the parity of integer partitioning uninitialized; every use sets it first.
            Partitioning::Integer | Partitioning::FractionalEven => Parity::Even,
        };
        Tessellator {
            original_partitioning: partitioning,
            original_parity: parity,
            parity,
            clockwise,
            points: Vec::new(),
            indices: Vec::new(),
            num_points: 0,
            num_indices: 0,
            patched: false,
            patched2: false,
            patch: IndexPatch::default(),
            patch2: IndexPatch2::default(),
        }
    }

    fn integer(&self) -> bool {
        self.original_partitioning == Partitioning::Integer
    }

    fn odd(&self) -> bool {
        self.parity == Parity::Odd
    }

    fn set_parity(&mut self, p: Parity) {
        self.parity = p;
    }

    fn define_point(&mut self, u: Fxp, v: Fxp, offset: i32) {
        let o = offset as usize;
        if self.points.len() <= o {
            self.points.resize(o + 1, [0, 0]);
        }
        self.points[o] = [u, v];
    }

    fn patch_index(&self, mut index: i32) -> i32 {
        if self.patched {
            let p = &self.patch;
            if index >= p.outside_patch_base {
                if index == p.outside_bad {
                    index = p.outside_replacement;
                } else {
                    index += p.outside_delta;
                }
            } else if index == p.inside_bad {
                index = p.inside_replacement;
            } else {
                index += p.inside_delta;
            }
        } else if self.patched2 {
            let p = &self.patch2;
            if index >= p.base_to_invert {
                if index == p.corner_bad {
                    index = p.corner_replacement;
                } else {
                    index = p.inversion_end - index;
                }
            } else if index == p.corner_bad {
                index = p.corner_replacement;
            }
        }
        index
    }

    fn define_index(&mut self, index: i32, offset: i32) {
        let index = self.patch_index(index);
        let o = offset as usize;
        if self.indices.len() <= o {
            self.indices.resize(o + 1, 0);
        }
        self.indices[o] = index as u32;
    }

    /// Takes a clockwise triangle; stores it clockwise or counter-clockwise as the output winding asks.
    fn define_clockwise_triangle(&mut self, i0: i32, i1: i32, i2: i32, base: i32) {
        self.define_index(i0, base);
        if self.clockwise {
            self.define_index(i1, base + 1);
            self.define_index(i2, base + 2);
        } else {
            self.define_index(i2, base + 1);
            self.define_index(i1, base + 2);
        }
    }

    fn num_points_for_factor(&self, f: Fxp) -> i32 {
        if self.odd() {
            ((fxp_ceil(FXP_ONE_HALF.wrapping_add(f.wrapping_add(1) / 2)).wrapping_mul(2)) >> FXP_FRACTION_BITS) as i32
        } else {
            (((fxp_ceil(f.wrapping_add(1) / 2).wrapping_mul(2)) >> FXP_FRACTION_BITS) + 1) as i32
        }
    }

    fn factor_context(&self, f: Fxp) -> FactorContext {
        let mut ctx = FactorContext::default();
        let mut half = f.wrapping_add(1) / 2;
        if self.odd() || half == FXP_ONE_HALF {
            half = half.wrapping_add(FXP_ONE_HALF);
        }
        let floor_half = fxp_floor(half);
        let ceil_half = fxp_ceil(half);
        ctx.half_fraction = half.wrapping_sub(floor_half);
        ctx.num_half_points = (ceil_half >> FXP_FRACTION_BITS) as i32;
        if ceil_half == floor_half {
            ctx.split_point_on_floor_half = ctx.num_half_points + 1;
        } else if self.odd() {
            if floor_half == FXP_ONE {
                ctx.split_point_on_floor_half = 0;
            } else {
                ctx.split_point_on_floor_half = (remove_msb((floor_half >> FXP_FRACTION_BITS) as i32 - 1) << 1) + 1;
            }
        } else {
            ctx.split_point_on_floor_half = (remove_msb((floor_half >> FXP_FRACTION_BITS) as i32) << 1) + 1;
        }
        let mut floor_segments = (floor_half.wrapping_mul(2) >> FXP_FRACTION_BITS) as i32;
        let mut ceil_segments = (ceil_half.wrapping_mul(2) >> FXP_FRACTION_BITS) as i32;
        if self.odd() {
            floor_segments -= 1;
            ceil_segments -= 1;
        }
        ctx.inv_num_segments_on_floor = FIXED_RECIPROCAL[floor_segments as usize];
        ctx.inv_num_segments_on_ceil = FIXED_RECIPROCAL[ceil_segments as usize];
        ctx
    }

    fn place_point_in_1d(&self, ctx: &FactorContext, mut point: i32) -> Fxp {
        let flip = point >= ctx.num_half_points;
        if flip {
            point = (ctx.num_half_points << 1) - point;
            if self.odd() {
                point -= 1;
            }
        }
        if point == ctx.num_half_points {
            return FXP_ONE_HALF;
        }
        let on_ceil = point as u32;
        let mut on_floor = on_ceil;
        if point > ctx.split_point_on_floor_half {
            on_floor = on_floor.wrapping_sub(1);
        }
        let loc_floor = on_floor.wrapping_mul(ctx.inv_num_segments_on_floor);
        let loc_ceil = on_ceil.wrapping_mul(ctx.inv_num_segments_on_ceil);
        let mut location = loc_floor
            .wrapping_mul(FXP_ONE.wrapping_sub(ctx.half_fraction))
            .wrapping_add(loc_ceil.wrapping_mul(ctx.half_fraction));
        location = location.wrapping_add(FXP_ONE_HALF) >> FXP_FRACTION_BITS;
        if flip {
            location = FXP_ONE.wrapping_sub(location);
        }
        location
    }

    #[allow(clippy::too_many_arguments)]
    fn stitch_transition(
        &mut self,
        mut base: i32,
        inside_base: i32,
        mut inside_half_points: i32,
        inside_parity: Parity,
        outside_base: i32,
        mut outside_half_points: i32,
        outside_parity: Parity,
    ) {
        // Where vertex i ends up on the half-edge at the largest factor, in ruler-function split order.
        const FINAL_POINT_POSITION: [i32; 33] = [
            0, 32, 16, 8, 17, 4, 18, 9, 19, 2, 20, 10, 21, 5, 22, 11, 23, 1, 24, 12, 25, 6, 26, 13, 27, 3, 28, 14, 29,
            7, 30, 15, 31,
        ];
        const LOOP_START: [i32; 33] =
            [1, 1, 17, 9, 9, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        const LOOP_END: [i32; 33] = [
            0, 0, 17, 17, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
            31, 31, 31, 31, 31, 32,
        ];
        if inside_parity == Parity::Odd {
            inside_half_points -= 1;
        }
        if outside_parity == Parity::Odd {
            outside_half_points -= 1;
        }
        let mut outside = outside_base;
        let mut inside = inside_base;
        let i_start = LOOP_START[inside_half_points as usize].min(LOOP_START[outside_half_points as usize]);
        let i_end = LOOP_END[inside_half_points as usize].max(LOOP_END[outside_half_points as usize]);
        if FINAL_POINT_POSITION[0] < outside_half_points {
            self.define_clockwise_triangle(outside, outside + 1, inside, base);
            base += 3;
            outside += 1;
        }
        let mut i = i_start;
        while i <= i_end {
            if FINAL_POINT_POSITION[i as usize] < inside_half_points {
                self.define_clockwise_triangle(inside, outside, inside + 1, base);
                base += 3;
                inside += 1;
            }
            if FINAL_POINT_POSITION[i as usize] < outside_half_points {
                self.define_clockwise_triangle(outside, outside + 1, inside, base);
                base += 3;
                outside += 1;
            }
            i += 1;
        }
        if inside_parity != outside_parity || inside_parity == Parity::Odd {
            if inside_parity == outside_parity {
                // Quad in the middle.
                self.define_clockwise_triangle(inside, outside, inside + 1, base);
                base += 3;
                self.define_clockwise_triangle(outside, outside + 1, inside + 1, base);
                base += 3;
                inside += 1;
                outside += 1;
            } else if inside_parity == Parity::Even {
                // Triangle pointing inside.
                self.define_clockwise_triangle(outside, outside + 1, inside, base);
                base += 3;
                outside += 1;
            } else {
                // Triangle pointing outside.
                self.define_clockwise_triangle(inside, outside, inside + 1, base);
                base += 3;
                inside += 1;
            }
        }
        let mut i = i_end;
        while i >= i_start {
            if FINAL_POINT_POSITION[i as usize] < outside_half_points {
                self.define_clockwise_triangle(outside, outside + 1, inside, base);
                base += 3;
                outside += 1;
            }
            if FINAL_POINT_POSITION[i as usize] < inside_half_points {
                self.define_clockwise_triangle(inside, outside, inside + 1, base);
                base += 3;
                inside += 1;
            }
            i -= 1;
        }
        if FINAL_POINT_POSITION[0] < outside_half_points {
            self.define_clockwise_triangle(outside, outside + 1, inside, base);
        }
    }

    // Quad domain -----------------------------------------------------------------------------------------------------

    // The loops index the reference's parallel per-edge arrays.
    #[allow(clippy::needless_range_loop)]
    fn quad_process(&mut self, outside_in: [f32; 4], inside_in: [f32; 2]) -> QuadFactors {
        let mut pf = QuadFactors::default();
        if outside_in.iter().any(|f| f.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater)) {
            pf.culled = true;
            return pf;
        }
        let (mut lower, upper) = match self.original_partitioning {
            Partitioning::Integer => (MIN_ODD_TESSELLATION_FACTOR, MAX_EVEN_TESSELLATION_FACTOR),
            Partitioning::FractionalEven => (MIN_EVEN_TESSELLATION_FACTOR, MAX_EVEN_TESSELLATION_FACTOR),
            Partitioning::FractionalOdd => (MIN_ODD_TESSELLATION_FACTOR, MAX_ODD_TESSELLATION_FACTOR),
        };
        let mut outside = outside_in.map(|f| fmin(upper, fmax(lower, f)));
        if self.integer() {
            outside = outside.map(f32::ceil);
        }
        let mut inside = inside_in;
        if self.original_partitioning == Partitioning::FractionalOdd
            && (outside.iter().any(|f| *f > MIN_ODD_TESSFACTOR_PLUS_HALF_EPSILON)
                || inside.iter().any(|f| *f > MIN_ODD_TESSFACTOR_PLUS_HALF_EPSILON))
        {
            // Force a picture frame.
            lower = MIN_ODD_TESSELLATION_FACTOR + EPSILON;
        }
        inside = inside.map(|f| fmin(upper, fmax(lower, f)));
        if self.integer() {
            inside = inside.map(f32::ceil);
        }
        self.num_points = 0;
        self.num_indices = 0;
        if self.integer() {
            for e in 0..QUAD_EDGES {
                pf.outside_parity[e] = Some(if is_even(outside[e]) { Parity::Even } else { Parity::Odd });
            }
            for a in 0..2 {
                pf.inside_parity[a] =
                    Some(if is_even(inside[a]) || inside[a] == 1.0 { Parity::Even } else { Parity::Odd });
            }
        } else {
            pf.outside_parity = [Some(self.original_parity); 4];
            pf.inside_parity = [Some(self.original_parity); 2];
        }
        for e in 0..QUAD_EDGES {
            pf.outside[e] = float_to_fixed(outside[e]);
        }
        for a in 0..2 {
            pf.inside[a] = float_to_fixed(inside[a]);
        }
        if (self.integer() || self.odd())
            && pf.inside[U] == FXP_ONE
            && pf.inside[V] == FXP_ONE
            && pf.outside.iter().all(|f| *f == FXP_ONE)
        {
            pf.minimum = true;
            return pf;
        }
        pf.minimum = false;
        for e in 0..QUAD_EDGES {
            self.set_parity(par(pf.outside_parity[e]));
            pf.outside_ctx[e] = self.factor_context(pf.outside[e]);
        }
        for a in 0..2 {
            self.set_parity(par(pf.inside_parity[a]));
            pf.inside_ctx[a] = self.factor_context(pf.inside[a]);
        }
        for e in 0..QUAD_EDGES {
            self.set_parity(par(pf.outside_parity[e]));
            pf.num_points_outside[e] = self.num_points_for_factor(pf.outside[e]);
            self.num_points += pf.num_points_outside[e];
        }
        self.num_points -= 4;
        for a in 0..2 {
            self.set_parity(par(pf.inside_parity[a]));
            pf.num_points_inside[a] = self.num_points_for_factor(pf.inside[a]);
            let min = if par(pf.inside_parity[a]) == Parity::Odd { 4 } else { 3 };
            pf.num_points_inside[a] = pf.num_points_inside[a].max(min);
        }
        pf.inside_edge_point_base = self.num_points;
        self.num_points += (pf.num_points_inside[U] - 2) * (pf.num_points_inside[V] - 2);
        pf
    }

    #[allow(clippy::needless_range_loop)]
    fn quad_points(&mut self, pf: &QuadFactors) {
        let mut offset = 0;
        for edge in 0..QUAD_EDGES {
            let parity = edge & 1;
            let end = pf.num_points_outside[edge] - 1;
            for p in 0..end {
                let q = if edge == 1 || edge == 2 { p } else { end - p };
                self.set_parity(par(pf.outside_parity[edge]));
                let param = self.place_point_in_1d(&pf.outside_ctx[edge], q);
                if parity != 0 {
                    self.define_point(param, if edge == 3 { FXP_ONE } else { 0 }, offset);
                } else {
                    self.define_point(if edge == 2 { FXP_ONE } else { 0 }, param, offset);
                }
                offset += 1;
            }
        }
        let min_points = pf.num_points_inside[U].min(pf.num_points_inside[V]);
        let num_rings = min_points >> 1;
        for ring in 1..num_rings {
            let start = ring;
            let end = [pf.num_points_inside[U] - 1 - start, pf.num_points_inside[V] - 1 - start];
            for edge in 0..QUAD_EDGES {
                let parity = [edge & 1, (edge + 1) & 1];
                let perp_point = if edge < 2 { start } else { end[parity[0]] };
                self.set_parity(par(pf.inside_parity[parity[0]]));
                let perp = self.place_point_in_1d(&pf.inside_ctx[parity[0]], perp_point);
                self.set_parity(par(pf.inside_parity[parity[1]]));
                for p in start..end[parity[1]] {
                    let q = if edge == 1 || edge == 2 { p } else { end[parity[1]] - (p - start) };
                    let param = self.place_point_in_1d(&pf.inside_ctx[parity[1]], q);
                    if parity[1] != 0 {
                        self.define_point(perp, param, offset);
                    } else {
                        self.define_point(param, perp, offset);
                    }
                    offset += 1;
                }
            }
        }
        // For even tessellation the inner ring is degenerate: a row of points.
        if pf.num_points_inside[U] > pf.num_points_inside[V] && par(pf.inside_parity[V]) == Parity::Even {
            let start = num_rings;
            let end = pf.num_points_inside[U] - 1 - start;
            self.set_parity(par(pf.inside_parity[U]));
            for p in start..=end {
                let param = self.place_point_in_1d(&pf.inside_ctx[U], p);
                self.define_point(param, FXP_ONE_HALF, offset);
                offset += 1;
            }
        } else if pf.num_points_inside[V] >= pf.num_points_inside[U] && par(pf.inside_parity[U]) == Parity::Even {
            let start = num_rings;
            let end = pf.num_points_inside[V] - 1 - start;
            self.set_parity(par(pf.inside_parity[V]));
            let mut p = end;
            while p >= start {
                let param = self.place_point_in_1d(&pf.inside_ctx[V], p);
                self.define_point(FXP_ONE_HALF, param, offset);
                offset += 1;
                p -= 1;
            }
        }
    }

    /// The ring-1 transition between the outer edges and the outermost ring of the inside grid, edge by edge, as the
    /// reference stitches it (its `QuadGenerateConnectivity` for ring 1), then the inside grid ([`Self::quad_interior`]).
    #[allow(clippy::needless_range_loop)]
    fn quad_connectivity(&mut self, pf: &QuadFactors) {
        let rows_to_center = [(pf.num_points_inside[U] + 1) >> 1, (pf.num_points_inside[V] + 1) >> 1];
        let degenerate_ring = [
            if par(pf.inside_parity[V]) == Parity::Even { rows_to_center[V] - 1 } else { -1 },
            if par(pf.inside_parity[U]) == Parity::Even { rows_to_center[U] - 1 } else { -1 },
        ];
        let num_points_outside = pf.num_points_outside;
        let mut inside_edge_point_base = pf.inside_edge_point_base;
        let mut outside_edge_point_base = 0;
        // Every inside factor gives at least three points per axis, so ring 1 always exists.
        let ring = 1;
        let num_points_inside_edge = [pf.num_points_inside[U] - 2 * ring, pf.num_points_inside[V] - 2 * ring];
        let edge0_inside_base = inside_edge_point_base;
        let edge0_outside_base = outside_edge_point_base;
        for edge in 0..QUAD_EDGES {
            let parity = (edge + 1) & 1;
            let num_triangles = num_points_inside_edge[parity] + num_points_outside[edge] - 2;
            let inside_base;
            let outside_base;
            if edge == 3 {
                if ring == degenerate_ring[parity] {
                    self.patch2.base_to_invert = inside_edge_point_base + 1;
                    self.patch2.corner_bad = outside_edge_point_base + num_points_outside[edge] - 1;
                    self.patch2.corner_replacement = edge0_outside_base;
                    self.patch2.inversion_end = (self.patch2.base_to_invert << 1) - 1;
                    inside_base = self.patch2.base_to_invert;
                    outside_base = outside_edge_point_base;
                    self.patched2 = true;
                } else {
                    self.patch.inside_delta = inside_edge_point_base;
                    self.patch.inside_bad = num_points_inside_edge[parity] - 1;
                    self.patch.inside_replacement = edge0_inside_base;
                    self.patch.outside_patch_base = self.patch.inside_bad + 1;
                    self.patch.outside_delta = outside_edge_point_base - self.patch.outside_patch_base;
                    self.patch.outside_bad = self.patch.outside_patch_base + num_points_outside[edge] - 1;
                    self.patch.outside_replacement = edge0_outside_base;
                    inside_base = 0;
                    outside_base = self.patch.outside_patch_base;
                    self.patched = true;
                }
            } else if edge == 2 && ring == degenerate_ring[parity] {
                self.patch2.base_to_invert = inside_edge_point_base;
                self.patch2.corner_bad = -1;
                self.patch2.corner_replacement = -1;
                self.patch2.inversion_end = self.patch2.base_to_invert << 1;
                inside_base = self.patch2.base_to_invert;
                outside_base = outside_edge_point_base;
                self.patched2 = true;
            } else {
                inside_base = inside_edge_point_base;
                outside_base = outside_edge_point_base;
            }
            self.stitch_transition(
                self.num_indices,
                inside_base,
                pf.inside_ctx[parity].num_half_points,
                par(pf.inside_parity[parity]),
                outside_base,
                pf.outside_ctx[edge].num_half_points,
                par(pf.outside_parity[edge]),
            );
            self.patched = false;
            self.patched2 = false;
            self.num_indices += num_triangles * 3;
            outside_edge_point_base += num_points_outside[edge] - 1;
            if edge == 2 && ring == degenerate_ring[parity] {
                inside_edge_point_base -= num_points_inside_edge[parity] - 1;
            } else {
                inside_edge_point_base += num_points_inside_edge[parity] - 1;
            }
        }
        self.quad_interior(pf);
    }

    /// The inside grid's quads in Metal's order: blade by blade ([`quad_blade`]), each blade's quads with the outer
    /// loop along `a` and the inner along `b`, each quad as its two triangles `(X, F, N)` and `(N, F, Y)`: `N` the
    /// corner nearest the center, `F` the farthest, `X` far along `a` and near along `b`, `Y` near along `a` and far
    /// along `b`.
    fn quad_interior(&mut self, pf: &QuadFactors) {
        let g = QuadGrid::new(pf);
        for k in 0..4 {
            let blade = quad_blade(g.nu, g.nv, k);
            let emit = |t: &mut Self, i: i32, j: i32| {
                let corner = |da: i32, db: i32| {
                    let (x, y) = blade.corner(i, j, da, db);
                    g.index(x, y)
                };
                let (n, x, f, y) = (corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1));
                t.metal_triangle(x, f, n);
                t.metal_triangle(n, f, y);
            };
            if let Some((i, j)) = blade.center {
                emit(self, i, j);
            }
            for o in 0..blade.outer.2 {
                for n in 0..blade.inner.2 {
                    let (i, j) = blade.quad(o, n);
                    emit(self, i, j);
                }
            }
        }
    }

    /// A triangle given in the vertex order Metal emits for `MTLWindingClockwise`.
    fn metal_triangle(&mut self, x: i32, y: i32, z: i32) {
        let o = self.num_indices;
        if self.clockwise {
            // The reference's clockwise output, which is Metal's MTLWindingCounterClockwise.
            self.define_index(x, o);
            self.define_index(z, o + 1);
            self.define_index(y, o + 2);
        } else {
            self.define_index(x, o);
            self.define_index(y, o + 1);
            self.define_index(z, o + 2);
        }
        self.num_indices += 3;
    }

    fn quad(&mut self, outside: [f32; 4], inside: [f32; 2]) {
        let pf = self.quad_process(outside, inside);
        if pf.culled {
            self.num_points = 0;
            self.num_indices = 0;
            return;
        }
        if pf.minimum {
            self.define_point(0, 0, 0);
            self.define_point(FXP_ONE, 0, 1);
            self.define_point(FXP_ONE, FXP_ONE, 2);
            self.define_point(0, FXP_ONE, 3);
            self.num_points = 4;
            self.define_clockwise_triangle(0, 1, 3, 0);
            self.define_clockwise_triangle(3, 1, 2, 3);
            self.num_indices = 6;
            return;
        }
        self.quad_points(&pf);
        self.quad_connectivity(&pf);
    }

    // Triangle domain -------------------------------------------------------------------------------------------------

    #[allow(clippy::needless_range_loop)]
    fn tri_process(&mut self, outside_in: [f32; 3], inside_in: f32) -> TriFactors {
        let mut pf = TriFactors::default();
        if outside_in.iter().any(|f| f.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater)) {
            pf.culled = true;
            return pf;
        }
        let (mut lower, upper) = match self.original_partitioning {
            Partitioning::Integer => (MIN_ODD_TESSELLATION_FACTOR, MAX_EVEN_TESSELLATION_FACTOR),
            Partitioning::FractionalEven => (MIN_EVEN_TESSELLATION_FACTOR, MAX_EVEN_TESSELLATION_FACTOR),
            Partitioning::FractionalOdd => (MIN_ODD_TESSELLATION_FACTOR, MAX_ODD_TESSELLATION_FACTOR),
        };
        let mut outside = outside_in.map(|f| fmin(upper, fmax(lower, f)));
        if self.integer() {
            outside = outside.map(f32::ceil);
        }
        if self.original_partitioning == Partitioning::FractionalOdd
            && outside.iter().any(|f| *f > MIN_ODD_TESSFACTOR_PLUS_HALF_EPSILON)
        {
            // Force a picture frame. A triangle has one inside factor, so it is not consulted, unlike a quad's two.
            lower = MIN_ODD_TESSELLATION_FACTOR + EPSILON;
        }
        let mut inside = fmin(upper, fmax(lower, inside_in));
        if self.integer() {
            inside = inside.ceil();
        }
        self.num_points = 0;
        self.num_indices = 0;
        if self.integer() {
            for e in 0..TRI_EDGES {
                pf.outside_parity[e] = Some(if is_even(outside[e]) { Parity::Even } else { Parity::Odd });
            }
            pf.inside_parity = Some(if is_even(inside) || inside == 1.0 { Parity::Even } else { Parity::Odd });
        } else {
            pf.outside_parity = [Some(self.original_parity); 3];
            pf.inside_parity = Some(self.original_parity);
        }
        for e in 0..TRI_EDGES {
            pf.outside[e] = float_to_fixed(outside[e]);
        }
        pf.inside = float_to_fixed(inside);
        if (self.integer() || self.odd()) && pf.inside == FXP_ONE && pf.outside.iter().all(|f| *f == FXP_ONE) {
            pf.minimum = true;
            return pf;
        }
        pf.minimum = false;
        for e in 0..TRI_EDGES {
            self.set_parity(par(pf.outside_parity[e]));
            pf.outside_ctx[e] = self.factor_context(pf.outside[e]);
        }
        self.set_parity(par(pf.inside_parity));
        pf.inside_ctx = self.factor_context(pf.inside);
        for e in 0..TRI_EDGES {
            self.set_parity(par(pf.outside_parity[e]));
            pf.num_points_outside[e] = self.num_points_for_factor(pf.outside[e]);
            self.num_points += pf.num_points_outside[e];
        }
        self.num_points -= 3;
        self.set_parity(par(pf.inside_parity));
        pf.num_points_inside = self.num_points_for_factor(pf.inside);
        let min = if self.odd() { 4 } else { 3 };
        pf.num_points_inside = pf.num_points_inside.max(min);
        pf.inside_edge_point_base = self.num_points;
        let rings = (pf.num_points_inside >> 1) - 1;
        let interior = if self.odd() { 3 * (rings * (rings + 1) - rings) } else { 3 * (rings * (rings + 1)) + 1 };
        self.num_points += interior;
        pf
    }

    #[allow(clippy::needless_range_loop)]
    fn tri_points(&mut self, pf: &TriFactors) {
        let mut offset = 0;
        for edge in 0..TRI_EDGES {
            let parity = edge & 1;
            let end = pf.num_points_outside[edge] - 1;
            for p in 0..end {
                let q = if parity != 0 { p } else { end - p };
                self.set_parity(par(pf.outside_parity[edge]));
                let param = self.place_point_in_1d(&pf.outside_ctx[edge], q);
                if edge == 0 {
                    self.define_point(0, param, offset);
                } else {
                    self.define_point(param, if edge == 2 { FXP_ONE.wrapping_sub(param) } else { 0 }, offset);
                }
                offset += 1;
            }
        }
        self.set_parity(par(pf.inside_parity));
        let num_rings = pf.num_points_inside >> 1;
        for ring in 1..num_rings {
            let start = ring;
            let end = pf.num_points_inside - 1 - start;
            for edge in 0..TRI_EDGES {
                let parity = edge & 1;
                let mut perp = self.place_point_in_1d(&pf.inside_ctx, start);
                // Map the location to its size in barycentric space.
                perp = perp.wrapping_mul(FXP_TWO_THIRDS);
                perp = perp.wrapping_add(FXP_ONE_HALF) >> FXP_FRACTION_BITS;
                for p in start..end {
                    let q = if parity != 0 { p } else { end - (p - start) };
                    let param = self.place_point_in_1d(&pf.inside_ctx, q);
                    // The rate of change of edge-parallel parameters as they are pushed into the triangle.
                    const DERIV: u32 = 2;
                    let shifted = param.wrapping_sub(perp.wrapping_add(1) / DERIV);
                    match edge {
                        0 => self.define_point(perp, shifted, offset),
                        1 => self.define_point(shifted, perp, offset),
                        _ => self.define_point(shifted, FXP_ONE.wrapping_sub(shifted).wrapping_sub(perp), offset),
                    }
                    offset += 1;
                }
            }
        }
        if !self.odd() {
            // The last point is the center.
            self.define_point(FXP_ONE_THIRD, FXP_ONE_THIRD, offset);
        }
    }

    /// The ring-1 transition between the outer edges and the outermost inside ring, edge by edge, as the reference
    /// stitches it (its `TriGenerateConnectivity` for ring 1), then the inside ([`Self::tri_interior`]).
    #[allow(clippy::needless_range_loop)]
    fn tri_connectivity(&mut self, pf: &TriFactors) {
        let num_points_outside = pf.num_points_outside;
        let mut inside_edge_point_base = pf.inside_edge_point_base;
        let mut outside_edge_point_base = 0;
        let num_points_inside_edge = pf.num_points_inside - 2;
        let edge0_inside_base = inside_edge_point_base;
        let edge0_outside_base = outside_edge_point_base;
        for edge in 0..TRI_EDGES {
            let num_triangles = num_points_inside_edge + num_points_outside[edge] - 2;
            let (inside_base, outside_base) = if edge == 2 {
                self.patch.inside_delta = inside_edge_point_base;
                self.patch.inside_bad = num_points_inside_edge - 1;
                self.patch.inside_replacement = edge0_inside_base;
                self.patch.outside_patch_base = self.patch.inside_bad + 1;
                self.patch.outside_delta = outside_edge_point_base - self.patch.outside_patch_base;
                self.patch.outside_bad = self.patch.outside_patch_base + num_points_outside[edge] - 1;
                self.patch.outside_replacement = edge0_outside_base;
                self.patched = true;
                (0, self.patch.outside_patch_base)
            } else {
                (inside_edge_point_base, outside_edge_point_base)
            };
            self.stitch_transition(
                self.num_indices,
                inside_base,
                pf.inside_ctx.num_half_points,
                par(pf.inside_parity),
                outside_base,
                pf.outside_ctx[edge].num_half_points,
                par(pf.outside_parity[edge]),
            );
            self.patched = false;
            self.num_indices += num_triangles * 3;
            outside_edge_point_base += num_points_outside[edge] - 1;
            inside_edge_point_base += num_points_inside_edge - 1;
        }
        self.tri_interior(pf);
    }

    /// The inside of a triangle patch in Metal's order: with an odd inside factor first the center triangle, then
    /// for each corner `k` (the start of edge `k`: V, W, U) its blade ([`TriGrid`]), quads as in
    /// [`Self::quad_interior`].
    fn tri_interior(&mut self, pf: &TriFactors) {
        let g = TriGrid::new(pf);
        if g.odd {
            let c = g.ring_base(g.rings - 1);
            if g.a_max == 0 {
                self.metal_triangle(c + 1, c, c + 2);
            } else {
                self.metal_triangle(c + 2, c + 1, c);
            }
        }
        for k in 0..3 {
            for a in g.a_min..=g.a_max {
                for b in 0..=g.a_max {
                    let (n, x, f, y) =
                        (g.point(k, a, b), g.point(k, a + 1, b), g.point(k, a + 1, b + 1), g.point(k, a, b + 1));
                    self.metal_triangle(x, f, n);
                    self.metal_triangle(n, f, y);
                }
            }
        }
    }

    fn tri(&mut self, outside: [f32; 3], inside: f32) {
        let pf = self.tri_process(outside, inside);
        if pf.culled {
            self.num_points = 0;
            self.num_indices = 0;
            return;
        }
        if pf.minimum {
            self.define_point(0, FXP_ONE, 0);
            self.define_point(0, 0, 1);
            self.define_point(FXP_ONE, 0, 2);
            self.num_points = 3;
            self.define_clockwise_triangle(1, 2, 0, self.num_indices);
            self.num_indices = 3;
            return;
        }
        self.tri_points(&pf);
        self.tri_connectivity(&pf);
    }

    fn finish(mut self) -> Patch {
        self.points.truncate(self.num_points as usize);
        self.indices.truncate(self.num_indices as usize);
        debug_assert_eq!(self.points.len(), self.num_points as usize);
        debug_assert_eq!(self.indices.len(), self.num_indices as usize);
        Patch { points: self.points, indices: self.indices }
    }
}

/// The inside grid of a quad patch: points `(i, j)` for `i` in `1..nu - 1` and `j` in `1..nv - 1` (`nu`, `nv` the
/// numbers of points of the inside factors, the outer edges at 0 and `n - 1`), stored as the reference stores them:
/// ring after ring from the outermost (`QuadGeneratePoints`), then the middle row or column an even inside factor
/// leaves.
struct QuadGrid {
    nu: i32,
    nv: i32,
    base: i32,
    /// The rings with points, 1..rings.
    rings: i32,
    /// The middle row (along u, at the middle v) when the u axis is the longer, else the middle column.
    row: bool,
}

impl QuadGrid {
    fn new(pf: &QuadFactors) -> QuadGrid {
        let (nu, nv) = (pf.num_points_inside[U], pf.num_points_inside[V]);
        QuadGrid {
            nu,
            nv,
            base: pf.inside_edge_point_base,
            rings: nu.min(nv) >> 1,
            row: nu > nv && par(pf.inside_parity[V]) == Parity::Even,
        }
    }

    /// The first point of ring `r`: each ring before it has `2 * ((nu - 1 - 2r) + (nv - 1 - 2r))` points.
    fn ring_base(&self, r: i32) -> i32 {
        self.base + 2 * (r - 1) * (self.nu + self.nv - 2) - 4 * r * (r - 1)
    }

    /// The point index of grid point `(i, j)`.
    fn index(&self, i: i32, j: i32) -> i32 {
        let r = i.min(j).min(self.nu - 1 - i).min(self.nv - 1 - j);
        if r < self.rings {
            // Ring r runs down edge u = r from (r, end_v), along v = r, up edge u = end_u, back along v = end_v.
            let (end_u, end_v) = (self.nu - 1 - r, self.nv - 1 - r);
            let (lu, lv) = (end_u - r, end_v - r);
            let b = self.ring_base(r);
            if i == r && j > r {
                b + (end_v - j)
            } else if j == r && i < end_u {
                b + lv + (i - r)
            } else if i == end_u && j < end_v {
                b + lv + lu + (j - r)
            } else {
                b + 2 * lv + lu + (end_u - i)
            }
        } else {
            let b = self.ring_base(self.rings);
            if self.row { b + (i - self.rings) } else { b + (self.nv - 1 - self.rings - j) }
        }
    }
}

/// One blade of a quad patch's inside: the quads `[i, i + 1] x [j, j + 1]` (`i` in `1..nu - 2`, `j` in `1..nv - 2`)
/// whose offset `(du, dv) = (2i + 2 - nu, 2j + 2 - nv)` from the center lies in the blade's half-open quadrant: 0 (the
/// corner u = 0, v = 1) `du < 0, dv >= 0`; 1 (u = 0, v = 0) `du <= 0, dv < 0`; 2 (u = 1, v = 0) `du > 0, dv <= 0`; 3
/// (u = 1, v = 1) `du >= 0, dv > 0`; and the quad on both center lines, which only odd inside factors on both axes
/// leave, first in blade 0. Blade `k` lies at the start of edge `k`; its `a` axis points toward edge `k` and its `b`
/// axis toward edge `k - 1`, the outer loop running along `a` away from the center and the inner along `b`.
struct QuadBlade {
    k: i32,
    /// (first, step, count) of the quad index along `a` (u for blades 0 and 2, v for 1 and 3).
    outer: (i32, i32, i32),
    /// (first, step, count) along `b`.
    inner: (i32, i32, i32),
    center: Option<(i32, i32)>,
}

impl QuadBlade {
    /// The quad of the `o`-th outer and `n`-th inner step.
    fn quad(&self, o: i32, n: i32) -> (i32, i32) {
        let a = self.outer.0 + o * self.outer.1;
        let b = self.inner.0 + n * self.inner.1;
        if self.k % 2 == 0 { (a, b) } else { (b, a) }
    }

    /// The grid point of quad `(i, j)` at `da` steps along `a` and `db` along `b` from its corner nearest the center.
    fn corner(&self, i: i32, j: i32, da: i32, db: i32) -> (i32, i32) {
        match self.k {
            0 => (i + 1 - da, j + db),
            1 => (i + 1 - db, j + 1 - da),
            2 => (i + da, j + 1 - db),
            _ => (i + db, j + da),
        }
    }
}

fn quad_blade(nu: i32, nv: i32, k: i32) -> QuadBlade {
    // Index ranges of the quads by the sign of their offset: below the center line, on it, above it.
    let below = |n: i32| (1, (n + 1) / 2 - 2);
    let on = |n: i32| if n % 2 == 0 { Some(n / 2 - 1) } else { None };
    let above = |n: i32| (n / 2, n - 3);
    let down = |(lo, hi): (i32, i32)| (hi, -1, (hi - lo + 1).max(0));
    let up = |(lo, hi): (i32, i32)| (lo, 1, (hi - lo + 1).max(0));
    // Ranges including the center line: from it outward.
    let down_on = |n: i32| {
        let (lo, hi) = below(n);
        let hi = on(n).unwrap_or(hi);
        (hi, -1, (hi - lo + 1).max(0))
    };
    let up_on = |n: i32| {
        let (lo, hi) = above(n);
        let lo = on(n).unwrap_or(lo);
        (lo, 1, (hi - lo + 1).max(0))
    };
    match k {
        0 => QuadBlade { k, outer: down(below(nu)), inner: up_on(nv), center: on(nu).zip(on(nv)) },
        1 => QuadBlade { k, outer: down(below(nv)), inner: down_on(nu), center: None },
        2 => QuadBlade { k, outer: up(above(nu)), inner: down_on(nv), center: None },
        _ => QuadBlade { k, outer: up(above(nv)), inner: up_on(nu), center: None },
    }
}

/// The inside of a triangle patch, as blades: for corner `k` (the start of edge `k`) the points `(a, b)` at depth
/// `d = max(a, b)` from the center lie on the inside ring with `2d - odd` points per edge, `(d, b)` on edge `k` at `d
/// - b` points from its start and `(a, d)` (`a < d`) on edge `k - 1` at `d - odd + a`. An even inside factor has a
/// center point at depth 0 and quads `a, b` in `0..=a_max`; an odd one a center triangle at depth 1 and quads `a` in
/// `1..=a_max`, `b` in `0..=a_max`.
struct TriGrid {
    n: i32,
    base: i32,
    /// The rings with points, 1..rings (TriGeneratePoints), the even center point after them.
    rings: i32,
    odd: bool,
    a_min: i32,
    a_max: i32,
}

impl TriGrid {
    fn new(pf: &TriFactors) -> TriGrid {
        let n = pf.num_points_inside;
        let odd = par(pf.inside_parity) == Parity::Odd;
        let rings = n >> 1;
        TriGrid { n, base: pf.inside_edge_point_base, rings, odd, a_min: odd as i32, a_max: rings - 2 }
    }

    /// The first point of ring `r`: each ring before it has `3 * (n - 1 - 2r)` points.
    fn ring_base(&self, r: i32) -> i32 {
        self.base + 3 * (r - 1) * (self.n - 1) - 3 * r * (r - 1)
    }

    fn point(&self, k: i32, a: i32, b: i32) -> i32 {
        let d = a.max(b);
        if d == 0 {
            return self.ring_base(self.rings);
        }
        let o = self.odd as i32;
        let l = 2 * d - o;
        let r = (self.n - 1 - l) / 2;
        let (mut e, mut p) = if a == d { (k, d - b) } else { ((k + 2) % 3, d - o + a) };
        if p == l {
            e = (e + 1) % 3;
            p = 0;
        }
        self.ring_base(r) + e * l + p
    }
}

/// Tessellate a quad patch with edge factors for the edges u = 0, v = 0, u = 1 and v = 1 and inside factors for u and
/// v. `metal_clockwise` is `tessellationOutputWindingOrder == MTLWindingClockwise`, the reference's counter-clockwise
/// output.
pub fn quad(partitioning: Partitioning, metal_clockwise: bool, outside: [f32; 4], inside: [f32; 2]) -> Patch {
    let mut t = Tessellator::new(partitioning, !metal_clockwise);
    t.quad(outside, inside);
    t.finish()
}

/// Tessellate a triangle patch with edge factors for the edges u = 0, v = 0 and w = 0 and its inside factor.
/// `metal_clockwise` as for [`quad`].
pub fn triangle(partitioning: Partitioning, metal_clockwise: bool, outside: [f32; 3], inside: f32) -> Patch {
    let mut t = Tessellator::new(partitioning, !metal_clockwise);
    t.tri(outside, inside);
    t.finish()
}

/// The sizes of a generated patch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub points: u32,
    pub indices: u32,
    /// Whether a blade of the inside spans more than 36 grid points (`(A + 1) * (B + 1)` for `A` by `B` quads, the
    /// bounding box of blade 0 including an odd center). The M4 emits such a blade in tiles whose sizes depend on what
    /// the draw emitted before (measured), which [`quad`] and [`triangle`] do not reproduce; every blade of at most 36
    /// points is emitted whole, in the order they produce.
    pub batched: bool,
}

/// The largest inside blade the M4 emits in one piece whatever came before it: 36 grid points (measured; its tiles
/// span 36, 64 or 100 points).
pub const WHOLE_BLADE_POINTS: i32 = 36;

/// The sizes [`quad`] generates, without generating the patch.
pub fn quad_counts(partitioning: Partitioning, outside: [f32; 4], inside: [f32; 2]) -> Counts {
    let mut t = Tessellator::new(partitioning, true);
    let pf = t.quad_process(outside, inside);
    if pf.culled {
        return Counts::default();
    }
    if pf.minimum {
        return Counts { points: 4, indices: 6, batched: false };
    }
    let (nu, nv) = (pf.num_points_inside[U], pf.num_points_inside[V]);
    let inside_edge = [nu - 2, nv - 2];
    let mut triangles = 0;
    for edge in 0..QUAD_EDGES {
        triangles += inside_edge[(edge + 1) & 1] + pf.num_points_outside[edge] - 2;
    }
    triangles += 2 * (nu - 3).max(0) * (nv - 3).max(0);
    let batched = (0..4).any(|k| {
        let b = quad_blade(nu, nv, k);
        let a = b.outer.2 + b.center.is_some() as i32;
        a > 0 && b.inner.2 > 0 && (a + 1) * (b.inner.2 + 1) > WHOLE_BLADE_POINTS
    });
    Counts { points: t.num_points as u32, indices: 3 * triangles as u32, batched }
}

/// The sizes [`triangle`] generates, without generating the patch.
pub fn triangle_counts(partitioning: Partitioning, outside: [f32; 3], inside: f32) -> Counts {
    let mut t = Tessellator::new(partitioning, true);
    let pf = t.tri_process(outside, inside);
    if pf.culled {
        return Counts::default();
    }
    if pf.minimum {
        return Counts { points: 3, indices: 3, batched: false };
    }
    let n = pf.num_points_inside;
    let mut triangles = 0;
    for edge in 0..TRI_EDGES {
        triangles += n - 2 + pf.num_points_outside[edge] - 2;
    }
    let g = TriGrid::new(&pf);
    let span = g.a_max + 1;
    let quads = if g.odd { (span - 1) * span } else { span * span };
    triangles += 6 * quads + g.odd as i32;
    // Blade 0's bounding box includes the odd center.
    let batched = (span + 1) * (span + 1) > WHOLE_BLADE_POINTS;
    Counts { points: t.num_points as u32, indices: 3 * triangles as u32, batched }
}
