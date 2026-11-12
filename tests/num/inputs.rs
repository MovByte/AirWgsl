//! Test inputs: a deterministic generator biased towards the cases floating-point code gets wrong (special values,
//! subnormals, exponent extremes, operands that cancel or produce exact rounding ties).

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    /// splitmix64.
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }

    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next() >> 32) * n as u64 >> 32) as u32
    }
}

/// binary32 values that are boundaries of some rule.
pub const SPECIAL_F32: &[u32] = &[
    0x0000_0000, 0x8000_0000, // zeros
    0x0000_0001, 0x8000_0001, 0x0000_0002, 0x007f_ffff, 0x807f_ffff, 0x0040_0000, // subnormals
    0x0080_0000, 0x8080_0000, 0x0080_0001, 0x00ff_ffff, // smallest normals
    0x3f80_0000, 0xbf80_0000, 0x3f7f_ffff, 0x3f80_0001, 0x3f00_0000, 0xbf00_0000, 0x3fc0_0000, 0x4000_0000,
    0x4b00_0000, 0x4b7f_ffff, 0x4b80_0000, 0x4f00_0000, 0xcf00_0000, 0x4f80_0000, 0x5f00_0000, 0xdf00_0000,
    0x7f7f_ffff, 0xff7f_ffff, 0x7f7f_fffe, 0x7f00_0000, // largest finite
    0x7f80_0000, 0xff80_0000, // infinities
    0x7fc0_0000, 0xffc0_0000, 0x7fc0_1234, 0x7f80_0001, 0xff80_0001, 0x7fbf_ffff, 0x7fff_ffff, // NaNs
];

fn mantissa_pattern(rng: &mut Rng) -> u32 {
    match rng.below(8) {
        0 => 0,
        1 => 0x7f_ffff,
        2 => 1 << rng.below(23),
        3 => rng.u32() & 0x7f_ffff & !((1 << rng.below(23)) - 1),
        4 => rng.u32() & ((1 << rng.below(24)) - 1),
        // A few set bits: products and sums of such values are often exact or exact ties.
        5 => (1 << rng.below(23)) | (1 << rng.below(23)) | (1 << rng.below(23)),
        6 => rng.u32() & 0x7f_f000,
        _ => rng.u32() & 0x7f_ffff,
    }
}

/// One binary32 operand.
pub fn f32_any(rng: &mut Rng) -> u32 {
    match rng.below(10) {
        0 => SPECIAL_F32[rng.below(SPECIAL_F32.len() as u32) as usize],
        1..=3 => rng.u32(),
        _ => {
            let s = rng.below(2) << 31;
            let e = match rng.below(6) {
                0 => rng.below(4),
                1 => 254 - rng.below(4),
                2 => 127 + rng.below(9) - 4,
                _ => rng.below(255),
            };
            s | (e << 23) | mantissa_pattern(rng)
        }
    }
}

/// An operand related to `a`: close in exponent (cancellation, alignment by 0..26 places, ties) or arbitrary.
pub fn f32_near(rng: &mut Rng, a: u32) -> u32 {
    match rng.below(4) {
        0 => f32_any(rng),
        _ => {
            let ea = ((a >> 23) & 0xff) as i32;
            let shift = match rng.below(4) {
                0 => rng.below(3) as i32,
                1 => 23 + rng.below(4) as i32,
                _ => rng.below(30) as i32,
            };
            let e = (ea + if rng.below(2) == 0 { shift } else { -shift }).clamp(0, 254) as u32;
            let m = match rng.below(3) {
                0 => (a & 0x7f_ffff).wrapping_add(rng.below(5)).wrapping_sub(2) & 0x7f_ffff,
                _ => mantissa_pattern(rng),
            };
            (rng.below(2) << 31) | (e << 23) | m
        }
    }
}

/// binary16 values that are boundaries of some rule.
pub const SPECIAL_F16: &[u16] = &[
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3bff, 0x3c01, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
    0x7e00, 0xfe00, 0x7c01, 0x7dff, 0x7fff,
];
