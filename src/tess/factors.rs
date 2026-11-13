//! Metal's tessellation factor processing: from the half-precision factors of the factor buffer to the factors the
//! tessellator partitions edges with.

use super::hw::Partitioning;

/// `MTLTessellationPartitionMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionMode {
    Pow2,
    Integer,
    FractionalOdd,
    FractionalEven,
}

impl PartitionMode {
    pub fn from_metal(v: u64) -> Option<PartitionMode> {
        Some(match v {
            0 => PartitionMode::Pow2,
            1 => PartitionMode::Integer,
            2 => PartitionMode::FractionalOdd,
            3 => PartitionMode::FractionalEven,
            _ => return None,
        })
    }
}

/// `MTLPatchType` of a post-tessellation vertex function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchType {
    Triangle,
    Quad,
}

impl PatchType {
    /// Edge factors, inside factors.
    pub fn factor_counts(self) -> (usize, usize) {
        match self {
            PatchType::Triangle => (3, 1),
            PatchType::Quad => (4, 2),
        }
    }

    /// The size of `MTLTriangleTessellationFactorsHalf` or `MTLQuadTessellationFactorsHalf`.
    pub fn factors_size(self) -> u64 {
        let (e, i) = self.factor_counts();
        2 * (e + i) as u64
    }
}

/// The pipeline's tessellation state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub partition: PartitionMode,
    /// `maxTessellationFactor`.
    pub max_factor: u32,
    /// `tessellationFactorScaleEnabled`.
    pub scale_enabled: bool,
    /// `tessellationOutputWindingOrder` is `MTLWindingClockwise`.
    pub clockwise: bool,
}

impl Settings {
    /// The range Metal clamps the factors to: `[1, max]` for pow2 and integer partitioning, `[1, max - 1]` for
    /// fractional odd, `[2, max]` for fractional even (`MTLTessellationPartitionMode` documentation).
    pub fn clamp(&self) -> (f32, f32) {
        let max = self.max_factor as f32;
        match self.partition {
            PartitionMode::Pow2 | PartitionMode::Integer => (1.0, max),
            PartitionMode::FractionalOdd => (1.0, max - 1.0),
            PartitionMode::FractionalEven => (2.0, max),
        }
    }
}

/// The factors the tessellator partitions with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Final {
    pub partitioning: Partitioning,
    /// Edge factors; a triangle uses the first three.
    pub edges: [f32; 4],
    /// Inside factors; a triangle uses the first.
    pub inside: [f32; 2],
}

/// Half to single precision, exactly.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Single to half precision, rounding to nearest even; NaN stays NaN, magnitudes past the largest finite half become
/// infinities.
pub fn f32_to_half(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    // The significand with its hidden bit, and the number of low bits to drop to reach binary16's precision.
    let (sig, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (man | 0x80_0000, (14 - e) as u32)
    } else {
        (man | 0x80_0000, 13)
    };
    let kept = sig >> shift;
    let rest = sig & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let mut h = if e <= 0 { kept } else { ((e as u32) << 10) | (kept & 0x3ff) };
    if rest > half || (rest == half && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The factors of one patch, or `None` when the patch is discarded.
pub fn process(settings: &Settings, patch: PatchType, halves: &[u16], scale: f32) -> Option<Final> {
    let (ne, ni) = patch.factor_counts();
    let mut edges = [0.0f32; 4];
    let mut inside = [0.0f32; 2];
    for i in 0..ne {
        edges[i] = half_to_f32(halves[i]);
    }
    for i in 0..ni {
        inside[i] = half_to_f32(halves[ne + i]);
    }
    if settings.scale_enabled {
        let scale = half_to_f32(f32_to_half(scale));
        for f in edges[..ne].iter_mut().chain(inside[..ni].iter_mut()) {
            *f *= scale;
        }
    }
    // "If the value of an edge tessellation factor is either negative, zero, or corresponds to a floating-point NaN,
    // the tessellator discards the patch." Real Metal tests the scaled factors: a scale of zero, a negative or NaN
    // scale, or one that rounds to a binary16 zero discards every patch (measured).
    if edges[..ne].iter().any(|f| f.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater)) {
        return None;
    }
    let (lower, upper) = settings.clamp();
    let clamp = |f: f32| if f.is_nan() { lower } else { f.clamp(lower, upper) };
    for f in edges[..ne].iter_mut().chain(inside[..ni].iter_mut()) {
        *f = clamp(*f);
        if settings.partition == PartitionMode::Pow2 {
            // The next power of two at or above a value of at least 1: drop the significand, carrying into the
            // exponent when it had bits.
            let b = f.to_bits();
            if b & 0x7f_ffff != 0 {
                *f = f32::from_bits((b & 0x7f80_0000) + 0x80_0000);
            }
        }
    }
    let partitioning = match settings.partition {
        PartitionMode::Pow2 | PartitionMode::Integer => Partitioning::Integer,
        PartitionMode::FractionalOdd => Partitioning::FractionalOdd,
        PartitionMode::FractionalEven => Partitioning::FractionalEven,
    };
    Some(Final { partitioning, edges, inside })
}
