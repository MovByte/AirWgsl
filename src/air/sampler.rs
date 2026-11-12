//! Constant samplers: the 64-bit `air.sampler_state` word of an MSL `constexpr sampler`, decoded into the
//! `MTLSamplerDescriptor` the target builds from it.
//!
//! The layout and the code tables are the target's: `setDescriptorFromBitmask` in iPadOS 17.7.11 (21H461) Metal,
//! called by `_MTLCreateConstantSamplerDescriptorsFromBitmasks`, executed over every bit and every code of every field
//! and cross-checked against constexpr samplers compiled by Apple's compiler (the `[sampler_state]` table of WebMetal's
//! device profile, `profile/ipados-17.7.11-21H461-a10x.toml`). A word the target refuses (bit 63 clear, a reserved
//! bit set, or a code the target fails on) is refused here with the same reason.

use crate::api::{Refusal, RefusalKind};

/// The descriptor fields a constant sampler sets, as `MTLSamplerDescriptor` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDescriptor {
    /// `MTLSamplerAddressMode` for s, t and r.
    pub address_modes: [u64; 3],
    /// `MTLSamplerMinMagFilter`. Codes 2 and 3 decode to values with no SDK enumerator; the target stores them as
    /// they are.
    pub mag_filter: u64,
    pub min_filter: u64,
    /// `MTLSamplerMipFilter`; code 3 has no SDK enumerator.
    pub mip_filter: u64,
    pub normalized_coordinates: bool,
    /// `MTLCompareFunction`.
    pub compare_function: u64,
    pub max_anisotropy: u64,
    /// IEEE binary16 bits of `lodMinClamp`.
    pub lod_min_clamp: u16,
    /// IEEE binary16 bits of `lodMaxClamp`.
    pub lod_max_clamp: u16,
    /// `MTLSamplerBorderColor`; code 3 has no SDK enumerator.
    pub border_color: u64,
}

/// `MTLSamplerAddressMode` by code (bits 0..2, 3..5, 6..8); codes 5 to 7 fail.
const ADDRESS: [u64; 5] = [4, 0, 2, 3, 5];
/// `MTLCompareFunction` by code (bits 16..19); codes 9 to 15 fail.
const COMPARE: [u64; 9] = [0, 1, 3, 4, 6, 2, 5, 7, 0];

fn field(word: u64, offset: u32, width: u32) -> u64 {
    (word >> offset) & ((1u64 << width) - 1)
}

fn refuse(word: u64, reason: String) -> Refusal {
    Refusal {
        kind: RefusalKind::RejectedByDevice,
        construct: "air.sampler_state".into(),
        location: None,
        reason: format!("sampler state {:#018x}: {}", word, reason),
        reference: Some(
            "setDescriptorFromBitmask in iPadOS 17.7.11 Metal (WebMetal device profile, [sampler_state])".into(),
        ),
    }
}

/// Decodes a sampler-state word as the target does.
pub fn decode(word: u64) -> Result<SamplerDescriptor, Refusal> {
    if word >> 63 == 0 {
        return Err(refuse(word, "bit 63 is clear; the target requires it".into()));
    }
    if field(word, 58, 5) != 0 {
        return Err(refuse(word, "a reserved bit (58 to 62) is set".into()));
    }
    let mut address_modes = [0; 3];
    for (i, mode) in address_modes.iter_mut().enumerate() {
        let code = field(word, 3 * i as u32, 3) as usize;
        *mode = *ADDRESS.get(code).ok_or_else(|| refuse(word, format!("address mode code {} fails", code)))?;
    }
    let compare = field(word, 16, 4) as usize;
    let compare_function =
        *COMPARE.get(compare).ok_or_else(|| refuse(word, format!("compare function code {} fails", compare)))?;
    Ok(SamplerDescriptor {
        address_modes,
        mag_filter: field(word, 9, 2),
        min_filter: field(word, 11, 2),
        mip_filter: field(word, 13, 2),
        normalized_coordinates: field(word, 15, 1) == 0,
        compare_function,
        max_anisotropy: field(word, 20, 4) + 1,
        lod_min_clamp: field(word, 24, 16) as u16,
        lod_max_clamp: field(word, 40, 16) as u16,
        border_color: field(word, 56, 2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The profile's table, field by field.
    #[test]
    fn decodes_each_field_as_the_profile_states() {
        let base = 1u64 << 63;
        let d = decode(base).unwrap();
        assert_eq!(d.address_modes, [4, 4, 4]);
        assert_eq!((d.mag_filter, d.min_filter, d.mip_filter), (0, 0, 0));
        assert!(d.normalized_coordinates);
        assert_eq!(d.compare_function, 0);
        assert_eq!(d.max_anisotropy, 1);
        assert_eq!(d.border_color, 0);
        for (code, mode) in [(1u64, 0u64), (2, 2), (3, 3), (4, 5)] {
            let d = decode(base | code | code << 3 | code << 6).unwrap();
            assert_eq!(d.address_modes, [mode; 3]);
        }
        for code in 5..8u64 {
            assert!(decode(base | code).is_err());
            assert!(decode(base | code << 3).is_err());
            assert!(decode(base | code << 6).is_err());
        }
        assert_eq!(decode(base | 1 << 9).unwrap().mag_filter, 1);
        assert_eq!(decode(base | 1 << 11).unwrap().min_filter, 1);
        assert_eq!(decode(base | 2 << 13).unwrap().mip_filter, 2);
        assert!(!decode(base | 1 << 15).unwrap().normalized_coordinates);
        for (code, f) in [(1u64, 1u64), (2, 3), (3, 4), (4, 6), (5, 2), (6, 5), (7, 7), (8, 0)] {
            assert_eq!(decode(base | code << 16).unwrap().compare_function, f);
        }
        for code in 9..16u64 {
            assert!(decode(base | code << 16).is_err());
        }
        assert_eq!(decode(base | 15 << 20).unwrap().max_anisotropy, 16);
        assert_eq!(decode(base | 0x3c00 << 24).unwrap().lod_min_clamp, 0x3c00);
        assert_eq!(decode(base | 0x4b80 << 40).unwrap().lod_max_clamp, 0x4b80);
        assert_eq!(decode(base | 2 << 56).unwrap().border_color, 2);
        for bit in 58..63 {
            assert!(decode(base | 1 << bit).is_err());
        }
        assert!(decode(0).is_err());
    }

    /// The word Apple's compiler wrote for `constexpr sampler smp_v1(coord::normalized)` in the generated corpus
    /// (`samplers.metal`, variant 1, `@__air_sampler_state`, `i64 -9188470239253757879`): every other field at its
    /// MSL default, so clamp_to_edge, nearest filters, no mipmaps, no comparison, anisotropy 1, and the lod range
    /// [0, 0x7bff] in binary16.
    #[test]
    fn decodes_a_compiled_word() {
        let word = (-9188470239253757879i64) as u64;
        let d = decode(word).unwrap();
        assert_eq!(d.address_modes, [0, 0, 0]);
        assert_eq!((d.mag_filter, d.min_filter, d.mip_filter), (0, 0, 0));
        assert!(d.normalized_coordinates);
        assert_eq!(d.compare_function, 0);
        assert_eq!(d.max_anisotropy, 1);
        assert_eq!((d.lod_min_clamp, d.lod_max_clamp), (0, 0x7bff));
        assert_eq!(d.border_color, 0);
        // `.1` in the same module is `coord::pixel`: only bit 15 differs.
        let pixel = decode((-9188470239253725111i64) as u64).unwrap();
        assert!(!pixel.normalized_coordinates);
    }
}
