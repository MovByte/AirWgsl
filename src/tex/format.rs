//! What a shader sees of a bound texture's pixel format.
//!
//! The executor stores each Metal pixel format the A10X reports either in the WebGPU format of the same definition or
//! in another WebGPU format with the conversions done at the boundary and in the shader (WebMetal's
//! `host/src/format/handling.rs`, generated from WebMetal `tables/formats.toml`). This table restates, for the formats
//! a shader can bind, the part of that decision the shader lowering depends on: the components the Metal format
//! specifies (MSL 4.1 section 6.13: an out-of-bounds read, and an address mode that reads outside the texture, give 0
//! for them and the default value, 0 or alpha 1, for the others), the WebGPU format the texels are stored in and the
//! texel class WGSL reads it as (WebGPU section 26.1), whether the host can bind it for write-only storage, and the
//! shader conversion an emulated format needs ([`Shim`]).

use crate::api::HostCaps;

/// The components a format specifies: bit 0 red, 1 green, 2 blue, 3 alpha.
pub type Components = u8;

pub const R: Components = 1;
pub const RG: Components = 3;
pub const RGB: Components = 7;
pub const RGBA: Components = 15;
pub const A: Components = 8;

/// What WGSL reads a stored WebGPU format as (the `GPUTextureSampleType`s of WebGPU section 26.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Texel {
    /// `"float"`: filterable.
    Float,
    /// `"float"` with `float32-filterable`, else `"unfilterable-float"` (the 32-bit float formats).
    Float32,
    /// `"unfilterable-float"` only (the 16-bit normalized formats).
    UnfilterableFloat,
    Uint,
    Sint,
    /// `"depth"` (and `"unfilterable-float"`).
    Depth,
}

/// When a stored format can be bound as a write-only storage texture (the `STORAGE_BINDING "write-only"` cells).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Always,
    Feature(&'static str),
    Never,
}

/// The shader conversion between the stored texel and the value Metal defines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shim {
    /// The stored format's own read is Metal's value.
    None,
    /// `A8Unorm` in `r8unorm`: reads as (0, 0, 0, r).
    AlphaInRed,
    /// `R8Unorm_sRGB` and `RG8Unorm_sRGB`: the codes in `r8unorm` / `rg8unorm`, decoded with MSL 8.7.7's rule.
    Srgb8,
    /// A packed 16-bit format in `r16uint`: (component, bit offset, bits) of each stored channel, read as
    /// `code / (2^bits - 1)` (MSL 8.7.1.1); a missing color component reads 0 and a missing alpha 1.
    Packed(&'static [(usize, u32, u32)]),
    /// `BGRA10_XR` in `rgba16uint` (each stored 16-bit value holds a code in bits 6 to 15) or `BGR10_XR` in `r32uint`
    /// (blue, green and red codes from bit 0): each code reads as `(code - 384) / 510`; the alpha of `BGRA10_XR` is
    /// clamped to [0, 1] and a missing alpha reads 1.
    ExtendedRange { alpha: bool },
    /// `ASTC_*_LDR` decoded to `rgba16uint` UNORM16 results `C` when the host has no ASTC: reads as `(C >> 4) / 4095`.
    AstcUnorm12,
}

/// A Metal pixel format as the shader sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    /// `MTLPixelFormat`.
    pub value: u64,
    pub name: &'static str,
    pub components: Components,
    /// The class Metal reads it as: a depth or stencil format reads only its first component, and the other
    /// components of such a read are undefined (MSL 4.1 section 6.13).
    pub class: Class,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Float,
    Sint,
    Uint,
    Depth,
    Stencil,
}

/// The texels as the executor stores them on a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stored {
    /// The `GPUTextureFormat`.
    pub webgpu: &'static str,
    pub texel: Texel,
    pub storage: Storage,
    pub shim: Shim,
}

/// Why a format cannot be bound on this host or at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unavailable {
    /// The A10X does not offer the format, or the executor refuses it on every host (the reason).
    NotProvided(&'static str),
    /// The host lacks the feature the executor needs for it.
    Host(&'static str),
}

const fn f(value: u64, name: &'static str, components: Components, class: Class) -> Format {
    Format { value, name, components, class }
}

use Class::{Depth as D, Float as F, Sint as S, Stencil as St, Uint as U};

/// The formats a shader can bind, by `MTLPixelFormat` value (MTLPixelFormat.h).
pub const FORMATS: &[Format] = &[
    f(1, "A8Unorm", A, F),
    f(10, "R8Unorm", R, F),
    f(11, "R8Unorm_sRGB", R, F),
    f(12, "R8Snorm", R, F),
    f(13, "R8Uint", R, U),
    f(14, "R8Sint", R, S),
    f(20, "R16Unorm", R, F),
    f(22, "R16Snorm", R, F),
    f(23, "R16Uint", R, U),
    f(24, "R16Sint", R, S),
    f(25, "R16Float", R, F),
    f(30, "RG8Unorm", RG, F),
    f(31, "RG8Unorm_sRGB", RG, F),
    f(32, "RG8Snorm", RG, F),
    f(33, "RG8Uint", RG, U),
    f(34, "RG8Sint", RG, S),
    f(40, "B5G6R5Unorm", RGB, F),
    f(41, "A1BGR5Unorm", RGBA, F),
    f(42, "ABGR4Unorm", RGBA, F),
    f(43, "BGR5A1Unorm", RGBA, F),
    f(53, "R32Uint", R, U),
    f(54, "R32Sint", R, S),
    f(55, "R32Float", R, F),
    f(60, "RG16Unorm", RG, F),
    f(62, "RG16Snorm", RG, F),
    f(63, "RG16Uint", RG, U),
    f(64, "RG16Sint", RG, S),
    f(65, "RG16Float", RG, F),
    f(70, "RGBA8Unorm", RGBA, F),
    f(71, "RGBA8Unorm_sRGB", RGBA, F),
    f(72, "RGBA8Snorm", RGBA, F),
    f(73, "RGBA8Uint", RGBA, U),
    f(74, "RGBA8Sint", RGBA, S),
    f(80, "BGRA8Unorm", RGBA, F),
    f(81, "BGRA8Unorm_sRGB", RGBA, F),
    f(90, "RGB10A2Unorm", RGBA, F),
    f(91, "RGB10A2Uint", RGBA, U),
    f(92, "RG11B10Float", RGB, F),
    f(93, "RGB9E5Float", RGB, F),
    f(94, "BGR10A2Unorm", RGBA, F),
    f(103, "RG32Uint", RG, U),
    f(104, "RG32Sint", RG, S),
    f(105, "RG32Float", RG, F),
    f(110, "RGBA16Unorm", RGBA, F),
    f(112, "RGBA16Snorm", RGBA, F),
    f(113, "RGBA16Uint", RGBA, U),
    f(114, "RGBA16Sint", RGBA, S),
    f(115, "RGBA16Float", RGBA, F),
    f(123, "RGBA32Uint", RGBA, U),
    f(124, "RGBA32Sint", RGBA, S),
    f(125, "RGBA32Float", RGBA, F),
    f(160, "PVRTC_RGB_2BPP", RGB, F),
    f(161, "PVRTC_RGB_2BPP_sRGB", RGB, F),
    f(162, "PVRTC_RGB_4BPP", RGB, F),
    f(163, "PVRTC_RGB_4BPP_sRGB", RGB, F),
    f(164, "PVRTC_RGBA_2BPP", RGBA, F),
    f(165, "PVRTC_RGBA_2BPP_sRGB", RGBA, F),
    f(166, "PVRTC_RGBA_4BPP", RGBA, F),
    f(167, "PVRTC_RGBA_4BPP_sRGB", RGBA, F),
    f(170, "EAC_R11Unorm", R, F),
    f(172, "EAC_R11Snorm", R, F),
    f(174, "EAC_RG11Unorm", RG, F),
    f(176, "EAC_RG11Snorm", RG, F),
    f(178, "EAC_RGBA8", RGBA, F),
    f(179, "EAC_RGBA8_sRGB", RGBA, F),
    f(180, "ETC2_RGB8", RGB, F),
    f(181, "ETC2_RGB8_sRGB", RGB, F),
    f(182, "ETC2_RGB8A1", RGBA, F),
    f(183, "ETC2_RGB8A1_sRGB", RGBA, F),
    f(186, "ASTC_4x4_sRGB", RGBA, F),
    f(187, "ASTC_5x4_sRGB", RGBA, F),
    f(188, "ASTC_5x5_sRGB", RGBA, F),
    f(189, "ASTC_6x5_sRGB", RGBA, F),
    f(190, "ASTC_6x6_sRGB", RGBA, F),
    f(192, "ASTC_8x5_sRGB", RGBA, F),
    f(193, "ASTC_8x6_sRGB", RGBA, F),
    f(194, "ASTC_8x8_sRGB", RGBA, F),
    f(195, "ASTC_10x5_sRGB", RGBA, F),
    f(196, "ASTC_10x6_sRGB", RGBA, F),
    f(197, "ASTC_10x8_sRGB", RGBA, F),
    f(198, "ASTC_10x10_sRGB", RGBA, F),
    f(199, "ASTC_12x10_sRGB", RGBA, F),
    f(200, "ASTC_12x12_sRGB", RGBA, F),
    f(204, "ASTC_4x4_LDR", RGBA, F),
    f(205, "ASTC_5x4_LDR", RGBA, F),
    f(206, "ASTC_5x5_LDR", RGBA, F),
    f(207, "ASTC_6x5_LDR", RGBA, F),
    f(208, "ASTC_6x6_LDR", RGBA, F),
    f(210, "ASTC_8x5_LDR", RGBA, F),
    f(211, "ASTC_8x6_LDR", RGBA, F),
    f(212, "ASTC_8x8_LDR", RGBA, F),
    f(213, "ASTC_10x5_LDR", RGBA, F),
    f(214, "ASTC_10x6_LDR", RGBA, F),
    f(215, "ASTC_10x8_LDR", RGBA, F),
    f(216, "ASTC_10x10_LDR", RGBA, F),
    f(217, "ASTC_12x10_LDR", RGBA, F),
    f(218, "ASTC_12x12_LDR", RGBA, F),
    f(250, "Depth16Unorm", R, D),
    f(252, "Depth32Float", R, D),
    f(253, "Stencil8", R, St),
    f(260, "Depth32Float_Stencil8", R, D),
    f(261, "X32_Stencil8", R, St),
    f(552, "BGRA10_XR", RGBA, F),
    f(553, "BGRA10_XR_sRGB", RGBA, F),
    f(554, "BGR10_XR", RGB, F),
    f(555, "BGR10_XR_sRGB", RGB, F),
];

pub fn format(value: u64) -> Option<&'static Format> {
    FORMATS.iter().find(|f| f.value == value)
}

const fn st(webgpu: &'static str, texel: Texel, storage: Storage, shim: Shim) -> Stored {
    Stored { webgpu, texel, storage, shim }
}

const TIER1: Storage = Storage::Feature("texture-formats-tier1");

/// The channels of the packed 16-bit formats, bit 0 first (WebMetal `host/src/format/packed.rs`, measured on the
/// device by the reference harness): (component index R 0 ... A 3, offset, bits).
const B5G6R5: &[(usize, u32, u32)] = &[(2, 0, 5), (1, 5, 6), (0, 11, 5)];
const A1BGR5: &[(usize, u32, u32)] = &[(3, 0, 1), (2, 1, 5), (1, 6, 5), (0, 11, 5)];
const ABGR4: &[(usize, u32, u32)] = &[(3, 0, 4), (2, 4, 4), (1, 8, 4), (0, 12, 4)];
const BGR5A1: &[(usize, u32, u32)] = &[(2, 0, 5), (1, 5, 5), (0, 10, 5), (3, 15, 1)];

fn has(host: &HostCaps, feature: &str) -> bool {
    host.features.contains(feature)
}

/// How the executor stores `value` on `host` (WebMetal `handling::resolve`).
pub fn stored(value: u64, host: &HostCaps) -> Result<Stored, Unavailable> {
    use Storage::{Always, Feature, Never};
    use Texel::*;
    let native = |webgpu, texel, storage| Ok(st(webgpu, texel, storage, Shim::None));
    let tier1 = |webgpu, texel, storage| {
        if has(host, "texture-formats-tier1") {
            native(webgpu, texel, storage)
        } else {
            Err(Unavailable::Host("texture-formats-tier1"))
        }
    };
    let etc2 = has(host, "texture-compression-etc2");
    let astc = has(host, "texture-compression-astc");
    let core = Feature("core-features-and-limits");
    match value {
        1 => Ok(st("r8unorm", Float, TIER1, Shim::AlphaInRed)),
        10 => native("r8unorm", Float, TIER1),
        11 => Ok(st("r8unorm", Float, TIER1, Shim::Srgb8)),
        12 => native("r8snorm", Float, TIER1),
        13 => native("r8uint", Uint, TIER1),
        14 => native("r8sint", Sint, TIER1),
        20 => tier1("r16unorm", UnfilterableFloat, Always),
        22 => tier1("r16snorm", UnfilterableFloat, Always),
        23 => native("r16uint", Uint, TIER1),
        24 => native("r16sint", Sint, TIER1),
        25 => native("r16float", Float, TIER1),
        30 => native("rg8unorm", Float, TIER1),
        31 => Ok(st("rg8unorm", Float, TIER1, Shim::Srgb8)),
        32 => native("rg8snorm", Float, TIER1),
        33 => native("rg8uint", Uint, TIER1),
        34 => native("rg8sint", Sint, TIER1),
        40 => Ok(st("r16uint", Uint, TIER1, Shim::Packed(B5G6R5))),
        41 => Ok(st("r16uint", Uint, TIER1, Shim::Packed(A1BGR5))),
        42 => Ok(st("r16uint", Uint, TIER1, Shim::Packed(ABGR4))),
        43 => Ok(st("r16uint", Uint, TIER1, Shim::Packed(BGR5A1))),
        53 => native("r32uint", Uint, Always),
        54 => native("r32sint", Sint, Always),
        55 => native("r32float", Float32, Always),
        60 => tier1("rg16unorm", UnfilterableFloat, Always),
        62 => tier1("rg16snorm", UnfilterableFloat, Always),
        63 => native("rg16uint", Uint, TIER1),
        64 => native("rg16sint", Sint, TIER1),
        65 => native("rg16float", Float, TIER1),
        70 => native("rgba8unorm", Float, Always),
        71 => native("rgba8unorm-srgb", Float, Never),
        72 => native("rgba8snorm", Float, Always),
        73 => native("rgba8uint", Uint, Always),
        74 => native("rgba8sint", Sint, Always),
        80 => native("bgra8unorm", Float, Feature("bgra8unorm-storage")),
        81 => native("bgra8unorm-srgb", Float, Never),
        90 => native("rgb10a2unorm", Float, TIER1),
        91 => native("rgb10a2uint", Uint, TIER1),
        92 => native("rg11b10ufloat", Float, TIER1),
        93 => native("rgb9e5ufloat", Float, Never),
        // Stored in rgb10a2unorm with red and blue exchanged when the bytes cross the boundary.
        94 => native("rgb10a2unorm", Float, TIER1),
        103 => native("rg32uint", Uint, core),
        104 => native("rg32sint", Sint, core),
        105 => native("rg32float", Float32, core),
        110 => tier1("rgba16unorm", UnfilterableFloat, Always),
        112 => tier1("rgba16snorm", UnfilterableFloat, Always),
        113 => native("rgba16uint", Uint, Always),
        114 => native("rgba16sint", Sint, Always),
        115 => native("rgba16float", Float, Always),
        123 => native("rgba32uint", Uint, Always),
        124 => native("rgba32sint", Sint, Always),
        125 => native("rgba32float", Float32, Always),
        // Decoded on upload: WebGPU has no PVRTC.
        160..=167 => native(if value % 2 == 1 { "rgba8unorm-srgb" } else { "rgba8unorm" }, Float, Never),
        170 | 172 | 174 | 176 if etc2 => native(
            match value {
                170 => "eac-r11unorm",
                172 => "eac-r11snorm",
                174 => "eac-rg11unorm",
                _ => "eac-rg11snorm",
            },
            Float,
            Never,
        ),
        170 | 172 | 174 | 176 => Err(Unavailable::Host("texture-compression-etc2")),
        178..=183 => {
            let srgb = value % 2 == 1;
            if etc2 {
                let base = match value {
                    178 | 179 => "etc2-rgba8unorm",
                    180 | 181 => "etc2-rgb8unorm",
                    _ => "etc2-rgb8a1unorm",
                };
                native(if srgb { srgb_name(base) } else { base }, Float, Never)
            } else {
                native(if srgb { "rgba8unorm-srgb" } else { "rgba8unorm" }, Float, Never)
            }
        }
        186..=200 if astc => native(astc_name(value), Float, Never),
        186..=200 => native("rgba8unorm-srgb", Float, Never),
        204..=218 if astc => native(astc_name(value), Float, Never),
        204..=218 => Ok(st("rgba16uint", Uint, Never, Shim::AstcUnorm12)),
        250 => native("depth16unorm", Depth, Never),
        252 => native("depth32float", Depth, Never),
        253 => native("stencil8", Uint, Never),
        260 | 261 if !has(host, "depth32float-stencil8") => Err(Unavailable::Host("depth32float-stencil8")),
        260 => native("depth32float-stencil8", Depth, Never),
        261 => native("depth32float-stencil8", Uint, Never),
        552 => Ok(st("rgba16uint", Uint, Always, Shim::ExtendedRange { alpha: true })),
        554 => Ok(st("r32uint", Uint, Always, Shim::ExtendedRange { alpha: false })),
        553 | 555 => Err(Unavailable::NotProvided(
            "the sRGB variants of the XR formats apply a transfer function to values outside [0, 1] that neither the \
             SDK header nor the pinned documentation defines (WebMetal formats.toml question 6.3-xr-encoding)",
        )),
        _ => Err(Unavailable::NotProvided("not a pixel format the A10X binds to a shader (WebMetal format handling)")),
    }
}

/// The WebGPU ASTC format of an `ASTC_*_sRGB` (186 to 200) or `ASTC_*_LDR` (204 to 218) value.
fn astc_name(value: u64) -> &'static str {
    const SRGB: [&str; 15] = [
        "astc-4x4-unorm-srgb",
        "astc-5x4-unorm-srgb",
        "astc-5x5-unorm-srgb",
        "astc-6x5-unorm-srgb",
        "astc-6x6-unorm-srgb",
        "",
        "astc-8x5-unorm-srgb",
        "astc-8x6-unorm-srgb",
        "astc-8x8-unorm-srgb",
        "astc-10x5-unorm-srgb",
        "astc-10x6-unorm-srgb",
        "astc-10x8-unorm-srgb",
        "astc-10x10-unorm-srgb",
        "astc-12x10-unorm-srgb",
        "astc-12x12-unorm-srgb",
    ];
    let srgb = SRGB[(if value >= 204 { value - 204 } else { value - 186 }) as usize];
    if value >= 204 { srgb.strip_suffix("-srgb").unwrap() } else { srgb }
}

fn srgb_name(base: &'static str) -> &'static str {
    match base {
        "etc2-rgba8unorm" => "etc2-rgba8unorm-srgb",
        "etc2-rgb8unorm" => "etc2-rgb8unorm-srgb",
        _ => "etc2-rgb8a1unorm-srgb",
    }
}

impl Stored {
    /// Whether the host can filter the stored texels.
    pub fn filterable(&self, host: &HostCaps) -> bool {
        match self.texel {
            Texel::Float => true,
            Texel::Float32 => has(host, "float32-filterable"),
            _ => false,
        }
    }

    /// Whether the host can bind the stored format as a write-only storage texture.
    pub fn writable(&self, host: &HostCaps) -> bool {
        match self.storage {
            Storage::Always => true,
            Storage::Feature(f) => has(host, f),
            Storage::Never => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_are_listed_once() {
        for (i, a) in FORMATS.iter().enumerate() {
            assert!(FORMATS[i + 1..].iter().all(|b| b.value != a.value), "{}", a.name);
            let host = HostCaps::default();
            match stored(a.value, &host) {
                Ok(s) => assert!(!s.webgpu.is_empty()),
                Err(Unavailable::Host(_)) | Err(Unavailable::NotProvided(_)) => {}
            }
        }
    }

    #[test]
    fn storage_and_filtering_follow_the_host() {
        let mut host = HostCaps::default();
        let r32 = stored(55, &host).unwrap();
        assert!(!r32.filterable(&host) && r32.writable(&host));
        host.features.insert("float32-filterable".into());
        assert!(r32.filterable(&host));
        let r8 = stored(10, &host).unwrap();
        assert!(!r8.writable(&host));
        host.features.insert("texture-formats-tier1".into());
        assert!(r8.writable(&host));
        assert_eq!(stored(41, &host).unwrap().shim, Shim::Packed(A1BGR5));
        assert!(matches!(stored(20, &HostCaps::default()), Err(Unavailable::Host("texture-formats-tier1"))));
        host.features.insert("texture-compression-astc".into());
        assert_eq!(stored(186, &host).unwrap().webgpu, "astc-4x4-unorm-srgb");
        assert_eq!(stored(218, &host).unwrap().webgpu, "astc-12x12-unorm");
        assert_eq!(stored(192, &host).unwrap().webgpu, "astc-8x5-unorm-srgb");
    }
}
