//! MSL type names as AIR records them (`air.arg_type_name`, struct member type names), and the Metal values they
//! reflect as: `MTLDataType` for data, `MTLTextureType` and the texture's data type for textures.
//!
//! Enumerator values are the iPhoneOS SDK's (`MTLDataType.h`, `MTLTexture.h`); which name reflects as which value is
//! checked against real Metal's reflection for the corpora (`examples/reflect_corpus.rs`).

/// `MTLDataType` values.
pub mod data_type {
    pub const NONE: u64 = 0;
    pub const STRUCT: u64 = 1;
    pub const ARRAY: u64 = 2;
    pub const FLOAT: u64 = 3;
    pub const HALF: u64 = 16;
    pub const INT: u64 = 29;
    pub const UINT: u64 = 33;
    pub const SHORT: u64 = 37;
    pub const USHORT: u64 = 41;
    pub const CHAR: u64 = 45;
    pub const UCHAR: u64 = 49;
    pub const BOOL: u64 = 53;
    pub const TEXTURE: u64 = 58;
    pub const SAMPLER: u64 = 59;
    pub const POINTER: u64 = 60;
    pub const RENDER_PIPELINE: u64 = 78;
    pub const COMPUTE_PIPELINE: u64 = 79;
    pub const INDIRECT_COMMAND_BUFFER: u64 = 80;
    pub const LONG: u64 = 81;
    pub const ULONG: u64 = 85;
    pub const VISIBLE_FUNCTION_TABLE: u64 = 115;
    pub const INTERSECTION_FUNCTION_TABLE: u64 = 116;
    pub const PRIMITIVE_ACCELERATION_STRUCTURE: u64 = 117;
    pub const INSTANCE_ACCELERATION_STRUCTURE: u64 = 118;
    pub const BFLOAT: u64 = 121;
}

/// The scalar element names, the `MTLDataType` of the scalar, and whether `NxM` matrices of it exist.
const SCALARS: &[(&str, u64, bool)] = &[
    ("float", data_type::FLOAT, true),
    ("half", data_type::HALF, true),
    ("int", data_type::INT, false),
    ("uint", data_type::UINT, false),
    ("short", data_type::SHORT, false),
    ("ushort", data_type::USHORT, false),
    ("char", data_type::CHAR, false),
    ("uchar", data_type::UCHAR, false),
    ("bool", data_type::BOOL, false),
    ("long", data_type::LONG, false),
    ("ulong", data_type::ULONG, false),
    ("bfloat", data_type::BFLOAT, false),
];

/// The `MTLDataType` of a scalar, vector or matrix type name (`float`, `uint3`, `half4x4`), or `None`. A packed
/// vector (`packed_float3`) reflects as the vector type (measured on real Metal).
pub fn data_type(name: &str) -> Option<u64> {
    if let Some(vector) = name.strip_prefix("packed_") {
        let is_vector =
            SCALARS.iter().any(|s| vector.strip_prefix(s.0).is_some_and(|n| matches!(n.as_bytes(), [b'2'..=b'4'])));
        return if is_vector { data_type(vector) } else { None };
    }
    for &(scalar, base, matrices) in SCALARS {
        let Some(rest) = name.strip_prefix(scalar) else { continue };
        if rest.is_empty() {
            return Some(base);
        }
        let bytes = rest.as_bytes();
        let digit = |b: u8| (b'2'..=b'4').contains(&b);
        match bytes {
            [n] if digit(*n) => return Some(base + (*n - b'1') as u64),
            [c, b'x', r] if matrices && digit(*c) && digit(*r) => {
                // Columns 2..4 by rows 2..4 follow the scalar's vector types: float2x2 = Float4 + 1.
                let index = (*c - b'2') as u64 * 3 + (*r - b'2') as u64;
                return Some(base + 4 + index);
            }
            _ => {}
        }
    }
    None
}

/// `MTLTextureType` values.
pub mod texture_type {
    pub const D1: u64 = 0;
    pub const D1_ARRAY: u64 = 1;
    pub const D2: u64 = 2;
    pub const D2_ARRAY: u64 = 3;
    pub const D2_MULTISAMPLE: u64 = 4;
    pub const CUBE: u64 = 5;
    pub const CUBE_ARRAY: u64 = 6;
    pub const D3: u64 = 7;
    pub const D2_MULTISAMPLE_ARRAY: u64 = 8;
    pub const TEXTURE_BUFFER: u64 = 9;
}

/// A texture type name decoded: `texture2d<float, sample>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureName {
    /// `MTLTextureType`.
    pub texture_type: u64,
    /// `MTLDataType` of the texel components.
    pub data_type: u64,
    pub is_depth: bool,
    /// The access token: `sample`, `read`, `write` or `read_write`.
    pub access: TextureAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureAccess {
    Sample,
    Read,
    Write,
    ReadWrite,
}

const TEXTURES: &[(&str, u64, bool)] = &[
    ("texture1d", texture_type::D1, false),
    ("texture1d_array", texture_type::D1_ARRAY, false),
    ("texture2d", texture_type::D2, false),
    ("texture2d_array", texture_type::D2_ARRAY, false),
    ("texture2d_ms", texture_type::D2_MULTISAMPLE, false),
    ("texture2d_ms_array", texture_type::D2_MULTISAMPLE_ARRAY, false),
    ("texture3d", texture_type::D3, false),
    ("texturecube", texture_type::CUBE, false),
    ("texturecube_array", texture_type::CUBE_ARRAY, false),
    ("texture_buffer", texture_type::TEXTURE_BUFFER, false),
    ("depth2d", texture_type::D2, true),
    ("depth2d_array", texture_type::D2_ARRAY, true),
    ("depth2d_ms", texture_type::D2_MULTISAMPLE, true),
    ("depth2d_ms_array", texture_type::D2_MULTISAMPLE_ARRAY, true),
    ("depthcube", texture_type::CUBE, true),
    ("depthcube_array", texture_type::CUBE_ARRAY, true),
];

/// Decodes a texture type name, or `None` when `name` is not one.
pub fn texture(name: &str) -> Option<TextureName> {
    let (head, args) = name.split_once('<')?;
    let args = args.strip_suffix('>')?;
    let &(_, texture_type, is_depth) = TEXTURES.iter().find(|t| t.0 == head)?;
    let (component, access) = args.split_once(", ")?;
    let data_type = data_type(component).filter(|d| {
        matches!(
            *d,
            data_type::FLOAT
                | data_type::HALF
                | data_type::INT
                | data_type::UINT
                | data_type::SHORT
                | data_type::USHORT
                | data_type::ULONG
        )
    })?;
    let access = match access {
        "sample" => TextureAccess::Sample,
        "read" => TextureAccess::Read,
        "write" => TextureAccess::Write,
        "read_write" => TextureAccess::ReadWrite,
        _ => return None,
    };
    Some(TextureName { texture_type, data_type, is_depth, access })
}

/// `array<T, N>`: the element type name and the length.
pub fn array(name: &str) -> Option<(&str, u64)> {
    let inner = name.strip_prefix("array<")?.strip_suffix('>')?;
    let (elem, len) = inner.rsplit_once(", ")?;
    Some((elem, len.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_types_follow_the_sdk_numbering() {
        assert_eq!(data_type("float"), Some(3));
        assert_eq!(data_type("float4"), Some(6));
        assert_eq!(data_type("float2x2"), Some(7));
        assert_eq!(data_type("float2x4"), Some(9));
        assert_eq!(data_type("float3x2"), Some(10));
        assert_eq!(data_type("float4x4"), Some(15));
        assert_eq!(data_type("half4x3"), Some(27));
        assert_eq!(data_type("uint3"), Some(35));
        assert_eq!(data_type("ushort4"), Some(44));
        assert_eq!(data_type("uchar2"), Some(50));
        assert_eq!(data_type("bool4"), Some(56));
        assert_eq!(data_type("ulong4"), Some(88));
        assert_eq!(data_type("bfloat3"), Some(123));
        assert_eq!(data_type("packed_float3"), Some(5));
        assert_eq!(data_type("packed_char4"), Some(48));
        assert_eq!(data_type("packed_float"), None);
        assert_eq!(data_type("packed_float2x2"), None);
        assert_eq!(data_type("int2x2"), None);
        assert_eq!(data_type("float5"), None);
        assert_eq!(data_type("Foo"), None);
    }

    #[test]
    fn texture_names() {
        let t = texture("texture2d<float, sample>").unwrap();
        assert_eq!((t.texture_type, t.data_type, t.is_depth, t.access), (2, 3, false, TextureAccess::Sample));
        let t = texture("depthcube_array<float, read>").unwrap();
        assert_eq!((t.texture_type, t.is_depth), (6, true));
        assert_eq!(texture("texture2d<half, read_write>").unwrap().data_type, 16);
        assert!(texture("sampler").is_none());
        assert_eq!(array("array<texture2d<float, sample>, 4>"), Some(("texture2d<float, sample>", 4)));
    }
}
