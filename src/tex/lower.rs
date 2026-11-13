//! The lowering of each texture function: operands in the order the front end writes them (measured on
//! `metal-32023.922`, `air64-apple-ios17.7`, and checked against the generated and system corpora), then the WGSL.

use super::format::{self, Shim};
use super::*;
use crate::api::{SamplerBindingType, TextureSampleType as S, TextureViewDimension as D};

/// The operands of a call, read in order.
struct Operands<'a> {
    args: &'a [Value],
    k: usize,
    construct: &'a str,
}

impl<'a> Operands<'a> {
    fn next(&mut self) -> R<Value> {
        let v = self.args.get(self.k).copied().ok_or_else(|| call_form(self.construct, "too few operands"))?;
        self.k += 1;
        Ok(v)
    }

    fn done(&self) -> R<()> {
        if self.k != self.args.len() {
            return Err(call_form(self.construct, "too many operands"));
        }
        Ok(())
    }
}

fn call_form(construct: &str, detail: &str) -> Refusal {
    refuse("texture-call-form", construct, format!("{} for its name", detail))
}

/// An offset operand pair (`i1 has_offset`, the offset vector): the constant offset, `None` when it is zero.
fn offset(cx: &dyn RuleCx, construct: &str, has: Value, off: Value, lanes: usize) -> R<Option<Vec<i32>>> {
    let m = cx.module();
    let has = const_int(m, has).ok_or_else(|| call_form(construct, "a non-constant offset flag"))?;
    let v = const_ivec(m, off).ok_or_else(|| {
        refuse(
            "texture-dynamic-offset",
            construct,
            "a texel offset computed at run time: WGSL offsets are constant expressions",
        )
    })?;
    if v.len() != lanes {
        return Err(call_form(construct, "an offset of the wrong length"));
    }
    if has == 0 || v.iter().all(|x| *x == 0) {
        return Ok(None);
    }
    if v.iter().any(|x| !(-8..=7).contains(x)) {
        return Err(call_form(construct, "an offset outside -8 to 7 (MSL 4.1 section 6.13)"));
    }
    Ok(Some(v))
}

fn wgsl_offset(off: &Option<Vec<i32>>) -> String {
    match off {
        None => String::new(),
        Some(v) => format!(", vec{}<i32>({})", v.len(), v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

/// The access code operand: 0 sample, 1 read, 2 write, 3 read_write (measured).
fn access_code(cx: &dyn RuleCx, construct: &str, v: Value) -> R<i64> {
    let a = const_int(cx.module(), v).ok_or_else(|| call_form(construct, "a non-constant access operand"))?;
    if a == 3 {
        return Err(above_family(construct, "access::read_write textures", "getters.readWriteTextureSupport = 0"));
    }
    if !(0..=2).contains(&a) {
        return Err(call_form(construct, "an access operand other than 0, 1 or 2"));
    }
    Ok(a)
}

/// The depth functions' leading `i32` operand: 1 in every call the front end writes.
fn depth_flag(cx: &dyn RuleCx, construct: &str, v: Value) -> R<()> {
    if const_int(cx.module(), v) != Some(1) {
        return Err(call_form(construct, "a depth function operand other than the constant 1"));
    }
    Ok(())
}

pub(crate) fn lower(cx: &mut dyn RuleCx, call: &CallSite<'_>, n: &Name) -> R<Option<String>> {
    let c = call.name;
    if n.op == Op2::IsNull {
        return is_null(cx, call, c).map(Some);
    }
    match n.dim {
        Dim::Buffer => {
            return Err(refuse(
                "texture-buffer",
                c,
                "texture buffers are read through the memory model over the bound buffer, which is not implemented",
            ));
        }
        Dim::CubeArray => {
            return Err(above_family(c, "cube texture arrays", "getters.supportsTextureCubeArray = false"));
        }
        Dim::D2MsArray => {
            return Err(above_family(
                c,
                "2D multisample texture arrays",
                "the A10X creates no MTLTextureType2DMultisampleArray texture (plan 2.7)",
            ));
        }
        _ => {}
    }
    let t = texture(cx, n, c, call.args[0])?;
    if t.access == Access::ReadWrite {
        return Err(above_family(c, "access::read_write textures", "getters.readWriteTextureSupport = 0"));
    }
    if n.dim == Dim::Cube {
        let u = uses(cx, Some(t.arg), None);
        if u.texel && u.sampled {
            return Err(refuse(
                "texture-binding-conflict",
                c,
                "a cube texture both sampled (a cube view) and read or written texel by texel (a 2D array view of its \
                 faces) in one function",
            ));
        }
    }
    let r = match n.op {
        Op2::Width | Op2::Height | Op2::Depth | Op2::ArraySize | Op2::MipLevels | Op2::Samples => {
            query(cx, call, n, &t)?
        }
        Op2::Read => read(cx, call, n, &t)?,
        Op2::Write => {
            write(cx, call, n, &t)?;
            return Ok(None);
        }
        Op2::Sample | Op2::SampleCompare => sample(cx, call, n, &t)?,
        Op2::Gather | Op2::GatherCompare => gather(cx, call, n, &t)?,
        Op2::IsNull => unreachable!(),
    };
    Ok(Some(r))
}

// Null textures ---------------------------------------------------------------------------------------------------

/// `is_null_texture`: whether the slot holds a texture. Only argument textures reach here; a null handle made in the
/// shader (`air.get_null_*`) is refused by the memory model as a pointer of unknown provenance.
fn is_null(cx: &mut dyn RuleCx, call: &CallSite<'_>, c: &str) -> R<String> {
    if call.args.len() != 1 {
        return Err(call_form(c, "an operand count other than one"));
    }
    let Some(arg) = cx.handle_arg(call.args[0]) else {
        return Err(refuse("texture-dynamic-handle", c, "the texture operand is not one texture argument"));
    };
    let Some(Role::Texture { location, .. }) = cx.arg_record(arg).map(|r| r.role) else {
        return Err(refuse("entry-metadata", c, "the texture operand's argument is not a texture"));
    };
    let Slot::Value(index) = location.index else {
        return Err(tracked::refuse_family("function-constants", c.to_string(), None));
    };
    let bound = cx.bindings().is_some_and(|b| b.textures.iter().any(|t| t.index == index as u64));
    Ok(if bound { "false".into() } else { "true".into() })
}

// Views -----------------------------------------------------------------------------------------------------------

/// The view dimension of a sampled binding of an AIR texture type (1D textures are stored as 2D).
fn sampled_dim(dim: Dim, texel: bool) -> D {
    match dim {
        Dim::D1 | Dim::D2 | Dim::D2Ms => D::D2,
        Dim::D1Array | Dim::D2Array => D::D2Array,
        Dim::D3 => D::D3,
        Dim::Cube if texel => D::D2Array,
        Dim::Cube => D::Cube,
        Dim::CubeArray | Dim::D2MsArray | Dim::Buffer => unreachable!("refused before"),
    }
}

/// The storage format of a write binding: the bound format's stored WebGPU format, or `rgba32` of the declared
/// type for a null texture.
fn storage_format(cx: &dyn RuleCx, t: &Tex, c: &str) -> R<&'static str> {
    match &t.format {
        None => Ok(match t.component {
            Component::Float => "rgba32float",
            Component::Sint => "rgba32sint",
            Component::Uint => "rgba32uint",
        }),
        Some((f, st)) => {
            if !st.writable(cx.host()) {
                return Err(refuse(
                    "storage-texture-format",
                    c,
                    format!(
                        "the bound format {} is stored as {}, which the host cannot bind as a write-only storage \
                         texture",
                        f.name, st.webgpu
                    ),
                ));
            }
            Ok(st.webgpu)
        }
    }
}

/// Bind the texture for the call's op: a written texture's binding is its level 0 ([`bind_level`]).
fn bind(cx: &mut dyn RuleCx, n: &Name, t: &Tex, c: &str) -> R<String> {
    if t.access == Access::Write {
        return bind_level(cx, n, t, c, 0);
    }
    let texel = n.dim == Dim::Cube && uses(cx, Some(t.arg), None).texel;
    let ms = n.dim == Dim::D2Ms;
    let mut st = sample_type(cx, t, n.depth);
    // WebGPU 8.1.1: a multisampled texture binding's sample type is not "float" (a multisampled texture is never
    // filtered); its float texels are bound as "unfilterable-float".
    if ms && st == S::Float {
        st = S::UnfilterableFloat;
    }
    bind_texture(cx, t, View::Sampled(sampled_dim(n.dim, texel), st, ms), 0)
}

/// Bind level `level` of a written texture (counted from a bound view's first level) as a storage texture.
fn bind_level(cx: &mut dyn RuleCx, n: &Name, t: &Tex, c: &str, level: u32) -> R<String> {
    if cx.stage() == MetalStage::Vertex {
        return Err(tracked::refuse_row(
            "vertex-stage-writes",
            RefusalKind::NoExactLowering,
            c.to_string(),
            None,
            "a vertex function writes a texture",
            "WebGPU 8.1.1 bind group layout creation",
        ));
    }
    let fmt = storage_format(cx, t, c)?;
    bind_texture(cx, t, View::Storage(sampled_dim(n.dim, true), fmt), level)
}

// Queries ---------------------------------------------------------------------------------------------------------

fn query(cx: &mut dyn RuleCx, call: &CallSite<'_>, n: &Name, t: &Tex) -> R<String> {
    let c = call.name;
    let has_lod = matches!(n.op, Op2::Width | Op2::Height | Op2::Depth) && n.dim != Dim::D2Ms;
    let want = if has_lod { 2 } else { 1 };
    if call.args.len() != want {
        return Err(call_form(c, "an operand count"));
    }
    let storage = t.access == Access::Write;
    if n.op == Op2::MipLevels && storage {
        // The level count of the bound texture (a view's own) is variant state; a null texture's placeholder has one
        // level.
        return Ok(format!("{}u", levels(t).unwrap_or(1)));
    }
    let v = bind(cx, n, t, c)?;
    let lane = match n.op {
        Op2::Width => "x",
        Op2::Height => "y",
        _ => "z",
    };
    Ok(match n.op {
        Op2::Width | Op2::Height | Op2::Depth => {
            if !has_lod || n.dim == Dim::D2Ms {
                format!("textureDimensions({}).{}", v, lane)
            } else {
                let lod = cx.operand(call.args[1])?;
                if storage {
                    // A storage binding is level 0 of the bound texture; level L of a Metal texture is
                    // max(1, size >> L) in each dimension.
                    format!("max(textureDimensions({}).{} >> min({}, 31u), 1u)", v, lane, lod)
                } else {
                    format!("textureDimensions({}, {}).{}", v, lod, lane)
                }
            }
        }
        Op2::ArraySize => format!("textureNumLayers({})", v),
        Op2::Samples => format!("textureNumSamples({})", v),
        Op2::MipLevels => format!("textureNumLevels({})", v),
        _ => unreachable!(),
    })
}

// Texel values ----------------------------------------------------------------------------------------------------

/// The bits of 1.0 (or integer 1) in the result element type.
fn one_bits(e: Elem) -> &'static str {
    match e {
        Elem::F32 => "0x3f800000u",
        Elem::F16 => "0x3c00u",
        _ => "1u",
    }
}

/// The default texel of MSL 4.1 section 6.13 for an out-of-bounds read, as result-element bits: 0 for the components
/// the format specifies and for missing color components, 1 for a missing alpha. The `clampToZero` address mode reads
/// the same value (MTLSampler.h: transparent zero with alpha, opaque zero without).
fn default_bits(t: &Tex, e: Elem) -> String {
    let alpha = if super::components(t) & format::A != 0 { "0u" } else { one_bits(e) };
    format!("vec4<u32>(0u, 0u, 0u, {})", alpha)
}

/// The format's own component mapping composed with the bound view's swizzle (MTLTextureSwizzle: 0 zero, 1 one,
/// 2..5 red..alpha), when the shader applies it; `None` when the view applies it or there is none.
fn swizzle(cx: &mut dyn RuleCx, t: &Tex) -> Option<[u8; 4]> {
    let fmt: [u8; 4] = if shim(t) == Shim::AlphaInRed { [0, 0, 0, 2] } else { [2, 3, 4, 5] };
    let view = t.slot.map(|s| s.swizzle).unwrap_or([2, 3, 4, 5]);
    let composed: [u8; 4] = std::array::from_fn(|i| match view[i] {
        0 | 1 => view[i],
        k => fmt[(k - 2) as usize],
    });
    if composed == [2, 3, 4, 5] {
        return None;
    }
    if cx.host().features.contains("texture-component-swizzle") {
        cx.require_feature("texture-component-swizzle");
        return None;
    }
    Some(composed)
}

fn apply_swizzle(v: &str, sw: Option<[u8; 4]>, e: Elem) -> String {
    let Some(sw) = sw else { return v.to_string() };
    let lanes: Vec<String> = sw
        .iter()
        .map(|&k| match k {
            0 => "0u".to_string(),
            1 => one_bits(e).to_string(),
            k => format!("{}.{}", v, ["x", "y", "z", "w"][(k - 2) as usize]),
        })
        .collect();
    format!("vec4<u32>({})", lanes.join(", "))
}

/// A vec4 of f32 texel values as result-element bits.
fn float_bits(cx: &mut dyn RuleCx, v: &str, e: Elem) -> R<String> {
    Ok(match e {
        Elem::F16 => {
            cx.prelude("nm_f32_to_f16")?;
            cx.module_item(
                "aw_tex_f16x4",
                "// binary32 texel values to binary16 bits, rounding to nearest even (MSL 4.1 section 8.7.2).\n\
                 fn aw_tex_f16x4(v: vec4<f32>) -> vec4<u32> {\n    let b = bitcast<vec4<u32>>(v);\n    \
                 return vec4<u32>(nm_f32_to_f16(b.x), nm_f32_to_f16(b.y), nm_f32_to_f16(b.z), nm_f32_to_f16(b.w));\n}\n"
                    .into(),
            );
            format!("aw_tex_f16x4({})", v)
        }
        _ => format!("bitcast<vec4<u32>>({})", v),
    })
}

/// Integer texel values as result-element bits: 16-bit results keep the low 16 bits.
fn int_bits(v: &str, e: Elem) -> String {
    match e {
        Elem::I16(_) => format!("(bitcast<vec4<u32>>({}) & vec4<u32>(0xffffu))", v),
        _ => format!("bitcast<vec4<u32>>({})", v),
    }
}

/// The value Metal reads from one stored texel `raw` (the WGSL load's result) as result-element bits.
fn texel_bits(cx: &mut dyn RuleCx, t: &Tex, raw: &str, e: Elem) -> R<String> {
    let f16 = e == Elem::F16;
    let conv = |cx: &mut dyn RuleCx, code: String, max: &str| -> R<String> {
        let f = if f16 { "nm_unorm_f16" } else { "nm_unorm_f32" };
        cx.prelude(if f16 { "nm_unorm_f16" } else { "nm_unorm_f32" })?;
        Ok(format!("{}({}, {})", f, code, max))
    };
    match shim(t) {
        Shim::None | Shim::AlphaInRed => match t.component {
            Component::Float => float_bits(cx, raw, e),
            _ => Ok(int_bits(raw, e)),
        },
        Shim::Srgb8 => {
            let f = if f16 { "nm_srgb_f16" } else { "nm_srgb_f32" };
            cx.prelude(f)?;
            let two = super::components(t) & 2 != 0;
            let code = |lane: &str| format!("{}(u32(round({}.{} * 255.0)))", f, raw, lane);
            let g = if two { code("y") } else { "0u".into() };
            Ok(format!("vec4<u32>({}, {}, 0u, {})", code("x"), g, one_bits(e)))
        }
        Shim::Packed(channels) => {
            let mut lanes = ["0u".to_string(), "0u".to_string(), "0u".to_string(), one_bits(e).to_string()];
            for &(comp, off, bits) in channels {
                let max = (1u32 << bits) - 1;
                lanes[comp] = conv(cx, format!("(({}.x >> {}u) & {}u)", raw, off, max), &format!("{}u", max))?;
            }
            Ok(format!("vec4<u32>({})", lanes.join(", ")))
        }
        Shim::ExtendedRange { alpha } => {
            // (code - 384) / 510, correctly rounded (Apple's documentation of MTLPixelFormatBGR10_XR).
            cx.prelude("nm_f32_from_i32")?;
            let xr = |cx: &mut dyn RuleCx, code: String| -> R<String> {
                let q = if f16 {
                    cx.prelude("nm_f32_div_ur")?;
                    cx.prelude("nm_f16_finish")?;
                    format!("nm_f16_finish(nm_f32_div_ur(nm_f32_from_i32(({}) - 384u), 0x43ff0000u))", code)
                } else {
                    cx.prelude("nm_f32_div")?;
                    format!("nm_f32_div(nm_f32_from_i32(({}) - 384u), 0x43ff0000u)", code)
                };
                Ok(q)
            };
            if alpha {
                // BGRA10_XR in rgba16uint: blue, green, red, alpha codes in bits 6 to 15 of the four values; alpha
                // clamped to [0, 1], which is codes 384 to 894.
                let r = xr(cx, format!("({}.z >> 6u)", raw))?;
                let g = xr(cx, format!("({}.y >> 6u)", raw))?;
                let b = xr(cx, format!("({}.x >> 6u)", raw))?;
                let a = xr(cx, format!("clamp({}.w >> 6u, 384u, 894u)", raw))?;
                Ok(format!("vec4<u32>({}, {}, {}, {})", r, g, b, a))
            } else {
                // BGR10_XR in r32uint: blue, green and red from bit 0.
                let r = xr(cx, format!("(({}.x >> 20u) & 0x3ffu)", raw))?;
                let g = xr(cx, format!("(({}.x >> 10u) & 0x3ffu)", raw))?;
                let b = xr(cx, format!("({}.x & 0x3ffu)", raw))?;
                Ok(format!("vec4<u32>({}, {}, {}, {})", r, g, b, one_bits(e)))
            }
        }
        Shim::AstcUnorm12 => {
            let lanes: R<Vec<String>> =
                ["x", "y", "z", "w"].iter().map(|l| conv(cx, format!("({}.{} >> 4u)", raw, l), "4095u")).collect();
            Ok(format!("vec4<u32>({})", lanes?.join(", ")))
        }
    }
}

/// Whether reads of the texture go through a shim that decodes stored integer words.
fn integer_shim(t: &Tex) -> bool {
    matches!(shim(t), Shim::Packed(_) | Shim::ExtendedRange { .. } | Shim::AstcUnorm12)
}

/// Package a `{ <4 x T>, i8 }` or `{ T, i8 }` result: the value and a resident flag of 0.
fn result(cx: &mut dyn RuleCx, call: &CallSite<'_>, value: String) -> R<String> {
    let s = result_struct(cx, call.ret)?;
    Ok(format!("{}({}, 0u)", s, value))
}

// Reads -----------------------------------------------------------------------------------------------------------

fn read(cx: &mut dyn RuleCx, call: &CallSite<'_>, n: &Name, t: &Tex) -> R<String> {
    let c = call.name;
    let mut o = Operands { args: call.args, k: 1, construct: c };
    if n.depth {
        depth_flag(cx, c, o.next()?)?;
    }
    let coord = o.next()?;
    let face = if n.dim == Dim::Cube { Some(o.next()?) } else { None };
    let slice = if n.dim.arrayed() { Some(o.next()?) } else { None };
    let lod_or_sample = o.next()?;
    let acc = o.next()?;
    o.done()?;
    if access_code(cx, c, acc)? == 2 {
        return Err(call_form(c, "a read with the write access code"));
    }
    if t.access == Access::Write {
        return Err(call_form(c, "a read of a write-only texture"));
    }
    if n.elem == Elem::None {
        return Err(call_form(c, "no result type"));
    }
    let v = bind(cx, n, t, c)?;
    let p = cx.fresh();
    let coord_e = cx.operand(coord)?;
    let coords = match n.dim {
        Dim::D1 | Dim::D1Array => format!("vec2<u32>({}, 0u)", coord_e),
        _ => coord_e,
    };
    cx.line(format!("let {}_c = {};", p, coords));
    let ls = cx.operand(lod_or_sample)?;
    cx.line(format!("let {}_l = {};", p, ls));
    // MSL 4.1 section 6.13: a read is out of bounds when the coordinates, the level, the slice, the face or the
    // sample is.
    let mut extra = String::new();
    let mut layer = String::new();
    if let Some(f) = face {
        let fe = cx.operand(f)?;
        cx.line(format!("let {}_f = {};", p, fe));
        extra.push_str(&format!(" && {}_f < 6u", p));
        layer = format!(", {}_f", p);
    }
    if let Some(s) = slice {
        let se = cx.operand(s)?;
        cx.line(format!("let {}_s = {};", p, se));
        extra.push_str(&format!(" && {}_s < textureNumLayers({})", p, v));
        layer = format!(", {}_s", p);
    }
    if n.depth {
        // A depth read returns the depth component only; out of bounds it is 0.
        cx.line(format!("var {}_r = 0u;", p));
    } else {
        cx.line(format!("var {}_r = {};", p, default_bits(t, n.elem)));
    }
    if n.dim == Dim::D2Ms {
        cx.line(format!(
            "if (all({p}_c < textureDimensions({v})) && {p}_l < textureNumSamples({v}){e}) {{",
            p = p,
            v = v,
            e = extra
        ));
    } else {
        cx.line(format!("if ({}_l < textureNumLevels({})) {{", p, v));
        cx.line(format!("if (all({p}_c < textureDimensions({v}, {p}_l)){e}) {{", p = p, v = v, e = extra));
    }
    let load = format!("textureLoad({}, {}_c{}, {}_l)", v, p, layer, p);
    let val = if n.depth {
        format!("bitcast<u32>({})", load)
    } else {
        cx.line(format!("let {}_t = {};", p, load));
        texel_bits(cx, t, &format!("{}_t", p), n.elem)?
    };
    cx.line(format!("{}_r = {};", p, val));
    cx.line("}".into());
    if n.dim != Dim::D2Ms {
        cx.line("}".into());
    }
    let out = if n.depth {
        format!("{}_r", p)
    } else {
        let sw = swizzle(cx, t);
        apply_swizzle(&format!("{}_r", p), sw, n.elem)
    };
    result(cx, call, out)
}

// Writes ----------------------------------------------------------------------------------------------------------

/// The WGSL value a write stores into the stored format, rounded to the format's precision in the shader so that
/// WebGPU's conversion of an exactly representable value is exact: Metal rounds float to half to nearest even (MSL
/// 4.1 section 8.7.2) and float to normalized codes to nearest even after clamping (section 8.7.1.2), and saturates
/// integers to 8- and 16-bit formats (section 8.7.6, Table 8.8).
fn store_value(cx: &mut dyn RuleCx, t: &Tex, n: &Name, v: &str, c: &str) -> R<String> {
    let e = n.elem;
    // The operand as vec4<f32> (float types), vec4<i32> or vec4<u32>.
    let float = |cx: &mut dyn RuleCx| -> R<String> {
        Ok(match e {
            Elem::F16 => {
                cx.prelude("nm_f16_to_f32")?;
                format!(
                    "bitcast<vec4<f32>>(vec4<u32>(nm_f16_to_f32({v}.x), nm_f16_to_f32({v}.y), nm_f16_to_f32({v}.z), \
                     nm_f16_to_f32({v}.w)))"
                )
            }
            _ => format!("bitcast<vec4<f32>>({})", v),
        })
    };
    let ints = match e {
        Elem::I16(true) => format!("(bitcast<vec4<i32>>({} << vec4<u32>(16u)) >> vec4<u32>(16u))", v),
        Elem::I16(false) => format!("({} & vec4<u32>(0xffffu))", v),
        Elem::I32(true) => format!("bitcast<vec4<i32>>({})", v),
        _ => v.to_string(),
    };
    let Some((f, st)) = t.format else {
        return match t.component {
            Component::Float => float(cx),
            _ => Ok(ints),
        };
    };
    // Per-lane conversion of float bits (as a vec4<u32> `b`) to the exact stored value.
    let lanes = |cx: &mut dyn RuleCx, item: &'static str, call: &dyn Fn(&str) -> String| -> R<String> {
        cx.prelude(item)?;
        let b = cx.fresh();
        let fv = float(cx)?;
        cx.line(format!("let {} = bitcast<vec4<u32>>({});", b, fv));
        Ok(format!(
            "bitcast<vec4<f32>>(vec4<u32>({}, {}, {}, {}))",
            call(&format!("{}.x", b)),
            call(&format!("{}.y", b)),
            call(&format!("{}.z", b)),
            call(&format!("{}.w", b))
        ))
    };
    let unorm = |cx: &mut dyn RuleCx, max: [u32; 4]| -> R<String> {
        cx.prelude("nm_unorm_code")?;
        cx.prelude("nm_unorm_f32")?;
        let b = cx.fresh();
        let fv = float(cx)?;
        cx.line(format!("let {} = bitcast<vec4<u32>>({});", b, fv));
        let l: Vec<String> = ["x", "y", "z", "w"]
            .iter()
            .zip(max)
            .map(|(k, m)| format!("nm_unorm_f32(nm_unorm_code({}.{}, {}u), {}u)", b, k, m, m))
            .collect();
        Ok(format!("bitcast<vec4<f32>>(vec4<u32>({}))", l.join(", ")))
    };
    let snorm = |cx: &mut dyn RuleCx, bits: u32| -> R<String> {
        let max = (1u32 << (bits - 1)) - 1;
        lanes(cx, "nm_snorm_f32", &|x| {
            format!("nm_snorm_f32(nm_snorm_code({}, {}u, {}u), {}u, {}u)", x, max, bits, max, bits)
        })
        .and_then(|s| cx.prelude("nm_snorm_code").map(|_| s))
    };
    let sat = |lo: i64, hi: i64, signed: bool| -> String {
        if signed {
            format!("clamp({}, vec4<i32>({}), vec4<i32>({}))", ints, lo, hi)
        } else {
            format!("min({}, vec4<u32>({}u))", ints, hi)
        }
    };
    match st.shim {
        Shim::None => {}
        Shim::Srgb8 => {
            cx.prelude("nm_srgb_code")?;
            cx.prelude("nm_unorm_f32")?;
            let b = cx.fresh();
            let fv = float(cx)?;
            cx.line(format!("let {} = bitcast<vec4<u32>>({});", b, fv));
            let l = |k: &str| format!("bitcast<f32>(nm_unorm_f32(nm_srgb_code({}.{}), 255u))", b, k);
            return Ok(format!("vec4<f32>({}, {}, 0.0, 0.0)", l("x"), l("y")));
        }
        Shim::Packed(channels) => {
            cx.prelude("nm_unorm_code")?;
            let b = cx.fresh();
            let fv = float(cx)?;
            cx.line(format!("let {} = bitcast<vec4<u32>>({});", b, fv));
            let word: Vec<String> = channels
                .iter()
                .map(|&(comp, off, bits)| {
                    format!(
                        "(nm_unorm_code({}.{}, {}u) << {}u)",
                        b,
                        ["x", "y", "z", "w"][comp],
                        (1u32 << bits) - 1,
                        off
                    )
                })
                .collect();
            return Ok(format!("vec4<u32>({}, 0u, 0u, 0u)", word.join(" | ")));
        }
        _ => {
            return Err(refuse(
                "texture-format-shim",
                c,
                format!("a write to {}: the shader encoding of its stored format is not implemented", f.name),
            ));
        }
    }
    Ok(match st.webgpu {
        "r32float" | "rg32float" | "rgba32float" => float(cx)?,
        "r16float" | "rg16float" | "rgba16float" if e == Elem::F16 => float(cx)?,
        "r16float" | "rg16float" | "rgba16float" => {
            // A float value rounds to binary16 by the write's rounding mode (MSL 4.1 section 1.6.7); the native mode
            // is the hardware's (this Mac's Apple M4 rounds toward zero, measured by the texture tests), which the A10X
            // profile does not record.
            cx.prelude("nm_f16_to_f32")?;
            match n.round {
                Some(true) => lanes(cx, "nm_f32_to_f16", &|x| format!("nm_f16_to_f32(nm_f32_to_f16({}))", x))?,
                Some(false) => {
                    cx.module_item("aw_tex_f16_rtz", F16_RTZ.into());
                    lanes(cx, "nm_f16_to_f32", &|x| format!("nm_f16_to_f32(aw_tex_f16_rtz({}))", x))?
                }
                None => {
                    return Err(refuse(
                        "texture-write-rounding",
                        c,
                        format!(
                            "a float write to {} with the native rounding mode: the A10X's native rounding of texture \
                             writes is not measured",
                            f.name
                        ),
                    ));
                }
            }
        }
        "r8unorm" | "rg8unorm" | "rgba8unorm" | "bgra8unorm" => unorm(cx, [255; 4])?,
        "r16unorm" | "rg16unorm" | "rgba16unorm" => unorm(cx, [65535; 4])?,
        "rgb10a2unorm" => unorm(cx, [1023, 1023, 1023, 3])?,
        "r8snorm" | "rg8snorm" | "rgba8snorm" => snorm(cx, 8)?,
        "r16snorm" | "rg16snorm" | "rgba16snorm" => snorm(cx, 16)?,
        "r8sint" | "rg8sint" | "rgba8sint" => sat(-128, 127, true),
        "r16sint" | "rg16sint" | "rgba16sint" => sat(-32768, 32767, true),
        "r8uint" | "rg8uint" | "rgba8uint" => sat(0, 255, false),
        "r16uint" | "rg16uint" | "rgba16uint" => sat(0, 65535, false),
        "r32sint" | "rg32sint" | "rgba32sint" | "r32uint" | "rg32uint" | "rgba32uint" => ints.clone(),
        other => {
            return Err(refuse(
                "texture-format-shim",
                c,
                format!("a write to {} stored as {}: its rounding is not reproduced in the shader", f.name, other),
            ));
        }
    })
}

const F16_RTZ: &str = "// binary32 bits to binary16 bits rounding toward zero (MSL 4.1 section 1.6.7, rtz): finite values never \
round to infinity; binary16 subnormals truncate.\n\
fn aw_tex_f16_rtz(a: u32) -> u32 {\n    let s = (a >> 16u) & 0x8000u;\n    let e = i32((a >> 23u) & 0xffu);\n    \
let f = a & 0x7fffffu;\n    if (e == 255) {\n        return select(s | 0x7e00u | (f >> 13u), s | 0x7c00u, f == 0u);\n    }\n    \
let h = e - 112;\n    if (h >= 31) {\n        return s | 0x7bffu;\n    }\n    if (h >= 1) {\n        \
return s | (u32(h) << 10u) | (f >> 13u);\n    }\n    if (e == 0 || 126 - e >= 32) {\n        return s;\n    }\n    \
return s | ((f | 0x800000u) >> u32(126 - e));\n}\n";

/// A constant integer operand as the unsigned value of its width (a `uint` or `ushort` level).
fn const_uint(m: &Module, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    let bits = match m.types.get(m.constant(c).ty) {
        Type::Int(b) => *b,
        _ => return None,
    };
    let x = const_int(m, v)? as u64;
    Some(if bits >= 64 { x } else { x & ((1u64 << bits) - 1) })
}

/// A write: `textureStore` to the storage binding of the level written, under the bounds of MSL 4.1 section 6.13 ("A
/// read from or write to a texture is out-of-bounds if and only if ... the level of detail argument is
/// out-of-bounds"; "an out-of-bounds write to a texture is ignored"). The level is out of bounds when it is not below
/// the mipmap level count of the bound texture, a view's own count from its first level. A level computed at run time
/// selects among a storage binding of each level ([`crate::api::BindingSource::MetalTextureLevel`]); a constant level
/// binds that level alone.
///
/// Without a bound texture at the slot the level count is not known: a constant level binds that level, and a level
/// computed at run time selects level 0 alone, the one level of the executor's placeholder. The executor refuses a
/// bound texture whose levels the variant's bindings do not cover.
fn write(cx: &mut dyn RuleCx, call: &CallSite<'_>, n: &Name, t: &Tex) -> R<()> {
    let c = call.name;
    if n.depth {
        return Err(call_form(c, "a depth texture write"));
    }
    let mut o = Operands { args: call.args, k: 1, construct: c };
    let coord = o.next()?;
    let face = if n.dim == Dim::Cube { Some(o.next()?) } else { None };
    let slice = if n.dim.arrayed() { Some(o.next()?) } else { None };
    let value = o.next()?;
    let lod = o.next()?;
    let acc = o.next()?;
    o.done()?;
    if access_code(cx, c, acc)? != 2 || t.access != Access::Write {
        return Err(call_form(c, "a write without write access"));
    }
    if n.elem == Elem::None {
        return Err(call_form(c, "no texel type"));
    }
    let level = const_uint(cx.module(), lod);
    let written: Vec<u32> = match (level, levels(t)) {
        // A constant level out of range: the write is ignored.
        (Some(l), Some(count)) if l >= count => return Ok(()),
        (Some(l), _) => vec![u32::try_from(l).map_err(|_| call_form(c, "a level operand wider than 32 bits"))?],
        (None, count) => {
            let count = u32::try_from(count.unwrap_or(1))
                .map_err(|_| refuse("texture-slot-mismatch", c, "a bound texture of more than 2^32 levels"))?;
            (0..count).collect()
        }
    };
    let mut views = Vec::new();
    for &l in &written {
        views.push((l, bind_level(cx, n, t, c, l)?));
    }
    let p = cx.fresh();
    let coord_e = cx.operand(coord)?;
    let coords = match n.dim {
        Dim::D1 | Dim::D1Array => format!("vec2<u32>({}, 0u)", coord_e),
        _ => coord_e,
    };
    cx.line(format!("let {}_c = {};", p, coords));
    let mut layer = String::new();
    let mut extra = String::new();
    if let Some(f) = face {
        let fe = cx.operand(f)?;
        cx.line(format!("let {}_f = {};", p, fe));
        extra = format!(" && {}_f < 6u", p);
        layer = format!(", {}_f", p);
    }
    let sliced = match slice {
        Some(s) => {
            let se = cx.operand(s)?;
            cx.line(format!("let {}_s = {};", p, se));
            layer = format!(", {}_s", p);
            true
        }
        None => false,
    };
    let val = cx.operand(value)?;
    cx.line(format!("let {}_v = {};", p, val));
    let stored = store_value(cx, t, n, &format!("{}_v", p), c)?;
    cx.line(format!("let {}_x = {};", p, stored));
    // Each level's own extent and layers bound the coordinates and the slice; an out-of-bounds write is ignored.
    let store = |cx: &mut dyn RuleCx, v: &str| {
        let mut cond = format!("all({}_c < textureDimensions({})){}", p, v, extra);
        if sliced {
            cond = format!("{} && {}_s < textureNumLayers({})", cond, p, v);
        }
        cx.line(format!("if ({}) {{", cond));
        cx.line(format!("textureStore({}, {}_c{}, {}_x);", v, p, layer, p));
        cx.line("}".into());
    };
    if level.is_some() {
        store(cx, &views[0].1);
        return Ok(());
    }
    let l = cx.operand(lod)?;
    cx.line(format!("switch ({}) {{", l));
    for (k, v) in &views {
        cx.line(format!("case {}u: {{", k));
        store(cx, v);
        cx.line("}".into());
    }
    cx.line("default: {}".into());
    cx.line("}".into());
    Ok(())
}

// Sampling --------------------------------------------------------------------------------------------------------

/// The level of detail of a sample.
enum Lod {
    Implicit,
    Bias(Value),
    Level(Value),
    Grad(Value, Value),
}

/// How a sample or gather is lowered.
enum Path {
    /// The WGSL sampling function with the sampler bound.
    Native,
    /// Texel fetches of level 0 with the address modes applied in the shader.
    Fetch(SamplerSlot),
}

/// The descriptor of an unbound sampler: sampling with it is undefined, and the default `MTLSamplerDescriptor`
/// (nearest filters, no mipmaps, clamp to edge, normalized coordinates) stands for it.
fn default_sampler() -> SamplerSlot {
    SamplerSlot {
        index: u64::MAX,
        min_filter: 0,
        mag_filter: 0,
        mip_filter: 0,
        address_modes: [0; 3],
        border_color: 0,
        normalized_coordinates: true,
        compare_function: 0,
        max_anisotropy: 1,
        lod_min_clamp: 0.0,
        lod_max_clamp: f32::MAX,
        lod_average: false,
    }
}

fn axes(dim: Dim) -> usize {
    match dim {
        Dim::D1 | Dim::D1Array => 1,
        Dim::D3 | Dim::Cube | Dim::CubeArray => 3,
        _ => 2,
    }
}

/// Decide the path of a sample or gather: texel fetches when WebGPU cannot sample the texture with the sampler.
fn path(t: &Tex, n: &Name, s: &Smp, gather: bool, c: &str) -> R<Path> {
    let d = s.desc.unwrap_or_else(default_sampler);
    check_sampler(&d, axes(n.dim), c)?;
    let integer = t.component != Component::Float && !gather;
    let emulated_mode = d.address_modes[..axes(n.dim)].contains(&4);
    let fetch = integer || integer_shim(t) || emulated_mode || !d.normalized_coordinates;
    if !fetch {
        if shim(t) == Shim::Srgb8 && !gather && (d.min_filter != 0 || d.mag_filter != 0 || d.mip_filter == 2) {
            return Err(refuse(
                "texture-fetch-filtering",
                c,
                "linear filtering of R8Unorm_sRGB or RG8Unorm_sRGB: Metal filters after the sRGB conversion (MSL 4.1 \
                 section 8.7.7), which the stored codes cannot be filtered for",
            ));
        }
        return Ok(Path::Native);
    }
    let why = if integer {
        "an integer texture"
    } else if integer_shim(t) {
        "a format stored as integer words"
    } else if emulated_mode {
        "the clampToZero address mode"
    } else {
        "unnormalized coordinates"
    };
    if matches!(n.op, Op2::SampleCompare | Op2::GatherCompare) || n.depth {
        return Err(refuse("texture-fetch-filtering", c, format!("a depth texture with {}", why)));
    }
    if matches!(n.dim, Dim::Cube | Dim::CubeArray) {
        return Err(refuse("texture-fetch-filtering", c, format!("a cube texture with {}", why)));
    }
    if !gather {
        if d.mip_filter != 0 || d.max_anisotropy > 1 {
            return Err(tracked::refuse_row(
                "address-modes-with-mips-or-anisotropy",
                RefusalKind::NoExactLowering,
                c.to_string(),
                None,
                format!("{} sampled with a mipmap filter or anisotropy", why),
                "WebGPU GPUSamplerDescriptor; plan 5.7",
            ));
        }
        if d.min_filter != 0 || d.mag_filter != 0 {
            return Err(refuse(
                "texture-fetch-filtering",
                c,
                format!(
                    "{} sampled with linear filtering: Metal states no filter-weight precision (WebMetal formats.toml \
                     question 5.7-filter-weight-precision)",
                    why
                ),
            ));
        }
    }
    Ok(Path::Fetch(d))
}

/// One axis of a nearest fetch: the integer texel index along an axis of `n` texels for the unwrapped index `i` (an
/// integer-valued f32) plus a constant offset, under MTLSamplerAddressMode `mode`; -1 when clampToZero falls outside.
fn wrap(i: &str, off: i32, n: &str, mode: u64) -> String {
    // Indices beyond +-2^30 are clamped first: no f32 coordinate that large addresses a texel exactly.
    let k = format!("(i32(clamp({}, -1073741824.0, 1073741824.0)) + {})", i, off);
    match mode {
        2 => format!("((({} % {n}) + {n}) % {n})", k, n = n),
        3 => format!("aw_tex_mirror({}, {})", k, n),
        4 => format!("select(-1, {k}, {k} >= 0 && {k} < {n})", k = k, n = n),
        _ => format!("clamp({}, 0, {} - 1)", k, n),
    }
}

const MIRROR: &str = "// MTLSamplerAddressModeMirrorRepeat on a texel index: the period of 2n texels mirrored at n.\n\
fn aw_tex_mirror(k: i32, n: i32) -> i32 {\n    let p = 2 * n;\n    let m = ((k % p) + p) % p;\n    \
return select(p - 1 - m, m, m < n);\n}\n";

/// The level-0 texel fetch of a nearest sample at `coord` (the AIR coordinate operand's WGSL f32 value) with the
/// fetch path's sampler; returns the result-element bits.
#[allow(clippy::too_many_arguments)]
fn fetch_sample(
    cx: &mut dyn RuleCx,
    t: &Tex,
    n: &Name,
    v: &str,
    d: &SamplerSlot,
    coord: &str,
    slice: Option<String>,
    off: &Option<Vec<i32>>,
) -> R<String> {
    cx.module_item("aw_tex_mirror", MIRROR.into());
    let p = cx.fresh();
    let lanes = axes(n.dim);
    let dims = match n.dim {
        Dim::D1 | Dim::D1Array | Dim::D2 | Dim::D2Array => "vec2<i32>",
        _ => "vec3<i32>",
    };
    cx.line(format!("let {}_n = {}(textureDimensions({}, 0u));", p, dims, v));
    // 1D coordinates arrive as (x, 0.5) on the stored row; only x is addressed.
    let cvec = coord.to_string();
    let scale = if d.normalized_coordinates {
        format!(" * vec{}<f32>({}_n)", if n.dim == Dim::D3 { 3 } else { 2 }, p)
    } else {
        String::new()
    };
    cx.line(format!("let {}_x = floor({}{});", p, cvec, scale));
    let names = ["x", "y", "z"];
    let mut idx = Vec::new();
    for (a, lane) in names.iter().enumerate().take(lanes) {
        let o = off.as_ref().map(|v| v[a]).unwrap_or(0);
        let w = wrap(&format!("{}_x.{}", p, lane), o, &format!("{}_n.{}", p, lane), d.address_modes[a]);
        cx.line(format!("let {}_i{} = {};", p, a, w));
        idx.push(format!("{}_i{}", p, a));
    }
    if lanes == 1 {
        idx.push("0".into());
    }
    let inside = idx.iter().map(|i| format!("{} >= 0", i)).collect::<Vec<_>>().join(" && ");
    let cvec_i = format!("vec{}<i32>({})", idx.len(), idx.join(", "));
    let layer = match &slice {
        Some(s) => format!(", min({}, textureNumLayers({}) - 1u)", s, v),
        None => String::new(),
    };
    cx.line(format!("var {}_r = {};", p, default_bits(t, n.elem)));
    cx.line(format!("if ({}) {{", inside));
    cx.line(format!("let {}_t = textureLoad({}, {}{}, 0);", p, v, cvec_i, layer));
    let bits = texel_bits(cx, t, &format!("{}_t", p), n.elem)?;
    cx.line(format!("{}_r = {};", p, bits));
    cx.line("}".into());
    Ok(format!("{}_r", p))
}

/// The WGSL coordinates of a sample: f32 vectors from the operand's bits; 1D textures sample the middle of their one
/// row.
fn coords(cx: &mut dyn RuleCx, n: &Name, v: Value) -> R<String> {
    let e = cx.operand(v)?;
    Ok(match n.dim {
        Dim::D1 | Dim::D1Array => format!("vec2<f32>(bitcast<f32>({}), 0.5)", e),
        Dim::D2 | Dim::D2Array => format!("bitcast<vec2<f32>>({})", e),
        _ => format!("bitcast<vec3<f32>>({})", e),
    })
}

fn sample(cx: &mut dyn RuleCx, call: &CallSite<'_>, n: &Name, t: &Tex) -> R<String> {
    let c = call.name;
    let compare = n.op == Op2::SampleCompare;
    let m = cx.module();
    let mut o = Operands { args: call.args, k: 1, construct: c };
    let sv = o.next()?;
    if n.depth {
        depth_flag(cx, c, o.next()?)?;
    }
    let coord = o.next()?;
    let slice = if n.dim.arrayed() { Some(o.next()?) } else { None };
    let reference = if compare { Some(o.next()?) } else { None };
    let has_offset_slot = !matches!(n.dim, Dim::Cube | Dim::CubeArray);
    let lanes = match n.dim {
        Dim::D1 | Dim::D1Array => 1,
        Dim::D3 | Dim::Cube | Dim::CubeArray => 3,
        _ => 2,
    };
    let (lod, off) = if n.grad {
        let dx = o.next()?;
        let dy = o.next()?;
        let min_lod = o.next()?;
        let off = if has_offset_slot { offset(cx, c, o.next()?, o.next()?, lanes)? } else { None };
        check_min_lod(m, c, min_lod)?;
        (Lod::Grad(dx, dy), off)
    } else {
        let off = if has_offset_slot { offset(cx, c, o.next()?, o.next()?, lanes)? } else { None };
        let is_level = const_int(m, o.next()?).ok_or_else(|| call_form(c, "a non-constant level flag"))?;
        let value = o.next()?;
        let min_lod = o.next()?;
        check_min_lod(m, c, min_lod)?;
        let lod = if is_level != 0 {
            Lod::Level(value)
        } else if const_f32(m, value) == Some(0.0) {
            Lod::Implicit
        } else {
            Lod::Bias(value)
        };
        (lod, off)
    };
    let acc = o.next()?;
    o.done()?;
    if access_code(cx, c, acc)? != 0 || t.access != Access::Sample {
        return Err(call_form(c, "a sample of a texture without sample access"));
    }
    let s = sampler(cx, c, sv)?;
    let u = uses(cx, None, Some(&s.source));
    if u.compare && u.plain {
        return Err(refuse(
            "texture-binding-conflict",
            c,
            "one sampler used both by compare functions (a comparison sampler binding) and by other sampling \
             functions",
        ));
    }
    let path = path(t, n, &s, false, c)?;
    let v = bind(cx, n, t, c)?;
    let coord_e = coords(cx, n, coord)?;
    let slice_e = match slice {
        Some(s) => Some(cx.operand(s)?),
        None => None,
    };
    if let Path::Fetch(d) = path {
        let bits = fetch_sample(cx, t, n, &v, &d, &coord_e, slice_e, &off)?;
        let sw = swizzle(cx, t);
        let out = apply_swizzle(&bits, sw, n.elem);
        return result(cx, call, out);
    }
    let frag = fragment(cx);
    let st = sample_type(cx, t, n.depth);
    let sty = if compare { SamplerBindingType::Comparison } else { plain_sampler_type(&s) };
    if sty == SamplerBindingType::Filtering && st != S::Float {
        return Err(unfilterable(t, c));
    }
    let sname = bind_sampler(cx, &s, sty)?;
    let arr = slice_e.map(|s| format!(", {}", s)).unwrap_or_default();
    let offs = wgsl_offset(&off);
    let head = format!("{}, {}, {}{}", v, sname, coord_e, arr);
    let dref = match reference {
        Some(r) => format!(", bitcast<f32>({})", cx.operand(r)?),
        None => String::new(),
    };
    let depth_level = |cx: &mut dyn RuleCx, l: Value| -> R<String> {
        match const_f32(cx.module(), l) {
            Some(x) if x >= 0.0 && x.fract() == 0.0 && x < 32.0 => Ok(format!("{}u", x as u32)),
            _ => Err(refuse(
                "texture-depth-lod",
                c,
                "a depth texture sampled or compared at a computed level: WGSL samples depth textures at integer \
                 levels and compares at level 0 only",
            )),
        }
    };
    let call_e = if compare {
        let level0 = |cx: &mut dyn RuleCx, l: Value| -> R<()> {
            if depth_level(cx, l)? != "0u" {
                return Err(refuse("texture-depth-lod", c, "a comparison at a level other than 0"));
            }
            Ok(())
        };
        match lod {
            Lod::Implicit if frag => {
                cx.derivatives();
                format!("textureSampleCompare({}{}{})", head, dref, offs)
            }
            Lod::Implicit => format!("textureSampleCompareLevel({}{}{})", head, dref, offs),
            Lod::Level(l) => {
                level0(cx, l)?;
                format!("textureSampleCompareLevel({}{}{})", head, dref, offs)
            }
            Lod::Bias(_) | Lod::Grad(..) => {
                return Err(refuse(
                    "texture-depth-lod",
                    c,
                    "a comparison with a bias or gradients: WGSL compares at the implicit level or level 0 only",
                ));
            }
        }
    } else if n.depth {
        match lod {
            Lod::Implicit if frag => {
                cx.derivatives();
                format!("textureSample({}{})", head, offs)
            }
            Lod::Implicit => format!("textureSampleLevel({}, 0u{})", head, offs),
            Lod::Level(l) => {
                let l = depth_level(cx, l)?;
                format!("textureSampleLevel({}, {}{})", head, l, offs)
            }
            Lod::Bias(_) | Lod::Grad(..) => {
                return Err(refuse(
                    "texture-depth-lod",
                    c,
                    "a depth texture sampled with a bias or gradients, which WGSL has no depth form of",
                ));
            }
        }
    } else {
        match lod {
            Lod::Implicit if frag => {
                cx.derivatives();
                format!("textureSample({}{})", head, offs)
            }
            // Outside fragment functions the default LOD is 0 (MSL 4.1 section 6.13) and a bias adds to it.
            Lod::Implicit => format!("textureSampleLevel({}, 0.0{})", head, offs),
            Lod::Bias(b) if frag => {
                cx.derivatives();
                let b = cx.operand(b)?;
                format!("textureSampleBias({}, bitcast<f32>({}){})", head, b, offs)
            }
            Lod::Bias(b) | Lod::Level(b) => {
                let b = cx.operand(b)?;
                format!("textureSampleLevel({}, bitcast<f32>({}){})", head, b, offs)
            }
            Lod::Grad(dx, dy) => {
                let g = |cx: &mut dyn RuleCx, x: Value| -> R<String> {
                    let e = cx.operand(x)?;
                    Ok(if lanes == 2 {
                        format!("bitcast<vec2<f32>>({})", e)
                    } else {
                        format!("bitcast<vec3<f32>>({})", e)
                    })
                };
                let (dx, dy) = (g(cx, dx)?, g(cx, dy)?);
                format!("textureSampleGrad({}, {}, {}{})", head, dx, dy, offs)
            }
        }
    };
    if n.depth {
        let out = format!("bitcast<u32>({})", call_e);
        return result(cx, call, out);
    }
    let p = cx.fresh();
    cx.line(format!("let {} = {};", p, call_e));
    let bits = texel_bits(cx, t, &p, n.elem)?;
    let sw = swizzle(cx, t);
    let out = apply_swizzle(&bits, sw, n.elem);
    result(cx, call, out)
}

fn check_min_lod(m: &Module, c: &str, v: Value) -> R<()> {
    if const_f32(m, v) == Some(0.0) {
        return Ok(());
    }
    Err(refuse(
        "texture-min-lod-clamp",
        c,
        "min_lod_clamp: the A10X reports supportsShaderMinLODClamp = false, what its compiler does with the option is \
         not established, and WGSL sampling takes no minimum LOD",
    ))
}

fn unfilterable(t: &Tex, c: &str) -> Refusal {
    let name = t.format.map(|(f, _)| f.name).unwrap_or("of a depth texture argument");
    if matches!(t.format.map(|(f, _)| f.value), Some(20 | 22 | 60 | 62 | 110 | 112)) {
        return tracked::refuse_row(
            "unorm16-filtering",
            RefusalKind::NoExactLowering,
            c.to_string(),
            None,
            format!("{} sampled with linear filtering", name),
            "WebGPU 26.1 plain color formats",
        );
    }
    refuse(
        "texture-unfilterable",
        c,
        format!("a linear-filtering sampler with the bound format {}, which the host cannot filter", name),
    )
}

// Gathers ---------------------------------------------------------------------------------------------------------

fn gather(cx: &mut dyn RuleCx, call: &CallSite<'_>, n: &Name, t: &Tex) -> R<String> {
    let c = call.name;
    let compare = n.op == Op2::GatherCompare;
    let m = cx.module();
    let mut o = Operands { args: call.args, k: 1, construct: c };
    let sv = o.next()?;
    if n.depth {
        depth_flag(cx, c, o.next()?)?;
    }
    let coord = o.next()?;
    let slice = if n.dim.arrayed() { Some(o.next()?) } else { None };
    let reference = if compare { Some(o.next()?) } else { None };
    let off = if matches!(n.dim, Dim::D2 | Dim::D2Array) { offset(cx, c, o.next()?, o.next()?, 2)? } else { None };
    let component = if n.depth {
        0
    } else {
        let k = const_int(m, o.next()?).ok_or_else(|| call_form(c, "a non-constant component"))?;
        if !(0..4).contains(&k) {
            return Err(call_form(c, "a component outside 0 to 3"));
        }
        k as usize
    };
    let acc = o.next()?;
    o.done()?;
    if access_code(cx, c, acc)? != 0 || t.access != Access::Sample {
        return Err(call_form(c, "a gather of a texture without sample access"));
    }
    let elem = if n.depth { Elem::F32 } else { n.elem };
    let s = sampler(cx, c, sv)?;
    let u = uses(cx, None, Some(&s.source));
    if u.compare && u.plain {
        return Err(refuse(
            "texture-binding-conflict",
            c,
            "one sampler used both by compare functions (a comparison sampler binding) and by other sampling \
             functions",
        ));
    }
    let path = path(t, n, &s, true, c)?;
    let v = bind(cx, n, t, c)?;
    let coord_e = coords(cx, n, coord)?;
    let slice_e = match slice {
        Some(s) => Some(cx.operand(s)?),
        None => None,
    };
    let sw = if n.depth { None } else { swizzle(cx, t) };
    let effective = sw.map(|s| s[component]).unwrap_or(component as u8 + 2);
    if let Path::Fetch(d) = path {
        return gather_fetch(cx, call, t, &v, &d, &coord_e, slice_e, &off, effective, elem);
    }
    let st = sample_type(cx, t, n.depth);
    let sty = if compare { SamplerBindingType::Comparison } else { plain_sampler_type(&s) };
    if sty == SamplerBindingType::Filtering && !matches!(st, S::Float | S::Sint | S::Uint) {
        return Err(unfilterable(t, c));
    }
    if sty == SamplerBindingType::Filtering && matches!(st, S::Sint | S::Uint) {
        return Err(refuse(
            "texture-unfilterable",
            c,
            "a gather of an integer texture with a linear-filtering sampler: WebGPU binds integer textures with \
             non-filtering samplers only",
        ));
    }
    let sname = bind_sampler(cx, &s, sty)?;
    let arr = slice_e.map(|s| format!(", {}", s)).unwrap_or_default();
    let offs = wgsl_offset(&off);
    if n.depth {
        let call_e = match reference {
            Some(r) => {
                let r = cx.operand(r)?;
                format!("textureGatherCompare({}, {}, {}{}, bitcast<f32>({}){})", v, sname, coord_e, arr, r, offs)
            }
            None => format!("textureGather({}, {}, {}{}{})", v, sname, coord_e, arr, offs),
        };
        return result(cx, call, format!("bitcast<vec4<u32>>({})", call_e));
    }
    // A component the swizzle makes constant gathers that constant from each texel.
    let out = match effective {
        0 => "vec4<u32>(0u)".to_string(),
        1 => format!("vec4<u32>({})", one_bits(elem)),
        k => {
            let p = cx.fresh();
            cx.line(format!("let {} = textureGather({}, {}, {}, {}{}{});", p, k - 2, v, sname, coord_e, arr, offs));
            gathered_bits(cx, t, &p, elem, (k - 2) as usize)?
        }
    };
    result(cx, call, out)
}

/// Four gathered values of stored channel `ch` as result-element bits: each is the channel of one texel, converted as
/// a read of that texel converts it.
fn gathered_bits(cx: &mut dyn RuleCx, t: &Tex, g: &str, e: Elem, ch: usize) -> R<String> {
    match shim(t) {
        Shim::None | Shim::AlphaInRed => match t.component {
            Component::Float => float_bits(cx, g, e),
            _ => Ok(int_bits(g, e)),
        },
        Shim::Srgb8 => {
            if ch >= 2 {
                // Stored formats have no such channel; the decoded texel has blue 0 and alpha 1.
                return Ok(format!("vec4<u32>({})", if ch == 2 { "0u" } else { one_bits(e) }));
            }
            let f = if e == Elem::F16 { "nm_srgb_f16" } else { "nm_srgb_f32" };
            cx.prelude(f)?;
            let l = |k: &str| format!("{}(u32(round({}.{} * 255.0)))", f, g, k);
            Ok(format!("vec4<u32>({}, {}, {}, {})", l("x"), l("y"), l("z"), l("w")))
        }
        _ => unreachable!("integer-word formats gather by fetches"),
    }
}

/// A gather by four level-0 fetches: the texels around the coordinate at half-texel offsets (-,+), (+,+), (+,-),
/// (-,-) (MSL 4.1 section 6.13), each addressed by the sampler's modes.
#[allow(clippy::too_many_arguments)]
fn gather_fetch(
    cx: &mut dyn RuleCx,
    call: &CallSite<'_>,
    t: &Tex,
    v: &str,
    d: &SamplerSlot,
    coord: &str,
    slice: Option<String>,
    off: &Option<Vec<i32>>,
    effective: u8,
    e: Elem,
) -> R<String> {
    if effective < 2 {
        let k = if effective == 0 { "0u".to_string() } else { one_bits(e).to_string() };
        return result(cx, call, format!("vec4<u32>({})", k));
    }
    cx.module_item("aw_tex_mirror", MIRROR.into());
    let ch = ["x", "y", "z", "w"][(effective - 2) as usize];
    let p = cx.fresh();
    cx.line(format!("let {}_n = vec2<i32>(textureDimensions({}, 0u));", p, v));
    let scale = if d.normalized_coordinates { format!(" * vec2<f32>({}_n)", p) } else { String::new() };
    cx.line(format!("let {}_x = floor({}{} - vec2<f32>(0.5));", p, coord, scale));
    let layer = match &slice {
        Some(s) => format!(", min({}, textureNumLayers({}) - 1u)", s, v),
        None => String::new(),
    };
    let (ox, oy) = off.as_ref().map(|o| (o[0], o[1])).unwrap_or((0, 0));
    let mut lanes = Vec::new();
    // x: (i0, j1), y: (i1, j1), z: (i1, j0), w: (i0, j0), with j down the texture.
    for (k, (di, dj)) in [(0, 1), (1, 1), (1, 0), (0, 0)].iter().enumerate() {
        let i = wrap(&format!("{}_x.x", p), ox + di, &format!("{}_n.x", p), d.address_modes[0]);
        let j = wrap(&format!("{}_x.y", p), oy + dj, &format!("{}_n.y", p), d.address_modes[1]);
        cx.line(format!("let {}_i{} = {};", p, k, i));
        cx.line(format!("let {}_j{} = {};", p, k, j));
        cx.line(format!("var {}_r{} = {};", p, k, default_bits(t, e)));
        cx.line(format!("if ({p}_i{k} >= 0 && {p}_j{k} >= 0) {{", p = p, k = k));
        cx.line(format!(
            "let {p}_t{k} = textureLoad({v}, vec2<i32>({p}_i{k}, {p}_j{k}){layer}, 0);",
            p = p,
            k = k,
            v = v,
            layer = layer
        ));
        let bits = texel_bits(cx, t, &format!("{}_t{}", p, k), e)?;
        cx.line(format!("{}_r{} = {};", p, k, bits));
        cx.line("}".into());
        lanes.push(format!("{}_r{}.{}", p, k, ch));
    }
    result(cx, call, format!("vec4<u32>({})", lanes.join(", ")))
}
