// Moving binary16 carriers (the low 16 bits of a u32) in and out of native arithmetic: WGSL f16 when the device has
// shader-f16 (these items need `enable f16;`), binary32 otherwise.

// The f16 value of a carrier and back. unpack2x16float widens the low half exactly and pack2x16float narrows a value
// binary16 holds exactly (WGSL 17.10.7, 17.9.9, 15.7.6), so both are exact for finite values; infinities and NaNs,
// for which the packing result is indeterminate, do not occur in the fast-math code that uses them (MSL 8.1).
fn nm_h(x: u32) -> f16 {
    return f16(unpack2x16float(x).x);
}

fn nm_hb(h: f16) -> u32 {
    return pack2x16float(vec2<f32>(f32(h), 0.0)) & 0xffffu;
}

// Without shader-f16: the exactly widened binary32 value, and a binary32 result rounded to nearest even binary16
// with integer operations (MSL 8.6; numerics.toml question 5.3-quantize-to-f16).
fn nm_hw(x: u32) -> f32 {
    return bitcast<f32>(nm_f16_to_f32(x));
}

fn nm_hn(v: f32) -> u32 {
    return nm_f32_to_f16(bitcast<u32>(v));
}
