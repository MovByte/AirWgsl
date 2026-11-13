//! The tessellator as WebGPU compute work ([`WGSL`], `kernel.wgsl`): the executor's half of a patch draw.
//!
//! The factors are read on the GPU, as Metal reads them: a compute pass or an earlier draw of the command buffer may
//! have written them. Three dispatches run in order on one bind group ([`Binding`]):
//!
//! 1. [`ENTRY_COUNT`], `ceil(slots / 64)` workgroups: each patch's point and index counts into the slot buffer, and
//!    whether any patch has a blade the M4 batches (word [`DRAW_BATCHED`] of the draw buffer, set non-zero);
//! 2. [`ENTRY_SCAN`], one workgroup: each patch's first point and first index into the slot buffer, the
//!    `drawIndexedIndirect` arguments into words 0 to 4 of the draw buffer, the point count into word
//!    [`DRAW_POINTS`];
//! 3. [`ENTRY_GENERATE`], `ceil(slots / 64)` workgroups: the points (each a [`VERTEX_WORDS`]-word record: u and v in
//!    16.16 fixed point, the patch identifier, the instance identifier) and the indices (into the point records).
//!
//! A slot is one patch of one instance, instance-major: Metal draws every patch of an instance before the next
//! instance (measured, V1 `tessellation.draws`). The draw then is `drawIndexedIndirect` over the index buffer with
//! the draw buffer's arguments; the vertex stage reads its point from the records by `vertex_index`.
//!
//! The draw buffer's word [`DRAW_BATCHED`] must be zero before the count dispatch.

use super::factors::{PartitionMode, PatchType, Settings, f32_to_half};

/// The kernels' WGSL source.
pub const WGSL: &str = include_str!("kernel.wgsl");

pub const ENTRY_COUNT: &str = "tess_count";
pub const ENTRY_SCAN: &str = "tess_scan";
pub const ENTRY_GENERATE: &str = "tess_generate";
/// Workgroup size of the count and generate dispatches.
pub const WORKGROUP: u32 = 64;

/// The `u32` words of one point record.
pub const VERTEX_WORDS: u32 = 4;
/// The `u32` words of one slot in the slot buffer.
pub const SLOT_WORDS: u32 = 2;
/// The `u32` words of the draw buffer: the five `drawIndexedIndirect` arguments, the batched flag, the point count.
pub const DRAW_WORDS: u32 = 7;
pub const DRAW_BATCHED: u32 = 5;
pub const DRAW_POINTS: u32 = 6;

/// The kernels' bindings, all in group 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// Uniform, [`Params::bytes`].
    Params = 0,
    /// Read-only storage: the factor buffer, from an aligned base at or before the Metal offset.
    Factors = 1,
    /// Read-only storage: the patch index buffer (any buffer of at least four bytes when the draw has none).
    PatchIndices = 2,
    /// Storage, `slots * SLOT_WORDS` words.
    Slots = 3,
    /// Storage, the point records.
    Vertices = 4,
    /// Storage, the indices: also the draw's index buffer (`uint32`).
    Indices = 5,
    /// Storage, [`DRAW_WORDS`] words: also the draw's indirect buffer.
    Draw = 6,
}

/// The uniform block of the kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Params {
    pub quad: bool,
    /// `MTLTessellationPartitionMode`.
    pub partition: u32,
    /// Metal's clamp range of the processed factors, binary32 bit patterns.
    pub lower: u32,
    pub upper: u32,
    pub clockwise: bool,
    pub scale_enabled: bool,
    /// The factor scale as a binary16 bit pattern (`setTessellationFactorScale:` rounds it so).
    pub scale: u16,
    /// `MTLTessellationFactorStepFunction`.
    pub step: u32,
    /// The byte offset of the draw's factors from the factor binding's start.
    pub factor_offset: u32,
    pub instance_stride: u32,
    pub patch_start: u32,
    pub patch_count: u32,
    pub base_instance: u32,
    pub instance_count: u32,
    /// The byte offset of the patch index buffer's first element from its binding's start, when the draw has one.
    pub patch_index_offset: Option<u32>,
}

impl Params {
    /// The pipeline's tessellation state and the draw's patch range; offsets and strides are the caller's.
    pub fn new(settings: &Settings, patch: PatchType) -> Params {
        let (lower, upper) = settings.clamp();
        Params {
            quad: patch == PatchType::Quad,
            partition: match settings.partition {
                PartitionMode::Pow2 => 0,
                PartitionMode::Integer => 1,
                PartitionMode::FractionalOdd => 2,
                PartitionMode::FractionalEven => 3,
            },
            lower: lower.to_bits(),
            upper: upper.to_bits(),
            clockwise: settings.clockwise,
            scale_enabled: settings.scale_enabled,
            ..Params::default()
        }
    }

    /// Set the encoder's factor scale (`setTessellationFactorScale:`).
    pub fn with_scale(mut self, scale: f32) -> Params {
        self.scale = f32_to_half(scale);
        self
    }

    /// Patches times instances.
    pub fn slots(&self) -> u32 {
        self.patch_count * self.instance_count
    }

    /// The uniform block's bytes (the `Params` struct of `kernel.wgsl`: sixteen `u32`).
    pub fn bytes(&self) -> Vec<u8> {
        let words = [
            self.quad as u32,
            self.partition,
            self.lower,
            self.upper,
            self.clockwise as u32,
            self.scale_enabled as u32,
            self.scale as u32,
            self.step,
            self.factor_offset,
            self.instance_stride,
            self.patch_start,
            self.patch_count,
            self.base_instance,
            self.slots(),
            self.patch_index_offset.is_some() as u32,
            self.patch_index_offset.unwrap_or(0),
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// The most points and indices one patch generates with the pipeline's settings: the counts at the largest factors
/// the partition mode allows, which no other factors exceed (checked over the sweep by the tests).
pub fn bounds(settings: &Settings, patch: PatchType) -> (u32, u32) {
    let (_, upper) = settings.clamp();
    let h = f32_to_half(upper);
    let halves = [h; 6];
    let plain = Settings { scale_enabled: false, ..*settings };
    let c = super::counts(&plain, patch, &halves, 1.0);
    (c.points, c.indices)
}
