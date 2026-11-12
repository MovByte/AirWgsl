//! The overlap prover: which primitives of a framebuffer-fetch draw may share a WebGPU pass (plan 5.9, "disjoint
//! batches").
//!
//! The exact way to run a fetching draw is one primitive per pass with a fresh snapshot before each
//! ([`crate::api::PassPlan::PerPrimitive`]). Primitives `[a, b)` may share a pass, with one snapshot taken before
//! them, exactly when no two of them cover a common sample: then every fragment of the group reads in the snapshot
//! the value its sample had after primitive `a - 1`, which is the value it has after its own predecessors in the
//! group too, since none of them touched it. Blending, depth and stencil only meet the attachments at samples a
//! primitive covers, so they see no difference either. The equivalence is between two runs on the same host: Metal's
//! own rasterization precision does not enter it (limits.toml question `5.9-subpixel-precision` asks for the A10X's,
//! which the comparison with real Metal's coverage needs, not this proof).
//!
//! # The proof
//!
//! 1. **Positions.** The vertex function runs on the CPU in the IR interpreter ([`crate::interp`]) for each vertex of
//!    the draw. Its clip-space position equals the GPU's bit for bit when every operation it depends on is lowered
//!    exactly: integer, memory and bit operations, and floating-point operations the numerics lowering implements
//!    with the soft-float library or exact integer code ([`crate::num::How::SoftFloat`], [`crate::num::How::Exact`]).
//!    Native WGSL arithmetic (fast-math rows) may differ from the correctly rounded result (fusion, flushed
//!    subnormals), so a vertex function whose position depends on it gets no prover. The function is first sliced to
//!    the position: the other members of the returned value are replaced by `undef` and dead code is removed, so
//!    varyings computed with any operation do not disqualify it.
//! 2. **Framebuffer coordinates.** WebGPU 23.2.5 maps a clip position `p` to `vp.x + 0.5 * (p.x / p.w + 1) *
//!    vp.width`, `vp.y + 0.5 * (1 - p.y / p.w) * vp.height`, as Metal does (WebMetal limits.toml question
//!    `5.10-clip-space-and-framebuffer-origin`). With every `w > 0` the projected primitive is the convex hull of its
//!    projected vertices, and clipping only removes parts of it. A primitive with a vertex at `w <= 0`, a position
//!    that is not finite, or a coordinate beyond 2^23 pixels proves nothing and is drawn alone.
//! 3. **Coverage.** WebGPU 23.2.5.4 includes a sample strictly inside a polygon and excludes one strictly outside;
//!    one on an edge is left to the rasterizer's precision, and the transform of step 2 is computed in binary32. Each
//!    primitive's footprint is therefore taken as every pixel within `d` of its exact region (in the maximum norm),
//!    with `d = 1 + 2^-10 * m` for triangles and points (`m` the largest coordinate magnitude in pixels of its
//!    vertices) and one pixel more for lines, whose algorithm WebGPU 23.2.5.2 leaves open (a 1-pixel-wide rectangle
//!    or Bresenham's, both within a pixel of the segment). The margin is many times the snapping of any rasterizer
//!    with sub-pixel precision and the error of binary32 evaluation, and makes the footprint contain the samples the
//!    host covers. Culling, scissoring and the viewport rectangle only remove coverage and are ignored.
//! 4. **Groups.** In primitive order, a primitive joins the current group when its footprint shares no pixel with
//!    the group's; otherwise it starts the next group. A primitive drawn alone ends the group before it and forms its
//!    own.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::air::{EntryPoint, OutputRole, Role, Slot};
use crate::api::HostCaps;
use crate::interp::{self, Memory, RegionKind, Scalar, Val};
use crate::ir::{ConstKind, Constant, GlobalId, Module, Op, Type, TypeId, Value};
use crate::num::{self, How, Rounding};

/// What the prover supplies for an argument of the vertex function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgInput {
    /// The vertex buffer argument table slot `index`, in address space `space` (1 device, 2 constant).
    Buffer { index: u64, space: u32 },
    /// `[[vertex_id]]`.
    VertexId,
    /// `[[instance_id]]`.
    InstanceId,
    /// `[[base_vertex]]`.
    BaseVertex,
    /// `[[base_instance]]`.
    BaseInstance,
    /// An argument the sliced function does not use.
    Unused,
}

struct Inner {
    /// SHA-256 of the inputs the prover was built from (the stage modules, the entry point, the constants).
    key: [u8; 32],
    module: Module,
    function: usize,
    args: Vec<ArgInput>,
    /// The member of the returned struct holding the position, or `None` when the function returns the position.
    position: Option<u32>,
}

/// The overlap prover of a pipeline's vertex function; cheap to clone.
#[derive(Clone)]
pub struct OverlapProver(Arc<Inner>);

impl PartialEq for OverlapProver {
    fn eq(&self, other: &OverlapProver) -> bool {
        self.0.key == other.0.key
    }
}

impl Eq for OverlapProver {}

impl fmt::Debug for OverlapProver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key: String = self.0.key.iter().map(|b| format!("{:02x}", b)).collect();
        write!(f, "OverlapProver({}, {:?})", key, self.0.args)
    }
}

/// The primitive class of a draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Point,
    Line,
    Triangle,
}

impl Topology {
    pub fn vertices(self) -> usize {
        match self {
            Topology::Point => 1,
            Topology::Line => 2,
            Topology::Triangle => 3,
        }
    }
}

/// One primitive of a draw: the `[[vertex_id]]` values of its vertices (the first [`Topology::vertices`] are used),
/// as the vertex function sees them (index plus base vertex for an indexed draw), and its `[[instance_id]]` (which
/// includes the base instance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Primitive {
    pub vertices: [u32; 3],
    pub instance: u32,
}

/// The viewport of a draw (`MTLViewport` origin and size), in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// What the prover needs of a draw besides its primitives.
pub struct Draw<'a> {
    pub topology: Topology,
    /// The vertex stage's bound buffers: (argument table index, the buffer's bytes from the binding offset to its
    /// end).
    pub buffers: &'a [(u64, &'a [u8])],
    /// `[[base_vertex]]` and `[[base_instance]]`.
    pub base_vertex: i32,
    pub base_instance: u32,
    pub viewport: Viewport,
    /// Width and height of the render pass's attachments.
    pub target: [u32; 2],
}

impl OverlapProver {
    /// The vertex buffer slots the prover reads.
    pub fn buffers(&self) -> Vec<u64> {
        self.0
            .args
            .iter()
            .filter_map(|a| match a {
                ArgInput::Buffer { index, .. } => Some(*index),
                _ => None,
            })
            .collect()
    }

    /// What the prover supplies for each argument of the vertex function.
    pub fn arguments(&self) -> &[ArgInput] {
        &self.0.args
    }

    /// The groups of `primitives` (ranges of their indices, in order, covering all of them) that may each be drawn in
    /// one WebGPU pass after one snapshot.
    pub fn groups(&self, draw: &Draw<'_>, primitives: &[Primitive]) -> Vec<Range<usize>> {
        let mut runner = Runner::new(&self.0, draw);
        let footprints: Vec<Option<Footprint>> = primitives
            .iter()
            .map(|p| {
                let mut pts = Vec::new();
                for &v in &p.vertices[..draw.topology.vertices()] {
                    pts.push(runner.position(v, p.instance)?);
                }
                footprint(&pts, draw.topology, &draw.viewport, draw.target)
            })
            .collect();
        group(&footprints, draw.target)
    }
}

// Building ------------------------------------------------------------------------------------------------------------

/// The identity of a prover: SHA-256 of the modules a pipeline is compiled from (each module's hash, entry point,
/// function type and constant values), which determine its canonical vertex function.
pub fn key(modules: &[crate::api::AirModule]) -> [u8; 32] {
    let mut bytes = Vec::new();
    for m in modules {
        bytes.extend_from_slice(&m.hash);
        if let Some(e) = &m.entry {
            bytes.extend_from_slice(&(e.function.len() as u64).to_le_bytes());
            bytes.extend_from_slice(e.function.as_bytes());
            bytes.extend_from_slice(&e.function_type.to_le_bytes());
            for c in &e.constants {
                bytes.extend_from_slice(&c.index.to_le_bytes());
                let name = c.name.as_deref().unwrap_or("");
                bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
                bytes.extend_from_slice(name.as_bytes());
                bytes.extend_from_slice(&c.data_type.to_le_bytes());
                bytes.extend_from_slice(&(c.value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(&c.value);
            }
        }
        bytes.push(0xff);
    }
    crate::mtlb::sha256(&bytes)
}

/// Build the prover of a canonical vertex function, or say why it does not qualify.
pub fn build(
    module: &Module,
    entry: &EntryPoint,
    rounding: &Rounding,
    host: &HostCaps,
    key: [u8; 32],
) -> Result<OverlapProver, String> {
    let f = entry.function as usize;
    let pos = entry
        .outputs
        .iter()
        .position(|o| o.role == OutputRole::Builtin("air.position"))
        .ok_or("the vertex function has no position output")?;
    let mut m = module.clone();
    let ret = m.types.function(m.functions[f].ty).map(|(r, _, _)| r).ok_or("not a function type")?;
    let position = match m.types.get(ret) {
        Type::Struct { .. } | Type::LiteralStruct { .. } => Some(pos as u32),
        _ if entry.outputs.len() == 1 => None,
        _ => return Err("an unexpected return type".into()),
    };
    if let Some(member) = position {
        slice(&mut m, f, member);
    }
    let mut args = Vec::new();
    let used = used_args(&m, f);
    for (i, a) in m.functions[f].args.iter().enumerate() {
        if !used[i] {
            args.push(ArgInput::Unused);
            continue;
        }
        let r = entry.arguments.iter().find(|r| r.position as usize == i).ok_or("an argument without a record")?;
        args.push(match &r.role {
            Role::Buffer(b) => {
                let Slot::Value(index) = b.location.index else {
                    return Err("a buffer index a function constant gives".into());
                };
                let space = b.address_space.or(m.types.pointer_addrspace(a.ty)).unwrap_or(1);
                if !matches!(space, 1 | 2) {
                    return Err(format!("a buffer in address space {}", space));
                }
                ArgInput::Buffer { index: index as u64, space }
            }
            Role::Builtin { key: "air.vertex_id", .. } => ArgInput::VertexId,
            Role::Builtin { key: "air.instance_id", .. } => ArgInput::InstanceId,
            Role::Builtin { key: "air.base_vertex", .. } => ArgInput::BaseVertex,
            Role::Builtin { key: "air.base_instance", .. } => ArgInput::BaseInstance,
            other => return Err(format!("the position depends on {}", crate::wgsl::io::role_key(other))),
        });
    }
    let target = num::Target { rounding, host };
    exact(&mut m, f, target)?;
    Ok(OverlapProver(Arc::new(Inner { key, module: m, function: f, args, position })))
}

fn used_args(m: &Module, f: usize) -> Vec<bool> {
    let func = &m.functions[f];
    let mut used = vec![false; func.args.len()];
    for b in &func.blocks {
        for i in &b.insts {
            for v in func.insts[i.0 as usize].op.operands() {
                if let Value::Arg(a) = v {
                    used[a as usize] = true;
                }
            }
        }
    }
    used
}

/// Replace every member but `member` that an `insertvalue` puts into a value of the function by `undef`, then remove
/// the code only those members needed.
fn slice(m: &mut Module, f: usize, member: u32) {
    let n = m.functions[f].insts.len();
    for k in 0..n {
        let Op::InsertValue { value, indices, .. } = &m.functions[f].insts[k].op else { continue };
        if indices.first() == Some(&member) {
            continue;
        }
        let ty = m.value_type(f, *value);
        let undef = m.intern_const(Constant { ty, kind: ConstKind::Undef });
        if let Op::InsertValue { value, .. } = &mut m.functions[f].insts[k].op {
            *value = Value::Const(undef);
        }
    }
    while crate::passes::dce::dce(m, f) {}
}

fn fmf(f: crate::ir::FastMath) -> num::FastMath {
    use crate::ir::FastMath as F;
    num::FastMath {
        nnan: f.0 & F::NNAN != 0,
        ninf: f.0 & F::NINF != 0,
        nsz: f.0 & F::NSZ != 0,
        arcp: f.0 & F::ARCP != 0,
        contract: f.0 & F::CONTRACT != 0,
        afn: f.0 & F::AFN != 0,
        reassoc: f.0 & F::REASSOC != 0,
    }
}

/// Whether the numerics lowering of an operation gives the correctly rounded IEEE result (or exact integer code).
#[allow(clippy::too_many_arguments)]
fn exact_numeric(
    m: &mut Module,
    f: usize,
    op: num::Op<'_>,
    args: &[Value],
    ret: TypeId,
    fm: crate::ir::FastMath,
    target: num::Target<'_>,
    what: &str,
) -> Result<(), String> {
    let mut tys = Vec::new();
    let mut consts = Vec::new();
    for &a in args {
        let t = m.value_type(f, a);
        if m.types.is_ptr(t) {
            tys.push(num::Ty::scalar(num::Scalar::I32));
            consts.push(None);
            continue;
        }
        tys.push(crate::wgsl::types::num_ty(&m.types, t).ok_or_else(|| format!("{}: an operand type", what))?);
        consts.push(match a {
            Value::Const(c) => match &m.constant(c).kind {
                ConstKind::Float(b) => Some(*b as u64),
                ConstKind::Int(w) => Some(w[0]),
                _ => None,
            },
            _ => None,
        });
    }
    let exprs: Vec<String> = (0..args.len()).map(|k| format!("a{}", k)).collect();
    let operands: Vec<num::Operand<'_>> =
        exprs.iter().zip(&tys).zip(&consts).map(|((e, t), c)| num::Operand { expr: e, ty: *t, constant: *c }).collect();
    let rty = if m.types.is_void(ret) { None } else { crate::wgsl::types::num_ty(&m.types, ret) };
    let l = num::lower(op, &operands, rty, fmf(fm), target, "p").map_err(|r| format!("{}: {}", what, r))?;
    match l.rule.how {
        How::SoftFloat | How::Exact => Ok(()),
        How::Native | How::NativeRne => {
            Err(format!("the position depends on {}, which is lowered to native WGSL arithmetic", what))
        }
    }
}

/// Check that every operation of the sliced function is lowered exactly.
fn exact(m: &mut Module, f: usize, target: num::Target<'_>) -> Result<(), String> {
    use crate::ir::{BinOp, CastOp};
    let live: Vec<crate::ir::InstId> = m.functions[f].blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
    for i in live {
        let inst = m.functions[f].insts[i.0 as usize].clone();
        match &inst.op {
            Op::Binary { op, lhs, rhs, fmf, .. } => {
                let ni = match op {
                    BinOp::FAdd => num::Inst::FAdd,
                    BinOp::FSub => num::Inst::FSub,
                    BinOp::FMul => num::Inst::FMul,
                    BinOp::FDiv => num::Inst::FDiv,
                    BinOp::FRem => num::Inst::FRem,
                    _ => continue,
                };
                exact_numeric(m, f, num::Op::Inst(ni), &[*lhs, *rhs], inst.ty, *fmf, target, op.name())?;
            }
            Op::FNeg { value, fmf } => {
                exact_numeric(m, f, num::Op::Inst(num::Inst::FNeg), &[*value], inst.ty, *fmf, target, "fneg")?
            }
            Op::FCmp { .. } => {
                // Comparisons are exact integer code on the carriers (numerics lowering, relational rows).
            }
            Op::Cast { op, value, fmf, .. } => {
                let ni = match op {
                    CastOp::FPToSI => num::Inst::FPToSI,
                    CastOp::FPToUI => num::Inst::FPToUI,
                    CastOp::SIToFP => num::Inst::SIToFP,
                    CastOp::UIToFP => num::Inst::UIToFP,
                    CastOp::FPTrunc => num::Inst::FPTrunc,
                    CastOp::FPExt => num::Inst::FPExt,
                    _ => continue,
                };
                exact_numeric(m, f, num::Op::Inst(ni), &[*value], inst.ty, *fmf, target, op.name())?;
            }
            Op::Call { callee, args, fmf, .. } => {
                let name = match callee {
                    Value::Global(GlobalId::Function(k)) => m.functions[*k as usize].common.name.clone(),
                    _ => return Err("an indirect call".into()),
                };
                if !(name.starts_with("air.") || name.starts_with("llvm.")) {
                    return Err(format!("a call of {}", name));
                }
                exact_numeric(m, f, num::Op::Call(&name), args, inst.ty, *fmf, target, &name)?;
            }
            Op::AtomicRmw { .. } | Op::CmpXchg { .. } | Op::Fence { .. } | Op::VaArg { .. } => {
                return Err("an atomic operation, fence or va_arg".into());
            }
            _ => {}
        }
    }
    Ok(())
}

// Running -------------------------------------------------------------------------------------------------------------

struct Runner<'a> {
    inner: &'a Inner,
    draw: &'a Draw<'a>,
    memory: Option<(Memory, Vec<Val>)>,
    cache: HashMap<(u32, u32), Option<[f64; 4]>>,
}

impl<'a> Runner<'a> {
    fn new(inner: &'a Inner, draw: &'a Draw<'a>) -> Runner<'a> {
        Runner { inner, draw, memory: None, cache: HashMap::new() }
    }

    /// The clip-space position of a vertex, when the run gives a defined one.
    fn position(&mut self, vertex: u32, instance: u32) -> Option<[f64; 4]> {
        if let Some(p) = self.cache.get(&(vertex, instance)) {
            return *p;
        }
        let p = self.run(vertex, instance);
        self.cache.insert((vertex, instance), p);
        p
    }

    fn run(&mut self, vertex: u32, instance: u32) -> Option<[f64; 4]> {
        if self.memory.is_none() {
            let mut mem = Memory::new();
            let mut ptrs = Vec::new();
            for a in &self.inner.args {
                ptrs.push(match a {
                    ArgInput::Buffer { index, space } => {
                        let bytes = self.draw.buffers.iter().find(|(i, _)| i == index)?.1;
                        let id = mem.add_data(RegionKind::Buffer, *space, bytes, false, format!("buffer {}", index));
                        Val::S(mem.pointer(id, 0))
                    }
                    _ => Val::S(Scalar::POISON),
                });
            }
            self.memory = Some((mem, ptrs));
        }
        let (mem, ptrs) = self.memory.as_mut()?;
        let int = |v: u32| Val::S(Scalar::int(v as u128));
        let args: Vec<Val> = self
            .inner
            .args
            .iter()
            .zip(ptrs.iter())
            .map(|(a, p)| match a {
                ArgInput::Buffer { .. } | ArgInput::Unused => p.clone(),
                ArgInput::VertexId => int(vertex),
                ArgInput::InstanceId => int(instance),
                ArgInput::BaseVertex => int(self.draw.base_vertex as u32),
                ArgInput::BaseInstance => int(self.draw.base_instance),
            })
            .collect();
        let opts = interp::Options { step_limit: 1_000_000, ..interp::Options::default() };
        let out = interp::run(&self.inner.module, self.inner.function, &args, mem, &opts).ok()?;
        if out.choice_path {
            return None;
        }
        let ret = out.ret?;
        let v = match (self.inner.position, ret) {
            (Some(k), Val::A(members)) => members.get(k as usize)?.clone(),
            (None, v) => v,
            _ => return None,
        };
        let lanes = v.lanes();
        if lanes.len() != 4 {
            return None;
        }
        let mut p = [0f64; 4];
        for (k, s) in lanes.iter().enumerate() {
            if !s.is_defined() || s.choice != 0 {
                return None;
            }
            p[k] = f32::from_bits(s.bits as u32) as f64;
        }
        Some(p)
    }
}

// Footprints ----------------------------------------------------------------------------------------------------------

/// The pixels a primitive may cover: for each row from `top`, the inclusive column range (empty when `lo > hi`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Footprint {
    pub top: u32,
    pub rows: Vec<(u32, u32)>,
}

/// Coordinates beyond this many pixels prove nothing.
const LIMIT: f64 = 8_388_608.0;

/// The footprint of a primitive with clip-space vertex positions `clip`, or `None` when it proves nothing.
pub fn footprint(clip: &[[f64; 4]], topology: Topology, vp: &Viewport, target: [u32; 2]) -> Option<Footprint> {
    let mut pts = Vec::new();
    let mut magnitude: f64 = 0.0;
    for p in clip {
        if !p.iter().all(|c| c.is_finite()) || p[3] <= 0.0 {
            return None;
        }
        let x = vp.x + 0.5 * (p[0] / p[3] + 1.0) * vp.width;
        let y = vp.y + 0.5 * (1.0 - p[1] / p[3]) * vp.height;
        if !(x.abs() <= LIMIT && y.abs() <= LIMIT) {
            return None;
        }
        magnitude = magnitude.max(x.abs()).max(y.abs());
        pts.push((x, y));
    }
    let mut d = 1.0 + magnitude / 1024.0;
    if topology == Topology::Line {
        d += 1.0;
    }
    Some(dilated_pixels(&pts, d, target))
}

/// The pixels whose square, grown by `d` on every side, meets the convex hull of `pts`, inside a `target` of pixels.
pub fn dilated_pixels(pts: &[(f64, f64)], d: f64, target: [u32; 2]) -> Footprint {
    let (w, h) = (target[0] as f64, target[1] as f64);
    let ymin = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let ymax = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    // Row r meets when [r - d, r + 1 + d] meets [ymin, ymax].
    let r0 = (ymin - 1.0 - d).floor().max(0.0);
    let r1 = (ymax + d).floor().min(h - 1.0);
    if r0 > r1 || w < 1.0 {
        return Footprint { top: 0, rows: Vec::new() };
    }
    let mut rows = Vec::new();
    let mut r = r0;
    while r <= r1 {
        let (b0, b1) = (r - d, r + 1.0 + d);
        match band_extent(pts, b0, b1) {
            Some((x0, x1)) => {
                let lo = (x0 - 1.0 - d).floor().max(0.0);
                let hi = (x1 + d).floor().min(w - 1.0);
                rows.push(if lo <= hi { (lo as u32, hi as u32) } else { (1, 0) });
            }
            None => rows.push((1, 0)),
        }
        r += 1.0;
    }
    Footprint { top: r0 as u32, rows }
}

/// The x extent of the convex hull of `pts` within the band `b0 <= y <= b1`.
fn band_extent(pts: &[(f64, f64)], b0: f64, b1: f64) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut take = |x: f64| {
        lo = lo.min(x);
        hi = hi.max(x);
    };
    for &(x, y) in pts {
        if y >= b0 && y <= b1 {
            take(x);
        }
    }
    let n = pts.len();
    for i in 0..n {
        for j in i + 1..n {
            let (p, q) = (pts[i], pts[j]);
            for b in [b0, b1] {
                if (p.1 - b) * (q.1 - b) < 0.0 {
                    let t = (b - p.1) / (q.1 - p.1);
                    take(p.0 + t * (q.0 - p.0));
                }
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Group primitives with these footprints (`None`: drawn alone) in order.
pub fn group(footprints: &[Option<Footprint>], target: [u32; 2]) -> Vec<Range<usize>> {
    let (w, h) = (target[0] as usize, target[1] as usize);
    // A pixel belongs to the current group when its stamp is the group's number.
    let mut stamps: Vec<u32> = vec![0; w * h];
    let mut groups = Vec::new();
    let mut start = 0;
    let mut number = 1u32;
    for (i, fp) in footprints.iter().enumerate() {
        let Some(fp) = fp else {
            if start < i {
                groups.push(start..i);
            }
            groups.push(i..i + 1);
            start = i + 1;
            number += 1;
            continue;
        };
        let pixels = || {
            fp.rows.iter().enumerate().flat_map(move |(k, &(lo, hi))| {
                let y = fp.top as usize + k;
                (lo as usize..=hi as usize).filter(move |_| lo <= hi).map(move |x| y * w + x)
            })
        };
        if pixels().any(|p| stamps[p] == number) {
            groups.push(start..i);
            start = i;
            number += 1;
        }
        for p in pixels() {
            stamps[p] = number;
        }
    }
    if start < footprints.len() {
        groups.push(start..footprints.len());
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(pts: &[(f64, f64)], d: f64) -> Option<Footprint> {
        Some(dilated_pixels(pts, d, [16, 16]))
    }

    #[test]
    fn footprints_hold_every_pixel_within_the_margin() {
        // A point at the centre of pixel (5, 7) with margin 0.25 covers that pixel only... and its neighbours whose
        // grown squares reach it: none at 0.25 from the centre.
        let f = dilated_pixels(&[(5.5, 7.5)], 0.25, [16, 16]);
        let cells: Vec<(u32, u32)> = f
            .rows
            .iter()
            .enumerate()
            .flat_map(|(k, &(lo, hi))| (lo..=hi).map(move |x| (x, f.top + k as u32)))
            .collect();
        assert!(cells.contains(&(5, 7)));
        for (x, y) in &cells {
            // Every listed pixel's square grown by the margin, plus the conservative extra pixel, reaches the point.
            assert!((*x as f64 - 5.5).abs() <= 2.25 && (*y as f64 - 7.5).abs() <= 2.25, "{} {}", x, y);
        }
        // Outside the target: nothing.
        assert!(dilated_pixels(&[(-50.0, -50.0)], 1.0, [16, 16]).rows.iter().all(|r| r.0 > r.1));
    }

    #[test]
    fn disjoint_primitives_share_a_group_and_overlapping_ones_do_not() {
        let a = fp(&[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)], 1.0);
        let b = fp(&[(10.0, 10.0), (14.0, 10.0), (10.0, 14.0)], 1.0);
        let c = fp(&[(1.0, 1.0), (3.0, 1.0), (1.0, 3.0)], 1.0);
        assert_eq!(group(&[a.clone(), b.clone(), c.clone()], [16, 16]), vec![0..2, 2..3]);
        // A primitive that proves nothing is alone.
        assert_eq!(group(&[a.clone(), None, b.clone()], [16, 16]), vec![0..1, 1..2, 2..3]);
        // Adjacent primitives sharing an edge are within the margin of each other.
        let left = fp(&[(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)], 1.0);
        let right = fp(&[(8.0, 0.0), (8.0, 8.0), (0.0, 8.0)], 1.0);
        assert_eq!(group(&[left, right], [16, 16]), vec![0..1, 1..2]);
    }

    #[test]
    fn positions_behind_the_eye_or_unbounded_prove_nothing() {
        let vp = Viewport { x: 0.0, y: 0.0, width: 16.0, height: 16.0 };
        let ok = [[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0]];
        assert!(footprint(&ok, Topology::Triangle, &vp, [16, 16]).is_some());
        let behind = [[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, -1.0], [0.0, 0.5, 0.0, 1.0]];
        assert!(footprint(&behind, Topology::Triangle, &vp, [16, 16]).is_none());
        let nan = [[f64::NAN, 0.0, 0.0, 1.0]];
        assert!(footprint(&nan, Topology::Point, &vp, [16, 16]).is_none());
        let far = [[1e9, 0.0, 0.0, 1.0]];
        assert!(footprint(&far, Topology::Point, &vp, [16, 16]).is_none());
    }
}
