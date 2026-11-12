//! The interface the lowering packages (S6a texture and sampler, S6b stage interface, S6c framebuffer fetch, S6d
//! atomics, barriers, argument buffers and function tables, S6e tessellation) register their rules with.
//!
//! A [`CallRule`] lowers calls of the `air.*` functions it claims. The emitter asks the registered rules first, in
//! registration order, then its own lowerings (numerics through [`crate::num::lower`], the memory intrinsics); a call
//! nobody lowers is refused with the intrinsics.toml family of its name. A rule works through [`RuleCx`]: it reads
//! operands as WGSL expressions in their carriers, emits statements at the call, declares module-scope items and
//! bindings, and states the WebGPU features and WGSL extensions its code needs. Before any code is emitted the memory
//! model asks every rule which pointer operands a call reads, writes or accesses atomically ([`CallRule::effects`]), so
//! that a region an atomic rule touches is declared atomic and every other access to it is atomic too (plan 5.8).
//!
//! A [`StageRule`] lowers entry-point argument and output records (vertex input, varyings beyond the direct ones,
//! built-ins that need the executor's help, outputs): the stage interface asks the registered stage rules before its
//! own direct mappings, so the stage-interface package replaces or extends them without editing them.
//!
//! Registration is static: a package adds its rule set to [`registered`]. [`crate::wgsl::compile_with`] takes an
//! explicit set (tests register their own).

use crate::air::{Argument, Output};
use crate::api::{
    BindingResource, BindingSource, HostCaps, MetalStage, PipelineDesc, Refusal, StageBindings, UniformValue,
};
use crate::ir::{InstId, Module, TypeId, Value};
use crate::mem::access::Ptr;

/// How a call accesses memory through one of its pointer operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Read,
    /// A write of `bytes` bytes at an offset aligned to `align` (a write that is not whole aligned words makes a
    /// shared region atomic).
    Write {
        bytes: u64,
        align: u64,
    },
    /// An atomic access: the region is declared atomic.
    Atomic,
}

/// A call being lowered.
pub struct CallSite<'a> {
    /// The callee's name.
    pub name: &'a str,
    pub inst: InstId,
    pub args: &'a [Value],
    /// The result type (`void` for none).
    pub ret: TypeId,
}

/// What a rule can do at a call.
pub trait RuleCx {
    fn module(&self) -> &Module;
    /// The function being lowered.
    fn function(&self) -> usize;
    /// The WGSL expression of an operand in its carrier.
    fn operand(&mut self, v: Value) -> Result<String, Refusal>;
    /// A pointer operand for word access (the memory model's form).
    fn pointer(&mut self, v: Value) -> Result<Ptr, Refusal>;
    /// The WGSL name of region `r`'s array (for atomic built-ins: `&name[word]`).
    fn region_name(&self, r: usize) -> String;
    /// Region `r` of the memory model (its kind, and whether it is declared atomic).
    fn region(&self, r: usize) -> &crate::mem::Region;
    /// The IR argument a handle operand (texture, sampler) comes from, when it does.
    fn handle_arg(&self, v: Value) -> Option<u32>;
    /// The carrier type of an IR type.
    fn carrier(&mut self, t: TypeId) -> Result<String, Refusal>;
    /// A fresh identifier.
    fn fresh(&mut self) -> String;
    /// A statement at the call.
    fn line(&mut self, s: String);
    /// A module-scope declaration, keyed so that it is emitted once.
    fn module_item(&mut self, key: &str, text: String);
    /// A binding in the pipeline layout for a resource (texture, sampler, argument-buffer member) of the stage,
    /// declared in the stage's module as `var NAME: wgsl_type`; returns NAME. One per source.
    fn resource(
        &mut self,
        resource: BindingResource,
        source: BindingSource,
        wgsl_type: &str,
    ) -> Result<String, Refusal>;
    /// The entry point's record of an IR argument (texture location and access, sampler, ...).
    fn arg_record(&self, arg: u32) -> Option<Argument>;
    /// A WebGPU feature the code needs.
    fn require_feature(&mut self, feature: &str);
    /// A WGSL `enable` directive the code needs.
    fn enable(&mut self, extension: &str);
    /// The stage being lowered.
    fn stage(&self) -> MetalStage;
    /// What the pipeline description binds to the stage at the draw or dispatch, when it lists the stage.
    fn bindings(&self) -> Option<&StageBindings>;
    /// The host's capabilities.
    fn host(&self) -> &HostCaps;
    /// A numerics library item the generated expressions call.
    fn prelude(&mut self, item: &'static str) -> Result<(), Refusal>;
    /// The code takes implicit derivatives (implicit-LOD sampling): the module is treated as the derivative functions'
    /// modules are (see [`crate::wgsl`]).
    fn derivatives(&mut self);
}

/// A rule lowering calls of some `air.*` functions.
pub trait CallRule: Send + Sync {
    /// Whether the rule lowers calls of `name`.
    fn handles(&self, name: &str) -> bool;
    /// The memory the call accesses through its pointer operands: (operand index, effect).
    fn effects(&self, _call: &CallSite<'_>) -> Vec<(usize, Effect)> {
        Vec::new()
    }
    /// Whether the rule lowers the uses of entry arguments of this record kind (`air.texture`, `air.sampler`): such
    /// arguments become handles ([`RuleCx::handle_arg`]) instead of being refused by the stage interface.
    fn claims_argument(&self, _kind: &str) -> bool {
        false
    }
    /// Lower the call; returns the result's WGSL expression (in the result type's carrier), `None` for void.
    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal>;
    /// Check the finished WGSL module of a stage (its bindings declared) whose entry function `function` is: a rule
    /// whose lowering is exact only when the module as a whole has a property (WGSL's uniformity analysis accepting
    /// a barrier, for one) refuses the stage when it does not.
    fn check_stage(&self, _stage: MetalStage, _function: &str, _wgsl: &str) -> Result<(), Refusal> {
        Ok(())
    }
}

/// What a stage-interface rule can do: declare entry parameters, read executor uniforms, ask for bindings and
/// library items.
pub trait StageCx {
    fn module(&self) -> &Module;
    /// The entry function.
    fn function(&self) -> usize;
    /// The stage being lowered.
    fn stage(&self) -> MetalStage;
    /// The pipeline state.
    fn desc(&self) -> &PipelineDesc;
    /// An entry parameter (`@builtin(...) name: type`, `@location(n) ... name: type`).
    fn param(&mut self, p: String);
    /// The WGSL expression (a `u32`) of a field of the executor's uniform block.
    fn uniform(&mut self, value: UniformValue) -> String;
    /// A binding for a resource, as [`RuleCx::resource`].
    fn resource(
        &mut self,
        resource: BindingResource,
        source: BindingSource,
        wgsl_type: &str,
    ) -> Result<String, Refusal>;
    /// Bind IR argument `arg` as the memory of the Metal buffer at `index` in address space `space` (1 device, 2
    /// constant), declared, bound and accessed as a buffer argument is; returns the argument's value (the offset of
    /// its region's first byte).
    fn buffer_region(&mut self, arg: u32, index: u64, space: u32) -> String;
    /// A numerics library item the generated expressions call.
    fn prelude(&mut self, item: &'static str);
    /// A WebGPU feature the code needs.
    fn require_feature(&mut self, feature: &str);
}

/// How a returned value becomes a stage output.
pub struct OutputField {
    /// The output struct member (`@builtin(...) NAME: type` or `@location(n) NAME: type`), with `{name}` where the
    /// member's name goes.
    pub decl: String,
    /// The member's value from the IR value's carrier (`{}`).
    pub value: String,
}

/// A rule lowering entry-point argument and output records (the stage-interface lowering).
pub trait StageRule: Send + Sync {
    /// Whether the rule lowers argument records of this kind (the record key of the role, [`super::io::role_key`])
    /// in this stage.
    fn handles_argument(&self, _stage: MetalStage, _kind: &str) -> bool {
        false
    }
    /// Lower argument `arg` (record `record`): the WGSL expression of its value in its type's carrier.
    fn argument(&self, _cx: &mut dyn StageCx, _arg: u32, _record: &Argument) -> Result<String, Refusal> {
        unreachable!("argument() of a rule that handles no argument")
    }
    /// Whether the rule lowers output records of this kind in this stage.
    fn handles_output(&self, _stage: MetalStage, _kind: &str) -> bool {
        false
    }
    /// Lower output `index` (record `record`, member type `ty`); `None` when the output is not declared (nothing
    /// reads it).
    fn output(
        &self,
        _cx: &mut dyn StageCx,
        _index: usize,
        _record: &Output,
        _ty: TypeId,
    ) -> Result<Option<OutputField>, Refusal> {
        unreachable!("output() of a rule that handles no output")
    }
}

/// A set of rules.
#[derive(Default)]
pub struct Rules {
    pub calls: Vec<Box<dyn CallRule>>,
    pub stage: Vec<Box<dyn StageRule>>,
}

impl Rules {
    pub fn new() -> Rules {
        Rules::default()
    }

    pub fn with(mut self, rule: Box<dyn CallRule>) -> Rules {
        self.calls.push(rule);
        self
    }

    pub fn with_stage(mut self, rule: Box<dyn StageRule>) -> Rules {
        self.stage.push(rule);
        self
    }

    /// The stage rule lowering argument records of `kind`, if one is registered.
    pub fn stage_argument(&self, stage: MetalStage, kind: &str) -> Option<&dyn StageRule> {
        self.stage.iter().find(|r| r.handles_argument(stage, kind)).map(|r| r.as_ref())
    }

    /// The stage rule lowering output records of `kind`, if one is registered.
    pub fn stage_output(&self, stage: MetalStage, kind: &str) -> Option<&dyn StageRule> {
        self.stage.iter().find(|r| r.handles_output(stage, kind)).map(|r| r.as_ref())
    }

    /// Whether a registered rule lowers uses of arguments of this record kind.
    pub fn claims_argument(&self, kind: &str) -> bool {
        self.calls.iter().any(|r| r.claims_argument(kind))
    }

    pub fn find(&self, name: &str) -> Option<&dyn CallRule> {
        self.calls.iter().find(|r| r.handles(name)).map(|r| r.as_ref())
    }

    /// Run every call rule's check of a finished stage module.
    pub fn check_stage(&self, stage: MetalStage, function: &str, wgsl: &str) -> Result<(), Refusal> {
        for r in &self.calls {
            r.check_stage(stage, function, wgsl)?;
        }
        Ok(())
    }
}

/// The rules of the lowering packages that have landed: each package appends its rule set here.
pub fn registered() -> Rules {
    let mut r = Rules::new();
    for rule in crate::sync::rules() {
        r = r.with(rule);
    }
    r.with(Box::new(crate::icb::Encoding))
        .with(Box::new(crate::ftable::Calls))
        .with_stage(Box::new(crate::argbuf::ArgumentBuffers))
        .with_stage(Box::new(crate::icb::Arguments))
        .with_stage(Box::new(crate::ftable::Arguments))
}
