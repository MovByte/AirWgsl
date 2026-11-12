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
//! Registration is static: a package adds its rule set to [`registered`]. [`crate::wgsl::compile_with`] takes an
//! explicit set (tests register their own).

use crate::api::Refusal;
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
    /// A WebGPU feature the code needs.
    fn require_feature(&mut self, feature: &str);
    /// A WGSL `enable` directive the code needs.
    fn enable(&mut self, extension: &str);
}

/// A rule lowering calls of some `air.*` functions.
pub trait CallRule: Send + Sync {
    /// Whether the rule lowers calls of `name`.
    fn handles(&self, name: &str) -> bool;
    /// The memory the call accesses through its pointer operands: (operand index, effect).
    fn effects(&self, _call: &CallSite<'_>) -> Vec<(usize, Effect)> {
        Vec::new()
    }
    /// Lower the call; returns the result's WGSL expression (in the result type's carrier), `None` for void.
    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal>;
}

/// A set of rules.
#[derive(Default)]
pub struct Rules {
    pub calls: Vec<Box<dyn CallRule>>,
}

impl Rules {
    pub fn new() -> Rules {
        Rules::default()
    }

    pub fn with(mut self, rule: Box<dyn CallRule>) -> Rules {
        self.calls.push(rule);
        self
    }

    pub fn find(&self, name: &str) -> Option<&dyn CallRule> {
        self.calls.iter().find(|r| r.handles(name)).map(|r| r.as_ref())
    }
}

/// The rules of the lowering packages that have landed. None has yet: each package appends its rule set here.
pub fn registered() -> Rules {
    Rules::new()
}
