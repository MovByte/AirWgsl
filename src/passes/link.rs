//! Linking: the stage module's declarations resolved against the definitions of the other modules (Metal's linked
//! functions and preloaded libraries), lazily, as LLVM's IR linker does with only the needed definitions.
//!
//! A declaration in the result with non-local linkage takes the definition of the first other module (in the order
//! given) that defines that name with non-local linkage, together with everything that definition refers to:
//! internal and private globals are copied under unique names, and names the result already has resolve to the
//! result's own. A name the stage module defines stays its own definition. Linking repeats until no declaration can
//! be resolved, so definitions brought in may bring in more. Modules with a data layout other than the stage
//! module's are refused.

use crate::api::{Refusal, RefusalKind};
use crate::ir::{GlobalId, Linkage, Module};

use super::copy::{Copier, Reach};
use super::refusal;

/// The stage module with the definitions it needs from `libraries`.
#[allow(clippy::result_large_err)]
pub fn link(stage: &Module, libraries: &[Module], entry: &str) -> Result<Module, Refusal> {
    let mut m = stage.clone();
    if libraries.is_empty() {
        return Ok(m);
    }
    if libraries.iter().any(|l| l.datalayout != stage.datalayout) {
        return Err(refusal(
            RefusalKind::RejectedByDevice,
            "linked module",
            entry,
            "a linked module has a different data layout",
        ));
    }
    loop {
        let mut progress = false;
        // Declarations of the result, by name.
        let decls: Vec<(usize, String)> = m
            .functions
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_declaration() && !f.common.linkage.is_local() && !intrinsic(&f.common.name))
            .map(|(i, f)| (i, f.common.name.clone()))
            .collect();
        for (d, name) in decls {
            let Some((lib, s)) = libraries.iter().enumerate().find_map(|(k, l)| {
                l.functions
                    .iter()
                    .position(|f| f.common.name == name && !f.is_declaration() && !f.common.linkage.is_local())
                    .map(|i| (k, i))
            }) else {
                continue;
            };
            let src = &libraries[lib];
            let mut c = Copier::new(src, Reach::CodeOnly);
            // Names the result already has, resolved to it: definitions and declarations alike.
            bind_existing(&mut c, src, &m);
            // This definition goes into the declaration's slot.
            c.bind(GlobalId::Function(s as u32), GlobalId::Function(d as u32));
            c.function_header(&mut m, s as u32, d as u32);
            let linkage = src.functions[s].common.linkage;
            m.functions[d].common.linkage = linkage;
            c.function_body(&mut m, s as u32, d as u32);
            c.finish(&mut m);
            progress = true;
        }
        // Global variables declared in the result and defined in a library.
        let gdecls: Vec<(usize, String)> = m
            .globals
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                g.initializer.is_none() && !g.common.linkage.is_local() && g.common.linkage != Linkage::Common
            })
            .map(|(i, g)| (i, g.common.name.clone()))
            .collect();
        for (d, name) in gdecls {
            let Some((lib, s)) = libraries.iter().enumerate().find_map(|(k, l)| {
                l.globals
                    .iter()
                    .position(|g| g.common.name == name && g.initializer.is_some() && !g.common.linkage.is_local())
                    .map(|i| (k, i))
            }) else {
                continue;
            };
            let src = &libraries[lib];
            let mut c = Copier::new(src, Reach::CodeOnly);
            bind_existing(&mut c, src, &m);
            c.bind(GlobalId::Var(s as u32), GlobalId::Var(d as u32));
            let init = src.globals[s].initializer.map(|v| c.value(&mut m, v));
            m.globals[d].initializer = init;
            m.globals[d].constant = src.globals[s].constant;
            m.globals[d].common.linkage = src.globals[s].common.linkage;
            c.finish(&mut m);
            progress = true;
        }
        if !progress {
            return Ok(m);
        }
    }
}

fn intrinsic(name: &str) -> bool {
    name.starts_with("llvm.") || name.starts_with("air.")
}

/// Binds every non-local global of `src` whose name `dst` already has to `dst`'s global.
fn bind_existing(c: &mut Copier, src: &Module, dst: &Module) {
    for (i, f) in src.functions.iter().enumerate() {
        if f.common.linkage.is_local() {
            continue;
        }
        if let Some(j) = dst.functions.iter().position(|g| g.common.name == f.common.name) {
            c.bind(GlobalId::Function(i as u32), GlobalId::Function(j as u32));
        }
    }
    for (i, g) in src.globals.iter().enumerate() {
        if g.common.linkage.is_local() {
            continue;
        }
        if let Some(j) = dst.globals.iter().position(|h| h.common.name == g.common.name) {
            c.bind(GlobalId::Var(i as u32), GlobalId::Var(j as u32));
        }
    }
}
