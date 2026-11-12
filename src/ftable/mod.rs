//! Visible function tables, function pointers and dynamic libraries (MSL 2.15, 5.1.4, 5.9).
//!
//! The A10X has none of them: D1 reads `supportsFunctionPointers`, `supportsFunctionPointersFromRender` and
//! `supportsDynamicLibraries` as false from the 21H461 driver, in agreement with Apple's family tables (function
//! pointers and dynamic libraries from Apple6). A pipeline with a function table argument or a call of a table
//! function is therefore refused as rejected by the device (family `function-tables-dynamic`), with the profile
//! values named. Visible functions called directly are linked and inlined by canonicalization (family
//! `visible-stitchable`), not here.

use crate::air::Argument;
use crate::api::{Location, MetalStage, Refusal, RefusalKind};
use crate::wgsl::hooks::{CallRule, CallSite, RuleCx, StageCx, StageRule};
use crate::wgsl::tracked;

const FAMILY: &str = "function-tables-dynamic";

/// The refusal of a function-table construct.
pub fn refusal(construct: &str, function: &str) -> Refusal {
    Refusal {
        kind: RefusalKind::RejectedByDevice,
        construct: construct.to_string(),
        location: Some(Location { function: function.to_string(), instruction: None }),
        reason: format!(
            "[family:{}] the A10X has no function pointers or function tables (D1: supportsFunctionPointers, \
             supportsFunctionPointersFromRender and supportsDynamicLibraries are false)",
            FAMILY
        ),
        reference: Some("MSL 5.1.4; D1 device profile".into()),
    }
}

/// The refusal of the function-table functions.
pub struct Calls;

impl CallRule for Calls {
    fn handles(&self, name: &str) -> bool {
        tracked::family_of(name) == Some(FAMILY)
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal> {
        let fi = cx.function();
        Err(refusal(call.name, &cx.module().functions[fi].common.name))
    }
}

/// The refusal of function-table arguments, used or not, as the family's decision (device-unsupported) states for
/// every construct of the family.
pub struct Arguments;

impl StageRule for Arguments {
    fn handles_argument(&self, _stage: MetalStage, kind: &str) -> bool {
        kind == "air.visible_function_table"
    }

    fn argument(&self, cx: &mut dyn StageCx, _arg: u32, record: &Argument) -> Result<String, Refusal> {
        let fi = cx.function();
        Err(refusal(crate::wgsl::io::role_key(&record.role), &cx.module().functions[fi].common.name))
    }
}
