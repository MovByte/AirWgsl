//! `air.struct_type_info`: the layout of a struct an argument points to, member by member.
//!
//! A struct's tuple lists its members in order. Each member is an optional nested-struct prefix
//! (`!"air.struct_type_info", !N`, the layout of the member's own struct type), then `i32` offset, `i32` size,
//! `i32` array length (0 for a member that is not an array), the member's type name and its name, then the member's
//! keys ([`super::grammar::Context::Member`]). The next member begins at the next `i32` or nested-struct prefix.
//!
//! In an argument buffer every member that holds resources carries `air.indirect_argument`: for a resource, the
//! resource's own argument record (as an entry point argument's); for a struct or array of them, the `[[id]]` the
//! aggregate starts at. Imageblock members carry `air.render_target` with the color attachment index.

use super::grammar::{self, Context, Item};
use super::md::{Cursor, Operand, refusal};
use super::{Argument, decode_argument};
use crate::api::{Refusal, RefusalKind};
use crate::ir::{MdId, Module};

/// A struct layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructInfo {
    pub members: Vec<Member>,
}

/// One struct member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    /// Byte offset in the struct.
    pub offset: u32,
    /// Size in bytes; for an array member, the size of one element.
    pub size: u32,
    /// The number of elements of an array member, 0 when the member is not an array.
    pub array_length: u32,
    /// The member's MSL type name.
    pub type_name: String,
    pub name: String,
    /// The layout of the member's struct type, when it is a struct (or an array of one).
    pub nested: Option<StructInfo>,
    /// The argument-buffer resource this member is.
    pub indirect: Option<IndirectArgument>,
    /// `air.render_target`: the color attachment of an imageblock member.
    pub render_target: Option<u32>,
}

/// `air.indirect_argument` of an argument-buffer member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndirectArgument {
    /// A resource: its argument record.
    Resource(Box<Argument>),
    /// A struct or array of resources: the `[[id]]` it starts at.
    Aggregate(u32),
}

/// Decodes the struct layout in tuple `id`. `depth` bounds nesting (the corpora nest at most a few levels).
pub fn struct_info(m: &Module, id: MdId, function: Option<&str>, depth: u32) -> Result<StructInfo, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.struct_type_info !{}", id.0), function)?;
    if depth > 64 {
        return Err(c.error("air.struct_type_info", "a struct nested at most 64 levels deep"));
    }
    let mut members = Vec::new();
    while !c.at_end() {
        let mut nested = None;
        if c.peek_str() == Some(b"air.struct_type_info") {
            c.advance();
            let n = c.node_id("air.struct_type_info")?;
            nested = Some(struct_info(m, n, function, depth + 1)?);
        }
        let offset = c.i32("struct member offset")?;
        let size = c.i32("struct member size")?;
        let array_length = c.i32("struct member array length")?;
        let type_name = c.string("struct member type name")?;
        let name = c.string("struct member name")?;
        let member_name = name.clone();
        let mut member =
            Member { offset, size, array_length, type_name, name, nested, indirect: None, render_target: None };
        loop {
            match c.peek() {
                None | Some(Operand::Int { .. }) => break,
                Some(Operand::Str(s)) if s == b"air.struct_type_info" => break,
                _ => {}
            }
            let Item { key, ops } = grammar::item(&mut c, Context::Member)?;
            let dup = || {
                refusal(
                    RefusalKind::NotImplemented,
                    key,
                    function,
                    format!("air.struct_type_info !{}: member {} has {} twice", id.0, member_name, key),
                )
            };
            match key {
                "air.indirect_argument" => {
                    if member.indirect.is_some() {
                        return Err(dup());
                    }
                    member.indirect = Some(match ops[0] {
                        Operand::Int { value, .. } => IndirectArgument::Aggregate(value as u32),
                        Operand::Node(n) => {
                            IndirectArgument::Resource(Box::new(decode_argument(m, n, function, true)?))
                        }
                        _ => unreachable!("the grammar admits an i32 or a tuple"),
                    });
                }
                "air.render_target" => {
                    if member.render_target.is_some() {
                        return Err(dup());
                    }
                    let Operand::Int { value, .. } = ops[0] else { unreachable!("the grammar admits an i32") };
                    member.render_target = Some(value as u32);
                }
                _ => unreachable!("{} is a member key of the grammar", key),
            }
        }
        members.push(member);
    }
    Ok(StructInfo { members })
}
