//! The WGSL library the numerics lowering calls into, and the prelude generator.
//!
//! The library is a set of WGSL source files under `wgsl/`. Each top-level item (a `fn`, `const`, `struct` or `alias` starting
//! in column 0, with the comment lines directly above it) is one unit; an item's dependencies are the other items it
//! names. Every identifier the library defines starts with `nm_`, a prefix the rest of the generated module must not
//! use. A [`Prelude`] collects the items a shader's lowered operations need and emits their transitive closure in
//! library order, so the same request always produces the same text.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

/// The prefix of every identifier the library defines.
pub const PREFIX: &str = "nm_";

const SOURCES: &[(&str, &str)] = &[
    ("wide.wgsl", include_str!("wgsl/wide.wgsl")),
    ("f32.wgsl", include_str!("wgsl/f32.wgsl")),
    ("conv.wgsl", include_str!("wgsl/conv.wgsl")),
    ("xf.wgsl", include_str!("wgsl/xf.wgsl")),
    ("xfconst.wgsl", include_str!("wgsl/xfconst.wgsl")),
    ("math.wgsl", include_str!("wgsl/math.wgsl")),
    ("common.wgsl", include_str!("wgsl/common.wgsl")),
    ("int.wgsl", include_str!("wgsl/int.wgsl")),
    ("half.wgsl", include_str!("wgsl/half.wgsl")),
];

/// One top-level library item.
#[derive(Debug)]
pub struct Item {
    pub name: String,
    /// The source file it comes from.
    pub file: &'static str,
    /// Item text, comments included, ending in a newline.
    pub text: String,
    /// Indices of the items this one names directly.
    pub deps: Vec<usize>,
}

/// The parsed library.
#[derive(Debug)]
pub struct Library {
    items: Vec<Item>,
    by_name: BTreeMap<String, usize>,
}

/// A library defect found while parsing the sources: an item named twice or a reference to an item that does not
/// exist. The library is part of this crate, so a defect is a bug the unit tests report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryError(pub String);

fn item_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("fn ")
        .or_else(|| line.strip_prefix("const "))
        .or_else(|| line.strip_prefix("struct "))
        .or_else(|| line.strip_prefix("alias "))?;
    let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len());
    Some(&rest[..end])
}

fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_alphabetic() || c == b'_' {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                return Some(&text[start..i]);
            }
            i += 1;
        }
        None
    })
}

fn strip_comments(text: &str) -> String {
    text.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n")
}

impl Library {
    fn parse() -> Result<Library, LibraryError> {
        let mut items: Vec<Item> = Vec::new();
        for (file, source) in SOURCES {
            let lines: Vec<&str> = source.lines().collect();
            let starts: Vec<usize> = (0..lines.len()).filter(|&i| item_name(lines[i]).is_some()).collect();
            for (k, &start) in starts.iter().enumerate() {
                let mut first = start;
                while first > 0 && lines[first - 1].starts_with("//") {
                    first -= 1;
                }
                let mut end = starts.get(k + 1).copied().unwrap_or(lines.len());
                // The next item's leading comment block belongs to it.
                while end > start + 1 && lines[end - 1].starts_with("//") {
                    end -= 1;
                }
                while end > start + 1 && lines[end - 1].trim().is_empty() {
                    end -= 1;
                }
                let name = item_name(lines[start]).unwrap_or_default().to_string();
                if !name.starts_with(PREFIX) {
                    return Err(LibraryError(format!("{}: item {} lacks the {} prefix", file, name, PREFIX)));
                }
                let mut text = lines[first..end].join("\n");
                text.push('\n');
                items.push(Item { name, file, text, deps: Vec::new() });
            }
        }
        let mut by_name = BTreeMap::new();
        for (i, item) in items.iter().enumerate() {
            if by_name.insert(item.name.clone(), i).is_some() {
                return Err(LibraryError(format!("{} is defined twice", item.name)));
            }
        }
        for i in 0..items.len() {
            let code = strip_comments(&items[i].text);
            let mut deps = BTreeSet::new();
            for id in identifiers(&code) {
                if !id.starts_with(PREFIX) || id == items[i].name {
                    continue;
                }
                match by_name.get(id) {
                    Some(&d) => {
                        deps.insert(d);
                    }
                    None => return Err(LibraryError(format!("{} names {}, which the library does not define", items[i].name, id))),
                }
            }
            items[i].deps = deps.into_iter().collect();
        }
        Ok(Library { items, by_name })
    }

    /// The library, parsed once.
    pub fn get() -> Result<&'static Library, LibraryError> {
        static LIB: OnceLock<Result<Library, LibraryError>> = OnceLock::new();
        LIB.get_or_init(Library::parse).as_ref().map_err(Clone::clone)
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }
}

/// The library items a shader needs. Requests are by item name; [`Prelude::emit`] writes them and everything they
/// depend on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Prelude {
    names: BTreeSet<String>,
}

impl Prelude {
    pub fn new() -> Prelude {
        Prelude::default()
    }

    /// Add one library item by name. Lowering rules only name items that exist; an unknown name is a defect of the
    /// lowering and is reported as an error rather than ignored.
    pub fn require(&mut self, name: &str) -> Result<(), LibraryError> {
        let lib = Library::get()?;
        if lib.find(name).is_none() {
            return Err(LibraryError(format!("no library item named {}", name)));
        }
        self.names.insert(name.to_string());
        Ok(())
    }

    pub fn extend(&mut self, other: &Prelude) {
        self.names.extend(other.names.iter().cloned());
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The requested item names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    /// The WGSL text of the requested items and their dependencies, in library order.
    pub fn emit(&self) -> Result<String, LibraryError> {
        let lib = Library::get()?;
        let mut keep = BTreeSet::new();
        let mut stack: Vec<usize> = Vec::new();
        for name in &self.names {
            stack.push(lib.find(name).ok_or_else(|| LibraryError(format!("no library item named {}", name)))?);
        }
        while let Some(i) = stack.pop() {
            if keep.insert(i) {
                stack.extend(lib.items[i].deps.iter().copied());
            }
        }
        let mut out = String::new();
        for i in keep {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&lib.items[i].text);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_parses_and_every_reference_resolves() {
        let lib = Library::get().unwrap();
        assert!(lib.items().len() > 20);
        assert!(lib.find("nm_f32_add").is_some());
    }

    #[test]
    fn prelude_emits_the_dependency_closure_in_library_order() {
        let mut p = Prelude::new();
        p.require("nm_f32_add").unwrap();
        let text = p.emit().unwrap();
        let rpack = text.find("fn nm_f32_rpack").unwrap();
        let add = text.find("fn nm_f32_add(").unwrap();
        let jam = text.find("fn nm_shr32_jam").unwrap();
        assert!(jam < rpack && rpack < add);
        assert!(!text.contains("fn nm_f32_div"));
        // Requesting the same items in a different order gives the same text.
        let mut q = Prelude::new();
        q.require("nm_f32_rpack").unwrap();
        q.require("nm_f32_add").unwrap();
        assert_eq!(q.emit().unwrap(), text);
    }

    #[test]
    fn unknown_items_are_errors() {
        assert!(Prelude::new().require("nm_no_such_item").is_err());
    }

    #[test]
    fn whole_library_is_valid_wgsl() {
        let lib = Library::get().unwrap();
        let mut p = Prelude::new();
        for item in lib.items() {
            p.require(&item.name).unwrap();
        }
        let text = format!("enable f16;\n{}", p.emit().unwrap());
        let module = naga::front::wgsl::parse_str(&text).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    }
}
