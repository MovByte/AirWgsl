//! Compares this crate's metallib reader with `metal-readobj -a` for a set of metallibs.
//!
//! Usage: `mtlb_check UNITS_DIR READOBJ_DIR REPORT.tsv`
//!
//! UNITS_DIR holds metallibs (`*.metallib`); READOBJ_DIR holds `metal-readobj -a FILE` output for each as
//! `FILE.txt`. For every metallib the header fields, every function entry, the module list (wrapper headers), the
//! record sizes of the public and private metadata, reflection and script sections, the dynamic header's install
//! name and the symbol list are compared. A field `metal-readobj` prints that this check does not know is a
//! mismatch, so nothing it shows goes unchecked. The nested content of metadata and reflection records (function
//! constants, argument reflection) is outside the container reader and is not compared.

use std::collections::BTreeMap;
use std::io::Write;

use airwgsl::mtlb::{self, Metallib};

#[derive(Debug, Default)]
struct Node {
    name: String,
    fields: Vec<(String, String)>,
    children: Vec<Node>,
}

/// Parses `metal-readobj`'s LLVM-style output into a tree of blocks (`Name {`/`Name [` ... `}`/`]`).
fn parse_readobj(text: &str) -> Node {
    let mut stack = vec![Node { name: "root".into(), ..Default::default() }];
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t == "}" || t == "]" {
            if stack.len() > 1 {
                let n = stack.pop().unwrap();
                stack.last_mut().unwrap().children.push(n);
            }
            continue;
        }
        // Block openers: "Name {", "Name [", and flag lists "Flags [ (0x842)".
        let opener = t.strip_suffix(" {").or_else(|| t.strip_suffix(" [")).or_else(|| t.split_once(" [ (").map(|x| x.0));
        if let Some(name) = opener {
            stack.push(Node { name: name.to_string(), ..Default::default() });
            continue;
        }
        if let Some((k, v)) = t.split_once(": ") {
            stack.last_mut().unwrap().fields.push((k.to_string(), v.to_string()));
        } else if let Some(k) = t.strip_suffix(':') {
            stack.last_mut().unwrap().fields.push((k.to_string(), String::new()));
        } else {
            stack.last_mut().unwrap().fields.push((t.to_string(), String::new()));
        }
    }
    while stack.len() > 1 {
        let n = stack.pop().unwrap();
        stack.last_mut().unwrap().children.push(n);
    }
    stack.pop().unwrap()
}

fn find<'a>(n: &'a Node, name: &str) -> Option<&'a Node> {
    n.children.iter().find(|c| c.name == name)
}

/// The value in parentheses of an enum field (`METALLIB_EXECUTE (0x0)`), or the field parsed as a number.
fn num(v: &str) -> Option<u64> {
    let v = match (v.rfind('('), v.rfind(')')) {
        (Some(a), Some(b)) if b > a => &v[a + 1..b],
        _ => v,
    };
    let v = v.trim();
    if let Some(h) = v.strip_prefix("0x") {
        u64::from_str_radix(h, 16).ok()
    } else {
        v.parse().ok()
    }
}

fn yes(v: &str) -> Option<bool> {
    match v {
        "Yes" => Some(true),
        "No" => Some(false),
        _ => None,
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02X}", x)).collect()
}

fn uuid(b: &[u8; 16]) -> String {
    let h = hex(b);
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

fn compare(lib: &Metallib, ro: &Node) -> Result<usize, String> {
    let checked = std::cell::Cell::new(0usize);
    let eq = |what: &str, got: Option<u64>, want: Option<u64>| -> Result<(), String> {
        checked.set(checked.get() + 1);
        if got != want {
            return Err(format!("{}: reader {:?}, readobj {:?}", what, got, want));
        }
        Ok(())
    };
    let h = &lib.header;
    let hdr = find(ro, "MetalLibHeader").ok_or("no MetalLibHeader")?;
    for (k, v) in &hdr.fields {
        let b = |x: bool| Some(x as u64);
        match k.as_str() {
            "MagicWord" => {
                if v != "MTLB" {
                    return Err("MagicWord".into());
                }
            }
            "MacOSTarget" => eq(k, b(h.macos_target), yes(v).map(|x| x as u64))?,
            "FileMajorVersion" => eq(k, Some(h.file_major as u64), num(v))?,
            "FileMinorVersion" => eq(k, Some(h.file_minor as u64), num(v))?,
            "FileBugFixVersion" => eq(k, Some(h.file_bugfix as u64), num(v))?,
            "Stub" => eq(k, b(h.stub), yes(v).map(|x| x as u64))?,
            "FileType" => eq(k, Some(h.file_type as u64), num(v))?,
            "Arch64Bit" => eq(k, b(h.arch64), yes(v).map(|x| x as u64))?,
            "Platform" => eq(k, Some(h.platform as u64), num(v))?,
            "PlatformMajor" => eq(k, Some(h.platform_major as u64), num(v))?,
            "PlatformMinor" => eq(k, Some(h.platform_minor as u64), num(v))?,
            "PlatformUpdate" => eq(k, Some(h.platform_update as u64), num(v))?,
            "FileSize" => eq(k, Some(h.file_size), num(v))?,
            "FunctionListOffset" => eq(k, Some(h.function_list.offset), num(v))?,
            // readobj shows the list's span, which includes its count word; the header's size does not.
            "FunctionListSize" => eq(k, Some(h.function_list.size + 4), num(v))?,
            "PublicMetadataOffset" => eq(k, Some(h.public_metadata.offset), num(v))?,
            "PublicMetadataSize" => eq(k, Some(h.public_metadata.size), num(v))?,
            "PrivateMetadataOffset" => eq(k, Some(h.private_metadata.offset), num(v))?,
            "PrivateMetadataSize" => eq(k, Some(h.private_metadata.size), num(v))?,
            "ModuleListOffset" => eq(k, Some(h.module_list.offset), num(v))?,
            "ModuleListSize" => eq(k, Some(h.module_list.size), num(v))?,
            "ReflectionListOffset" => eq(k, lib.reflection_list.map(|r| r.offset), num(v))?,
            "ReflectionListSize" => eq(k, lib.reflection_list.map(|r| r.size), num(v))?,
            "ScriptListOffset" => eq(k, lib.script_list.map(|r| r.offset), num(v))?,
            "ScriptListSize" => eq(k, lib.script_list.map(|r| r.size), num(v))?,
            "DynamicHeaderOffset" => eq(k, lib.dynamic_header.map(|r| r.offset), num(v))?,
            "DynamicHeaderSize" => eq(k, lib.dynamic_header.map(|r| r.size), num(v))?,
            "UUID" => {
                checked.set(checked.get() + 1);
                if lib.uuid.map(|u| uuid(&u)).as_deref() != Some(v.as_str()) {
                    return Err(format!("UUID: reader {:?}, readobj {}", lib.uuid.map(|u| uuid(&u)), v));
                }
            }
            other => return Err(format!("readobj header field {} is not checked", other)),
        }
    }
    // Optional sections the reader found must be printed by readobj too.
    let has = |k: &str| hdr.fields.iter().any(|(f, _)| f == k);
    if lib.reflection_list.is_some() != has("ReflectionListOffset")
        || lib.script_list.is_some() != has("ScriptListOffset")
        || lib.dynamic_header.is_some() != has("DynamicHeaderOffset")
        || lib.uuid.is_some() != has("UUID")
    {
        return Err("optional header sections differ".into());
    }
    let sections = find(ro, "Sections").ok_or("no Sections")?;
    let section = |name: &str| sections.children.iter().find(|s| s.fields.iter().any(|(k, v)| k == "Name" && v == name));
    // Functions.
    let fl = section("FUNCTION_LIST").ok_or("no FUNCTION_LIST")?;
    let fns = find(fl, "Functions").map(|n| &n.children[..]).unwrap_or(&[]);
    if fns.len() != lib.functions.len() {
        return Err(format!("{} functions, readobj {}", lib.functions.len(), fns.len()));
    }
    for (f, n) in lib.functions.iter().zip(fns) {
        let mut seen = std::collections::BTreeSet::new();
        for (k, v) in &n.fields {
            seen.insert(k.as_str());
            let what = format!("{}.{}", f.name, k);
            match k.as_str() {
                "DataSize" => eq(&what, Some(f.entry_size as u64), num(v))?,
                // For an alias entry readobj shows the aliased function as Name and the entry's own name as Alias.
                "Name" => {
                    checked.set(checked.get() + 1);
                    if v.as_str() != f.alias_of.as_deref().unwrap_or(&f.name) {
                        return Err(format!("{}: readobj {}", what, v));
                    }
                }
                "Alias" => {
                    checked.set(checked.get() + 1);
                    if f.alias_of.is_none() || *v != f.name {
                        return Err(format!("{}: reader {:?}, readobj {}", what, f.alias_of, v));
                    }
                }
                "Type" => eq(&what, Some(f.function_type as u64), num(v))?,
                "Hash" => {
                    checked.set(checked.get() + 1);
                    if hex(&f.hash) != *v {
                        return Err(format!("{}: reader {}, readobj {}", what, hex(&f.hash), v));
                    }
                }
                "PublicMetadataOffset" => eq(&what, Some(f.public_metadata_offset), num(v))?,
                "PrivateMetadataOffset" => eq(&what, Some(f.private_metadata_offset), num(v))?,
                "ModuleOffset" => eq(&what, Some(f.module_offset), num(v))?,
                "AIRMajorVersion" => eq(&what, Some(f.air_version.0 as u64), num(v))?,
                "AIRMinorVersion" => eq(&what, Some(f.air_version.1 as u64), num(v))?,
                "MetalMajorVersion" => eq(&what, Some(f.metal_version.0 as u64), num(v))?,
                "MetalMinorVersion" => eq(&what, Some(f.metal_version.1 as u64), num(v))?,
                "ModuleSize" => eq(&what, f.module.map(|m| m.len() as u64), num(v))?,
                "ReflectionOffset" => eq(&what, f.reflection_offset, num(v))?,
                "PostTessellationPatchType" => eq(&what, f.tessellation.map(|t| t.patch_type as u64), num(v))?,
                "PostTessellationControlPointCount" => eq(&what, f.tessellation.map(|t| t.control_points as u64), num(v))?,
                "RenderTargetArrayIndexDataType" => eq(&what, f.render_target_array_index_type.map(|x| x as u64), num(v))?,
                "UnspecializedFunctionHash" => {
                    checked.set(checked.get() + 1);
                    if f.specialization.map(|s| hex(&s.unspecialized_hash)).as_deref() != Some(v.as_str()) {
                        return Err(format!("{}: differs", what));
                    }
                }
                "SpecializationScriptOffset" => eq(&what, f.specialization.map(|s| s.script_offset), num(v))?,
                other => return Err(format!("readobj function field {} is not checked", other)),
            }
        }
        // Tags the reader decoded must be printed by readobj.
        let want = |b: bool, k: &str| -> Result<(), String> {
            if b && !seen.contains(k) {
                return Err(format!("{}: readobj does not print {}", f.name, k));
            }
            Ok(())
        };
        want(f.module.is_some(), "ModuleSize")?;
        // readobj shows RFLT only when the library has a reflection list (stub libraries carry RFLT 0 without one).
        want(f.reflection_offset.is_some() && lib.reflection_list.is_some(), "ReflectionOffset")?;
        if seen.contains("ReflectionOffset") && lib.reflection_list.is_none() {
            return Err(format!("{}: readobj shows a reflection offset without a reflection list", f.name));
        }
        want(f.tessellation.is_some(), "PostTessellationPatchType")?;
        want(f.specialization.is_some(), "SpecializationScriptOffset")?;
        want(f.alias_of.is_some(), "Alias")?;
        want(f.render_target_array_index_type.is_some(), "RenderTargetArrayIndexDataType")?;
    }
    // Module list: one entry per distinct module, in offset order.
    let mut mods: BTreeMap<u64, &[u8]> = BTreeMap::new();
    for f in &lib.functions {
        if let Some(m) = f.module {
            mods.insert(f.module_offset, m);
        }
    }
    let ml = section("MODULE_LIST");
    let ro_mods = ml.and_then(|s| find(s, "ModuleList")).map(|n| &n.children[..]).unwrap_or(&[]);
    if ro_mods.len() != mods.len() {
        return Err(format!("{} modules, readobj {}", mods.len(), ro_mods.len()));
    }
    for ((_, m), n) in mods.iter().zip(ro_mods) {
        let (w, _) = airwgsl::bitcode::split_wrapper(m).map_err(|e| e.to_string())?;
        let w = w.ok_or("module without a wrapper header")?;
        for (k, v) in &n.fields {
            match k.as_str() {
                "WrapperMagic" => eq(k, Some(0x0B17C0DE), num(v))?,
                "WrapperVersion" => eq(k, Some(w.version as u64), num(v))?,
                "BitcodeOffset" => eq(k, Some(w.offset as u64), num(v))?,
                "BitcodeSize" => eq(k, Some(w.size as u64), num(v))?,
                other => return Err(format!("readobj module field {} is not checked", other)),
            }
        }
    }
    // Section record sizes.
    for (name, list, which) in [
        ("PUBLIC_METADATA", "PublicMetadata", mtlb::Section::PublicMetadata),
        ("PRIVATE_METADATA", "PrivateMetadata", mtlb::Section::PrivateMetadata),
        ("REFLECTION_LIST", "Reflections", mtlb::Section::ReflectionList),
        ("SCRIPT_LIST", "Scripts", mtlb::Section::ScriptList),
    ] {
        let ro_sizes: Vec<u64> = section(name)
            .and_then(|s| find(s, list))
            .map(|n| n.children.iter().filter_map(|c| c.fields.iter().find(|(k, _)| k == "DataSize").and_then(|(_, v)| num(v))).collect())
            .unwrap_or_default();
        let sizes: Vec<u64> = lib.section_records(which).map_err(|e| e.to_string())?.iter().map(|rec| rec.size as u64).collect();
        checked.set(checked.get() + 1);
        if sizes != ro_sizes {
            return Err(format!("{} record sizes: reader {:?}, readobj {:?}", name, sizes, ro_sizes));
        }
    }
    let ro_name = section("DYNAMIC_HEADER").and_then(|d| find(d, "DynamicHeader")).and_then(|n| n.fields.iter().find(|(k, _)| k == "LibraryInstallName")).map(|x| x.1.clone());
    let tags = lib.dynamic_header_tags().map_err(|e| e.to_string())?;
    let ours = tags.iter().find(|t| &t.name == b"NAME").map(|t| String::from_utf8_lossy(&t.data[..t.data.len().saturating_sub(1)]).to_string());
    checked.set(checked.get() + 1);
    if ours != ro_name {
        return Err(format!("install name: reader {:?}, readobj {:?}", ours, ro_name));
    }
    // Symbols: every function name.
    let syms: Vec<String> = find(ro, "Symbols")
        .map(|n| n.children.iter().filter_map(|c| c.fields.iter().find(|(k, _)| k == "Name").map(|(_, v)| v.rsplit_once(" (").map(|x| x.0.to_string()).unwrap_or(v.clone()))).collect())
        .unwrap_or_default();
    let names: Vec<String> = lib.functions.iter().map(|f| f.name.clone()).collect();
    checked.set(checked.get() + 1);
    if syms != names {
        return Err(format!("symbols differ: reader {} names, readobj {}", names.len(), syms.len()));
    }
    Ok(checked.get())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: mtlb_check UNITS_DIR READOBJ_DIR REPORT.tsv");
        std::process::exit(2);
    }
    let mut names: Vec<String> = std::fs::read_dir(&args[1])
        .expect("units dir")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.ends_with(".metallib"))
        .collect();
    names.sort();
    let mut report = std::fs::File::create(&args[3]).expect("report");
    let (mut ok, mut bad, mut fields, mut functions) = (0, 0, 0usize, 0usize);
    for n in &names {
        let data = std::fs::read(format!("{}/{}", args[1], n)).expect("read");
        let ro = std::fs::read_to_string(format!("{}/{}.txt", args[2], n)).unwrap_or_default();
        let result = match mtlb::parse_metallib(&data) {
            Err(e) => Err(e.to_string()),
            Ok(lib) if lib.header.file_size as usize != data.len() => Err("bytes after the metallib".into()),
            Ok(lib) => {
                functions += lib.functions.len();
                compare(&lib, &parse_readobj(&ro))
            }
        };
        match result {
            Ok(c) => {
                ok += 1;
                fields += c;
                writeln!(report, "{}\tok\t{}", n, c).unwrap();
            }
            Err(e) => {
                bad += 1;
                writeln!(report, "{}\tmismatch\t{}", n, e).unwrap();
            }
        }
    }
    println!("metallibs={} ok={} mismatch={} functions={} fields_compared={}", names.len(), ok, bad, functions, fields);
    if bad != 0 {
        std::process::exit(1);
    }
}
