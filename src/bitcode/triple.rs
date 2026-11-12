//! `UpgradeDataLayoutString` (`llvm/lib/IR/AutoUpgrade.cpp`, release 21.1.8), which the bitcode reader applies to
//! every module's data layout, and the parts of `llvm::Triple` (`llvm/lib/TargetParser/Triple.cpp`) it consults.
//!
//! AIR's architectures (`air64`, `air64_v26`) are unknown to LLVM, so their layouts are never changed; the upgrade
//! matters for the foreign (host) modules that ship next to AIR.

/// The architecture classes the upgrade distinguishes (`Triple::ArchType`, grouped by the predicates it calls).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arch {
    R600,
    Amdgcn,
    Spir,
    /// `spirv`: SPIR-V logical.
    SpirvLogical,
    Spirv,
    LoongArch64,
    Riscv64,
    AArch64,
    Sparc,
    Mips64,
    Ppc64,
    Wasm,
    X86,
    X86_64,
    /// Any other architecture LLVM knows, or one it does not: the upgrade leaves their layouts alone.
    Other,
}

/// `parseArch`. Names LLVM hands to its ARM target parser (`arm*`, `thumb*`, `aarch64*` beyond the exact names below)
/// are refused: that parser is not implemented here, and it decides whether the name is AArch64.
fn parse_arch(name: &str) -> Result<Arch, String> {
    use Arch::*;
    let a = match name {
        "i386" | "i486" | "i586" | "i686" | "i786" | "i886" | "i986" => X86,
        "amd64" | "x86_64" | "x86_64h" => X86_64,
        "powerpc64" | "ppu" | "ppc64" | "powerpc64le" | "ppc64le" => Ppc64,
        "aarch64" | "aarch64_be" | "aarch64_32" | "arm64" | "arm64_32" | "arm64e" | "arm64ec" => AArch64,
        "mips64" | "mips64eb" | "mipsn32" | "mipsisa64r6" | "mips64r6" | "mipsn32r6" | "mips64el" | "mipsn32el"
        | "mipsisa64r6el" | "mips64r6el" | "mipsn32r6el" => Mips64,
        "r600" => R600,
        "amdgcn" => Amdgcn,
        "riscv64" => Riscv64,
        "sparc" | "sparcel" | "sparcv9" | "sparc64" => Sparc,
        "spir" | "spir64" => Spir,
        "spirv" | "spirv1.5" | "spirv1.6" => SpirvLogical,
        "spirv32" | "spirv32v1.0" | "spirv32v1.1" | "spirv32v1.2" | "spirv32v1.3" | "spirv32v1.4" | "spirv32v1.5"
        | "spirv32v1.6" | "spirv64" | "spirv64v1.0" | "spirv64v1.1" | "spirv64v1.2" | "spirv64v1.3" | "spirv64v1.4"
        | "spirv64v1.5" | "spirv64v1.6" => Spirv,
        "wasm32" | "wasm64" => Wasm,
        "loongarch64" => LoongArch64,
        // The rest of LLVM's exact names map to architectures the upgrade does not single out.
        "xscale" | "xscaleeb" | "arc" | "arm" | "armeb" | "thumb" | "thumbeb" => Other,
        n if n.starts_with("arm") || n.starts_with("thumb") || n.starts_with("aarch64") => {
            return Err(format!("target architecture \"{}\" needs LLVM's ARM target parser to classify", n));
        }
        _ => Other,
    };
    Ok(a)
}

/// `parseOS`, reduced to the two answers the upgrade asks: `elfiamcu` and Windows. `StringSwitch::StartsWith` takes
/// the first prefix that matches, and none of the earlier prefixes is a prefix of these.
fn os_is(name: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| name.starts_with(p))
}

/// `parseEnvironment` for the two answers the upgrade asks: unknown, and MSVC. The table's earlier prefixes that are
/// prefixes of "msvc" do not exist, so `msvc*` is MSVC; a name matching none of the table is unknown.
const ENVIRONMENT_PREFIXES: &[&str] = &[
    "eabihf",
    "eabi",
    "gnuabin32",
    "gnuabi64",
    "gnueabihft64",
    "gnueabihf",
    "gnueabit64",
    "gnueabi",
    "gnuf32",
    "gnuf64",
    "gnusf",
    "gnux32",
    "gnu_ilp32",
    "code16",
    "gnut64",
    "gnu",
    "android",
    "muslabin32",
    "muslabi64",
    "musleabihf",
    "musleabi",
    "muslf32",
    "muslsf",
    "muslx32",
    "musl",
    "msvc",
    "itanium",
    "cygnus",
    "coreclr",
    "simulator",
    "macabi",
    "pixel",
    "vertex",
    "geometry",
    "hull",
    "domain",
    "compute",
    "library",
    "raygeneration",
    "intersection",
    "anyhit",
    "closesthit",
    "miss",
    "callable",
    "mesh",
    "amplification",
    "opencl",
    "ohos",
    "pauthtest",
    "llvm",
    "mlibc",
];

/// The datalayout upgrade for `dl` under the target triple `triple`.
pub(crate) fn upgrade_datalayout(dl: &str, triple: &str) -> Result<String, String> {
    use Arch::*;
    // `Triple(std::string)`: up to four '-'-separated components, parsed by position.
    let parts: Vec<&str> = triple.splitn(4, '-').collect();
    let arch = parse_arch(parts[0])?;
    let os = parts.get(2).copied();
    let env = parts.get(3).copied();

    if matches!(arch, R600 | Spir | Spirv) && !dl.contains("-G") && !dl.starts_with('G') {
        return Ok(if dl.is_empty() { "G1".to_string() } else { format!("{}-G1", dl) });
    }
    if matches!(arch, LoongArch64 | Riscv64) {
        return Ok(match dl.find("-n64-") {
            Some(i) => format!("{}-n32:64-{}", &dl[..i], &dl[i + 5..]),
            None => dl.to_string(),
        });
    }
    let mut res = dl.to_string();
    if arch == Amdgcn {
        if !dl.contains("-G") && !dl.starts_with('G') {
            res.push_str(if res.is_empty() { "G1" } else { "-G1" });
        }
        if !dl.contains("-ni") && !dl.starts_with("ni") {
            res.push_str("-ni:7:8:9");
        }
        if dl.ends_with("ni:7") {
            res.push_str(":8:9");
        }
        if dl.ends_with("ni:7:8") {
            res.push_str(":9");
        }
        if !dl.contains("-p7") && !dl.starts_with("p7") {
            res.push_str("-p7:160:256:256:32");
        }
        if !dl.contains("-p8") && !dl.starts_with("p8") {
            res.push_str("-p8:128:128:128:48");
        }
        const OLD_P8: &str = "-p8:128:128-";
        if dl.contains(OLD_P8)
            && let Some(i) = res.find(OLD_P8)
        {
            res.replace_range(i..i + OLD_P8.len(), "-p8:128:128:128:48-");
        }
        if !dl.contains("-p9") && !dl.starts_with("p9") {
            res.push_str("-p9:192:256:256:32");
        }
        return Ok(res);
    }
    let add_ptr32_ptr64 = |res: &mut String| {
        const ADDR_SPACES: &str = "-p270:32:32-p271:32:32-p272:64:64";
        if !dl.contains(ADDR_SPACES)
            && let Some((head, tail)) = match_ptr_prefix(res)
        {
            *res = format!("{}{}{}", head, ADDR_SPACES, tail);
        }
    };
    if arch == AArch64 {
        if !dl.is_empty() && !dl.contains("-Fn32") {
            res.push_str("-Fn32");
        }
        add_ptr32_ptr64(&mut res);
        return Ok(res);
    }
    if matches!(arch, Sparc | Ppc64 | Wasm) || (arch == Mips64 && !dl.contains("m:m")) {
        const I64: &str = "-i64:64";
        if !res.contains("-i128:128")
            && let Some(p) = res.find(I64)
        {
            res.insert_str(p + I64.len(), "-i128:128");
        }
        return Ok(res);
    }
    if !matches!(arch, X86 | X86_64) {
        return Ok(res);
    }
    add_ptr32_ptr64(&mut res);
    let iamcu = os.map(|o| os_is(o, &["elfiamcu"])).unwrap_or(false);
    if !iamcu
        && !res.contains("-i128:128")
        && let Some((head, tail)) = match_i128_split(&res)
    {
        res = format!("{}-i128:128{}", head, tail);
    }
    // `isWindowsMSVCEnvironment() && !isArch64Bit()`.
    let windows = os.map(|o| os_is(o, &["win32", "windows"])).unwrap_or(false);
    let env_unknown = env.map(|e| !ENVIRONMENT_PREFIXES.iter().any(|p| e.starts_with(p))).unwrap_or(true);
    let env_msvc = env.map(|e| e.starts_with("msvc")).unwrap_or(false);
    if windows
        && (env_msvc || env_unknown)
        && arch == X86
        && let Some(i) = res.find("-f80:32-")
    {
        res = format!("{}-f80:128-{}", &res[..i], &res[i + 8..]);
    }
    Ok(res)
}

/// `^([Ee]-m:[a-z](-p:32:32)?)(-.*)$` with POSIX submatch rules (the first group as long as possible): the two groups.
fn match_ptr_prefix(s: &str) -> Option<(&str, &str)> {
    let b = s.as_bytes();
    if b.len() < 5 || !matches!(b[0], b'E' | b'e') || &b[1..4] != b"-m:" || !b[4].is_ascii_lowercase() {
        return None;
    }
    if s[5..].starts_with("-p:32:32") && s[13..].starts_with('-') {
        return Some((&s[..13], &s[13..]));
    }
    if s[5..].starts_with('-') {
        return Some((&s[..5], &s[5..]));
    }
    None
}

/// `^(e(-[mpi][^-]*)*)((-[^mpi][^-]*)*)$` with POSIX submatch rules: the longest first group whose remainder the
/// second group matches whole.
fn match_i128_split(s: &str) -> Option<(&str, &str)> {
    let b = s.as_bytes();
    if b.first() != Some(&b'e') {
        return None;
    }
    // Ends of the first group: after 'e', each iteration is '-', one of m/p/i, then any run of non-'-' bytes.
    let mut ends = vec![1usize];
    let mut p = 1;
    while p + 1 < b.len() && b[p] == b'-' && matches!(b[p + 1], b'm' | b'p' | b'i') {
        let mut q = p + 2;
        ends.push(q);
        while q < b.len() && b[q] != b'-' {
            q += 1;
            ends.push(q);
        }
        p = q;
    }
    ends.into_iter().rev().find(|&e| rest_matches(&b[e..])).map(|e| (&s[..e], &s[e..]))
}

/// Whether `(-[^mpi][^-]*)*` matches all of `b`.
fn rest_matches(b: &[u8]) -> bool {
    // Positions the star can reach; an iteration from `r` is '-', a byte other than m/p/i, then non-'-' bytes.
    let mut reach = vec![false; b.len() + 1];
    reach[0] = true;
    for r in 0..b.len() {
        if !reach[r] || b[r] != b'-' || r + 1 >= b.len() || matches!(b[r + 1], b'm' | b'p' | b'i') {
            continue;
        }
        let mut q = r + 2;
        reach[q] = true;
        while q < b.len() && b[q] != b'-' {
            q += 1;
            reach[q] = true;
        }
    }
    reach[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_layouts_are_unchanged() {
        let dl = "e-p:64:64:64-i1:8:8-i8:8:8-n8:16:32";
        assert_eq!(upgrade_datalayout(dl, "air64-apple-ios17.7.0").unwrap(), dl);
        assert_eq!(upgrade_datalayout(dl, "air64_v26-apple-ios17.7.0").unwrap(), dl);
    }

    #[test]
    fn x86_64_gets_pointer_spaces_and_i128() {
        // The layout of the shared cache's x86_64 modules, and llvm-dis's result for it.
        let dl = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";
        assert_eq!(
            upgrade_datalayout(dl, "x86_64-apple-macosx11.0.0").unwrap(),
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
        );
        let dl = "e-m:o-i64:64-f80:128-n8:16:32:64-S128";
        assert_eq!(
            upgrade_datalayout(dl, "x86_64-apple-macosx11.0.0").unwrap(),
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
        );
    }

    #[test]
    fn i128_split_follows_posix_rules() {
        assert_eq!(match_i128_split("e-m:e-p:32:32-i64:64-f80:32-n8"), Some(("e-m:e-p:32:32-i64:64", "-f80:32-n8")));
        assert_eq!(match_i128_split("e-f80:32-i64:64"), None);
        assert_eq!(match_i128_split("E-m:e"), None);
        assert_eq!(match_i128_split("e"), Some(("e", "")));
    }

    #[test]
    fn arm_names_beyond_the_table_are_refused() {
        assert!(upgrade_datalayout("e", "armv7-apple-ios").is_err());
        assert!(upgrade_datalayout("e", "arm64-apple-ios").is_ok());
    }
}
