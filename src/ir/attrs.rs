//! Attributes, as LLVM 21 defines them (`llvm/include/llvm/IR/Attributes.td`, release 21.1.8).
//!
//! [`AttrKind`] lists the attribute kinds in the order of LLVM's `Attribute::AttrKind` enumeration, which TableGen
//! generates by class (enum, type, int, constant range, constant range list attributes), each class sorted by
//! definition name. That order is the order attributes take inside a set ([`AttrSet`]), and so the order they print
//! in.

use super::types::TypeId;

macro_rules! attr_kinds {
    ($($variant:ident = $name:literal, $class:ident;)*) => {
        /// An attribute kind. The discriminants follow `Attribute::AttrKind` (`None` is 0).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u8)]
        pub enum AttrKind {
            $($variant,)*
        }

        impl AttrKind {
            /// Every kind, in enumeration order.
            pub const ALL: &'static [AttrKind] = &[$(AttrKind::$variant,)*];

            /// The attribute's spelling in LLVM assembly.
            pub fn name(self) -> &'static str {
                match self {
                    $(AttrKind::$variant => $name,)*
                }
            }

            /// What kind of value the attribute carries.
            pub fn class(self) -> AttrClass {
                match self {
                    $(AttrKind::$variant => AttrClass::$class,)*
                }
            }
        }
    };
}

/// The value an attribute kind carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrClass {
    /// No value.
    Enum,
    /// A type (`byval(<ty>)` and the like).
    Type,
    /// An integer.
    Int,
    /// A constant range.
    ConstantRange,
    /// A list of constant ranges.
    ConstantRangeList,
}

attr_kinds! {
    AllocAlign = "allocalign", Enum;
    AllocatedPointer = "allocptr", Enum;
    AlwaysInline = "alwaysinline", Enum;
    Builtin = "builtin", Enum;
    Cold = "cold", Enum;
    Convergent = "convergent", Enum;
    CoroDestroyOnlyWhenComplete = "coro_only_destroy_when_complete", Enum;
    CoroElideSafe = "coro_elide_safe", Enum;
    DeadOnReturn = "dead_on_return", Enum;
    DeadOnUnwind = "dead_on_unwind", Enum;
    DisableSanitizerInstrumentation = "disable_sanitizer_instrumentation", Enum;
    FnRetThunkExtern = "fn_ret_thunk_extern", Enum;
    Hot = "hot", Enum;
    HybridPatchable = "hybrid_patchable", Enum;
    ImmArg = "immarg", Enum;
    InReg = "inreg", Enum;
    InlineHint = "inlinehint", Enum;
    JumpTable = "jumptable", Enum;
    MinSize = "minsize", Enum;
    MustProgress = "mustprogress", Enum;
    Naked = "naked", Enum;
    Nest = "nest", Enum;
    NoAlias = "noalias", Enum;
    NoBuiltin = "nobuiltin", Enum;
    NoCallback = "nocallback", Enum;
    NoCfCheck = "nocf_check", Enum;
    NoDivergenceSource = "nodivergencesource", Enum;
    NoDuplicate = "noduplicate", Enum;
    NoExt = "noext", Enum;
    NoFree = "nofree", Enum;
    NoImplicitFloat = "noimplicitfloat", Enum;
    NoInline = "noinline", Enum;
    NoMerge = "nomerge", Enum;
    NoProfile = "noprofile", Enum;
    NoRecurse = "norecurse", Enum;
    NoRedZone = "noredzone", Enum;
    NoReturn = "noreturn", Enum;
    NoSanitizeBounds = "nosanitize_bounds", Enum;
    NoSanitizeCoverage = "nosanitize_coverage", Enum;
    NoSync = "nosync", Enum;
    NoUndef = "noundef", Enum;
    NoUnwind = "nounwind", Enum;
    NonLazyBind = "nonlazybind", Enum;
    NonNull = "nonnull", Enum;
    NullPointerIsValid = "null_pointer_is_valid", Enum;
    OptForFuzzing = "optforfuzzing", Enum;
    OptimizeForDebugging = "optdebug", Enum;
    OptimizeForSize = "optsize", Enum;
    OptimizeNone = "optnone", Enum;
    PresplitCoroutine = "presplitcoroutine", Enum;
    ReadNone = "readnone", Enum;
    ReadOnly = "readonly", Enum;
    Returned = "returned", Enum;
    ReturnsTwice = "returns_twice", Enum;
    SExt = "signext", Enum;
    SafeStack = "safestack", Enum;
    SanitizeAddress = "sanitize_address", Enum;
    SanitizeHWAddress = "sanitize_hwaddress", Enum;
    SanitizeMemTag = "sanitize_memtag", Enum;
    SanitizeMemory = "sanitize_memory", Enum;
    SanitizeNumericalStability = "sanitize_numerical_stability", Enum;
    SanitizeRealtime = "sanitize_realtime", Enum;
    SanitizeRealtimeBlocking = "sanitize_realtime_blocking", Enum;
    SanitizeThread = "sanitize_thread", Enum;
    SanitizeType = "sanitize_type", Enum;
    ShadowCallStack = "shadowcallstack", Enum;
    SkipProfile = "skipprofile", Enum;
    Speculatable = "speculatable", Enum;
    SpeculativeLoadHardening = "speculative_load_hardening", Enum;
    StackProtect = "ssp", Enum;
    StackProtectReq = "sspreq", Enum;
    StackProtectStrong = "sspstrong", Enum;
    StrictFP = "strictfp", Enum;
    SwiftAsync = "swiftasync", Enum;
    SwiftError = "swifterror", Enum;
    SwiftSelf = "swiftself", Enum;
    WillReturn = "willreturn", Enum;
    Writable = "writable", Enum;
    WriteOnly = "writeonly", Enum;
    ZExt = "zeroext", Enum;
    ByRef = "byref", Type;
    ByVal = "byval", Type;
    ElementType = "elementtype", Type;
    InAlloca = "inalloca", Type;
    Preallocated = "preallocated", Type;
    StructRet = "sret", Type;
    Alignment = "align", Int;
    AllocKind = "allockind", Int;
    AllocSize = "allocsize", Int;
    Captures = "captures", Int;
    Dereferenceable = "dereferenceable", Int;
    DereferenceableOrNull = "dereferenceable_or_null", Int;
    Memory = "memory", Int;
    NoFPClass = "nofpclass", Int;
    StackAlignment = "alignstack", Int;
    UWTable = "uwtable", Int;
    VScaleRange = "vscale_range", Int;
    Range = "range", ConstantRange;
    Initializes = "initializes", ConstantRangeList;
}

impl AttrKind {
    /// The kind a bitcode `ATTR_KIND_*` code names (`getAttrFromCode` in `BitcodeReader.cpp`). Codes whose
    /// attributes LLVM 21 no longer has (`nocapture`, `argmemonly`, `inaccessiblememonly`,
    /// `inaccessiblemem_or_argmemonly`) give `None`; the reader upgrades them itself.
    pub fn from_bitcode(code: u64) -> Option<AttrKind> {
        use AttrKind::*;
        Some(match code {
            1 => Alignment,
            2 => AlwaysInline,
            3 => ByVal,
            4 => InlineHint,
            5 => InReg,
            6 => MinSize,
            7 => Naked,
            8 => Nest,
            9 => NoAlias,
            10 => NoBuiltin,
            12 => NoDuplicate,
            13 => NoImplicitFloat,
            14 => NoInline,
            15 => NonLazyBind,
            16 => NoRedZone,
            17 => NoReturn,
            18 => NoUnwind,
            19 => OptimizeForSize,
            20 => ReadNone,
            21 => ReadOnly,
            22 => Returned,
            23 => ReturnsTwice,
            24 => SExt,
            25 => StackAlignment,
            26 => StackProtect,
            27 => StackProtectReq,
            28 => StackProtectStrong,
            29 => StructRet,
            30 => SanitizeAddress,
            31 => SanitizeThread,
            32 => SanitizeMemory,
            33 => UWTable,
            34 => ZExt,
            35 => Builtin,
            36 => Cold,
            37 => OptimizeNone,
            38 => InAlloca,
            39 => NonNull,
            40 => JumpTable,
            41 => Dereferenceable,
            42 => DereferenceableOrNull,
            43 => Convergent,
            44 => SafeStack,
            46 => SwiftSelf,
            47 => SwiftError,
            48 => NoRecurse,
            51 => AllocSize,
            52 => WriteOnly,
            53 => Speculatable,
            54 => StrictFP,
            55 => SanitizeHWAddress,
            56 => NoCfCheck,
            57 => OptForFuzzing,
            58 => ShadowCallStack,
            59 => SpeculativeLoadHardening,
            60 => ImmArg,
            61 => WillReturn,
            62 => NoFree,
            63 => NoSync,
            64 => SanitizeMemTag,
            65 => Preallocated,
            66 => NoMerge,
            67 => NullPointerIsValid,
            68 => NoUndef,
            69 => ByRef,
            70 => MustProgress,
            71 => NoCallback,
            72 => Hot,
            73 => NoProfile,
            74 => VScaleRange,
            75 => SwiftAsync,
            76 => NoSanitizeCoverage,
            77 => ElementType,
            78 => DisableSanitizerInstrumentation,
            79 => NoSanitizeBounds,
            80 => AllocAlign,
            81 => AllocatedPointer,
            82 => AllocKind,
            83 => PresplitCoroutine,
            84 => FnRetThunkExtern,
            85 => SkipProfile,
            86 => Memory,
            87 => NoFPClass,
            88 => OptimizeForDebugging,
            89 => Writable,
            90 => CoroDestroyOnlyWhenComplete,
            91 => DeadOnUnwind,
            92 => Range,
            93 => SanitizeNumericalStability,
            94 => Initializes,
            96 => SanitizeRealtime,
            97 => SanitizeRealtimeBlocking,
            98 => CoroElideSafe,
            99 => NoExt,
            100 => NoDivergenceSource,
            101 => SanitizeType,
            102 => Captures,
            103 => DeadOnReturn,
            _ => return None,
        })
    }
}

/// `ConstantRange`: the half-open range `[lower, upper)` of `bits`-bit integers, wrapping; `lower == upper` is the
/// full set when both are the maximum value and the empty set when both are 0. Words are little-endian.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstantRange {
    pub bits: u32,
    pub lower: Vec<u64>,
    pub upper: Vec<u64>,
}

/// One attribute.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Attribute {
    /// An attribute without a value.
    Enum(AttrKind),
    /// An integer attribute, holding LLVM's in-memory value: `align`, `alignstack`, `dereferenceable` and
    /// `dereferenceable_or_null` hold bytes; `allocsize` and `vscale_range` their packed forms; `uwtable` the
    /// `UWTableKind`; `allockind` the `AllocFnKind` bits; `memory` the `MemoryEffects` bits (two per location:
    /// argmem, inaccessiblemem, errnomem, other); `captures` the `CaptureInfo` bits (other components << 4 | return
    /// components); `nofpclass` the `FPClassTest` mask.
    Int(AttrKind, u64),
    /// A type attribute. The type is `None` only transiently while the reader upgrades a typeless `byval`, `sret` or
    /// `inalloca`.
    Type(AttrKind, Option<TypeId>),
    Range(AttrKind, ConstantRange),
    RangeList(AttrKind, Vec<ConstantRange>),
    /// A string attribute: key and value (empty when it has none).
    Str(Vec<u8>, Vec<u8>),
}

impl Attribute {
    /// The enum kind, or `None` for a string attribute.
    pub fn kind(&self) -> Option<AttrKind> {
        match self {
            Attribute::Enum(k) | Attribute::Int(k, _) | Attribute::Type(k, _) => Some(*k),
            Attribute::Range(k, _) | Attribute::RangeList(k, _) => Some(*k),
            Attribute::Str(..) => None,
        }
    }

    /// LLVM's order: kinds by enumeration value, then string attributes by key and value (bytewise). Two attributes
    /// of the same kind only coexist as strings with different keys.
    fn sort_key(&self) -> (u8, &[u8], &[u8]) {
        match self {
            Attribute::Str(k, v) => (u8::MAX, k, v),
            other => (other.kind().unwrap() as u8, &[], &[]),
        }
    }
}

/// A set of attributes for one position (function, return value or a parameter): at most one per kind and per
/// string key, in LLVM's order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttrSet {
    attrs: Vec<Attribute>,
}

impl AttrSet {
    pub fn new() -> AttrSet {
        AttrSet::default()
    }

    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Attribute> {
        self.attrs.iter()
    }

    pub fn len(&self) -> usize {
        self.attrs.len()
    }

    /// Adds `a`, replacing an attribute of the same kind (or string key), as `AttrBuilder` does.
    pub fn add(&mut self, a: Attribute) {
        let pos = self.attrs.iter().position(|b| match (&a, b) {
            (Attribute::Str(k1, _), Attribute::Str(k2, _)) => k1 == k2,
            (Attribute::Str(..), _) | (_, Attribute::Str(..)) => false,
            (x, y) => x.kind() == y.kind(),
        });
        match pos {
            Some(i) => self.attrs[i] = a,
            None => self.attrs.push(a),
        }
        self.attrs.sort_by(|x, y| x.sort_key().cmp(&y.sort_key()));
    }

    pub fn get(&self, kind: AttrKind) -> Option<&Attribute> {
        self.attrs.iter().find(|a| a.kind() == Some(kind))
    }

    pub fn get_str(&self, key: &[u8]) -> Option<&[u8]> {
        self.attrs.iter().find_map(|a| match a {
            Attribute::Str(k, v) if k.as_slice() == key => Some(v.as_slice()),
            _ => None,
        })
    }

    pub fn has(&self, kind: AttrKind) -> bool {
        self.get(kind).is_some()
    }

    pub fn remove(&mut self, kind: AttrKind) {
        self.attrs.retain(|a| a.kind() != Some(kind));
    }

    pub fn remove_str(&mut self, key: &[u8]) {
        self.attrs.retain(|a| !matches!(a, Attribute::Str(k, _) if k.as_slice() == key));
    }

    /// Merges `other` into this set; `other`'s attributes replace same-kind ones.
    pub fn merge(&mut self, other: &AttrSet) {
        for a in other.iter() {
            self.add(a.clone());
        }
    }
}

/// Index of an interned [`AttrSet`] in a module; [`AttrSetId::EMPTY`] is the empty set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct AttrSetId(pub u32);

impl AttrSetId {
    pub const EMPTY: AttrSetId = AttrSetId(0);
}

/// The attributes of a function or call: function, return value, and each parameter.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttrList {
    pub function: AttrSetId,
    pub ret: AttrSetId,
    /// Parameter `i` at position `i`; positions past the end are empty.
    pub params: Vec<AttrSetId>,
}

impl AttrList {
    pub fn param(&self, i: usize) -> AttrSetId {
        self.params.get(i).copied().unwrap_or(AttrSetId::EMPTY)
    }

    pub fn is_empty(&self) -> bool {
        self.function == AttrSetId::EMPTY
            && self.ret == AttrSetId::EMPTY
            && self.params.iter().all(|p| *p == AttrSetId::EMPTY)
    }

    /// Drops trailing empty parameter sets, the canonical form.
    pub fn trim(&mut self) {
        while self.params.last() == Some(&AttrSetId::EMPTY) {
            self.params.pop();
        }
    }
}

/// `ModRefInfo` access kinds as `memory(...)` spells them.
pub fn mod_ref_name(mr: u64) -> &'static str {
    match mr & 3 {
        0 => "none",
        1 => "read",
        2 => "write",
        _ => "readwrite",
    }
}

/// `MemoryEffects` locations in order: argmem, inaccessiblemem, errnomem, other.
pub const MEMORY_LOCATIONS: usize = 4;

/// The `memory(...)` spelling of a `MemoryEffects` value (`Attribute::getAsString`).
pub fn memory_string(me: u64) -> String {
    let get = |loc: usize| (me >> (2 * loc)) & 3;
    let other = get(3);
    let mut s = String::from("memory(");
    let mut first = true;
    // "other" is printed as the default access kind, unless it is none and some other location differs.
    let all_same = (0..MEMORY_LOCATIONS).all(|l| get(l) == other);
    if other != 0 || all_same {
        first = false;
        s.push_str(mod_ref_name(other));
    }
    for (loc, name) in [(0usize, "argmem: "), (1, "inaccessiblemem: "), (2, "errnomem: ")] {
        let mr = get(loc);
        if mr == other {
            continue;
        }
        if !first {
            s.push_str(", ");
        }
        first = false;
        s.push_str(name);
        s.push_str(mod_ref_name(mr));
    }
    s.push(')');
    s
}

fn capture_components(cc: u64) -> String {
    if cc == 0 {
        return "none".into();
    }
    let mut parts = Vec::new();
    // Address = 0b11, AddressIsNull = 0b01; Provenance = 0b1100, ReadProvenance = 0b0100.
    if cc & 3 == 1 {
        parts.push("address_is_null");
    } else if cc & 3 != 0 {
        parts.push("address");
    }
    if cc & 0xc == 4 {
        parts.push("read_provenance");
    }
    if cc & 0xc == 0xc {
        parts.push("provenance");
    }
    parts.join(", ")
}

/// The `captures(...)` spelling of a `CaptureInfo` value.
pub fn captures_string(ci: u64) -> String {
    let other = (ci >> 4) & 0xf;
    let ret = ci & 0xf;
    let mut parts = Vec::new();
    if other != 0 || other == ret {
        parts.push(capture_components(other));
    }
    if other != ret {
        parts.push(format!("ret: {}", capture_components(ret)));
    }
    format!("captures({})", parts.join(", "))
}

/// The `nofpclass(...)` spelling of an `FPClassTest` mask.
pub fn nofpclass_string(mut mask: u64) -> String {
    const NAMES: &[(u64, &str)] = &[
        (0x3ff, "all"),
        (0x003, "nan"),
        (0x001, "snan"),
        (0x002, "qnan"),
        (0x204, "inf"),
        (0x004, "ninf"),
        (0x200, "pinf"),
        (0x060, "zero"),
        (0x020, "nzero"),
        (0x040, "pzero"),
        (0x090, "sub"),
        (0x010, "nsub"),
        (0x080, "psub"),
        (0x108, "norm"),
        (0x008, "nnorm"),
        (0x100, "pnorm"),
    ];
    if mask == 0 {
        return "nofpclass(none)".into();
    }
    let mut parts = Vec::new();
    for &(bits, name) in NAMES {
        if mask & bits == bits {
            parts.push(name);
            mask &= !bits;
        }
    }
    format!("nofpclass({})", parts.join(" "))
}
