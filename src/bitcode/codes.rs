//! Block ids and record codes of LLVM bitcode (`llvm/include/llvm/Bitcode/LLVMBitCodes.h`, release 21.1.8).

// Block ids.
pub const MODULE_BLOCK_ID: u32 = 8;
pub const PARAMATTR_BLOCK_ID: u32 = 9;
pub const PARAMATTR_GROUP_BLOCK_ID: u32 = 10;
pub const CONSTANTS_BLOCK_ID: u32 = 11;
pub const FUNCTION_BLOCK_ID: u32 = 12;
pub const IDENTIFICATION_BLOCK_ID: u32 = 13;
pub const VALUE_SYMTAB_BLOCK_ID: u32 = 14;
pub const METADATA_BLOCK_ID: u32 = 15;
pub const METADATA_ATTACHMENT_ID: u32 = 16;
pub const TYPE_BLOCK_ID_NEW: u32 = 17;
pub const USELIST_BLOCK_ID: u32 = 18;
pub const MODULE_STRTAB_BLOCK_ID: u32 = 19;
pub const GLOBALVAL_SUMMARY_BLOCK_ID: u32 = 20;
pub const OPERAND_BUNDLE_TAGS_BLOCK_ID: u32 = 21;
pub const METADATA_KIND_BLOCK_ID: u32 = 22;
pub const STRTAB_BLOCK_ID: u32 = 23;
pub const FULL_LTO_GLOBALVAL_SUMMARY_BLOCK_ID: u32 = 24;
pub const SYMTAB_BLOCK_ID: u32 = 25;
pub const SYNC_SCOPE_NAMES_BLOCK_ID: u32 = 26;

pub fn block_name(id: u32) -> &'static str {
    match id {
        0 => "BLOCKINFO",
        MODULE_BLOCK_ID => "MODULE_BLOCK",
        PARAMATTR_BLOCK_ID => "PARAMATTR_BLOCK",
        PARAMATTR_GROUP_BLOCK_ID => "PARAMATTR_GROUP_BLOCK",
        CONSTANTS_BLOCK_ID => "CONSTANTS_BLOCK",
        FUNCTION_BLOCK_ID => "FUNCTION_BLOCK",
        IDENTIFICATION_BLOCK_ID => "IDENTIFICATION_BLOCK",
        VALUE_SYMTAB_BLOCK_ID => "VALUE_SYMTAB_BLOCK",
        METADATA_BLOCK_ID => "METADATA_BLOCK",
        METADATA_ATTACHMENT_ID => "METADATA_ATTACHMENT_BLOCK",
        TYPE_BLOCK_ID_NEW => "TYPE_BLOCK",
        USELIST_BLOCK_ID => "USELIST_BLOCK",
        MODULE_STRTAB_BLOCK_ID => "MODULE_STRTAB_BLOCK",
        GLOBALVAL_SUMMARY_BLOCK_ID => "GLOBALVAL_SUMMARY_BLOCK",
        OPERAND_BUNDLE_TAGS_BLOCK_ID => "OPERAND_BUNDLE_TAGS_BLOCK",
        METADATA_KIND_BLOCK_ID => "METADATA_KIND_BLOCK",
        STRTAB_BLOCK_ID => "STRTAB_BLOCK",
        FULL_LTO_GLOBALVAL_SUMMARY_BLOCK_ID => "FULL_LTO_GLOBALVAL_SUMMARY_BLOCK",
        SYMTAB_BLOCK_ID => "SYMTAB_BLOCK",
        SYNC_SCOPE_NAMES_BLOCK_ID => "SYNC_SCOPE_NAMES_BLOCK",
        _ => "unknown block",
    }
}

pub const IDENTIFICATION_CODE_STRING: u32 = 1;
pub const IDENTIFICATION_CODE_EPOCH: u32 = 2;
pub const BITCODE_CURRENT_EPOCH: u64 = 0;

pub const MODULE_CODE_VERSION: u32 = 1;
pub const MODULE_CODE_TRIPLE: u32 = 2;
pub const MODULE_CODE_DATALAYOUT: u32 = 3;
pub const MODULE_CODE_ASM: u32 = 4;
pub const MODULE_CODE_SECTIONNAME: u32 = 5;
pub const MODULE_CODE_DEPLIB: u32 = 6;
pub const MODULE_CODE_GLOBALVAR: u32 = 7;
pub const MODULE_CODE_FUNCTION: u32 = 8;
pub const MODULE_CODE_ALIAS_OLD: u32 = 9;
pub const MODULE_CODE_GCNAME: u32 = 11;
pub const MODULE_CODE_COMDAT: u32 = 12;
pub const MODULE_CODE_VSTOFFSET: u32 = 13;
pub const MODULE_CODE_ALIAS: u32 = 14;
pub const MODULE_CODE_METADATA_VALUES_UNUSED: u32 = 15;
pub const MODULE_CODE_SOURCE_FILENAME: u32 = 16;
pub const MODULE_CODE_HASH: u32 = 17;
pub const MODULE_CODE_IFUNC: u32 = 18;

pub const USELIST_CODE_DEFAULT: u32 = 1;
pub const USELIST_CODE_BB: u32 = 2;

pub const PARAMATTR_CODE_ENTRY_OLD: u32 = 1;
pub const PARAMATTR_CODE_ENTRY: u32 = 2;
pub const PARAMATTR_GRP_CODE_ENTRY: u32 = 3;

pub const TYPE_CODE_NUMENTRY: u32 = 1;
pub const TYPE_CODE_VOID: u32 = 2;
pub const TYPE_CODE_FLOAT: u32 = 3;
pub const TYPE_CODE_DOUBLE: u32 = 4;
pub const TYPE_CODE_LABEL: u32 = 5;
pub const TYPE_CODE_OPAQUE: u32 = 6;
pub const TYPE_CODE_INTEGER: u32 = 7;
pub const TYPE_CODE_POINTER: u32 = 8;
pub const TYPE_CODE_FUNCTION_OLD: u32 = 9;
pub const TYPE_CODE_HALF: u32 = 10;
pub const TYPE_CODE_ARRAY: u32 = 11;
pub const TYPE_CODE_VECTOR: u32 = 12;
pub const TYPE_CODE_X86_FP80: u32 = 13;
pub const TYPE_CODE_FP128: u32 = 14;
pub const TYPE_CODE_PPC_FP128: u32 = 15;
pub const TYPE_CODE_METADATA: u32 = 16;
pub const TYPE_CODE_X86_MMX: u32 = 17;
pub const TYPE_CODE_STRUCT_ANON: u32 = 18;
pub const TYPE_CODE_STRUCT_NAME: u32 = 19;
pub const TYPE_CODE_STRUCT_NAMED: u32 = 20;
pub const TYPE_CODE_FUNCTION: u32 = 21;
pub const TYPE_CODE_TOKEN: u32 = 22;
pub const TYPE_CODE_BFLOAT: u32 = 23;
pub const TYPE_CODE_X86_AMX: u32 = 24;
pub const TYPE_CODE_OPAQUE_POINTER: u32 = 25;
pub const TYPE_CODE_TARGET_TYPE: u32 = 26;

pub const OPERAND_BUNDLE_TAG: u32 = 1;
pub const SYNC_SCOPE_NAME: u32 = 1;

pub const VST_CODE_ENTRY: u32 = 1;
pub const VST_CODE_BBENTRY: u32 = 2;
pub const VST_CODE_FNENTRY: u32 = 3;
pub const VST_CODE_COMBINED_ENTRY: u32 = 5;

pub const METADATA_STRING_OLD: u32 = 1;
pub const METADATA_VALUE: u32 = 2;
pub const METADATA_NODE: u32 = 3;
pub const METADATA_NAME: u32 = 4;
pub const METADATA_DISTINCT_NODE: u32 = 5;
pub const METADATA_KIND: u32 = 6;
pub const METADATA_LOCATION: u32 = 7;
pub const METADATA_OLD_NODE: u32 = 8;
pub const METADATA_OLD_FN_NODE: u32 = 9;
pub const METADATA_NAMED_NODE: u32 = 10;
pub const METADATA_ATTACHMENT: u32 = 11;
pub const METADATA_STRINGS: u32 = 35;
pub const METADATA_GLOBAL_DECL_ATTACHMENT: u32 = 36;
pub const METADATA_INDEX_OFFSET: u32 = 38;
pub const METADATA_INDEX: u32 = 39;
pub const METADATA_ARG_LIST: u32 = 46;

/// Names of the debug-information metadata records (codes 12 to 49 except the generic ones above).
pub fn debug_metadata_record_name(code: u32) -> Option<&'static str> {
    Some(match code {
        7 => "METADATA_LOCATION",
        12 => "METADATA_GENERIC_DEBUG",
        13 => "METADATA_SUBRANGE",
        14 => "METADATA_ENUMERATOR",
        15 => "METADATA_BASIC_TYPE",
        16 => "METADATA_FILE",
        17 => "METADATA_DERIVED_TYPE",
        18 => "METADATA_COMPOSITE_TYPE",
        19 => "METADATA_SUBROUTINE_TYPE",
        20 => "METADATA_COMPILE_UNIT",
        21 => "METADATA_SUBPROGRAM",
        22 => "METADATA_LEXICAL_BLOCK",
        23 => "METADATA_LEXICAL_BLOCK_FILE",
        24 => "METADATA_NAMESPACE",
        25 => "METADATA_TEMPLATE_TYPE",
        26 => "METADATA_TEMPLATE_VALUE",
        27 => "METADATA_GLOBAL_VAR",
        28 => "METADATA_LOCAL_VAR",
        29 => "METADATA_EXPRESSION",
        30 => "METADATA_OBJC_PROPERTY",
        31 => "METADATA_IMPORTED_ENTITY",
        32 => "METADATA_MODULE",
        33 => "METADATA_MACRO",
        34 => "METADATA_MACRO_FILE",
        37 => "METADATA_GLOBAL_VAR_EXPR",
        40 => "METADATA_LABEL",
        41 => "METADATA_STRING_TYPE",
        44 => "METADATA_COMMON_BLOCK",
        45 => "METADATA_GENERIC_SUBRANGE",
        47 => "METADATA_ASSIGN_ID",
        48 => "METADATA_SUBRANGE_TYPE",
        49 => "METADATA_FIXED_POINT_TYPE",
        _ => return None,
    })
}

pub const CST_CODE_SETTYPE: u32 = 1;
pub const CST_CODE_NULL: u32 = 2;
pub const CST_CODE_UNDEF: u32 = 3;
pub const CST_CODE_INTEGER: u32 = 4;
pub const CST_CODE_WIDE_INTEGER: u32 = 5;
pub const CST_CODE_FLOAT: u32 = 6;
pub const CST_CODE_AGGREGATE: u32 = 7;
pub const CST_CODE_STRING: u32 = 8;
pub const CST_CODE_CSTRING: u32 = 9;
pub const CST_CODE_CE_BINOP: u32 = 10;
pub const CST_CODE_CE_CAST: u32 = 11;
pub const CST_CODE_CE_GEP_OLD: u32 = 12;
pub const CST_CODE_CE_SELECT: u32 = 13;
pub const CST_CODE_CE_EXTRACTELT: u32 = 14;
pub const CST_CODE_CE_INSERTELT: u32 = 15;
pub const CST_CODE_CE_SHUFFLEVEC: u32 = 16;
pub const CST_CODE_CE_CMP: u32 = 17;
pub const CST_CODE_INLINEASM_OLD: u32 = 18;
pub const CST_CODE_CE_SHUFVEC_EX: u32 = 19;
pub const CST_CODE_CE_INBOUNDS_GEP: u32 = 20;
pub const CST_CODE_BLOCKADDRESS: u32 = 21;
pub const CST_CODE_DATA: u32 = 22;
pub const CST_CODE_INLINEASM_OLD2: u32 = 23;
pub const CST_CODE_CE_GEP_WITH_INRANGE_INDEX_OLD: u32 = 24;
pub const CST_CODE_CE_UNOP: u32 = 25;
pub const CST_CODE_POISON: u32 = 26;
pub const CST_CODE_DSO_LOCAL_EQUIVALENT: u32 = 27;
pub const CST_CODE_INLINEASM_OLD3: u32 = 28;
pub const CST_CODE_NO_CFI_VALUE: u32 = 29;
pub const CST_CODE_INLINEASM: u32 = 30;
pub const CST_CODE_CE_GEP_WITH_INRANGE: u32 = 31;
pub const CST_CODE_CE_GEP: u32 = 32;
pub const CST_CODE_PTRAUTH: u32 = 33;

pub const FUNC_CODE_DECLAREBLOCKS: u32 = 1;
pub const FUNC_CODE_INST_BINOP: u32 = 2;
pub const FUNC_CODE_INST_CAST: u32 = 3;
pub const FUNC_CODE_INST_GEP_OLD: u32 = 4;
pub const FUNC_CODE_INST_SELECT: u32 = 5;
pub const FUNC_CODE_INST_EXTRACTELT: u32 = 6;
pub const FUNC_CODE_INST_INSERTELT: u32 = 7;
pub const FUNC_CODE_INST_SHUFFLEVEC: u32 = 8;
pub const FUNC_CODE_INST_CMP: u32 = 9;
pub const FUNC_CODE_INST_RET: u32 = 10;
pub const FUNC_CODE_INST_BR: u32 = 11;
pub const FUNC_CODE_INST_SWITCH: u32 = 12;
pub const FUNC_CODE_INST_INVOKE: u32 = 13;
pub const FUNC_CODE_INST_UNREACHABLE: u32 = 15;
pub const FUNC_CODE_INST_PHI: u32 = 16;
pub const FUNC_CODE_INST_ALLOCA: u32 = 19;
pub const FUNC_CODE_INST_LOAD: u32 = 20;
pub const FUNC_CODE_INST_VAARG: u32 = 23;
pub const FUNC_CODE_INST_STORE_OLD: u32 = 24;
pub const FUNC_CODE_INST_EXTRACTVAL: u32 = 26;
pub const FUNC_CODE_INST_INSERTVAL: u32 = 27;
pub const FUNC_CODE_INST_CMP2: u32 = 28;
pub const FUNC_CODE_INST_VSELECT: u32 = 29;
pub const FUNC_CODE_INST_INBOUNDS_GEP_OLD: u32 = 30;
pub const FUNC_CODE_INST_INDIRECTBR: u32 = 31;
pub const FUNC_CODE_DEBUG_LOC_AGAIN: u32 = 33;
pub const FUNC_CODE_INST_CALL: u32 = 34;
pub const FUNC_CODE_DEBUG_LOC: u32 = 35;
pub const FUNC_CODE_INST_FENCE: u32 = 36;
pub const FUNC_CODE_INST_CMPXCHG_OLD: u32 = 37;
pub const FUNC_CODE_INST_ATOMICRMW_OLD: u32 = 38;
pub const FUNC_CODE_INST_RESUME: u32 = 39;
pub const FUNC_CODE_INST_LANDINGPAD_OLD: u32 = 40;
pub const FUNC_CODE_INST_LOADATOMIC: u32 = 41;
pub const FUNC_CODE_INST_STOREATOMIC_OLD: u32 = 42;
pub const FUNC_CODE_INST_GEP: u32 = 43;
pub const FUNC_CODE_INST_STORE: u32 = 44;
pub const FUNC_CODE_INST_STOREATOMIC: u32 = 45;
pub const FUNC_CODE_INST_CMPXCHG: u32 = 46;
pub const FUNC_CODE_INST_LANDINGPAD: u32 = 47;
pub const FUNC_CODE_INST_CLEANUPRET: u32 = 48;
pub const FUNC_CODE_INST_CATCHRET: u32 = 49;
pub const FUNC_CODE_INST_CATCHPAD: u32 = 50;
pub const FUNC_CODE_INST_CLEANUPPAD: u32 = 51;
pub const FUNC_CODE_INST_CATCHSWITCH: u32 = 52;
pub const FUNC_CODE_OPERAND_BUNDLE: u32 = 55;
pub const FUNC_CODE_INST_UNOP: u32 = 56;
pub const FUNC_CODE_INST_CALLBR: u32 = 57;
pub const FUNC_CODE_INST_FREEZE: u32 = 58;
pub const FUNC_CODE_INST_ATOMICRMW: u32 = 59;
pub const FUNC_CODE_BLOCKADDR_USERS: u32 = 60;
pub const FUNC_CODE_DEBUG_RECORD_VALUE: u32 = 61;
pub const FUNC_CODE_DEBUG_RECORD_DECLARE: u32 = 62;
pub const FUNC_CODE_DEBUG_RECORD_ASSIGN: u32 = 63;
pub const FUNC_CODE_DEBUG_RECORD_VALUE_SIMPLE: u32 = 64;
pub const FUNC_CODE_DEBUG_RECORD_LABEL: u32 = 65;

pub const STRTAB_BLOB: u32 = 1;
pub const SYMTAB_BLOB: u32 = 1;

// Call record flags (`CallMarkersFlags`).
pub const CALL_TAIL: u64 = 0;
pub const CALL_CCONV: u64 = 1;
pub const CALL_MUSTTAIL: u64 = 14;
pub const CALL_EXPLICIT_TYPE: u64 = 15;
pub const CALL_NOTAIL: u64 = 16;
pub const CALL_FMF: u64 = 17;

/// `OB_METADATA`: marks a metadata operand in an operand bundle record.
pub const OB_METADATA: u64 = 0x8000_0000;
