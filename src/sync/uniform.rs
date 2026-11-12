//! WGSL's uniformity analysis (WGSL 15.2) of a generated stage module.
//!
//! The synchronization built-ins (`workgroupBarrier`, `storageBarrier`, `textureBarrier`, `workgroupUniformLoad`) must
//! be called in uniform control flow, and a uniformity failure for them is a shader-creation error that no diagnostic
//! filter turns off (WGSL 15.2, 17.11). A barrier lowering is therefore exact only for a module in which WGSL's
//! analysis proves every such call uniform; [`check`] runs that analysis on the finished text of a module and names
//! the first call it cannot prove, so the stage is refused instead of handed to a WebGPU implementation that rejects
//! it.
//!
//! The analysis follows the rules of WGSL 15.2.5 to 15.2.8 on the entry point's body: the control-flow node of each
//! statement, the value node of each expression, and the flow-sensitive value of each function-scope variable with the
//! join rules of 15.2.5. A node is tracked as either possibly non-uniform (it reaches `MayBeNonUniform`, with the
//! first source found kept as the cause) or not; loops are iterated to a fixed point, which computes the same
//! reachability as the spec's graph. Where the text has a construct whose exact rule is not modeled here, the
//! analysis takes the conservative answer (non-uniform), so it can refuse a module WGSL accepts but never accepts a
//! module WGSL refuses. The conservative points are:
//!
//! - a call of a built-in function that reads memory other invocations write, or whose result WGSL does not make
//!   uniform (atomics, texture and derivative built-ins, subgroup and quad operations), has a non-uniform result;
//! - a user-defined function that names a mutable module-scope variable, calls such a built-in or calls such a
//!   function has a non-uniform result; any other has the uniformity of its arguments (its parameter return tags);
//! - a function-address-space pointer passed to a user-defined function makes the variable non-uniform;
//! - an assignment through anything but the variable itself (an element, a member, a pointer) is a partial assignment
//!   whose index expressions also count;
//! - an identifier the module does not declare and that is not a predeclared WGSL name is non-uniform.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A node's uniformity: `None` when uniform, otherwise what makes it possibly non-uniform.
type Nu = Option<String>;

fn or(a: &Nu, b: &Nu) -> Nu {
    a.clone().or_else(|| b.clone())
}

/// A call the analysis could not prove to be in uniform control flow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// The built-in called.
    pub call: String,
    /// The first source of non-uniformity found for the control flow at the call.
    pub cause: String,
}

/// Whether a module's text calls a synchronization built-in.
pub fn has_synchronization(wgsl: &str) -> bool {
    SYNC.iter().any(|s| wgsl.contains(s))
}

/// The synchronization built-ins: called only in uniform control flow (WGSL 17.11).
const SYNC: &[&str] = &["workgroupBarrier", "storageBarrier", "textureBarrier", "workgroupUniformLoad"];

/// Built-in functions whose result the analysis takes as non-uniform: atomics read memory other invocations write;
/// derivatives, `textureSample*` and subgroup and quad operations are `ReturnValueMayBeNonUniform` (WGSL 15.2.7), and
/// `textureLoad` is for read-write storage textures.
fn nonuniform_builtin(name: &str) -> bool {
    name.starts_with("atomic")
        || name.starts_with("texture")
        || name.starts_with("dpd")
        || name.starts_with("fwidth")
        || name.starts_with("subgroup")
        || name.starts_with("quad")
}

/// Predeclared names that are not values (types, type generators, `true` and `false`, enumerants).
fn predeclared(name: &str) -> bool {
    matches!(
        name,
        "true"
            | "false"
            | "bool"
            | "i32"
            | "u32"
            | "f32"
            | "f16"
            | "vec2"
            | "vec3"
            | "vec4"
            | "vec2i"
            | "vec3i"
            | "vec4i"
            | "vec2u"
            | "vec3u"
            | "vec4u"
            | "vec2f"
            | "vec3f"
            | "vec4f"
            | "vec2h"
            | "vec3h"
            | "vec4h"
            | "mat2x2"
            | "mat2x3"
            | "mat2x4"
            | "mat3x2"
            | "mat3x3"
            | "mat3x4"
            | "mat4x2"
            | "mat4x3"
            | "mat4x4"
            | "array"
            | "atomic"
            | "ptr"
            | "function"
            | "private"
            | "workgroup"
            | "uniform"
            | "storage"
            | "read"
            | "write"
            | "read_write"
    )
}

// Tokens ----------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok {
    Ident(String),
    /// A numeric literal (with its suffix).
    Num,
    Punct(&'static str),
}

const PUNCT: &[&str] = &[
    "<<=", ">>=", "->", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "++", "--", "(", ")", "{", "}", "[", "]", "<", ">", ",", ";", ":", ".", "=", "+", "-", "*", "/", "%", "&", "|",
    "^", "!", "~", "@", "_",
];

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if src[i..].starts_with("//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if src[i..].starts_with("/*") {
            let mut depth = 0;
            while i < b.len() {
                if src[i..].starts_with("/*") {
                    depth += 1;
                    i += 2;
                } else if src[i..].starts_with("*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if c.is_ascii_alphabetic() || (c == b'_' && i + 1 < b.len() && is_ident_byte(b[i + 1])) {
            let s = i;
            while i < b.len() && is_ident_byte(b[i]) {
                i += 1;
            }
            out.push(Tok::Ident(src[s..i].to_string()));
        } else if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            // Decimal, hexadecimal and floating-point literals with their suffixes and exponents.
            i += 1;
            while i < b.len() {
                let d = b[i];
                let exponent_sign = (d == b'+' || d == b'-') && matches!(b[i - 1], b'e' | b'E' | b'p' | b'P');
                if is_ident_byte(d) || d == b'.' || exponent_sign {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Num);
        } else {
            let p = PUNCT
                .iter()
                .find(|p| src[i..].starts_with(*p))
                .ok_or_else(|| format!("an unexpected character {:?}", src[i..].chars().next().unwrap_or(' ')))?;
            out.push(Tok::Punct(p));
            i += p.len();
        }
    }
    Ok(out)
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

// Syntax -----------------------------------------------------------------------------------------------------------

type Expr = Vec<Tok>;

#[derive(Clone, Debug)]
enum Stmt {
    Let(String, Expr),
    Var(String, Option<Expr>),
    /// An assignment, compound assignment, increment or decrement. `full` when the left side is the variable itself
    /// and the operator is `=`.
    Assign {
        root: Option<String>,
        full: bool,
        lhs: Expr,
        rhs: Expr,
    },
    /// A function call statement or `_ = e`.
    Eval(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    Loop {
        body: Vec<Stmt>,
        continuing: Vec<Stmt>,
        break_if: Option<Expr>,
    },
    Switch(Expr, Vec<Vec<Stmt>>),
    Block(Vec<Stmt>),
    Break,
    Continue,
    Return(Option<Expr>),
    Discard,
}

struct Parser<'a> {
    t: &'a [Tok],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn is(&self, p: &str) -> bool {
        matches!(self.peek(), Some(Tok::Punct(q)) if *q == p)
    }

    fn is_kw(&self, k: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == k)
    }

    fn expect(&mut self, p: &str) -> Result<(), String> {
        if self.is(p) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {} at token {}: {:?}", p, self.i, self.peek()))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.peek() {
            Some(Tok::Ident(s)) => {
                let s = s.clone();
                self.i += 1;
                Ok(s)
            }
            other => Err(format!("expected a name at token {}: {:?}", self.i, other)),
        }
    }

    /// Tokens up to (not including) the first of `stops` at bracket depth 0.
    fn until(&mut self, stops: &[&str]) -> Result<Expr, String> {
        let s = self.i;
        let mut depth = 0i32;
        while let Some(t) = self.peek() {
            if let Tok::Punct(p) = t {
                if depth == 0 && stops.contains(p) {
                    return Ok(self.t[s..self.i].to_vec());
                }
                match *p {
                    "(" | "[" => depth += 1,
                    ")" | "]" => depth -= 1,
                    _ => {}
                }
            }
            self.i += 1;
        }
        Err(format!("unterminated expression from token {}", s))
    }

    /// The tokens of a type up to the first of `stops` outside parentheses and template lists.
    fn type_until(&mut self, stops: &[&str]) -> Result<Expr, String> {
        let s = self.i;
        let mut depth = 0i32;
        while let Some(t) = self.peek() {
            if let Tok::Punct(p) = t {
                if depth == 0 && stops.contains(p) {
                    return Ok(self.t[s..self.i].to_vec());
                }
                match *p {
                    "(" | "[" | "<" => depth += 1,
                    ")" | "]" | ">" => depth -= 1,
                    ">>" => depth -= 2,
                    _ => {}
                }
            }
            self.i += 1;
        }
        Err(format!("an unterminated type from token {}", s))
    }

    /// Skip attributes (`@name` and `@name(...)`).
    fn attributes(&mut self) -> Result<Vec<(String, Expr)>, String> {
        let mut out = Vec::new();
        while self.is("@") {
            self.i += 1;
            let n = self.ident()?;
            let mut args = Vec::new();
            if self.is("(") {
                self.i += 1;
                args = self.until(&[")"])?;
                self.expect(")")?;
            }
            out.push((n, args));
        }
        Ok(out)
    }

    /// Skip a template list `<...>` when one starts here.
    fn template(&mut self) {
        if self.is("<") {
            let mut depth = 0;
            while let Some(t) = self.peek() {
                match t {
                    Tok::Punct("<") => depth += 1,
                    Tok::Punct(">") => {
                        depth -= 1;
                        if depth == 0 {
                            self.i += 1;
                            return;
                        }
                    }
                    Tok::Punct(">>") => {
                        depth -= 2;
                        if depth <= 0 {
                            self.i += 1;
                            return;
                        }
                    }
                    _ => {}
                }
                self.i += 1;
            }
        }
    }

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect("{")?;
        let mut out = Vec::new();
        while !self.is("}") {
            if self.peek().is_none() {
                return Err("unterminated block".into());
            }
            if let Some(s) = self.statement()? {
                out.push(s);
            }
        }
        self.i += 1;
        Ok(out)
    }

    fn statement(&mut self) -> Result<Option<Stmt>, String> {
        if self.is(";") {
            self.i += 1;
            return Ok(None);
        }
        self.attributes()?;
        if self.is("{") {
            return Ok(Some(Stmt::Block(self.block()?)));
        }
        let kw = match self.peek() {
            Some(Tok::Ident(s)) => s.clone(),
            Some(Tok::Punct("_")) => {
                self.i += 1;
                self.expect("=")?;
                let e = self.until(&[";"])?;
                self.expect(";")?;
                return Ok(Some(Stmt::Eval(e)));
            }
            Some(Tok::Punct("*")) | Some(Tok::Punct("(")) => String::new(),
            other => return Err(format!("unexpected {:?} at token {}", other, self.i)),
        };
        match kw.as_str() {
            "let" | "const" => {
                self.i += 1;
                let n = self.ident()?;
                if self.is(":") {
                    self.i += 1;
                    self.until(&["="])?;
                }
                self.expect("=")?;
                let e = self.until(&[";"])?;
                self.expect(";")?;
                Ok(Some(Stmt::Let(n, e)))
            }
            "var" => {
                self.i += 1;
                self.template();
                let n = self.ident()?;
                if self.is(":") {
                    self.i += 1;
                    self.until(&["=", ";"])?;
                }
                let init = if self.is("=") {
                    self.i += 1;
                    Some(self.until(&[";"])?)
                } else {
                    None
                };
                self.expect(";")?;
                Ok(Some(Stmt::Var(n, init)))
            }
            "if" => self.if_stmt().map(Some),
            "loop" => {
                self.i += 1;
                self.expect("{")?;
                let mut body = Vec::new();
                let mut continuing = Vec::new();
                let mut break_if = None;
                while !self.is("}") {
                    if self.is_kw("continuing") {
                        self.i += 1;
                        self.expect("{")?;
                        while !self.is("}") {
                            if self.is_kw("break") && matches!(self.t.get(self.i + 1), Some(Tok::Ident(s)) if s == "if")
                            {
                                self.i += 2;
                                let e = self.until(&[";"])?;
                                self.expect(";")?;
                                break_if = Some(e);
                            } else if let Some(s) = self.statement()? {
                                continuing.push(s);
                            }
                        }
                        self.i += 1;
                    } else if let Some(s) = self.statement()? {
                        body.push(s);
                    }
                }
                self.i += 1;
                Ok(Some(Stmt::Loop { body, continuing, break_if }))
            }
            "for" => {
                // for (init; cond; update) { body } is { init; loop { if !(cond) { break; } body continuing { update } } }
                self.i += 1;
                self.expect("(")?;
                let init = if self.is(";") { None } else { self.simple()? };
                self.expect(";")?;
                let cond = self.until(&[";"])?;
                self.expect(";")?;
                let update = if self.is(")") { None } else { self.simple()? };
                self.expect(")")?;
                let body = self.block()?;
                let mut lb = Vec::new();
                if !cond.is_empty() {
                    let mut neg = vec![Tok::Punct("!"), Tok::Punct("(")];
                    neg.extend(cond);
                    neg.push(Tok::Punct(")"));
                    lb.push(Stmt::If(neg, vec![Stmt::Break], Vec::new()));
                }
                lb.extend(body);
                let mut out = Vec::new();
                out.extend(init);
                out.push(Stmt::Loop { body: lb, continuing: update.into_iter().collect(), break_if: None });
                Ok(Some(Stmt::Block(out)))
            }
            "while" => {
                self.i += 1;
                let cond = self.until(&["{"])?;
                let body = self.block()?;
                let mut neg = vec![Tok::Punct("!"), Tok::Punct("(")];
                neg.extend(cond);
                neg.push(Tok::Punct(")"));
                let mut lb = vec![Stmt::If(neg, vec![Stmt::Break], Vec::new())];
                lb.extend(body);
                Ok(Some(Stmt::Loop { body: lb, continuing: Vec::new(), break_if: None }))
            }
            "switch" => {
                self.i += 1;
                let sel = self.until(&["{"])?;
                self.attributes()?;
                self.expect("{")?;
                let mut cases = Vec::new();
                while !self.is("}") {
                    if self.is_kw("case") {
                        self.i += 1;
                        self.until(&["{", ":"])?;
                        if self.is(":") {
                            self.i += 1;
                        }
                    } else if self.is_kw("default") {
                        self.i += 1;
                        if self.is(":") {
                            self.i += 1;
                        }
                    } else {
                        return Err(format!("unexpected token in switch at {}", self.i));
                    }
                    self.attributes()?;
                    cases.push(self.block()?);
                }
                self.i += 1;
                Ok(Some(Stmt::Switch(sel, cases)))
            }
            "break" => {
                self.i += 1;
                self.expect(";")?;
                Ok(Some(Stmt::Break))
            }
            "continue" => {
                self.i += 1;
                self.expect(";")?;
                Ok(Some(Stmt::Continue))
            }
            "return" => {
                self.i += 1;
                let e = self.until(&[";"])?;
                self.expect(";")?;
                Ok(Some(Stmt::Return(if e.is_empty() { None } else { Some(e) })))
            }
            "discard" => {
                self.i += 1;
                self.expect(";")?;
                Ok(Some(Stmt::Discard))
            }
            _ => {
                let s = self.simple()?;
                self.expect(";")?;
                Ok(s)
            }
        }
    }

    fn if_stmt(&mut self) -> Result<Stmt, String> {
        self.i += 1;
        let cond = self.until(&["{"])?;
        let then_ = self.block()?;
        let mut else_ = Vec::new();
        if self.is_kw("else") {
            self.i += 1;
            if self.is_kw("if") {
                else_.push(self.if_stmt()?);
            } else {
                else_ = self.block()?;
            }
        }
        Ok(Stmt::If(cond, then_, else_))
    }

    /// An assignment, increment, decrement, call or declaration (the statements `for` allows in its header).
    fn simple(&mut self) -> Result<Option<Stmt>, String> {
        if self.is_kw("let") || self.is_kw("const") || self.is_kw("var") {
            // Reuse the statement parser, which consumes the semicolon: put it back for the caller.
            let s = self.statement()?;
            self.i -= 1;
            return Ok(s);
        }
        let lhs =
            self.until(&["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "++", "--", ";", ")"])?;
        let op = match self.peek() {
            Some(Tok::Punct(p)) => *p,
            _ => return Err("an unterminated statement".into()),
        };
        if op == ";" || op == ")" {
            return Ok(Some(Stmt::Eval(lhs)));
        }
        self.i += 1;
        let rhs = if op == "++" || op == "--" { Vec::new() } else { self.until(&[";", ")"])? };
        let root = lhs.iter().find_map(|t| match t {
            Tok::Ident(s) => Some(s.clone()),
            _ => None,
        });
        let full = op == "=" && lhs.len() == 1;
        Ok(Some(Stmt::Assign { root, full, lhs, rhs }))
    }
}

// Behaviors (WGSL 9.7) ----------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Beh {
    next: bool,
    brk: bool,
    cont: bool,
    ret: bool,
}

const NEXT: Beh = Beh { next: true, brk: false, cont: false, ret: false };

impl Beh {
    fn union(self, o: Beh) -> Beh {
        Beh { next: self.next || o.next, brk: self.brk || o.brk, cont: self.cont || o.cont, ret: self.ret || o.ret }
    }

    fn only_next(self) -> bool {
        self == NEXT
    }
}

fn beh_list(l: &[Stmt]) -> Beh {
    let mut acc = NEXT;
    for s in l {
        if !acc.next {
            break;
        }
        acc = Beh { next: false, ..acc }.union(beh(s));
    }
    acc
}

fn beh(s: &Stmt) -> Beh {
    match s {
        Stmt::Break => Beh { brk: true, ..Default::default() },
        Stmt::Continue => Beh { cont: true, ..Default::default() },
        Stmt::Return(_) => Beh { ret: true, ..Default::default() },
        Stmt::If(_, a, b) => beh_list(a).union(beh_list(b)),
        Stmt::Block(b) => beh_list(b),
        Stmt::Switch(_, cases) => {
            let mut b = Beh::default();
            for c in cases {
                b = b.union(beh_list(c));
            }
            if b.brk { Beh { next: true, brk: false, ..b } } else { b }
        }
        Stmt::Loop { body, continuing, break_if } => {
            let mut b = beh_list(body);
            if b.next || b.cont {
                b = b.union(beh_list(continuing));
            }
            if break_if.is_some() {
                b.brk = true;
            }
            if !b.brk {
                Beh { next: false, cont: false, ..b }
            } else {
                Beh { next: true, brk: false, cont: false, ..b }
            }
        }
        _ => NEXT,
    }
}

// Module ------------------------------------------------------------------------------------------------------------

/// A parameter: its attributes (name and argument tokens), name and type tokens.
type Param = (Vec<(String, Expr)>, String, Expr);

struct Function {
    name: String,
    attrs: Vec<String>,
    params: Vec<Param>,
    body: Vec<Stmt>,
    body_tokens: Expr,
}

struct ModuleInfo {
    /// Module-scope variables and whether their loads may be non-uniform.
    vars: HashMap<String, bool>,
    /// Module-scope constants, overrides, struct and alias names.
    names: BTreeSet<String>,
    functions: Vec<Function>,
}

fn parse_module(src: &str) -> Result<ModuleInfo, String> {
    let toks = tokenize(src)?;
    let mut p = Parser { t: &toks, i: 0 };
    let mut m = ModuleInfo { vars: HashMap::new(), names: BTreeSet::new(), functions: Vec::new() };
    while p.peek().is_some() {
        let attrs = p.attributes()?;
        let kw = p.ident()?;
        match kw.as_str() {
            "enable" | "requires" => {
                p.until(&[";"])?;
                p.expect(";")?;
            }
            "diagnostic" => {
                p.expect("(")?;
                p.until(&[")"])?;
                p.expect(")")?;
                p.expect(";")?;
            }
            "struct" => {
                m.names.insert(p.ident()?);
                p.expect("{")?;
                let mut depth = 1;
                while depth > 0 {
                    match p.peek() {
                        Some(Tok::Punct("{")) => depth += 1,
                        Some(Tok::Punct("}")) => depth -= 1,
                        None => return Err("an unterminated struct".into()),
                        _ => {}
                    }
                    p.i += 1;
                }
            }
            "alias" | "const" | "override" => {
                m.names.insert(p.ident()?);
                p.until(&[";"])?;
                p.expect(";")?;
            }
            "const_assert" => {
                p.until(&[";"])?;
                p.expect(";")?;
            }
            "var" => {
                // var<space[, access]> name: type [= init];
                let mut space = Vec::new();
                if p.is("<") {
                    p.i += 1;
                    while !p.is(">") {
                        if let Some(Tok::Ident(s)) = p.peek() {
                            space.push(s.clone());
                        }
                        p.i += 1;
                    }
                    p.i += 1;
                }
                let n = p.ident()?;
                p.until(&[";"])?;
                p.expect(";")?;
                // Read-only module-scope variables: uniform buffers, read-only storage buffers and handles
                // (WGSL 15.2.8: identifiers resolving to them are uniform).
                let read_only = match space.first().map(|s| s.as_str()) {
                    None => true,
                    Some("uniform") => true,
                    Some("storage") => !space.iter().any(|s| s == "read_write"),
                    _ => false,
                };
                m.vars.insert(n, !read_only);
            }
            "fn" => {
                let name = p.ident()?;
                p.expect("(")?;
                let mut params = Vec::new();
                while !p.is(")") {
                    let pa = p.attributes()?;
                    let pn = p.ident()?;
                    p.expect(":")?;
                    let ty = p.type_until(&[",", ")"])?;
                    if p.is(",") {
                        p.i += 1;
                    }
                    params.push((pa, pn, ty));
                }
                p.i += 1;
                if p.is("->") {
                    p.i += 1;
                    p.attributes()?;
                    p.until(&["{"])?;
                }
                let start = p.i;
                let body = p.block()?;
                let body_tokens = toks[start..p.i].to_vec();
                m.functions.push(Function {
                    name,
                    attrs: attrs.into_iter().map(|a| a.0).collect(),
                    params,
                    body,
                    body_tokens,
                });
            }
            other => return Err(format!("an unexpected module-scope declaration {:?}", other)),
        }
    }
    Ok(m)
}

// Analysis ----------------------------------------------------------------------------------------------------------

/// What a user-defined function does to its callers' uniformity.
#[derive(Clone, Debug, Default)]
struct FnTag {
    /// The result may be non-uniform whatever the arguments (`ReturnValueMayBeNonUniform`).
    result: Nu,
    /// It must be called in uniform control flow (`CallSiteRequiredToBeUniform`).
    collective: bool,
    /// It has a function-address-space pointer parameter.
    pointer_params: bool,
}

/// Variable state: function-scope variable name to its value's uniformity.
type Vars = BTreeMap<String, Nu>;

fn join(a: &Vars, b: &Vars) -> Vars {
    let mut out = a.clone();
    for (k, v) in b {
        match out.get_mut(k) {
            Some(x) => *x = or(x, v),
            None => {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    out
}

fn same(a: &Vars, b: &Vars) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| w.is_some() == v.is_some()))
}

/// The states that leave a loop or switch by `break`, and a loop's `continue` states.
#[derive(Default)]
struct Frame {
    is_loop: bool,
    breaks: Vec<Vars>,
    continues: Vec<Vars>,
}

struct Analyzer<'a> {
    module: &'a ModuleInfo,
    tags: &'a HashMap<String, FnTag>,
    /// Entry parameters and their uniformity.
    params: HashMap<String, Nu>,
    /// `let` declarations in scope.
    lets: Vec<HashMap<String, Nu>>,
    frames: Vec<Frame>,
    failures: Vec<Failure>,
    /// Constructs of the text the analysis does not model (pointer `let`s, assignments through them).
    unsupported: Vec<String>,
}

impl Analyzer<'_> {
    fn let_value(&self, n: &str) -> Option<&Nu> {
        self.lets.iter().rev().find_map(|s| s.get(n))
    }

    /// The value node of an expression evaluated in control flow `cf`, with the variables' values `vars`; also
    /// checks the collective calls it makes. Function-address-space pointers passed to user functions make their
    /// variables non-uniform in `vars`.
    fn expr(&mut self, e: &[Tok], cf: &Nu, vars: &mut Vars) -> Nu {
        let mut nu = cf.clone();
        for (k, t) in e.iter().enumerate() {
            let Tok::Ident(n) = t else { continue };
            if k > 0 && e[k - 1] == Tok::Punct(".") {
                continue;
            }
            // The root of an address-of expression is not loaded.
            if k > 0 && e[k - 1] == Tok::Punct("&") {
                continue;
            }
            let value: Option<Nu> = if let Some(x) = self.let_value(n) {
                Some(x.clone())
            } else if let Some(x) = vars.get(n) {
                Some(x.clone())
            } else if let Some(x) = self.params.get(n) {
                Some(x.clone())
            } else {
                self.module
                    .vars
                    .get(n)
                    .map(|&nonuniform| if nonuniform { Some(format!("a load of {}", n)) } else { None })
            };
            if let Some(v) = value {
                if nu.is_none() {
                    nu = v;
                }
                continue;
            }
            // A name that is not a value followed by `(` or a template list is a function or type: a call.
            let call = matches!(e.get(k + 1), Some(Tok::Punct("(")) | Some(Tok::Punct("<")));
            if call {
                if SYNC.contains(&n.as_str()) {
                    if let Some(c) = cf {
                        self.failures.push(Failure { call: n.clone(), cause: c.clone() });
                    }
                    continue;
                }
                if let Some(tag) = self.tags.get(n) {
                    if tag.collective
                        && let Some(c) = cf
                    {
                        self.failures.push(Failure { call: n.clone(), cause: c.clone() });
                    }
                    if nu.is_none() {
                        nu = tag.result.clone();
                    }
                    if tag.pointer_params {
                        // Any variable whose address is passed may be written with a non-uniform value.
                        for (j, u) in e.iter().enumerate().skip(k + 1) {
                            if let (Tok::Ident(v), Some(Tok::Punct("&"))) = (u, e.get(j.wrapping_sub(1)))
                                && vars.contains_key(v)
                            {
                                vars.insert(v.clone(), Some(format!("{} through a pointer argument", n)));
                            }
                        }
                    }
                    continue;
                }
                if nonuniform_builtin(n) {
                    if nu.is_none() {
                        nu = Some(format!("the result of {}", n));
                    }
                    continue;
                }
                // Another built-in function, a type constructor or a struct constructor: the uniformity of its
                // arguments.
                continue;
            }
            let v: Nu = if self.module.names.contains(n) || predeclared(n) {
                None
            } else {
                Some(format!("{} (not a name the analysis knows)", n))
            };
            if nu.is_none() {
                nu = v;
            }
        }
        nu
    }

    /// Analyze a statement list from control flow `cf`: returns the resulting control flow and the variables' values
    /// where the list completes normally.
    fn list(&mut self, l: &[Stmt], cf: Nu, mut vars: Vars) -> (Nu, Vars) {
        self.lets.push(HashMap::new());
        let declared_before: BTreeSet<String> = vars.keys().cloned().collect();
        let mut cf = cf;
        for s in l {
            let (c, v) = self.stmt(s, cf.clone(), vars);
            cf = c;
            vars = v;
            if !beh(s).next {
                break;
            }
        }
        self.lets.pop();
        // Variables declared in the list go out of scope.
        vars.retain(|k, _| declared_before.contains(k));
        (cf, vars)
    }

    fn stmt(&mut self, s: &Stmt, cf: Nu, mut vars: Vars) -> (Nu, Vars) {
        match s {
            Stmt::Let(n, e) => {
                if e.first() == Some(&Tok::Punct("&")) {
                    self.unsupported.push(format!("the pointer declaration {}", n));
                }
                let v = self.expr(e, &cf, &mut vars);
                self.lets.last_mut().unwrap().insert(n.clone(), v);
                (cf, vars)
            }
            Stmt::Var(n, init) => {
                // The zero value of a variable without an initializer is the control flow's node (WGSL 15.2.6).
                let v = match init {
                    Some(e) => self.expr(e, &cf, &mut vars),
                    None => cf.clone(),
                };
                vars.insert(n.clone(), v);
                (cf, vars)
            }
            Stmt::Assign { root, full, lhs, rhs } => {
                let r = self.expr(rhs, &cf, &mut vars);
                if let Some(root) = root
                    && vars.contains_key(root)
                {
                    let new = if *full {
                        r
                    } else {
                        // A partial assignment keeps the previous value; the left side's index expressions count.
                        let idx = self.expr(&lhs[1..], &cf, &mut vars);
                        or(&or(&vars[root], &r), &idx)
                    };
                    vars.insert(root.clone(), new);
                } else {
                    if let Some(root) = root
                        && self.let_value(root).is_some()
                    {
                        self.unsupported.push(format!("an assignment through {}", root));
                    }
                    self.expr(lhs, &cf, &mut vars);
                }
                (cf, vars)
            }
            Stmt::Eval(e) => {
                self.expr(e, &cf, &mut vars);
                (cf, vars)
            }
            Stmt::Discard | Stmt::Return(None) => (cf, vars),
            Stmt::Return(Some(e)) => {
                self.expr(e, &cf, &mut vars);
                (cf, vars)
            }
            Stmt::Break => {
                if let Some(f) = self.frames.last_mut() {
                    f.breaks.push(vars.clone());
                }
                (cf, vars)
            }
            Stmt::Continue => {
                if let Some(f) = self.frames.iter_mut().rev().find(|f| f.is_loop) {
                    f.continues.push(vars.clone());
                }
                (cf, vars)
            }
            Stmt::Block(b) => self.list(b, cf, vars),
            Stmt::If(c, a, b) => {
                let v = self.expr(c, &cf, &mut vars);
                let (cf1, v1) = self.list(a, v.clone(), vars.clone());
                let (cf2, v2) = self.list(b, v, vars.clone());
                let (b1, b2) = (beh_list(a), beh_list(b));
                let out = match (b1.next, b2.next) {
                    (true, true) => join(&v1, &v2),
                    (true, false) => v1,
                    (false, true) => v2,
                    (false, false) => vars,
                };
                let cf_out = if b1.union(b2).only_next() { cf } else { or(&cf1, &cf2) };
                (cf_out, out)
            }
            Stmt::Switch(sel, cases) => {
                let v = self.expr(sel, &cf, &mut vars);
                self.frames.push(Frame { is_loop: false, ..Default::default() });
                let mut cf_end: Nu = None;
                let mut out: Option<Vars> = None;
                for c in cases {
                    let (cfi, vi) = self.list(c, v.clone(), vars.clone());
                    let bi = beh_list(c);
                    cf_end = or(&cf_end, &cfi);
                    if bi.next || bi.brk {
                        out = Some(match out {
                            Some(o) => join(&o, &vi),
                            None => vi,
                        });
                    }
                }
                let f = self.frames.pop().unwrap();
                let mut out = out.unwrap_or_else(|| vars.clone());
                for b in &f.breaks {
                    out = join(&out, b);
                }
                out.retain(|k, _| vars.contains_key(k));
                let b = beh(s);
                let cf_out = if b.only_next() { cf } else { cf_end };
                (cf_out, out)
            }
            Stmt::Loop { body, continuing, break_if } => self.loop_stmt(body, continuing, break_if, cf, vars),
        }
    }

    fn loop_stmt(
        &mut self,
        body: &[Stmt],
        continuing: &[Stmt],
        break_if: &Option<Expr>,
        cf: Nu,
        vars: Vars,
    ) -> (Nu, Vars) {
        let b1 = beh_list(body);
        let iterates = b1.next || b1.cont;
        let mut head_cf = cf.clone();
        let mut head_vars = vars.clone();
        loop {
            let failures = self.failures.len();
            self.frames.push(Frame { is_loop: true, ..Default::default() });
            let (cf1, v1) = self.list(body, head_cf.clone(), head_vars.clone());
            let f = self.frames.pop().unwrap();
            // The continuing block starts where the body completes or continues.
            let mut vin_s2: Option<Vars> = if b1.next { Some(v1.clone()) } else { None };
            for c in &f.continues {
                vin_s2 = Some(match vin_s2 {
                    Some(v) => join(&v, c),
                    None => c.clone(),
                });
            }
            let mut cf2 = cf1.clone();
            let mut vout_s2 = vin_s2.clone();
            if let Some(vin) = vin_s2.clone() {
                let (c2, mut v2) = self.list(continuing, cf1.clone(), vin);
                cf2 = c2;
                if let Some(e) = break_if {
                    // break if e: the resulting control flow depends on the condition.
                    cf2 = self.expr(e, &cf2, &mut v2);
                }
                vout_s2 = Some(v2);
            }
            let new_cf = if iterates { or(&cf, &cf2) } else { cf.clone() };
            let new_vars = match (&vout_s2, iterates) {
                (Some(v), true) => {
                    let mut j = join(&vars, v);
                    j.retain(|k, _| vars.contains_key(k));
                    j
                }
                _ => vars.clone(),
            };
            let stable = new_cf.is_some() == head_cf.is_some() && same(&new_vars, &head_vars);
            if !stable {
                // Analyze the body again from the widened state; failures found so far are re-found there.
                self.failures.truncate(failures);
                head_cf = new_cf;
                head_vars = new_vars;
                continue;
            }
            // The state after the loop: the continuing block's (when the body can break) and each break's.
            let mut out: Option<Vars> = None;
            if b1.brk || break_if.is_some() {
                out = vout_s2.clone();
            }
            for b in &f.breaks {
                out = Some(match out {
                    Some(o) => join(&o, b),
                    None => b.clone(),
                });
            }
            let mut out = out.unwrap_or_else(|| vars.clone());
            out.retain(|k, _| vars.contains_key(k));
            let res_cf = if !iterates {
                if b1.ret { cf1 } else { cf.clone() }
            } else if b1.ret {
                head_cf
            } else {
                cf.clone()
            };
            return (res_cf, out);
        }
    }
}

/// The uniformity of the entry point's built-in inputs: `workgroup_id` and `num_workgroups`, and `subgroup_size` and
/// `num_subgroups` in a compute shader, are uniform; every other input is not (WGSL 15.2.8).
fn param_nu(attrs: &[(String, Expr)], name: &str, compute: bool) -> Nu {
    let builtin = attrs.iter().find(|a| a.0 == "builtin").and_then(|a| match a.1.first() {
        Some(Tok::Ident(s)) => Some(s.clone()),
        _ => None,
    });
    match builtin.as_deref() {
        Some("workgroup_id" | "num_workgroups") => None,
        Some("subgroup_size" | "num_subgroups") if compute => None,
        Some(b) => Some(format!("the built-in {} ({})", b, name)),
        None => Some(format!("the stage input {}", name)),
    }
}

/// The tags of the module's user-defined functions other than the entry point, from callees to callers.
fn function_tags(m: &ModuleInfo) -> HashMap<String, FnTag> {
    let mut tags: HashMap<String, FnTag> = HashMap::new();
    let names: BTreeSet<&str> = m.functions.iter().map(|f| f.name.as_str()).collect();
    // WGSL forbids recursion and requires declarations before use only for nothing: iterate to a fixed point.
    loop {
        let mut changed = false;
        for f in &m.functions {
            let mut tag = FnTag {
                pointer_params: f.params.iter().any(|p| {
                    p.2.first() == Some(&Tok::Ident("ptr".into()))
                        && p.2.iter().any(|t| *t == Tok::Ident("function".into()))
                }),
                ..Default::default()
            };
            for (k, t) in f.body_tokens.iter().enumerate() {
                let Tok::Ident(n) = t else { continue };
                if k > 0 && f.body_tokens[k - 1] == Tok::Punct(".") {
                    continue;
                }
                let call = matches!(f.body_tokens.get(k + 1), Some(Tok::Punct("(")) | Some(Tok::Punct("<")));
                if call && SYNC.contains(&n.as_str()) {
                    tag.collective = true;
                } else if call && names.contains(n.as_str()) && n != &f.name {
                    if let Some(t) = tags.get(n) {
                        tag.collective |= t.collective;
                        if tag.result.is_none() {
                            tag.result = t.result.clone();
                        }
                    }
                } else if call && nonuniform_builtin(n) {
                    if tag.result.is_none() {
                        tag.result = Some(format!("{} in {}", n, f.name));
                    }
                } else if m.vars.get(n) == Some(&true) && tag.result.is_none() {
                    tag.result = Some(format!("{} in {}", n, f.name));
                }
            }
            let old = tags.get(&f.name);
            if old.is_none_or(|o| o.collective != tag.collective || o.result.is_some() != tag.result.is_some()) {
                tags.insert(f.name.clone(), tag);
                changed = true;
            }
        }
        if !changed {
            return tags;
        }
    }
}

/// Run WGSL's uniformity analysis on a module's entry point (the function with a stage attribute): the
/// synchronization calls it cannot prove to be in uniform control flow, or why the text could not be analyzed.
pub fn check(wgsl: &str) -> Result<Vec<Failure>, String> {
    let m = parse_module(wgsl)?;
    let tags = function_tags(&m);
    let mut failures = Vec::new();
    for f in &m.functions {
        let stage = f.attrs.iter().find(|a| matches!(a.as_str(), "compute" | "vertex" | "fragment"));
        let Some(stage) = stage else {
            // A helper with a collective call is analyzed where it is called (its call-site tag).
            continue;
        };
        let compute = stage == "compute";
        let params = f.params.iter().map(|(a, n, _)| (n.clone(), param_nu(a, n, compute))).collect();
        let mut an = Analyzer {
            module: &m,
            tags: &tags,
            params,
            lets: Vec::new(),
            frames: Vec::new(),
            failures: Vec::new(),
            unsupported: Vec::new(),
        };
        an.list(&f.body, None, Vars::new());
        if let Some(u) = an.unsupported.first() {
            return Err(format!("{} in {}", u, f.name));
        }
        failures.extend(an.failures);
    }
    Ok(failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fails(src: &str) -> Vec<Failure> {
        check(src).unwrap()
    }

    const HEAD: &str = "@group(0) @binding(0) var<storage, read> ro: array<u32>;\n\
                        @group(0) @binding(1) var<storage, read_write> rw: array<u32>;\n\
                        var<workgroup> wg: array<u32, 64>;\n";

    fn kernel(body: &str) -> String {
        format!(
            "{}@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_index) li: u32, \
             @builtin(workgroup_id) wid: vec3<u32>) {{\n{}\n}}\n",
            HEAD, body
        )
    }

    #[test]
    fn uniform_sources_pass() {
        assert!(fails(&kernel("let n = ro[0]; if n > 3u { workgroupBarrier(); } storageBarrier();")).is_empty());
        assert!(fails(&kernel("if wid.x == 0u { workgroupBarrier(); }")).is_empty());
    }

    #[test]
    fn non_uniform_sources_fail() {
        let f = fails(&kernel("if li < 3u { workgroupBarrier(); }"));
        assert_eq!(f.len(), 1);
        assert!(f[0].cause.contains("local_invocation_index"), "{:?}", f);
        assert_eq!(fails(&kernel("if rw[0] == 1u { workgroupBarrier(); }")).len(), 1);
        assert_eq!(fails(&kernel("if wg[0] == 1u { workgroupBarrier(); }")).len(), 1);
        assert_eq!(fails(&kernel("let a = atomicLoad(&x); if a == 1u { workgroupBarrier(); }")).len(), 1);
    }

    #[test]
    fn an_if_with_next_only_restores_uniformity() {
        // if e s1 else s2 with behavior {Next}: the control flow after it is the control flow before it.
        assert!(fails(&kernel("if li < 3u { rw[li] = 1u; } workgroupBarrier();")).is_empty());
        // A non-uniform return makes what follows non-uniform.
        assert_eq!(fails(&kernel("if li < 3u { return; } workgroupBarrier();")).len(), 1);
    }

    #[test]
    fn loops_with_non_uniform_exits_fail() {
        let uniform = "var i = 0u; loop { if i >= ro[0] { break; } workgroupBarrier(); i = i + 1u; }";
        assert!(fails(&kernel(uniform)).is_empty());
        let nonuniform = "var i = 0u; loop { if i >= li { break; } workgroupBarrier(); i = i + 1u; }";
        assert_eq!(fails(&kernel(nonuniform)).len(), 1);
        // A barrier after the loop: the loop's behavior is {Next}, so uniformity is restored.
        let after = "var i = 0u; loop { if i >= li { break; } i = i + 1u; } workgroupBarrier();";
        assert!(fails(&kernel(after)).is_empty());
    }

    #[test]
    fn variables_are_flow_sensitive() {
        // A variable assigned a non-uniform value and then a uniform one is uniform again.
        let body = "var x = li; x = 3u; if x == 3u { workgroupBarrier(); }";
        assert!(fails(&kernel(body)).is_empty());
        // Assigned under non-uniform control flow: the value is non-uniform.
        let body = "var x = 0u; if li == 0u { x = 1u; } if x == 1u { workgroupBarrier(); }";
        assert_eq!(fails(&kernel(body)).len(), 1);
        // Carried around a loop: the second iteration sees the non-uniform value.
        let body = "var x = 0u; var i = 0u; loop { if i >= 4u { break; } if x == 0u { workgroupBarrier(); } \
                    x = li; i = i + 1u; }";
        assert_eq!(fails(&kernel(body)).len(), 1);
    }

    #[test]
    fn switch_blocks_and_breaks() {
        // switch 0i { default { ... break ... } } as a labelled block: uniform when the break is.
        let body = "switch 0i { default { if ro[0] == 1u { break; } workgroupBarrier(); } }";
        assert!(fails(&kernel(body)).is_empty());
        let body = "switch 0i { default { if li == 1u { break; } workgroupBarrier(); } }";
        assert_eq!(fails(&kernel(body)).len(), 1);
        let body = "switch 0i { default { if li == 1u { break; } } } workgroupBarrier();";
        assert!(fails(&kernel(body)).is_empty());
    }

    #[test]
    fn helpers_are_transparent_unless_they_read_shared_state() {
        let src = format!(
            "{}fn pure(a: u32) -> u32 {{ var r = 0u; for (var i = 0u; i < a; i++) {{ r += i; }} return r; }}\n\
             fn shared() -> u32 {{ return wg[0]; }}\n\
             @compute @workgroup_size(64)\nfn main(@builtin(local_invocation_index) li: u32) {{\n\
             if pure(ro[1]) == 3u {{ workgroupBarrier(); }}\n\
             if shared() == 3u {{ storageBarrier(); }}\n}}\n",
            HEAD
        );
        let f = fails(&src);
        assert_eq!(f.len(), 1, "{:?}", f);
        assert_eq!(f[0].call, "storageBarrier");
    }
}
