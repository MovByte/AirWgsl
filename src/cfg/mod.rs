//! Control-flow structuring: a function's control-flow graph to WGSL's structured statements.
//!
//! LLVM's control flow is a graph of basic blocks; WGSL has only `if`, `switch`, `loop`, `break`, `continue` and
//! `return`. The translation is Ramsey's "Beyond Relooper" (Norman Ramsey, *Beyond Relooper: Recursive Translation of
//! Unstructured Control Flow to Structured Control Flow*, ICFP 2022), which places code by the dominator tree:
//!
//! - every node's code is emitted once, where its immediate dominator places it;
//! - a node with two or more forward in-edges (a *merge node*) follows a labelled block that encloses the code of
//!   every branch to it, so a branch to it exits that block;
//! - a loop header's code is wrapped in a loop, and a branch back to it continues that loop;
//! - any other node is emitted inline at the one branch that reaches it.
//!
//! The translation is defined for reducible graphs. An irreducible graph (a retreating edge whose target does not
//! dominate its source) is refused: MSL has no `goto`, so the front end does not produce one (plan 5.6).
//!
//! WGSL has no labelled blocks. A block becomes `switch 0i { default { ... } }`, whose `break` exits it and through
//! which `continue` reaches the enclosing loop. A branch becomes the native `break` or `continue` when the innermost
//! construct's exit, or the innermost loop's next iteration, is where it must go, nothing when it falls through to its
//! target, and otherwise a jump through a flag: the branch stores the target's number in a function variable and
//! breaks, and after each construct it crosses a dispatch tests the flag and continues the jump from there.
//!
//! [`Structured::behavior`] computes the WGSL statement behaviors (WGSL 9.7) of the result, which the emitter needs to
//! satisfy the function-body rules, and [`structure`] refuses a loop whose behavior is empty (a loop without an exit,
//! which WGSL rejects and which this translation does not rewrite).

use std::collections::{BTreeMap, BTreeSet};

/// A block's terminator, as far as control flow is concerned. Successors are block indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    Return,
    Unreachable,
    Br(usize),
    /// `br i1 %c, label %t, label %f`.
    CondBr(usize, usize),
    /// The default target, then one target per case in case order.
    Switch(usize, Vec<usize>),
}

impl Term {
    /// Successors in LLVM's order (switch: default first).
    pub fn successors(&self) -> Vec<usize> {
        match self {
            Term::Return | Term::Unreachable => Vec::new(),
            Term::Br(d) => vec![*d],
            Term::CondBr(t, f) => vec![*t, *f],
            Term::Switch(d, cases) => std::iter::once(*d).chain(cases.iter().copied()).collect(),
        }
    }
}

/// Why a graph has no structured form here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The edge `from -> to` retreats to a block that does not dominate `from`.
    Irreducible { from: usize, to: usize },
    /// The loop headed by this block has no exit: WGSL rejects a loop whose behavior is empty (WGSL 9.7).
    LoopWithoutExit { header: usize },
}

/// A structured statement. Before jump resolution the tree holds [`Node::Goto`]; after it, only the WGSL forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// The instructions of a block, without its terminator.
    Code(usize),
    /// A labelled block: `switch 0i { default { body } }`.
    Block {
        id: u32,
        body: Vec<Node>,
    },
    /// `loop { body }`; the body never falls off its end except to continue.
    Loop {
        id: u32,
        header: usize,
        body: Vec<Node>,
    },
    /// The conditional branch ending `block`.
    If {
        block: usize,
        then_: Vec<Node>,
        else_: Vec<Node>,
    },
    /// The switch ending `block`: each arm lists the case indices (positions in the switch's case list) that reach
    /// one target; cases reaching the default target are left to the default.
    Switch {
        block: usize,
        arms: Vec<(Vec<usize>, Vec<Node>)>,
        default: Vec<Node>,
    },
    /// The phi copies of the edge `from -> to`, done before control moves along it.
    Edge {
        from: usize,
        to: usize,
    },
    /// `ret` ending the block.
    Return(usize),
    /// `unreachable` ending the block.
    Unreachable(usize),
    /// An unresolved branch (only before jump resolution).
    Goto(Goto),
    Break,
    Continue,
    /// Store the flag value of a jump target.
    SetFlag(u32),
    /// `if flag == k0 { flag = 0; arm0 } else if flag == k1 { flag = 0; arm1 } ...`, placed after a construct.
    Dispatch(Vec<(u32, Vec<Node>)>),
}

/// The target of an unresolved branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Goto {
    /// Exit the block construct `id` (to the merge node that follows it).
    BlockExit(u32),
    /// Start the next iteration of loop construct `id`.
    LoopContinue(u32),
}

/// The analyses of a graph.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// Reachable blocks in reverse postorder.
    pub rpo: Vec<usize>,
    /// Position of each block in `rpo`, `None` when unreachable.
    pub rpo_index: Vec<Option<usize>>,
    /// Immediate dominator; the entry is its own, unreachable blocks have none.
    pub idom: Vec<Option<usize>>,
    /// Dominator-tree children, in reverse postorder.
    pub dom_children: Vec<Vec<usize>>,
    pub loop_header: Vec<bool>,
    /// Two or more forward in-edges (edges counted with multiplicity).
    pub merge: Vec<bool>,
    pre: Vec<usize>,
    post: Vec<usize>,
}

impl Analysis {
    /// Analyze the graph of `terms` (entry block 0).
    pub fn new(terms: &[Term]) -> Result<Analysis, Error> {
        let n = terms.len();
        let succ: Vec<Vec<usize>> = terms.iter().map(|t| t.successors()).collect();
        // Depth-first postorder from the entry, iteratively.
        let mut post_order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        seen[0] = true;
        while let Some((b, i)) = stack.pop() {
            if i < succ[b].len() {
                stack.push((b, i + 1));
                let s = succ[b][i];
                if !seen[s] {
                    seen[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post_order.push(b);
            }
        }
        let rpo: Vec<usize> = post_order.iter().rev().copied().collect();
        let mut rpo_index = vec![None; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_index[b] = Some(i);
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &rpo {
            for &s in &succ[b] {
                preds[s].push(b);
            }
        }
        // Cooper, Harvey and Kennedy, "A Simple, Fast Dominance Algorithm".
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| -> usize {
            while a != b {
                while rpo_index[a] > rpo_index[b] {
                    a = idom[a].unwrap();
                }
                while rpo_index[b] > rpo_index[a] {
                    b = idom[b].unwrap();
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_some() {
                        new = Some(match new {
                            None => p,
                            Some(q) => intersect(&idom, p, q),
                        });
                    }
                }
                if new != idom[b] {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut dom_children = vec![Vec::new(); n];
        for &b in rpo.iter().skip(1) {
            dom_children[idom[b].unwrap()].push(b);
        }
        // Preorder and postorder numbers of the dominator tree: a dominates b iff pre[a] <= pre[b] and
        // post[b] <= post[a].
        let mut pre = vec![usize::MAX; n];
        let mut post = vec![usize::MAX; n];
        let (mut pc, mut qc) = (0, 0);
        let mut st: Vec<(usize, usize)> = vec![(0, 0)];
        pre[0] = pc;
        pc += 1;
        while let Some((b, i)) = st.pop() {
            if i < dom_children[b].len() {
                st.push((b, i + 1));
                let c = dom_children[b][i];
                pre[c] = pc;
                pc += 1;
                st.push((c, 0));
            } else {
                post[b] = qc;
                qc += 1;
            }
        }
        let mut a = Analysis {
            rpo,
            rpo_index,
            idom,
            dom_children,
            loop_header: vec![false; n],
            merge: vec![false; n],
            pre,
            post,
        };
        let mut forward_in = vec![0usize; n];
        for &b in &a.rpo {
            for &s in &succ[b] {
                if a.rpo_index[s] <= a.rpo_index[b] {
                    // A retreating edge: reducible only when it is a back edge.
                    if !a.dominates(s, b) {
                        return Err(Error::Irreducible { from: b, to: s });
                    }
                    a.loop_header[s] = true;
                } else {
                    forward_in[s] += 1;
                }
            }
        }
        a.merge = forward_in.iter().map(|&c| c >= 2).collect();
        Ok(a)
    }

    pub fn reachable(&self, b: usize) -> bool {
        self.rpo_index[b].is_some()
    }

    /// Whether `a` dominates `b` (both reachable).
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        self.pre[a] <= self.pre[b] && self.post[b] <= self.post[a]
    }

    /// Whether `from -> to` is a back edge.
    pub fn is_back_edge(&self, from: usize, to: usize) -> bool {
        self.rpo_index[to] <= self.rpo_index[from]
    }
}

/// The structured form of a function.
#[derive(Clone, Debug)]
pub struct Structured {
    pub body: Vec<Node>,
    /// Whether any jump goes through the flag (the emitter then declares it).
    pub uses_flag: bool,
}

/// WGSL statement behaviors (WGSL 9.7.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Behavior {
    pub next: bool,
    pub brk: bool,
    pub cont: bool,
    pub ret: bool,
}

impl Behavior {
    const NEXT: Behavior = Behavior { next: true, brk: false, cont: false, ret: false };

    fn union(self, o: Behavior) -> Behavior {
        Behavior {
            next: self.next || o.next,
            brk: self.brk || o.brk,
            cont: self.cont || o.cont,
            ret: self.ret || o.ret,
        }
    }

    pub fn is_empty(self) -> bool {
        !(self.next || self.brk || self.cont || self.ret)
    }
}

/// Structure the graph of `terms` (entry block 0; unreachable blocks are left out).
pub fn structure(terms: &[Term]) -> Result<(Analysis, Structured), Error> {
    let a = Analysis::new(terms)?;
    let mut b = Builder { terms, a: &a, next_id: 0 };
    let tree = b.do_tree(0);
    let mut r = Resolver { frames: Vec::new(), flags: BTreeMap::new(), uses_flag: false, next_pos: 0 };
    let (body, pending) = r.list(tree, Cont::FunctionEnd);
    debug_assert!(pending.is_empty(), "a jump target outside every construct");
    let s = Structured { body, uses_flag: r.uses_flag };
    check_loops(&s.body)?;
    Ok((a, s))
}

struct Builder<'a> {
    terms: &'a [Term],
    a: &'a Analysis,
    next_id: u32,
}

/// An enclosing construct during the build: what a branch to a block becomes.
#[derive(Clone, Copy)]
enum Ctx {
    BlockFollowedBy(usize, u32),
    LoopHeadedBy(usize, u32),
}

impl Builder<'_> {
    fn id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }

    fn do_tree(&mut self, x: usize) -> Vec<Node> {
        self.do_tree_in(x, &mut Vec::new())
    }

    fn do_tree_in(&mut self, x: usize, ctx: &mut Vec<Ctx>) -> Vec<Node> {
        let merges: Vec<usize> = self.a.dom_children[x].iter().copied().filter(|&c| self.a.merge[c]).collect();
        if self.a.loop_header[x] {
            let id = self.id();
            ctx.push(Ctx::LoopHeadedBy(x, id));
            let body = self.node_within(x, &merges, ctx);
            ctx.pop();
            vec![Node::Loop { id, header: x, body }]
        } else {
            self.node_within(x, &merges, ctx)
        }
    }

    /// `merges` is in reverse postorder; the last is outermost.
    fn node_within(&mut self, x: usize, merges: &[usize], ctx: &mut Vec<Ctx>) -> Vec<Node> {
        match merges.split_last() {
            Some((&y, rest)) => {
                let id = self.id();
                ctx.push(Ctx::BlockFollowedBy(y, id));
                let body = self.node_within(x, rest, ctx);
                ctx.pop();
                let mut out = vec![Node::Block { id, body }];
                out.extend(self.do_tree_in(y, ctx));
                out
            }
            None => {
                let mut out = vec![Node::Code(x)];
                match &self.terms[x] {
                    Term::Return => out.push(Node::Return(x)),
                    Term::Unreachable => out.push(Node::Unreachable(x)),
                    Term::Br(d) => out.extend(self.branch(x, *d, ctx)),
                    Term::CondBr(t, f) => {
                        let then_ = self.branch(x, *t, ctx);
                        let else_ = self.branch(x, *f, ctx);
                        out.push(Node::If { block: x, then_, else_ });
                    }
                    Term::Switch(d, cases) => {
                        // Group the cases by target; a case reaching the default target is the default.
                        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
                        for (k, &t) in cases.iter().enumerate() {
                            if t == *d {
                                continue;
                            }
                            match groups.iter_mut().find(|(g, _)| *g == t) {
                                Some((_, ks)) => ks.push(k),
                                None => groups.push((t, vec![k])),
                            }
                        }
                        let mut arms = Vec::new();
                        for (t, ks) in groups {
                            arms.push((ks, self.branch(x, t, ctx)));
                        }
                        let default = self.branch(x, *d, ctx);
                        out.push(Node::Switch { block: x, arms, default });
                    }
                }
                out
            }
        }
    }

    fn branch(&mut self, x: usize, y: usize, ctx: &mut Vec<Ctx>) -> Vec<Node> {
        let mut out = vec![Node::Edge { from: x, to: y }];
        if self.a.is_back_edge(x, y) {
            let id = ctx
                .iter()
                .rev()
                .find_map(|c| match c {
                    Ctx::LoopHeadedBy(h, id) if *h == y => Some(*id),
                    _ => None,
                })
                .expect("a back edge to a loop that does not enclose it");
            out.push(Node::Goto(Goto::LoopContinue(id)));
        } else if self.a.merge[y] {
            let id = ctx
                .iter()
                .rev()
                .find_map(|c| match c {
                    Ctx::BlockFollowedBy(m, id) if *m == y => Some(*id),
                    _ => None,
                })
                .expect("a branch to a merge node outside its block");
            out.push(Node::Goto(Goto::BlockExit(id)));
        } else {
            out.extend(self.do_tree_in(y, ctx));
        }
        out
    }
}

/// Where control goes: a statement position (the statement's number), the next iteration of a loop, or the end of
/// the function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Cont {
    Pos(u64),
    LoopContinue(u32),
    FunctionEnd,
}

struct Frame {
    id: u32,
    is_loop: bool,
    /// Where control goes when the construct is exited with `break`.
    after: Cont,
}

struct Resolver {
    /// Enclosing constructs, innermost last.
    frames: Vec<Frame>,
    /// Flag value of each jump target.
    flags: BTreeMap<Cont, u32>,
    uses_flag: bool,
    next_pos: u64,
}

impl Resolver {
    fn pos(&mut self) -> Cont {
        self.next_pos += 1;
        Cont::Pos(self.next_pos)
    }

    fn flag(&mut self, c: Cont) -> u32 {
        let n = self.flags.len() as u32 + 1;
        *self.flags.entry(c).or_insert(n)
    }

    /// Resolve a statement list whose end falls through to `next`. Returns the list and the targets of flagged jumps
    /// that leave through the innermost enclosing construct and need a dispatch after it.
    fn list(&mut self, list: Vec<Node>, next: Cont) -> (Vec<Node>, BTreeSet<Cont>) {
        // The continuation after each statement: the next statement's position, or `next` for the last.
        let positions: Vec<Cont> = list.iter().map(|_| self.pos()).collect();
        let mut out = Vec::with_capacity(list.len());
        let mut pending = BTreeSet::new();
        let n = list.len();
        for (i, node) in list.into_iter().enumerate() {
            let after = if i + 1 < n { positions[i + 1] } else { next };
            match node {
                Node::Block { id, body } => {
                    self.frames.push(Frame { id, is_loop: false, after });
                    let (body, inner) = self.list(body, after);
                    self.frames.pop();
                    out.push(Node::Block { id, body });
                    self.dispatch(&mut out, inner, after, &mut pending);
                }
                Node::Loop { id, header, body } => {
                    self.frames.push(Frame { id, is_loop: true, after });
                    let (body, inner) = self.list(body, Cont::LoopContinue(id));
                    self.frames.pop();
                    out.push(Node::Loop { id, header, body });
                    self.dispatch(&mut out, inner, after, &mut pending);
                }
                Node::If { block, then_, else_ } => {
                    let (then_, p1) = self.list(then_, after);
                    let (else_, p2) = self.list(else_, after);
                    pending.extend(p1);
                    pending.extend(p2);
                    out.push(Node::If { block, then_, else_ });
                }
                Node::Switch { block, arms, default } => {
                    let mut new_arms = Vec::with_capacity(arms.len());
                    for (ks, body) in arms {
                        let (body, p) = self.list(body, after);
                        pending.extend(p);
                        new_arms.push((ks, body));
                    }
                    let (default, p) = self.list(default, after);
                    pending.extend(p);
                    out.push(Node::Switch { block, arms: new_arms, default });
                }
                Node::Goto(g) => {
                    let target = self.target(g);
                    self.jump(&mut out, target, after, &mut pending);
                }
                other => out.push(other),
            }
        }
        (out, pending)
    }

    fn target(&self, g: Goto) -> Cont {
        match g {
            Goto::LoopContinue(id) => Cont::LoopContinue(id),
            Goto::BlockExit(id) => self
                .frames
                .iter()
                .rev()
                .find(|f| f.id == id && !f.is_loop)
                .map(|f| f.after)
                .expect("block exit outside its block"),
        }
    }

    /// Emit a jump to `target` from a point whose fallthrough goes to `fall`.
    fn jump(&mut self, out: &mut Vec<Node>, target: Cont, fall: Cont, pending: &mut BTreeSet<Cont>) {
        if target == fall {
            return;
        }
        let innermost = self.frames.last().expect("a jump outside every construct");
        if innermost.after == target {
            out.push(Node::Break);
            return;
        }
        if let Cont::LoopContinue(l) = target
            && self.frames.iter().rev().find(|f| f.is_loop).map(|f| f.id) == Some(l)
        {
            out.push(Node::Continue);
            return;
        }
        let k = self.flag(target);
        self.uses_flag = true;
        out.push(Node::SetFlag(k));
        out.push(Node::Break);
        pending.insert(target);
    }

    /// After a construct that flagged jumps leave, test the flag and continue each jump from here.
    fn dispatch(&mut self, out: &mut Vec<Node>, targets: BTreeSet<Cont>, after: Cont, pending: &mut BTreeSet<Cont>) {
        if targets.is_empty() {
            return;
        }
        let mut arms = Vec::new();
        for t in targets {
            let k = self.flag(t);
            let mut body = Vec::new();
            self.jump(&mut body, t, after, pending);
            arms.push((k, body));
        }
        out.push(Node::Dispatch(arms));
    }
}

impl Structured {
    /// The WGSL behavior of a statement list (WGSL 9.7.1), with the emitter's lowering of each node.
    pub fn behavior(list: &[Node]) -> Behavior {
        let mut acc = Behavior::NEXT;
        for n in list {
            if !acc.next {
                // The rest is statically unreachable.
                break;
            }
            let b = Self::node_behavior(n);
            acc = Behavior { next: false, ..acc }.union(b);
        }
        acc
    }

    fn node_behavior(n: &Node) -> Behavior {
        match n {
            Node::Code(_) | Node::Edge { .. } | Node::SetFlag(_) => Behavior::NEXT,
            Node::Return(_) | Node::Unreachable(_) => Behavior { ret: true, ..Default::default() },
            Node::Break => Behavior { brk: true, ..Default::default() },
            Node::Continue => Behavior { cont: true, ..Default::default() },
            Node::Goto(_) => unreachable!("behavior of an unresolved jump"),
            Node::If { then_, else_, .. } => Self::behavior(then_).union(Self::behavior(else_)),
            Node::Switch { arms, default, .. } => {
                let mut b = Self::behavior(default);
                for (_, a) in arms {
                    b = b.union(Self::behavior(a));
                }
                b
            }
            Node::Dispatch(arms) => {
                let mut b = Behavior::NEXT;
                for (_, a) in arms {
                    b = b.union(Self::behavior(a));
                }
                b
            }
            Node::Block { body, .. } => {
                let b = Self::behavior(body);
                if b.brk { Behavior { next: true, brk: false, ..b } } else { b }
            }
            Node::Loop { body, .. } => loop_behavior(Self::behavior(body)),
        }
    }
}

fn loop_behavior(b1: Behavior) -> Behavior {
    if b1 == (Behavior { ret: true, ..Default::default() }) {
        return b1;
    }
    if !b1.brk {
        Behavior { cont: false, next: false, ..b1 }
    } else {
        Behavior { next: true, brk: false, cont: false, ..b1 }
    }
}

fn check_loops(list: &[Node]) -> Result<(), Error> {
    for n in list {
        match n {
            Node::Loop { header, body, .. } => {
                if loop_behavior(Structured::behavior(body)).is_empty() {
                    return Err(Error::LoopWithoutExit { header: *header });
                }
                check_loops(body)?;
            }
            Node::Block { body, .. } => check_loops(body)?,
            Node::If { then_, else_, .. } => {
                check_loops(then_)?;
                check_loops(else_)?;
            }
            Node::Switch { arms, default, .. } => {
                for (_, a) in arms {
                    check_loops(a)?;
                }
                check_loops(default)?;
            }
            Node::Dispatch(arms) => {
                for (_, a) in arms {
                    check_loops(a)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
