//! The structuring is checked by running both forms: a walk of the graph and an interpreter of the structured
//! statements with WGSL's semantics (`break` leaves the innermost loop or switch, `continue` goes to the innermost
//! loop's next iteration, a loop body that falls off its end iterates, a switch-block's body that falls off its end
//! leaves it). Both take each branch decision from the same oracle, keyed by the block and how often it has run, so
//! the two traces of executed blocks and taken edges must be identical.

use super::*;

/// A deterministic pseudo-random generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn choice(seed: u64, block: usize, count: usize, n: usize) -> usize {
    let mut r = Rng(seed ^ (block as u64).wrapping_mul(0x1000_0000_01b3) ^ (count as u64).wrapping_mul(0x9e37_79b9));
    r.next();
    r.below(n)
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Event {
    Code(usize),
    Edge(usize, usize),
    End,
}

const LIMIT: usize = 400;

/// Walk the graph.
fn walk(terms: &[Term], seed: u64) -> Vec<Event> {
    let mut trace = Vec::new();
    let mut counts = vec![0usize; terms.len()];
    let mut b = 0;
    loop {
        trace.push(Event::Code(b));
        if trace.len() > LIMIT {
            return trace;
        }
        let s = terms[b].successors();
        if s.is_empty() {
            trace.push(Event::End);
            return trace;
        }
        let k = choice(seed, b, counts[b], s.len());
        counts[b] += 1;
        trace.push(Event::Edge(b, s[k]));
        b = s[k];
    }
}

enum Flow {
    Next,
    Break,
    Continue,
    Return,
    Stop,
}

struct Interp<'a> {
    terms: &'a [Term],
    seed: u64,
    counts: Vec<usize>,
    flag: u32,
    trace: Vec<Event>,
    /// The successor index chosen at the last executed terminator, per block.
    chosen: Vec<usize>,
}

impl Interp<'_> {
    fn decide(&mut self, b: usize) -> usize {
        let s = self.terms[b].successors();
        let k = choice(self.seed, b, self.counts[b], s.len());
        self.counts[b] += 1;
        self.chosen[b] = k;
        k
    }

    fn run(&mut self, list: &[Node]) -> Flow {
        for n in list {
            match self.node(n) {
                Flow::Next => {}
                f => return f,
            }
        }
        Flow::Next
    }

    fn node(&mut self, n: &Node) -> Flow {
        if self.trace.len() > LIMIT {
            return Flow::Stop;
        }
        match n {
            Node::Code(b) => {
                self.trace.push(Event::Code(*b));
                Flow::Next
            }
            Node::Edge { from, to } => {
                self.trace.push(Event::Edge(*from, *to));
                Flow::Next
            }
            Node::Return(_) | Node::Unreachable(_) => {
                self.trace.push(Event::End);
                Flow::Return
            }
            Node::Break => Flow::Break,
            Node::Continue => Flow::Continue,
            Node::SetFlag(k) => {
                self.flag = *k;
                Flow::Next
            }
            Node::Goto(_) => panic!("unresolved jump"),
            Node::If { block, then_, else_ } => {
                let k = self.decide(*block);
                if k == 0 { self.run(then_) } else { self.run(else_) }
            }
            Node::Switch { block, arms, default } => {
                let k = self.decide(*block);
                // Successor 0 is the default; successor j is case j - 1.
                if k > 0
                    && let Some((_, body)) = arms.iter().find(|(ks, _)| ks.contains(&(k - 1)))
                {
                    return self.run(body);
                }
                self.run(default)
            }
            Node::Dispatch(arms) => {
                for (k, body) in arms {
                    if self.flag == *k {
                        self.flag = 0;
                        return self.run(body);
                    }
                }
                Flow::Next
            }
            Node::Block { body, .. } => match self.run(body) {
                Flow::Break | Flow::Next => Flow::Next,
                f => f,
            },
            Node::Loop { body, .. } => loop {
                match self.run(body) {
                    Flow::Next | Flow::Continue => continue,
                    Flow::Break => return Flow::Next,
                    f => return f,
                }
            },
        }
    }
}

fn interpret(terms: &[Term], s: &Structured, seed: u64) -> Vec<Event> {
    let mut i =
        Interp { terms, seed, counts: vec![0; terms.len()], flag: 0, trace: Vec::new(), chosen: vec![0; terms.len()] };
    match i.run(&s.body) {
        Flow::Return | Flow::Stop => {}
        Flow::Next => panic!("the structured body fell off its end"),
        Flow::Break | Flow::Continue => panic!("a jump left the function body"),
    }
    let _ = &i.chosen;
    i.trace
}

fn check(terms: &[Term], seeds: u64) {
    let (_, s) = match structure(terms) {
        Ok(r) => r,
        Err(e) => panic!("{:?} for {:?}", e, terms),
    };
    for seed in 0..seeds {
        let a = walk(terms, seed);
        let mut b = interpret(terms, &s, seed);
        // The walk stops one event later than the interpreter at the limit; compare the common prefix.
        let n = a.len().min(b.len());
        b.truncate(n);
        assert_eq!(&a[..n], &b[..], "traces differ for {:?}\nstructured: {:#?}", terms, s.body);
    }
}

#[test]
fn straight_line_and_diamond() {
    check(&[Term::Br(1), Term::Return], 4);
    check(&[Term::CondBr(1, 2), Term::Br(3), Term::Br(3), Term::Return], 16);
}

#[test]
fn while_loop_with_two_exits() {
    // 0 -> 1 (header) -> 2 | 4; 2 -> 3 | 4; 3 -> 1; 4 -> ret
    check(&[Term::Br(1), Term::CondBr(2, 4), Term::CondBr(3, 4), Term::Br(1), Term::Return], 64);
}

#[test]
fn nested_loops_with_outer_continue() {
    // Outer header 1, inner header 2; the inner body 3 continues the outer loop or the inner one or exits.
    let terms =
        [Term::Br(1), Term::CondBr(2, 5), Term::Br(3), Term::Switch(2, vec![1, 4, 5]), Term::Br(1), Term::Return];
    check(&terms, 128);
}

#[test]
fn switch_with_shared_targets() {
    let terms = [Term::Switch(1, vec![2, 2, 3, 1]), Term::Br(4), Term::Br(4), Term::Br(4), Term::Return];
    check(&terms, 64);
}

#[test]
fn irreducible_is_refused() {
    // 0 -> 1 | 2; 1 -> 2; 2 -> 1: a loop with two entries.
    let terms = [Term::CondBr(1, 2), Term::Br(2), Term::CondBr(1, 3), Term::Return];
    assert!(matches!(structure(&terms), Err(Error::Irreducible { .. })));
}

#[test]
fn loop_without_exit_is_refused() {
    let terms = [Term::Br(1), Term::Br(1)];
    assert_eq!(structure(&terms).unwrap_err(), Error::LoopWithoutExit { header: 1 });
}

#[test]
fn unreachable_blocks_are_left_out() {
    let terms = [Term::Return, Term::Br(0)];
    let (a, s) = structure(&terms).unwrap();
    assert!(!a.reachable(1));
    assert_eq!(s.body, vec![Node::Code(0), Node::Return(0)]);
}

/// A random graph: a random spanning structure plus random extra edges, kept when reducible.
fn random_graph(r: &mut Rng, n: usize) -> Vec<Term> {
    let mut terms = Vec::with_capacity(n);
    for b in 0..n {
        let pick = |r: &mut Rng| r.below(n);
        let t = if b + 1 == n {
            Term::Return
        } else {
            match r.below(10) {
                0 => Term::Return,
                1..=3 => Term::Br(b + 1 + r.below(n - b - 1).min(2)),
                4..=7 => Term::CondBr(pick(r), b + 1),
                _ => {
                    let cases = (0..1 + r.below(4)).map(|_| pick(r)).collect();
                    Term::Switch(b + 1, cases)
                }
            }
        };
        terms.push(t);
    }
    terms
}

#[test]
fn random_reducible_graphs_match() {
    let mut r = Rng(0x5eed);
    let mut tested = 0;
    let mut irreducible = 0;
    let mut no_exit = 0;
    let mut flagged = 0;
    for _ in 0..20000 {
        let n = 2 + r.below(12);
        let terms = random_graph(&mut r, n);
        match structure(&terms) {
            Ok((_, s)) => {
                check(&terms, 24);
                tested += 1;
                flagged += s.uses_flag as usize;
            }
            Err(Error::Irreducible { .. }) => irreducible += 1,
            Err(Error::LoopWithoutExit { .. }) => no_exit += 1,
        }
    }
    assert!(tested > 5000, "only {} reducible graphs ({} irreducible, {} without exit)", tested, irreducible, no_exit);
    // The flagged jumps must be exercised too.
    assert!(flagged > 100, "only {} graphs needed flagged jumps", flagged);
}

#[test]
fn behaviors_of_structured_functions_end_in_return() {
    let mut r = Rng(0xbeef);
    for _ in 0..5000 {
        let n = 2 + r.below(10);
        let terms = random_graph(&mut r, n);
        if let Ok((_, s)) = structure(&terms) {
            let b = Structured::behavior(&s.body);
            assert!(!b.brk && !b.cont, "a jump escapes the function body: {:?}", terms);
            assert!(b.ret, "no return: {:?}", terms);
        }
    }
}
