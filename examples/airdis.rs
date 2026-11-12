//! Prints a bitcode module (raw or wrapped) in LLVM's textual form.
//!
//! Usage: `airdis FILE`

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: airdis FILE");
            std::process::exit(2);
        }
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}: {}", path, e);
            std::process::exit(1);
        }
    };
    match airwgsl::parse_bitcode(&data) {
        Ok(m) => print!("{}", airwgsl::ir::print::print_module(&m)),
        Err(e) => {
            eprintln!("{}: {}", path, e);
            std::process::exit(1);
        }
    }
}
