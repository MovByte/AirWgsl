//! Prints the canonical module of one entry point in LLVM's textual form.
//!
//! Usage: `canon_dump BITCODE ENTRY [INDEX=HEXBYTES:MTLDATATYPE ...]`
//!
//! Function constants are given by index, as little-endian bytes in hexadecimal and the `MTLDataType` number. The
//! output is what `llvm-as` reads, so `opt -passes=verify` can check it.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: canon_dump BITCODE ENTRY [INDEX=HEXBYTES:MTLDATATYPE ...]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read");
    let m = airwgsl::parse_bitcode(&data).expect("parse");
    let mut constants = Vec::new();
    for a in &args[3..] {
        let (i, rest) = a.split_once('=').expect("INDEX=HEXBYTES:TYPE");
        let (hex, ty) = rest.split_once(':').expect("INDEX=HEXBYTES:TYPE");
        let value = (0..hex.len()).step_by(2).map(|k| u8::from_str_radix(&hex[k..k + 2], 16).unwrap()).collect();
        constants.push(airwgsl::api::FunctionConstant {
            index: i.parse().unwrap(),
            name: None,
            data_type: ty.parse().unwrap(),
            value,
        });
    }
    match airwgsl::passes::canonicalize(std::slice::from_ref(&m), &args[2], &constants) {
        Ok(c) => print!("{}", airwgsl::ir::print::print_module(&c.module)),
        Err(r) => {
            eprintln!("refused: {}", r);
            std::process::exit(1);
        }
    }
}
