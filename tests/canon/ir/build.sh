#!/bin/sh
# Assembles the IR fixtures of tests/canon into bitcode with LLVM 21's llvm-as (the bitcode is committed next to the
# text so the tests need no LLVM). Usage: build.sh [LLVM_AS]
set -e
cd "$(dirname "$0")"
AS="${1:-/opt/homebrew/opt/llvm@21/bin/llvm-as}"
for f in *.ll; do
    "$AS" "$f" -o "${f%.ll}.bc"
done
