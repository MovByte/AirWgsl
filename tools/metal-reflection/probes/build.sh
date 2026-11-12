#!/bin/sh
# Compiles the probes as the corpus is compiled (iOS 17 AIR) into DIR and lists the metallibs for
# examples/reflect_corpus.rs. Usage: build.sh DIR
set -e
out="$1"
[ -n "$out" ] || { echo "usage: build.sh DIR" >&2; exit 2; }
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
for src in "$here"/*.metal; do
    name=$(basename "$src" .metal)
    xcrun -sdk iphoneos metal -std=metal3.1 -target air64-apple-ios17.0 "$src" -o "$out/$name.metallib"
    echo "$out/$name.metallib"
done > "$out/metallibs.txt"
