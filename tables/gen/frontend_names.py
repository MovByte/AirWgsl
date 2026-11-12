#!/usr/bin/env python3
"""Collect the air.* strings of the target's Metal compiler stack.

MTLCompiler.framework (which MTLCompilerService loads) links libGPUCompiler.dylib, whose
implementation libraries carry the front end's tables of AIR names. The five images are
extracted from the root's dyld shared cache with `ipsw dyld extract` into the cache
directory (never under the root), and every printable string run of four or more bytes
is searched for air.* names. Names that are complete identifiers and the printf-style
templates (containing %) are recorded separately, per image, with the cache UUID.

Usage: frontend_names.py --root ROOT --ipsw PATH --cache DIR --out NAMES.json
"""

import argparse
import json
import os
import re
import struct
import subprocess
import uuid

CACHE = "System/Library/Caches/com.apple.dyld/dyld_shared_cache_arm64"
IMAGES = [
    "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler",
    "/System/Library/PrivateFrameworks/MTLCompiler.framework/libMTLCompilerHelper.dylib",
    "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompiler.dylib",
    "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompilerImpl.dylib",
    "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompilerImplLazy.dylib",
]
RUN_RE = re.compile(rb"[\x20-\x7e]{4,}")
AIR_RE = re.compile(r"(?:^|[^A-Za-z0-9_.])(air\.[A-Za-z0-9_.%]+)")
IDENT_RE = re.compile(r"^air\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\.?$")


def cache_uuid(path):
    with open(path, "rb") as f:
        head = f.read(0x68)
    if not head.startswith(b"dyld_v1"):
        raise SystemExit("%s is not a dyld shared cache" % path)
    return str(uuid.UUID(bytes=head[0x58:0x68])).upper()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--ipsw", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache_path = os.path.join(a.root, CACHE)
    out_dir = os.path.join(a.cache, "frontend-images")
    os.makedirs(out_dir, exist_ok=True)
    result = {"cache_uuid": cache_uuid(cache_path),
              "ipsw": subprocess.run([a.ipsw, "version"], capture_output=True, text=True).stdout.split("\n")[0],
              "images": {}}
    for image in IMAGES:
        base = os.path.basename(image)
        dst = os.path.join(out_dir, base)
        if not os.path.exists(dst):
            subprocess.run([a.ipsw, "dyld", "extract", cache_path, base, "--output", out_dir],
                           check=True, capture_output=True)
        with open(dst, "rb") as f:
            data = f.read()
        idents, templates = set(), set()
        for run in RUN_RE.findall(data):
            for m in AIR_RE.findall(run.decode("ascii")):
                if "%" in m:
                    templates.add(m)
                elif IDENT_RE.match(m):
                    idents.add(m.rstrip("."))
        result["images"][image] = {"names": sorted(idents), "templates": sorted(templates)}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    total = set()
    for v in result["images"].values():
        total.update(v["names"])
    print("cache %s: %d names" % (result["cache_uuid"], len(total)))


if __name__ == "__main__":
    main()
