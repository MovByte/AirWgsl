#!/usr/bin/env python3
"""Compile the generated MSL corpus to AIR at every language version and record what compiled.

Usage:

    build.py GEN_DIR OUT_DIR [--jobs N] [--versions 1.1,2.4,...] [--groups NAME,...]

``GEN_DIR`` is the output of ``corpus/generate.py``. For each group and each language version at
or above the group's minimum, the group's source is compiled with

    xcrun -sdk iphoneos metal -std=<version> -target air64-apple-ios17.7 -c

Errors are attributed to variants by source line (an error inside a shared template is
attributed through the "requested here" note that points into a variant). Rejected variants are
removed and the rest recompiled until the file compiles; a variant is never dropped for an error
it did not cause. If the harness variant ``v0`` is rejected, every variant of the group is
recorded as "harness" for that version instead of "rejected".

Each accepted module is disassembled with ``llvm-dis``, and every ``air.*`` and ``llvm.*``
function the module declares is attributed to the entry points that reach it through calls.

Output (OUT_DIR):

``air/<version>/<group>.<k>.air``, ``.metal``, ``.metallib``
    The compiled modules (one per accepted chunk; usually one per group), the exact source
    each was compiled from, and the module packaged by ``xcrun metallib`` where that tool
    accepts it (``results.json`` records the outcome per module).
``ll/<sha256>.ll``
    ``llvm-dis`` output, named by the sha256 of the ``.air`` file.
``modules.tsv``
    One line per accepted variant, in the format of the system corpus's ``modules.tsv``.
``results.json``
    Per group, version and variant: status (``accepted``; ``rejected`` by the compiler;
    ``harness`` when the group's harness is rejected; ``invalid_output`` when the compiler wrote a
    module that ``llvm-dis`` cannot read), the first diagnostic, and the intrinsics an accepted
    variant reaches. Also the compiler
    version and the exact command line.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys

from generate import PRELUDE

TARGET = "air64-apple-ios17.7"
DIAG = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+): (?P<kind>error|note|warning): (?P<msg>.*)$")
DEFINE = re.compile(r'^define [^@]*@("?)([^"(\s]+)\1\(')
CALL = re.compile(r'@("?)((?:air|llvm)\.[^"(\s]+|[A-Za-z_$][\w.$]*)\1\(')
DECLARE = re.compile(r'^declare [^@]*@("?)([^"(\s]+)\1\(')


def llvm_dis_path():
    candidate = "/opt/homebrew/opt/llvm@21/bin/llvm-dis"
    return candidate if os.path.exists(candidate) else "llvm-dis"


def render(group, keep):
    """Source text and the line range of each kept variant."""
    with open(group["source_path"], encoding="utf-8") as fh:
        text = fh.read()
    # Split the generated file at its variant markers.
    parts = re.split(r"(?m)^// variant (\d+): .*\n", text)
    head = parts[0]
    bodies = {int(parts[k]): parts[k + 1] for k in range(1, len(parts), 2)}
    lines = head.count("\n")
    out = [head]
    ranges = {}
    for index in sorted(keep):
        marker = f"// variant {index}\n"
        out.append(marker)
        start = lines + 2
        lines += 1 + bodies[index].count("\n")
        out.append(bodies[index])
        ranges[index] = (start, lines)
    return "".join(out), ranges, head.count("\n")


def attribute(stderr, ranges, source_name):
    """Map compiler errors to variants: {variant: first message}, plus unattributed errors."""
    by_variant = {}
    unattributed = []
    current = None
    entries = []
    for line in stderr.splitlines():
        m = DIAG.match(line)
        if not m:
            continue
        in_source = m.group("file").endswith(source_name)
        entries.append((m.group("kind"), int(m.group("line")) if in_source else -1, m.group("msg")))

    def owner(lineno):
        for index, (start, end) in ranges.items():
            if start <= lineno <= end:
                return index
        return None

    pending = None
    for kind, lineno, msg in entries:
        if kind == "error":
            if pending is not None:
                unattributed.append(pending)
            v = owner(lineno)
            if v is not None:
                by_variant.setdefault(v, msg)
                pending = None
            else:
                pending = msg
        elif kind == "note" and pending is not None:
            v = owner(lineno)
            if v is not None:
                by_variant.setdefault(v, pending)
                pending = None
    if pending is not None:
        unattributed.append(pending)
    return by_variant, unattributed


def reached_intrinsics(ll_text):
    """{entry function: sorted list of air./llvm. functions it reaches}."""
    calls = {}
    current = None
    declared = set()
    for line in ll_text.splitlines():
        if line.startswith("declare "):
            m = DECLARE.match(line)
            if m:
                declared.add(m.group(2))
            continue
        if line.startswith("define "):
            m = DEFINE.match(line)
            current = m.group(2) if m else None
            if current is not None:
                calls[current] = set()
            continue
        if line.startswith("}"):
            current = None
            continue
        if current is not None and "@" in line:
            for c in CALL.findall(line):
                calls[current].add(c[1])
    result = {}
    for fn in calls:
        seen = set()
        stack = [fn]
        out = set()
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            for callee in calls.get(f, ()):
                if callee in calls:
                    stack.append(callee)
                elif callee in declared:
                    out.add(callee)
        result[fn] = sorted(out)
    return result


INVALID = "\0invalid-output\0"


class HarnessRejected(Exception):
    pass


def build_one(group, version, flag, out, extra_options):
    """Compile one group at one language version.

    The harness ``v0`` is compiled alone first; if it is rejected, so is the group ("harness").
    Then all variants are compiled together, rejected variants are removed and the rest are
    recompiled. An error inside a class template is reported once, for the first variant that
    instantiates it, so a set that keeps failing that way is split in halves and each half is
    solved the same way; every accepted half becomes its own module. A variant is rejected only
    when an error is attributed to it or when it fails alone.
    """
    vdir = os.path.join(out, "air", version)
    os.makedirs(vdir, exist_ok=True)
    base = os.path.join(vdir, group["name"])
    work_src, work_air = base + ".work.metal", base + ".work.air"
    status = {}
    chunks = []
    compiles = [0]

    def compile_set(keep):
        text, ranges, _ = render(group, keep)
        with open(work_src, "w", encoding="utf-8") as fh:
            fh.write(text)
        command = ["xcrun", "-sdk", "iphoneos", "metal", f"-std={flag}", "-target", TARGET, "-ferror-limit=0",
                   *extra_options, *group["options"], "-c", work_src, "-o", work_air]
        proc = subprocess.run(["nice", "-n", "10"] + command, capture_output=True, text=True)
        compiles[0] += 1
        if proc.returncode == 0:
            # A module the compiler wrote but that LLVM cannot read (seen: intersection queries
            # tagged max_levels without instancing; Apple's air-opt rejects them too) is not an
            # accepted variant. It is isolated like a compile error and recorded as invalid output.
            dis = subprocess.run(["nice", "-n", "10", llvm_dis_path(), "-o", os.devnull, work_air],
                                 capture_output=True, text=True)
            if dis.returncode != 0:
                return False, {}, [INVALID + (dis.stderr.strip().splitlines() or ["?"])[0]]
            k = len(chunks)
            os.replace(work_src, f"{base}.{k}.metal")
            os.replace(work_air, f"{base}.{k}.air")
            chunks.append((sorted(keep), f"{base}.{k}.air"))
            return True, {}, []
        by_variant, unattributed = attribute(proc.stderr, ranges, os.path.basename(work_src))
        if not by_variant and not unattributed:
            raise RuntimeError(f"{group['name']} {version}: compiler failed without diagnostics: {proc.stderr[:400]}")
        return False, by_variant, unattributed

    def solve(keep):
        rounds = 0
        while keep:
            ok, by_variant, unattributed = compile_set(keep)
            if ok:
                return
            if 0 in by_variant:
                raise HarnessRejected(by_variant[0])
            for v, msg in by_variant.items():
                status[v] = ("rejected", msg)
                keep.discard(v)
            rounds += 1
            if by_variant and rounds < 3:
                continue
            if not keep:
                return
            if len(keep) == 1:
                v = next(iter(keep))
                message = (unattributed or ["(no diagnostic)"])[0]
                if message.startswith(INVALID):
                    status[v] = ("invalid_output", message[len(INVALID):])
                else:
                    status[v] = ("rejected", message)
                return
            items = sorted(keep)
            half = len(items) // 2
            solve(set(items[:half]))
            solve(set(items[half:]))
            return

    for stale in os.listdir(vdir):
        if stale.startswith(group["name"] + ".") and stale.split(".")[1].isdigit():
            os.remove(os.path.join(vdir, stale))
    try:
        ok, by_variant, unattributed = compile_set({0})
        if not ok:
            raise HarnessRejected(by_variant.get(0) or unattributed[0])
        chunks.clear()
        solve(set(range(1, len(group["variants"]))) | {0})
    except HarnessRejected as reason:
        for stale in os.listdir(vdir):
            if stale.startswith(group["name"] + ".") and stale.split(".")[1].isdigit():
                os.remove(os.path.join(vdir, stale))
        chunks.clear()
        for v in range(len(group["variants"])):
            status[v] = ("harness", str(reason))
    result = {"status": {}, "modules": []}
    for index, (members, air) in enumerate(chunks):
        lib = air[:-4] + ".metallib"
        # The module is the corpus artifact; the metallib is a convenience for container tests.
        # xcrun metallib rejects some valid modules ("Unexpected bitcode file!", seen on
        # intersection-query modules); that is recorded per module, not hidden.
        mproc = subprocess.run(["nice", "-n", "10", "xcrun", "-sdk", "iphoneos", "metallib", air, "-o", lib],
                               capture_output=True, text=True)
        packaged = "ok" if mproc.returncode == 0 else "failed: " + (mproc.stderr.strip().splitlines() or ["?"])[0]
        with open(air, "rb") as fh:
            blob = fh.read()
        sha = hashlib.sha256(blob).hexdigest()
        ll_path = os.path.join(out, "ll", sha + ".ll")
        dis = subprocess.run(["nice", "-n", "10", llvm_dis_path(), "-o", ll_path, air], capture_output=True, text=True)
        if dis.returncode != 0:
            raise RuntimeError(f"{group['name']} {version}: llvm-dis failed: {dis.stderr[:400]}")
        with open(ll_path, encoding="utf-8") as fh:
            reach = reached_intrinsics(fh.read())
        result["modules"].append({"sha256": sha, "size": len(blob), "path": os.path.relpath(air, out),
                                  "metallib": packaged})
        for v in members:
            status[v] = ("accepted", index, reach.get(f"v{v}", []))
    for v in range(len(group["variants"])):
        entry = status[v]
        record = {"status": entry[0]}
        if entry[0] == "accepted":
            record["module"] = entry[1]
            record["intrinsics"] = entry[2]
        else:
            record["diagnostic"] = entry[1]
        result["status"][str(v)] = record
    result["compiles"] = compiles[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gen_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    parser.add_argument("--versions", help="comma-separated subset of language versions")
    parser.add_argument("--groups", help="comma-separated subset of group names")
    args = parser.parse_args()

    with open(os.path.join(args.gen_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    versions = manifest["versions"]
    wanted_versions = args.versions.split(",") if args.versions else list(versions)
    wanted_groups = set(args.groups.split(",")) if args.groups else None
    os.makedirs(os.path.join(args.out_dir, "ll"), exist_ok=True)

    order = list(versions)
    jobs = []
    for group in manifest["groups"]:
        if wanted_groups and group["name"] not in wanted_groups:
            continue
        group["source_path"] = os.path.join(args.gen_dir, group["file"])
        for version in wanted_versions:
            if order.index(version) >= order.index(group["min_version"]):
                jobs.append((group, version))

    results_path = os.path.join(args.out_dir, "results.json")
    results = {"groups": {}}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            results = json.load(fh)
    compiler = subprocess.run(["xcrun", "-sdk", "iphoneos", "metal", "--version"], capture_output=True, text=True).stdout
    results["compiler"] = compiler.splitlines()[0] if compiler else "unknown"
    results["target"] = TARGET
    results["command"] = f"xcrun -sdk iphoneos metal -std=<flag> -target {TARGET} -ferror-limit=0 -c <group>.metal"
    results["versions"] = versions

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_one, g, v, versions[v], args.out_dir, []): (g, v) for g, v in jobs}
        for future in concurrent.futures.as_completed(futures):
            g, v = futures[future]
            try:
                res = future.result()
            except Exception as error:  # recorded and reported; the run fails at the end
                failures.append(f"{g['name']} {v}: {error}")
                continue
            entry = results["groups"].setdefault(g["name"], {"chapter": g["chapter"], "versions": {}})
            entry["variants"] = [v_["params"] for v_ in g["variants"]]
            entry["reaches"] = [v_["reaches"] for v_ in g["variants"]]
            entry["versions"][v] = res

    modules = []
    for name in sorted(results["groups"]):
        for version, res in sorted(results["groups"][name]["versions"].items()):
            for v, rec in sorted(res["status"].items(), key=lambda kv: int(kv[0])):
                if rec["status"] == "accepted":
                    m = res["modules"][rec["module"]]
                    modules.append("\t".join([m["path"], version, f"{name}/v{v}", "-", m["sha256"], "0", str(m["size"])]) + "\n")
    with open(os.path.join(args.out_dir, "modules.tsv"), "w", encoding="utf-8") as fh:
        fh.write("# path\tlocation\tentry\ttype\tmodule_sha256\toffset\tsize\n")
        fh.writelines(modules)
    tmp = results_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, results_path)
    counts = {}
    for name, entry in results["groups"].items():
        for version, res in entry["versions"].items():
            for rec in res["status"].values():
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for failure in failures:
        print("error:", failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
