#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SDK_LIB = REPO_ROOT / "ucf-sdk" / "src" / "lib.rs"
SDK_CARGO = REPO_ROOT / "ucf-sdk" / "Cargo.toml"
SNAPSHOT_PATH = REPO_ROOT / "docs" / "sdk_public_api_snapshot.txt"

PUB_ITEM_RE = re.compile(
    r"^(?:pub\s+use\s+.+;|pub\s+enum\s+\w+|pub\s+struct\s+\w+|pub\s+fn\s+\w+|\s*pub\s+\w[\w_]*\s*:\s*.+,)$"
)


def read_version() -> str:
    data = tomllib.loads(SDK_CARGO.read_text(encoding="utf-8"))
    return data["package"]["version"]


def parse_major(version: str) -> int:
    return int(version.split(".", 1)[0])


def collect_api_lines() -> list[str]:
    lines = SDK_LIB.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    current_struct = None
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("pub struct "):
            current_struct = line.split()[2]
        if line.startswith("}"):
            current_struct = None
        if PUB_ITEM_RE.match(line):
            if current_struct and line.strip().startswith("pub ") and ":" in line:
                out.append(f"field {current_struct}::{line.strip()}")
            else:
                out.append(line.strip())
    return sorted(set(out))


def render_snapshot(version: str, api_lines: list[str]) -> str:
    body = "\n".join(api_lines)
    return f"# ucf-sdk public API snapshot\nversion={version}\n\n{body}\n"


def parse_snapshot(content: str) -> tuple[str, set[str]]:
    lines = content.splitlines()
    if len(lines) < 2 or not lines[1].startswith("version="):
        raise ValueError("invalid snapshot header")
    version = lines[1].split("=", 1)[1].strip()
    entries = {line.strip() for line in lines[3:] if line.strip()}
    return version, entries


def baseline_snapshot(ref: str | None) -> tuple[str, set[str]] | None:
    if not ref:
        return None
    cmd = ["git", "show", f"{ref}:docs/sdk_public_api_snapshot.txt"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return parse_snapshot(proc.stdout)


def cmd_generate() -> int:
    version = read_version()
    api = collect_api_lines()
    SNAPSHOT_PATH.write_text(render_snapshot(version, api), encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
    return 0


def cmd_check(baseline_ref: str | None) -> int:
    version = read_version()
    current_api = set(collect_api_lines())

    if not SNAPSHOT_PATH.exists():
        print("snapshot file missing", file=sys.stderr)
        return 1

    snap_version, snap_api = parse_snapshot(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    if snap_version != version or snap_api != current_api:
        print(
            "snapshot is stale; run: python scripts/sdk_api_snapshot.py generate",
            file=sys.stderr,
        )
        return 1

    baseline = baseline_snapshot(baseline_ref)
    if baseline is None:
        print("no baseline snapshot found; skipping breaking-change gate")
        return 0

    base_version, base_api = baseline
    removed = sorted(base_api - current_api)
    if removed:
        if parse_major(version) <= parse_major(base_version):
            print("breaking public API change detected without major version bump", file=sys.stderr)
            for item in removed:
                print(f"- removed: {item}", file=sys.stderr)
            print(
                f"baseline version={base_version}, current version={version}",
                file=sys.stderr,
            )
            return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("generate")
    c = sub.add_parser("check")
    c.add_argument("--baseline-ref", default="HEAD^")

    args = parser.parse_args()
    if args.cmd == "generate":
        return cmd_generate()
    return cmd_check(args.baseline_ref)


if __name__ == "__main__":
    raise SystemExit(main())
