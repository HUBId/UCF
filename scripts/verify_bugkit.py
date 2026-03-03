#!/usr/bin/env python3
"""Offline verifier for ucf bug report kit zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a ucf bugkit zip offline")
    parser.add_argument("--bugkit", required=True, type=Path, help="Path to bugkit zip")
    parser.add_argument(
        "--ucf-ops",
        default="cargo run -p ucf-ops --",
        help="Command prefix for ucf-ops invocation",
    )
    args = parser.parse_args()

    if not args.bugkit.exists():
        print(f"FAIL: bugkit does not exist: {args.bugkit}")
        return 2

    with tempfile.TemporaryDirectory(prefix="bugkit_verify_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(args.bugkit, "r") as zf:
            zf.extractall(tmp_path)

        manifest_path = tmp_path / "BUGKIT_MANIFEST.json"
        if not manifest_path.exists():
            print("FAIL: BUGKIT_MANIFEST.json missing")
            return 2

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures: list[str] = []

        files = sorted(manifest.get("files", []), key=lambda x: x["path"])
        active = [f for f in files if not f.get("dropped_due_to_size_cap", False)]

        for entry in active:
            rel = entry["path"]
            path = tmp_path / rel
            if not path.exists():
                failures.append(f"missing file: {rel}")
                continue
            digest = sha256_bytes(path.read_bytes())
            if digest != entry["sha256"]:
                failures.append(f"sha256 mismatch: {rel}")

        # Verify manifest digest.
        canonical = dict(manifest)
        canonical["bugkit_digest"] = ""
        canonical["files"] = sorted(canonical.get("files", []), key=lambda x: x["path"])
        digest = sha256_bytes(json.dumps(canonical, separators=(",", ":")).encode("utf-8"))
        if digest != manifest.get("bugkit_digest"):
            failures.append("bugkit_digest mismatch")

        repro = tmp_path / "repro_pack.zip"
        if not repro.exists():
            failures.append("repro_pack.zip missing")
        else:
            verify_out = tmp_path / "repro_verify.json"
            cmd = f"{args.ucf_ops} repro verify --pack \"{repro}\" --out \"{verify_out}\""
            proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
            if proc.returncode != 0:
                failures.append(
                    "repro verify command failed: "
                    + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
                )

        if failures:
            print("FAIL")
            for item in failures:
                print(f" - {item}")
            return 1

        print("PASS")
        print(f"checked_files={len(active)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
