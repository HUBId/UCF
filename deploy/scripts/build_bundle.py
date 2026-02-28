#!/usr/bin/env python3
"""Build a portable UCF deployment bundle without network access."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable
import zipfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_tree_digest(root: Path) -> str:
    entries: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        entries.append(f"{rel}:{hash_file(path)}")
    return sha256_bytes("\n".join(entries).encode("utf-8"))


def compute_policy_graph_digest(repo_root: Path, bundle_root: Path) -> str:
    ucf_ops = bundle_root / "bin" / "ucf-ops"
    if ucf_ops.exists() and os.access(ucf_ops, os.X_OK):
        cmd = [
            str(ucf_ops),
            "policy",
            "validate",
            "--pack",
            "policies/packs/base_v1",
            "--overlay",
            "policies/packs/overlays/prod",
        ]
        env = os.environ.copy()
        env.setdefault("UCF_OFFLINE", "1")
        try:
            run = subprocess.run(
                cmd,
                cwd=bundle_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            run = None
        if run and run.returncode == 0:
            for line in run.stdout.splitlines():
                if line.startswith("policy_graph_digest="):
                    return line.split("=", 1)[1].strip()
    return canonical_tree_digest(repo_root / "policies" / "packs")


def detect_code_version(repo_root: Path) -> str:
    git_dir = repo_root / ".git"
    if git_dir.exists():
        run = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if run.returncode == 0:
            return run.stdout.strip()
    return "unknown"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_optional_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_binaries(bin_source: Path, bundle_bin: Path) -> None:
    bundle_bin.mkdir(parents=True, exist_ok=True)
    for name in ("ucf-runtime", "ucf-ops", "ucf-gateway", "ucf-client"):
        src = bin_source / name
        if src.exists():
            shutil.copy2(src, bundle_bin / name)


def build_bundle(repo_root: Path, target_dir: Path, profile: str, bin_source: Path) -> Path:
    bundle_root = target_dir.resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)

    copy_binaries(bin_source, bundle_root / "bin")
    copy_tree(repo_root / "configs", bundle_root / "configs")
    copy_tree(repo_root / "policies", bundle_root / "policies")
    copy_tree(repo_root / "models", bundle_root / "models")

    data_dir = bundle_root / "data" / "ess"
    out_dir = bundle_root / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_src = repo_root / "configs" / f"{profile}.toml"
    copy_optional_file(profile_src, bundle_root / "configs" / f"{profile}.toml")

    policy_graph_digest = compute_policy_graph_digest(repo_root, bundle_root)
    manifest_digest = hash_file(repo_root / "models" / "manifest.toml")
    code_version = detect_code_version(repo_root)

    version_body = "\n".join(
        [
            f"code_version_tag={code_version}",
            f"policy_graph_digest={policy_graph_digest}",
            f"manifest_digest={manifest_digest}",
            f"profile={profile}",
        ]
    )
    (bundle_root / "VERSION.txt").write_text(version_body + "\n", encoding="utf-8")
    return bundle_root


def archive_bundle(bundle_root: Path, fmt: str) -> Path:
    version_file = (bundle_root / "VERSION.txt").read_text(encoding="utf-8")
    version = next((l.split("=", 1)[1] for l in version_file.splitlines() if l.startswith("code_version_tag=")), "unknown")
    digest = next((l.split("=", 1)[1][:12] for l in version_file.splitlines() if l.startswith("policy_graph_digest=")), "unknown")
    out_name = f"bundle_{version}_{digest}"

    if fmt == "zip":
        out_path = bundle_root.parent / f"{out_name}.zip"
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in bundle_root.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(bundle_root.parent))
        return out_path

    out_path = bundle_root.parent / f"{out_name}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tf:
        for path in sorted(p for p in bundle_root.rglob("*")):
            tf.add(path, arcname=path.relative_to(bundle_root.parent))
    return out_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UCF portable bundle")
    parser.add_argument("--target", required=True, help="Bundle output directory")
    parser.add_argument("--profile", default="prod", choices=["dev", "test", "prod"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--bin-source", default="target/release")
    parser.add_argument("--archive", choices=["zip", "tar.gz"])
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    bundle_root = build_bundle(
        repo_root=repo_root,
        target_dir=Path(args.target),
        profile=args.profile,
        bin_source=(repo_root / args.bin_source).resolve(),
    )
    print(f"bundle_root={bundle_root}")
    if args.archive:
        archive = archive_bundle(bundle_root, args.archive)
        print(f"archive={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
