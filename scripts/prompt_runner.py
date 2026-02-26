#!/usr/bin/env python3
"""Copy-paste prompt queue helper with safety guards."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required (tomllib missing).") from exc

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "docs" / "prompt_queue.toml"
LOG_ROOT = ROOT / "out" / "prompt_runs"
DEFAULT_TEMPLATE_PATH = ROOT / "docs" / "codex_prompt_template.txt"
ALLOWED_STATUSES = {"pending", "running", "done", "failed"}
OFFLINE_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
    "FTP_PROXY",
    "ftp_proxy",
]
NETWORK_HINT_TOKENS = [
    "curl ",
    "wget ",
    "Invoke-WebRequest",
    "git clone",
    "npm install",
    "pip install",
    "cargo add",
    "apt-get",
    "choco install",
    "winget install",
]


@dataclass
class Entry:
    id: str
    title: str
    status: str
    created_at: str
    prompt_text: str
    notes: str | None = None
    fail_reason: str | None = None


def load_queue(path: Path = QUEUE_PATH) -> list[Entry]:
    if not path.exists():
        raise SystemExit(f"Queue file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        raise SystemExit("Invalid queue format: 'entries' must be an array.")

    entries: list[Entry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise SystemExit("Invalid queue format: each entry must be a table.")
        entry = Entry(
            id=str(raw.get("id", "")).strip(),
            title=str(raw.get("title", "")).strip(),
            status=str(raw.get("status", "")).strip(),
            created_at=str(raw.get("created_at", "")).strip(),
            prompt_text=str(raw.get("prompt_text", "")),
            notes=(str(raw["notes"]) if "notes" in raw and raw["notes"] is not None else None),
            fail_reason=(
                str(raw["fail_reason"])
                if "fail_reason" in raw and raw["fail_reason"] is not None
                else None
            ),
        )
        entries.append(entry)
    return entries


def toml_escape_multiline(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"""\n{escaped}\n"""'


def toml_escape_inline(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_queue(entries: list[Entry], path: Path = QUEUE_PATH) -> None:
    lines: list[str] = ["# Prompt queue for copy-paste Codex workflow", ""]
    for entry in entries:
        lines.append("[[entries]]")
        lines.append(f"id = {toml_escape_inline(entry.id)}")
        lines.append(f"title = {toml_escape_inline(entry.title)}")
        lines.append(f"status = {toml_escape_inline(entry.status)}")
        lines.append(f"created_at = {toml_escape_inline(entry.created_at)}")
        lines.append(f"prompt_text = {toml_escape_multiline(entry.prompt_text)}")
        if entry.notes:
            lines.append(f"notes = {toml_escape_multiline(entry.notes)}")
        if entry.fail_reason:
            lines.append(f"fail_reason = {toml_escape_multiline(entry.fail_reason)}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def is_git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Failed to read git status: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def enforce_clean_tree(allow_dirty: bool) -> None:
    if allow_dirty:
        return
    if is_git_dirty():
        raise SystemExit(
            "Refusing to run with a dirty working tree. Commit/stash changes or pass --allow-dirty."
        )


def find_by_id(entries: list[Entry], prompt_id: str) -> Entry:
    for entry in entries:
        if entry.id == prompt_id:
            return entry
    raise SystemExit(f"Prompt ID not found: {prompt_id}")


def ensure_log_dir(prompt_id: str) -> Path:
    out_dir = LOG_ROOT / prompt_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_env_snapshot(path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        f"timestamp_utc={now}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"system={platform.system()}",
        f"machine={platform.machine()}",
    ]
    active_net_env = [f"{key}={os.environ.get(key, '')}" for key in OFFLINE_ENV_KEYS if key in os.environ]
    if active_net_env:
        lines.append("network_env_vars_detected=yes")
        lines.extend(active_net_env)
    else:
        lines.append("network_env_vars_detected=no")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def network_warnings(prompt_text: str) -> list[str]:
    warnings: list[str] = []
    set_vars = [key for key in OFFLINE_ENV_KEYS if key in os.environ]
    if set_vars:
        warnings.append(
            "Network/proxy environment variables detected: " + ", ".join(sorted(set_vars))
        )
    lowered = prompt_text.lower()
    matched = [token.strip() for token in NETWORK_HINT_TOKENS if token.lower() in lowered]
    if matched:
        warnings.append(
            "Prompt contains network-like command hints (best effort): " + ", ".join(matched)
        )
    return warnings


def print_prompt(entry: Entry, out_dir: Path) -> None:
    print(f"ID: {entry.id}")
    print(f"Title: {entry.title}")
    print(f"Status: {entry.status}")
    print(f"Log dir: {out_dir}")
    print("\n--- PROMPT START ---")
    print(entry.prompt_text, end="" if entry.prompt_text.endswith("\n") else "\n")
    print("--- PROMPT END ---")

    for warning in network_warnings(entry.prompt_text):
        print(f"WARNING: {warning}")


def cmd_next(args: argparse.Namespace) -> int:
    enforce_clean_tree(args.allow_dirty)
    entries = load_queue()
    entry = next((e for e in entries if e.status == "pending"), None)
    if entry is None:
        print("No pending prompts found.")
        return 0

    out_dir = ensure_log_dir(entry.id)
    print_prompt(entry, out_dir)

    if args.dry_run:
        print("Dry run: queue status not updated.")
        return 0

    entry.status = "running"
    save_queue(entries)
    (out_dir / "prompt.txt").write_text(entry.prompt_text, encoding="utf-8")
    write_env_snapshot(out_dir / "env_snapshot.txt")
    print(f"Updated {entry.id} -> running")
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    enforce_clean_tree(args.allow_dirty)
    entries = load_queue()
    entry = find_by_id(entries, args.id)
    entry.status = "done"
    entry.fail_reason = None
    save_queue(entries)

    out_dir = ensure_log_dir(entry.id)
    now = datetime.now(timezone.utc).isoformat()
    (out_dir / "done.txt").write_text(f"done_at_utc={now}\n", encoding="utf-8")
    print(f"Updated {entry.id} -> done")
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    enforce_clean_tree(args.allow_dirty)
    entries = load_queue()
    entry = find_by_id(entries, args.id)
    entry.status = "failed"
    entry.fail_reason = args.reason
    save_queue(entries)

    out_dir = ensure_log_dir(entry.id)
    now = datetime.now(timezone.utc).isoformat()
    (out_dir / "failed.txt").write_text(
        f"failed_at_utc={now}\nreason={args.reason}\n", encoding="utf-8"
    )
    print(f"Updated {entry.id} -> failed")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    entries = load_queue()
    counts: dict[str, int] = {status: 0 for status in sorted(ALLOWED_STATUSES)}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    next_pending = next((e.id for e in entries if e.status == "pending"), "none")

    print("Queue summary:")
    for status in ["pending", "running", "done", "failed"]:
        print(f"  {status}: {counts.get(status, 0)}")
    print(f"  next_pending: {next_pending}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    enforce_clean_tree(args.allow_dirty)
    entries = load_queue()
    if any(e.id == args.id for e in entries):
        raise SystemExit(f"Prompt ID already exists: {args.id}")

    prompt_text = Path(args.file).read_text(encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    entries.append(
        Entry(
            id=args.id,
            title=args.title,
            status="pending",
            created_at=now,
            prompt_text=prompt_text,
            notes=args.notes,
        )
    )
    save_queue(entries)
    print(f"Added prompt {args.id} -> pending")
    return 0


def render_prompt_template(template_text: str, task_text: str) -> str:
    start_marker = "START_TASK_SPECIFIC"
    end_marker = "END_TASK_SPECIFIC"

    if start_marker not in template_text or end_marker not in template_text:
        raise SystemExit(
            "Template is missing START_TASK_SPECIFIC/END_TASK_SPECIFIC markers."
        )

    start_idx = template_text.index(start_marker) + len(start_marker)
    end_idx = template_text.index(end_marker)
    if end_idx <= start_idx:
        raise SystemExit("Template markers are malformed (END before START).")

    normalized_task = task_text.strip("\n")
    replacement = f"\n{normalized_task}\n"
    return template_text[:start_idx] + replacement + template_text[end_idx:]


def cmd_render(args: argparse.Namespace) -> int:
    entries = load_queue()
    entry = find_by_id(entries, args.id)

    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = ROOT / template_path

    if not template_path.exists():
        raise SystemExit(f"Template file not found: {template_path}")

    template_text = template_path.read_text(encoding="utf-8")
    rendered = render_prompt_template(template_text, entry.prompt_text)

    print(rendered, end="" if rendered.endswith("\n") else "\n")
    return 0


def cmd_self_check(_: argparse.Namespace) -> int:
    entries = load_queue()
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise SystemExit("Self-check failed: duplicate IDs found.")

    invalid = [e.id for e in entries if e.status not in ALLOWED_STATUSES]
    if invalid:
        raise SystemExit(f"Self-check failed: invalid statuses for IDs: {', '.join(invalid)}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    probe = LOG_ROOT / ".write_probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink(missing_ok=True)

    template_ok = DEFAULT_TEMPLATE_PATH.exists()
    placeholder_ok = False
    rendered_ok = False
    if template_ok:
        template_text = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
        placeholder_ok = (
            "START_TASK_SPECIFIC" in template_text and "END_TASK_SPECIFIC" in template_text
        )
        if placeholder_ok and entries:
            try:
                _ = render_prompt_template(template_text, entries[0].prompt_text)
                rendered_ok = True
            except SystemExit:
                rendered_ok = False

    print("Self-check passed:")
    print(f"  entries={len(entries)}")
    print("  unique_ids=yes")
    print("  statuses_valid=yes")
    print(f"  out_dir_writable=yes ({LOG_ROOT})")
    print(f"  template_exists={'yes' if template_ok else 'no'} ({DEFAULT_TEMPLATE_PATH})")
    print(f"  template_placeholders_valid={'yes' if placeholder_ok else 'no'}")
    print(f"  template_render_smoke={'yes' if rendered_ok else 'no'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt queue helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_next = sub.add_parser("next", help="Show next pending prompt")
    p_next.add_argument("--dry-run", action="store_true", help="Do not modify queue status")
    p_next.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree")
    p_next.set_defaults(func=cmd_next)

    p_done = sub.add_parser("done", help="Mark a prompt done")
    p_done.add_argument("id", help="Prompt ID")
    p_done.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree")
    p_done.set_defaults(func=cmd_done)

    p_fail = sub.add_parser("fail", help="Mark a prompt failed")
    p_fail.add_argument("id", help="Prompt ID")
    p_fail.add_argument("--reason", required=True, help="Failure reason")
    p_fail.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree")
    p_fail.set_defaults(func=cmd_fail)

    p_status = sub.add_parser("status", help="Queue summary")
    p_status.set_defaults(func=cmd_status)

    p_add = sub.add_parser("add", help="Add prompt from file")
    p_add.add_argument("--id", required=True, help="Prompt ID")
    p_add.add_argument("--title", required=True, help="Prompt title")
    p_add.add_argument("--file", required=True, help="Path to text file containing prompt")
    p_add.add_argument("--notes", default=None, help="Optional notes")
    p_add.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree")
    p_add.set_defaults(func=cmd_add)

    p_render = sub.add_parser("render", help="Render prompt template with queue entry text")
    p_render.add_argument("--id", required=True, help="Prompt ID")
    p_render.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH.relative_to(ROOT)),
        help="Template path (default: docs/codex_prompt_template.txt)",
    )
    p_render.set_defaults(func=cmd_render)

    p_self = sub.add_parser("self-check", help="Validate queue and writeability")
    p_self.set_defaults(func=cmd_self_check)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
