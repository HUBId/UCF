# Prompt Runner

`python scripts/prompt_runner.py` is a local helper for a copy-paste workflow where one prompt is handled at a time.

## Safety model

- **Human-in-the-loop**: the script only prints prompts and updates queue metadata.
- **No auto-push / no auto-PR**: it does not run git push, create PRs, or execute prompts.
- **Dirty tree guard**: mutating commands refuse to run if `git status --porcelain` is non-empty unless `--allow-dirty` is passed.
- **Deterministic logs**: every prompt uses `./out/prompt_runs/<id>/`.
- **Offline best effort**: warns when proxy/network environment variables are present or the prompt text contains network-like command hints.

## Queue file format

File: `docs/prompt_queue.toml`

Each prompt is stored as `[[entries]]` with fields:

- `id` (string)
- `title` (string)
- `status` (`pending`, `running`, `done`, `failed`)
- `created_at` (ISO timestamp)
- `prompt_text` (multiline)
- `notes` (optional)
- `fail_reason` (optional, filled by `fail`)

## Commands

```bash
python scripts/prompt_runner.py status
python scripts/prompt_runner.py self-check
python scripts/prompt_runner.py next
python scripts/prompt_runner.py done 00129
python scripts/prompt_runner.py fail 00129 --reason "test failed"
python scripts/prompt_runner.py add --id 00131 --title "New prompt" --file prompt.txt
```

### `next`

- Finds the first `pending` entry.
- Prints ID/title/prompt text verbatim and the suggested log directory.
- Creates `./out/prompt_runs/<id>/`.
- Writes:
  - `prompt.txt` (exact prompt text)
  - `env_snapshot.txt` (python, OS, time, offline env hints)
- Sets status to `running` unless `--dry-run` is used.

### `done <id>`

- Sets status to `done`.
- Writes `./out/prompt_runs/<id>/done.txt` with timestamp.

### `fail <id> --reason ...`

- Sets status to `failed` and records `fail_reason`.
- Writes `./out/prompt_runs/<id>/failed.txt` with timestamp and reason.

## Failure recovery workflow

1. Mark failure with reason:
   `python scripts/prompt_runner.py fail <id> --reason "..."`
2. Inspect artifacts in `./out/prompt_runs/<id>/`.
3. Fix the issue manually.
4. Optionally reset status in `docs/prompt_queue.toml` to `pending` (manual edit), then run `self-check`.
