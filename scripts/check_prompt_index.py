#!/usr/bin/env python3
"""Validate prompt index IDs are unique, contiguous, and sorted."""
from pathlib import Path
import re

path = Path('docs/prompt_series_index.md')
text = path.read_text(encoding='utf-8')
ids = [int(m.group(1)) for m in re.finditer(r'^\|\s*(\d+)\s*\|', text, flags=re.MULTILINE)]

if not ids:
    raise SystemExit('No prompt IDs found in docs/prompt_series_index.md')

if ids != sorted(ids):
    raise SystemExit('Prompt IDs are not sorted ascending.')

if len(ids) != len(set(ids)):
    raise SystemExit('Prompt IDs are not unique.')

expected = list(range(min(ids), max(ids) + 1))
if ids != expected:
    raise SystemExit(f'Prompt IDs are not contiguous: expected {expected[0]}..{expected[-1]}')

print(f'OK: {len(ids)} prompt IDs verified ({ids[0]}..{ids[-1]}).')
