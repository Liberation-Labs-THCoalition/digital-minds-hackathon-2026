#!/usr/bin/env python3
"""Generate or verify ORIENTATION_FREEZE.sha256 — the verbatim-text pins.

The replication bundle claims the 9 authored messages and every FAQ answer
are delivered verbatim. This makes that claim checkable: each pinned text
is hashed (sha256 of the UTF-8 encoding of the whitespace-stripped text),
and a replicating lab runs `--check` to prove their copy matches.

Usage:
    python freeze_hashes.py            # (re)write ORIENTATION_FREEZE.sha256
    python freeze_hashes.py --check    # verify; exit 1 on any mismatch
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE_FILE = HERE / "ORIENTATION_FREEZE.sha256"

sys.path.insert(0, str(HERE))
from run_orientation_v2 import ORIENTATION, SYSTEM_PROMPT  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def faq_answers(faq_path: Path):
    """Yield (question, answer_text) pairs from orientation_faq.md."""
    raw = faq_path.read_text()
    # Blocks start at '**Q: ...**'; an answer runs to the next question
    # or section header.
    pattern = re.compile(
        r"\*\*Q: (?P<q>.+?)\*\*\n(?P<a>.*?)(?=\n\*\*Q: |\n## |\Z)",
        re.DOTALL)
    for m in pattern.finditer(raw):
        yield m.group("q").strip(), m.group("a").strip()


def _slug(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:48]


def pinned_items():
    items = [("system_prompt", SYSTEM_PROMPT)]
    for msg in ORIENTATION:
        items.append((f"message:{msg['id']}", msg["text"]))
    for question, answer in faq_answers(HERE / "orientation_faq.md"):
        items.append((f"faq:{_slug(question)}", answer))
    return items


def write_freeze():
    lines = [f"{_sha(text)}  {label}" for label, text in pinned_items()]
    FREEZE_FILE.write_text("\n".join(lines) + "\n")
    print(f"wrote {FREEZE_FILE.name}: {len(lines)} pins")


def check_freeze() -> int:
    recorded = {}
    for line in FREEZE_FILE.read_text().splitlines():
        digest, label = line.split(None, 1)
        recorded[label] = digest
    current = {label: _sha(text) for label, text in pinned_items()}
    failures = []
    for label, digest in current.items():
        if recorded.get(label) != digest:
            failures.append(label)
    missing = set(recorded) - set(current)
    for label in sorted(failures):
        print(f"MISMATCH: {label}")
    for label in sorted(missing):
        print(f"MISSING FROM SOURCE: {label}")
    if failures or missing:
        return 1
    print(f"OK: all {len(current)} pins verified")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify against the recorded freeze file")
    args = ap.parse_args()
    sys.exit(check_freeze() if args.check else (write_freeze() or 0))
