#!/usr/bin/env python3
"""Methods-anchor check: does the artifact record every number the paper claims?

Lyra's proposed Agni check (2026-08-16), mechanized in its honest, dumb form:
three papers this weekend described protocols richer than the ones that ran
(the phantom 0.7, the 12.5%-that-was-a-gate, n=20 anchors that were n=5),
and each survived review because the shipped artifact simply DIDN'T RECORD
the parameter — so nothing contradicted the text. This tool sweeps a paper
for numeric claims and reports which of them have no anchor anywhere in the
run artifacts. It renders judgment on NOTHING: every unanchored number is a
question for a human, not an error. The output is a review worklist.

Usage:
    python methods_anchor_check.py --paper papers/foo.md \
        --artifacts data/run/*.json data/run/*.jsonl \
        [--sections Methods Results] [--json out.json]
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:%|)(?![\w.])")


def _norm_num(s: str) -> str:
    return f"{float(s.replace(',', '')):g}"

# Numbers that are almost never artifact parameters; reported but de-ranked.
BORING = {"0", "1", "2", "3"}


def is_boring(num: str, context: str) -> bool:
    if num in BORING:
        return True
    # citation years: 4-digit 1900-2035 (Nader 2000, Tulving 1973, ...)
    if re.fullmatch(r"\d{4}", num) and 1900 <= int(num) <= 2035:
        return True
    # section headings: "### 3.2 Experimental Design"
    if re.search(r"#+\s*" + re.escape(num), context):
        return True
    # model version strings: "Claude Opus 4.6", "Qwen3.5-27B"
    if re.search(r"(Opus|Sonnet|Haiku|Qwen|Gemma|GPT)[\w.\- ]*" + re.escape(num),
                 context):
        return True
    return False


def paper_numbers(text: str, sections: list[str] | None):
    """Yield (number_string, context) for numeric claims in the paper.

    If sections are given, only scan text under headings whose title
    contains one of them (case-insensitive) until the next same-or-higher
    heading.
    """
    if sections:
        keep, level = [], None
        for line in text.splitlines(keepends=True):
            m = re.match(r"(#+)\s*(.*)", line)
            if m:
                if any(s.lower() in m.group(2).lower() for s in sections):
                    level = len(m.group(1))
                    keep.append(line)
                    continue
                if level is not None and len(m.group(1)) <= level:
                    level = None
            if level is not None:
                keep.append(line)
        text = "".join(keep)

    for m in NUMBER_RE.finditer(text):
        start = max(0, m.start() - 60)
        context = text[start:m.end() + 60].replace("\n", " ").strip()
        yield m.group(1), context


def artifact_values(paths: list[str]) -> set[str]:
    """Collect every numeric value recorded in the artifacts, as normalized
    strings (so 77 matches 77.0, and 0.4615... matches a 0.462 claim at
    3-decimal rounding)."""
    values: set[str] = set()

    def add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            values.add(f"{v:g}")
            if isinstance(v, float):
                for nd in (1, 2, 3, 4):
                    values.add(f"{round(v, nd):g}")
        elif isinstance(v, str):
            for m in NUMBER_RE.finditer(v):
                values.add(f"{float(m.group(1)):g}")
        elif isinstance(v, dict):
            for k, x in v.items():
                add(k) if isinstance(k, (int, float)) else None
                add(x)
        elif isinstance(v, list):
            for x in v:
                add(x)

    for pattern in paths:
        for p in glob.glob(pattern):
            path = Path(p)
            try:
                if path.suffix == ".jsonl":
                    for line in path.open(encoding="utf-8", errors="replace"):
                        try:
                            add(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                else:
                    add(json.loads(path.read_text(encoding="utf-8",
                                                  errors="replace")))
            except (OSError, json.JSONDecodeError):
                print(f"warning: could not read {p}", file=sys.stderr)
    return values


def check(paper_path: str, artifact_paths: list[str],
          sections: list[str] | None) -> dict:
    text = Path(paper_path).read_text(encoding="utf-8", errors="replace")
    recorded = artifact_values(artifact_paths)

    seen: set[tuple[str, str]] = set()
    anchored, unanchored = [], []
    for num, context in paper_numbers(text, sections):
        norm = _norm_num(num)
        key = (norm, context)
        if key in seen:
            continue
        seen.add(key)
        entry = {"number": num, "context": context,
                 "boring": is_boring(num, context)}
        (anchored if norm in recorded else unanchored).append(entry)

    return {
        "paper": paper_path,
        "artifacts": artifact_paths,
        "n_recorded_values": len(recorded),
        "anchored": anchored,
        "unanchored": unanchored,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", required=True)
    ap.add_argument("--artifacts", nargs="+", required=True)
    ap.add_argument("--sections", nargs="*", default=None,
                    help="Only scan these sections (e.g. Methods Results)")
    ap.add_argument("--json", default=None, help="Write full JSON report")
    args = ap.parse_args()

    result = check(args.paper, args.artifacts, args.sections)

    interesting = [e for e in result["unanchored"] if not e["boring"]]
    boring = [e for e in result["unanchored"] if e["boring"]]
    print(f"paper: {result['paper']}")
    print(f"artifact numeric values collected: {result['n_recorded_values']}")
    print(f"anchored claims: {len(result['anchored'])}   "
          f"UNANCHORED: {len(interesting)} (+{len(boring)} trivial)")
    print()
    for e in interesting:
        print(f"  UNANCHORED {e['number']:>10}  ...{e['context']}...")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nfull report: {args.json}")
    # exit 1 when interesting unanchored numbers exist: usable as a gate
    sys.exit(1 if interesting else 0)


if __name__ == "__main__":
    main()
