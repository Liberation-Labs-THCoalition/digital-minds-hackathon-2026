# Repo Cleanup Report — 2026-08-14

Maintenance pass ahead of judging. Scope: .gitignore, oversized/sensitive files, README accuracy, repo index.

## Changes made

1. **`.gitignore` rewritten.** Previous version covered bytecode, model weights, `.env`, `.DS_Store`, `*.log`, and `data/raw/`. Added: `*.key` / `credentials*`, `Thumbs.db`, `tmp/` + `*.tmp` + `scratch/`, editor files (`.vscode/`, `.idea/`, `*.swp`, `*.swo`), and `.env.*`. Added explicit keep rules (`!data/circumplex_profiles/*.json`, `!data/butlin_scores/*.json`, `!data/orientation/*.jsonl`, `!infrastructure/*.md`) so results and documentation can never be accidentally swept by a future ignore rule. Verified with `git check-ignore`: pycache and `data/raw/` are ignored, the committed profile JSON is not.

2. **`README.md` Repository Structure section corrected** (see findings 3.x below). No other README prose changed.

3. **`infrastructure/REPO_INDEX.md` created** — one-line description of every tracked file, organized by directory, for judges.

## Findings

### 1. Files that shouldn't be in the repo

- **Files > 1MB:** two, both intentional submission media:
  - `videos/circumplex/circumplex.mp4` (3.2 MB)
  - `videos/ghost_dimensions/ghost_dimensions.mp4` (3.3 MB)
  These are the track intro videos and appear deliberate. Left in place; flag if the repo should stay lean, since they're also in git history now.
- **`mnemosyne/__pycache__/` exists on disk but is NOT tracked** — the old .gitignore was already catching it. No action needed beyond keeping the rule.
- **Credentials scan: clean.** Grepped all non-.md files for `Bearer`, `api_key`, `password`, `token`, `secret`, `sk-…`, `hf_…` patterns. Every hit is benign (`tokenizer`, `max_tokens`, `TOKENIZERS_PARALLELISM`, etc.). No keys, tokens, or passwords anywhere in the repo.
- **Working tree was clean** before this pass (no untracked strays besides pycache).

### 2. Absolute paths

- `infrastructure/hackathon_launch.sh` — hardcodes `~/lab/projects/hackathon-digital-minds/`, `~/lab/projects/mnemosyne-jlens/`, `~/lab/mechinterp-env/bin/modal`, `~/agents/nexus`. This is a machine-specific launch script kept as process documentation; it is not runnable by judges as-is. Left unchanged but noted — consider a header comment stating it's archival.
- `infrastructure/stray_files_report.md` — mentions `~/...` paths, but that's the point of the document (an audit of files outside the repo). Fine.
- **`data/circumplex_profiles/profile_dense_32b.json` records `"model": "/Users/[AGENT]/models/qwen3-32b"`** — leaks a personal macOS username from whoever ran the dense-32B profile. Harmless technically, but it's a small privacy leak and an inconsistency (the run wasn't on MTH). Left untouched because it's a committed experimental result under pre-registration ("already collected... committed" per preregister_circumplex.md) — editing it would tamper with data provenance. Decide as a team whether to redact the path in a follow-up commit with a note.
- Python code is clean: all experiment scripts use relative/`Path(__file__)` imports or model names, no `/home/admin` literals.

### 3. README review

Fixed in this pass (Repository Structure section):

- 3.1 Listed `mnemosyne/variable_landing.py`, which doesn't exist — the file is `mnemosyne/archive/variable_landing_old.py` (superseded), and the live Track 5 code is `experiments/variable_landing/`. Corrected.
- 3.2 `experiments/` showed only `moe_jlens/`; actual tree has `circumplex/`, `ghost_probe/`, `orientation/`, and `variable_landing/` too. Corrected.
- 3.3 `papers/` said "PDF + source" — there are no PDFs, only markdown + three reference reviews. Corrected to "Six submission papers + literature reviews". If PDFs are a submission requirement, they still need to be generated.
- 3.4 `ethics/` said "Orientation transcript, aftercare protocol, Butlin scoring" — the directory actually contains the Butlin instrument, the judge agent spec, and the metacognitive memory spec. Corrected the structure line. **Note:** the Ethics prose section (unchanged) still promises "orientation transcript, ... and welfare monitoring data" in `ethics/` — those files don't exist yet. Either add them when the Day 1 run's artifacts land or soften the prose before submission.
- 3.5 `videos/` was missing from the structure entirely. Added.
- 3.6 `data/` description now matches the whitelisted result dirs (only `circumplex_profiles/` is populated so far; `butlin_scores/` and `orientation/` are expected during the weekend).

Checked, no action needed:

- **All 6 submissions listed** with correct, existing paper paths: metacognitive_memory (T4), variable_landing (T5), circumplex_jspace (T2), ghost_dimensions (T3), moe_jlens (T6), butlin_observation (naturalistic). Both ethics links resolve.
- **No broken internal links.** External links (Mnemosyne GitHub repo, firstdonoharm.dev) not verified from here but consistent with LICENSE.md.
- **Team list** (Thomas, Nexus, Lyra, CC, Vera, Dwayne Wilkes, Kavi, Ang Jandak, Arc, Wren) is a superset of every paper's author list and includes Vera (videos). `infrastructure/weekend_spec.md`'s team line omits Vera — README appears to be the current one. Minor: several papers still carry `[affiliation]` / `Dwayne [surname]` placeholders that should be filled before submission.

### 4. data/ dirs referenced by .gitignore keeps

`data/butlin_scores/` and `data/orientation/` don't exist yet. The keep rules are in place so their results commit cleanly when the runs produce them.

## Not done / for the team

- Decide on the `margaret` path redaction in profile_dense_32b.json (data provenance vs. privacy).
- Fill `[affiliation]` and `[surname]` placeholders in paper author lists.
- Generate paper PDFs if the hackathon requires them.
- Add the promised orientation transcript + welfare monitoring data to `ethics/` once Day 1 completes, or amend the README Ethics prose.
- These changes are uncommitted (`.gitignore`, `README.md` modified; `REPO_INDEX.md`, this report new). Commit when reviewed.
