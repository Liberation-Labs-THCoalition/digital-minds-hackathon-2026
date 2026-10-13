# Tests for Run 3's generate(): the split, the reasoning budget, the forced close, the answer continuation, the
# thinking-mode guard, and the per-turn stats. No model: a fake tokenizer and a scripted sampler.
import importlib.util, json, sys, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location("rl", sys.argv[1])
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

class Tok:
    def __init__(self, tail="<|im_start|>assistant\n<think>\n"): self.tail = tail
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True): return "PROMPT" + self.tail
    def __call__(self, s): return type("E", (), {"input_ids": s.split()})()

def scripted(*replies):
    calls = []
    def sample(m, tok, text, n):
        calls.append((text, n))
        return replies[len(calls) - 1]
    return sample, calls

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print("OK  " if cond else "FAIL", name, "" if cond else detail)

# split_reply
check("split: closed", rl.split_reply("think a\n</think>\n\nAnswer.") == ("think a", "Answer."))
check("split: unclosed gives no reply", rl.split_reply("2.  **Analyze** still thinking") == ("2.  **Analyze** still thinking", ""))
check("split: stray <think> removed from reasoning", rl.split_reply("<think>x</think>y") == ("x", "y"))
check("split: second </think> stays in reply", rl.split_reply("a</think>b</think>c") == ("a", "b</think>c"))
# the serial run's real leak: what the OLD code returned for this string vs the new split
leak = "Thinking Process:\n\n1.  **Analyze the Request:** x\n\n2.  **Analyze the Facts:** y\n</think>\n\nI accept."
check("split: real leak shape -> reply is only the answer", rl.split_reply(leak)[1] == "I accept.", rl.split_reply(leak))

stats = Path(tempfile.mkdtemp()) / "gen.jsonl"
rl.GEN_STATS_PATH = stats
msgs = [{"role": "user", "content": "q"}]
# 1. closed and stopped: one call, at the reasoning budget
s, calls = scripted(("r1 r2\n</think>\n\nThe answer.", True))
v, th = rl.generate(None, Tok(), msgs, sample=s)
check("closed: reply", v == "The answer." and th == "r1 r2" and len(calls) == 1 and calls[0][1] == rl.THINK_BUDGET, (v, th, calls))
# 2. never closed: forced close, then an answer call of ANSWER_MAX on top of the reasoning
s, calls = scripted(("long reasoning *Final", False), ("Forced answer.", True))
v, th = rl.generate(None, Tok(), msgs, sample=s)
check("unclosed: forced close then answer", v == "Forced answer." and th == "long reasoning *Final"
      and len(calls) == 2 and calls[1][1] == rl.ANSWER_MAX and calls[1][0].endswith("*Final" + rl.FORCE_CLOSE), (v, th, calls))
# 3. EOS inside the reasoning, never closed: still forced, never returns the reasoning as the reply
s, calls = scripted(("reasoning then eos", True), ("", True))
v, th = rl.generate(None, Tok(), msgs, sample=s)
check("eos in reasoning: empty reply, not the reasoning", v == "" and th == "reasoning then eos" and len(calls) == 2, (v, th))
# 4. closed but the answer cut at the budget: continue the answer
s, calls = scripted(("r\n</think>\n\nPart one", False), (" and part two.", True))
v, th = rl.generate(None, Tok(), msgs, sample=s)
check("answer cut: continued", v == "Part one and part two." and len(calls) == 2 and calls[1][1] == rl.ANSWER_MAX, (v, calls))
# 5. not in thinking mode: refuse
try:
    rl.generate(None, Tok(tail="<|im_start|>assistant\n<think>\n\n</think>\n\n"), msgs, sample=scripted(("x", True))[0])
    check("non-thinking prompt refused", False, "no error raised")
except RuntimeError:
    check("non-thinking prompt refused", True)
# stats: one line per generate() call that ran (4), with the right flags
rows = [json.loads(l) for l in open(stats)]
check("stats: 4 rows", len(rows) == 4, len(rows))
check("stats: forced flags", [r["forced_close"] for r in rows] == [False, True, True, False], [r["forced_close"] for r in rows])
check("stats: continued flags", [r["answer_continued"] for r in rows] == [False, False, False, True])
check("stats: MPS memory recorded per turn (None off MPS)", all("mps_gb" in r for r in rows), rows[0])
check("stats: empty reply flagged", [r["visible_empty"] for r in rows] == [False, False, True, False])
# 6. every generate() call site keeps the reasoning: binds it to a name and records it as a "thinking" turn.
# The site count is checked against a plain text count, so a walker that finds nothing can't pass.
import ast
src = open(sys.argv[1]).read()
sites = {}
for fn in ast.walk(ast.parse(src)):
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
    kept = {c.args[1].id for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "record_turn"
            and len(c.args) == 2 and isinstance(c.args[0], ast.Constant) and c.args[0].value == "thinking"
            and isinstance(c.args[1], ast.Name)}
    for a in ast.walk(fn):
        if isinstance(a, ast.Assign) and isinstance(a.value, ast.Call) and getattr(a.value.func, "id", "") == "generate":
            tg = a.targets[0]
            name = tg.elts[1].id if isinstance(tg, ast.Tuple) and len(tg.elts) == 2 and isinstance(tg.elts[1], ast.Name) else None
            sites[a.lineno] = sites.get(a.lineno, False) or (name not in (None, "_") and name in kept)
check("call sites found", len(sites) == src.count("= generate(hf_model") > 0, (len(sites), src.count("= generate(hf_model")))
check("reasoning kept at every call site", all(sites.values()), sorted(l for l, g in sites.items() if not g))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
