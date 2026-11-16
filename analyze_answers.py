"""Summaries for answers.py outputs.   uv run --with numpy python analyze_answers.py"""
import json, re
from collections import defaultdict
import numpy as np
rng = np.random.default_rng(0)
def ci(a):
    a = np.array(a, float); b = rng.choice(a, (5000, len(a)), replace=True).mean(1)
    return f"{100*a.mean():.1f} [{100*np.percentile(b,2.5):.1f}, {100*np.percentile(b,97.5):.1f}]"
def per_problem(rows):
    d = defaultdict(list)
    for x in rows: d[x["id"]].append(x["correct"])
    return {k: np.mean(v) for k, v in d.items()}

A = json.load(open("results/answers_arms.json")); S = A["summary"]
print("== answer-phase variants at 8k (2,000 samples)")
for k in ("box_sampled@8k", "final_answer_sampled@8k", "final_answer_greedy@8k"):
    print(f"  {k:26s} acc {ci(list(per_problem([x for x in A['rows'] if x['arm']==k]).values()))}  boxed {S[k]['boxed']:.1%}")
base = {p["id"]: p for p in json.load(open("results/base_scores.json"))["per_problem"]}
sub = S["penalty_subset_ids"]
print("== growing penalties, 100 problems x 2 samples, 32k")
print(f"  base (same problems, samples 0-1)  acc {np.mean([base[i]['correct'][:2] for i in sub]):.1%} cut {np.mean([base[i]['cut'][:2] for i in sub]):.1%} tokens {np.mean([base[i]['tokens'][:2] for i in sub]):,.0f}")
for k in ("repetition_1.1", "frequency_0.2"):
    g = [x for x in A["rows"] if x["arm"] == k]
    print(f"  {k:34s} acc {ci(list(per_problem(g).values()))} cut {S[k]['cut_share']:.1%} tokens {S[k]['mean_tokens']:,.0f}")
    fin = [x for x in g if not x["cut"]]
    print(f"     finished {len(fin)}, correct among finished {np.mean([x['correct'] for x in fin]) if fin else float('nan'):.1%}; sample tail: {g[0]['tail'][-110:]!r}")

E = json.load(open("results/answers_edited.json")); pp = E["per_problem"]
print("== edited checkpoint through this pipeline (500 x 4)")
print("  ", {k: round(v, 4) for k, v in E["summary"].items()})
print(f"   acc {ci([np.mean(p['correct']) for p in pp])};  with forced box at 8k on top {ci([np.mean(p['forced8k']) for p in pp])}")
d = np.array([np.mean(p["forced8k"]) - np.mean(p["correct"]) for p in pp]); b = rng.choice(d, (5000, len(d)), replace=True).mean(1)
print(f"   forcing on the edit minus the edit, paired: {100*d.mean():+.1f} [{100*np.percentile(b,2.5):+.1f}, {100*np.percentile(b,97.5):+.1f}]")
f8 = defaultdict(list)
for x in json.load(open("results/forced.json"))["rows"]:
    if x["cut"] == "8k" and x["variant"] == "forced": f8[x["id"]].append(x["correct"])
d2 = np.array([np.mean(p["forced8k"]) - np.mean(f8[p["id"]]) for p in pp]); b2 = rng.choice(d2, (5000, len(d2)), replace=True).mean(1)
d3 = np.array([np.mean(f8[p["id"]]) - np.mean(p["correct"]) for p in pp]); b3 = rng.choice(d3, (5000, len(d3)), replace=True).mean(1)
print(f"   forced edit minus forced base: {100*d2.mean():+.1f} [{100*np.percentile(b2,2.5):+.1f}, {100*np.percentile(b2,97.5):+.1f}];  forced base minus my edit run: {100*d3.mean():+.1f} [{100*np.percentile(b3,2.5):+.1f}, {100*np.percentile(b3,97.5):+.1f}]")
tt = [t for p in pp for t, c in zip(p["think_tokens"], p["correct"]) if t is not None]
fin = [(c, t) for p in pp for c, t, cut in zip(p["correct"], p["tokens"], p["cut"]) if not cut]
print(f"   closed thinking {len(tt)/2000:.1%}, median thinking tokens when closed {np.median(tt):,.0f}; finished {len(fin)/2000:.1%}, correct among finished {np.mean([c for c,_ in fin]):.1%}, mean tokens finished {np.mean([t for _,t in fin]):,.0f}; finished without box {np.mean([(not b) and (not cut) for p in pp for b,cut in zip(p['boxed'],p['cut'])]):.1%}")

Z = json.load(open("results/answers_second.json")); T = Z["traces"]
print("== second model (Qwen3-1.7B, 300 problems, 16k budget)")
print("  ", {k: round(v, 3) for k, v in Z["summary"].items()})
gk = {k: np.median([t["gap_by_k"][str(k)] for t in T]) for k in (24, 12, 6, 3, 2, 1, 0)}
print("   median gap by tokens before the stop:", {k: round(float(v), 1) for k, v in gk.items()})
rows = [r for t in T for r in t["rows"]]
ns = [r for r in rows if not r["is_stop"]]
print(f"   line starts {len(rows)}; non-stop line start gap median {np.median([r['gap'] for r in ns]):.1f}; at stop {np.median([r['gap'] for r in rows if r['is_stop']]):.1f} (n={sum(r['is_stop'] for r in rows)})")
print(f"   stops preceded by newline: {np.mean([t['tail'].endswith(chr(10)) for t in T]):.0%}; windows (P>0.05, not taken) per trace {np.mean([sum(1 for r in t['rows'] if not r['is_stop'] and r['p_end_pen']>0.05) for t in T]):.2f}")
for t in T[:8]: print("     ", repr(t["tail"][-70:]))
print("== probe direction as an edit (25 problems):", {k: {kk: round(vv, 3) for kk, vv in v.items() if kk != "rows"} for k, v in json.load(open("results/smoke_edit_v3_probe.json")).items()})
