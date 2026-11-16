"""What makes a line a closing line, and the repeat curve.   uv run --with numpy python analyze_lines.py"""
import json, re
from collections import defaultdict
import numpy as np

T = json.load(open("results/answers_lines.json"))["traces"]
rows = [dict(r, kind=t["kind"], ls=t["ls"]) for t in T for r in t["rows"]]
print("line starts:", len(rows), "stops:", sum(r["is_stop"] for r in rows))
DONE = re.compile(r"done|ready|proceed|generat|output|response|finish|write|present|provide|produce|go with|stick with|good to go|looks (good|solid|perfect)|end of thought", re.I)
CONF = re.compile(r"confident|final answer|the answer is|result is|solution is", re.I)
def feats(r):
    raw = r["prev_line"].rstrip("\n"); last = raw.split("\n")[-1]
    return {"indented": last.startswith((" ", "\t")), "bullet": bool(re.match(r"\s*([*\-]|\d+\.)\s", last)), "blank_before": r["prev_line"].endswith("\n\n") or last.strip() == "",
            "done_words": bool(DONE.search(last)), "confident_words": bool(CONF.search(last)), "has_math": "$" in last, "short": len(last.strip()) < 40}
for r in rows:
    r.update(feats(r))
fin = [r for r in rows if r["kind"] == "fin"]
print(f"\nfinished traces: {len(fin)} line starts; reachable (P>0) {np.mean([r['p_end_pen']>0 for r in fin]):.1%}; gap<5: {np.mean([r['gap']<5 for r in fin]):.1%}")
print(f"{'feature':16s} {'share of lines':>14s} {'median gap if yes':>18s} {'if no':>7s} {'P(stop taken) yes':>18s} {'no':>7s}")
for f in ("done_words", "confident_words", "indented", "bullet", "blank_before", "has_math", "short"):
    y = [r for r in fin if r[f]]; n = [r for r in fin if not r[f]]
    print(f"{f:16s} {len(y)/len(fin):14.1%} {np.median([r['gap'] for r in y]):18.1f} {np.median([r['gap'] for r in n]):7.1f} {np.mean([r['is_stop'] for r in y]):18.2%} {np.mean([r['is_stop'] for r in n]):7.2%}")
print("\ncombos (finished traces): n, median gap, share with gap<5, share that are the stop")
for name, cond in (("done & not indented", lambda r: r["done_words"] and not r["indented"]), ("done & indented", lambda r: r["done_words"] and r["indented"]),
                   ("confident & not indented", lambda r: r["confident_words"] and not r["indented"]), ("confident & indented", lambda r: r["confident_words"] and r["indented"]),
                   ("neither & not indented", lambda r: not r["done_words"] and not r["confident_words"] and not r["indented"]),
                   ("neither & indented", lambda r: not r["done_words"] and not r["confident_words"] and r["indented"])):
    g = [r for r in fin if cond(r)]
    print(f"  {name:26s} n={len(g):5d} gap {np.median([r['gap'] for r in g]):5.1f}  gap<5 {np.mean([r['gap']<5 for r in g]):6.1%}  stop {np.mean([r['is_stop'] for r in g]):6.2%}")
st = [r for r in fin if r["is_stop"]]
print(f"\nthe {len(st)} stops: indented {np.mean([r['indented'] for r in st]):.0%}, bullet {np.mean([r['bullet'] for r in st]):.0%}, done words {np.mean([r['done_words'] for r in st]):.0%}, confident words {np.mean([r['confident_words'] for r in st]):.0%}")
# position in trace: relative position of low-gap line starts
for t in T:
    if t["kind"] == "fin":
        n = len(t["rows"])
        for i, r in enumerate(t["rows"]): r["rel"] = i / max(1, n - 1)
relfin = [r for t in T if t["kind"] == "fin" for r in t["rows"]]
for lo, hi in ((0, .5), (.5, .9), (.9, .98), (.98, 1.01)):
    g = [r for r in relfin if lo <= r["rel"] < hi]
    print(f"  lines at {lo:.2f}-{hi:.2f} of the trace: median gap {np.median([r['gap'] for r in g]):5.1f}, gap<5 {np.mean([r['gap']<5 for r in g]):6.1%}")
# near misses: low gap but not taken
nm = sorted([r for r in rows if not r["is_stop"] and r["gap"] < 3], key=lambda r: r["gap"])
print(f"\nnear misses (gap<3, not taken): {len(nm)} ({sum(r['kind']=='loop' for r in nm)} in traces that later looped)")
for r in nm[:14]: print(f"   gap {r['gap']:4.1f} p {r['p_end_pen']:.2f} [{r['kind']}] {r['prev_line'].rstrip()[-70:]!r}")
# repeat curve
print("\nrepeat curve (looping traces): mean per-token log-prob of a line by how many times the identical line already appeared")
by = defaultdict(list)
for r in rows:
    if r["kind"] == "loop" and r["line_lp"] is not None and r["line_len"] >= 5:
        k = r["n_prev"]; by[k if k < 5 else (5 if k < 10 else 10)].append(r["line_lp"])
for k in sorted(by):
    lab = {5: "5-9", 10: "10+"}.get(k, str(k))
    print(f"  seen {lab:>4s} times before: n={len(by[k]):6d}  mean log-prob/token {np.mean(by[k]):7.3f}  => per-token prob {np.exp(np.mean(by[k])):.3f}")
gl = defaultdict(list)
for r in rows:
    if r["kind"] == "loop": gl[min(r["n_prev"], 10)].append(r["gap"])
print("  stop-token gap at the line start after a line seen n times before:", {k: round(float(np.median(v)), 1) for k, v in sorted(gl.items())})

print("\nstopping windows per trace (line start with sampler P(stop) > 0.05, not taken)")
for kind in ("fin", "loop"):
    ts = [t for t in T if t["kind"] == kind]
    w = [sum(1 for r in t["rows"] if not r["is_stop"] and r["p_end_pen"] > 0.05 and (kind == "fin" or r["p"] < t["ls"])) for t in ts]
    print(f"  {kind}: {len(ts)} traces, share with at least one {np.mean([x > 0 for x in w]):.1%}, mean {np.mean(w):.2f} per trace")
d = [t["rows"][-1]["p"] - r["p"] for t in T if t["kind"] == "fin" for r in t["rows"] if not r["is_stop"] and r["p_end_pen"] > 0.05]
print(f"  finished traces: tokens between an untaken window and the eventual stop: median {np.median(d):.0f}, p90 {np.percentile(d, 90):.0f}")
print("  in-loop line starts with P(stop) > 0:", sum(1 for t in T if t["kind"] == "loop" for r in t["rows"] if r["p"] >= t["ls"] and r["p_end_pen"] > 0))
