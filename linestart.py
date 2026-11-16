"""Recut of penalty.json and checks_acts.json at line starts only (the token before the position ends with a newline),
since every natural stop follows a newline.   uv run --with numpy --with transformers --with modal python linestart.py"""
import gzip, json
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer
from fullrun import BASE, REV

tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
rows = {(r["id"], r["sample"]): r["token_ids"] for r in (json.loads(l) for l in gzip.open("results/base_samples.jsonl.gz", "rt"))}
nl = {}
def is_ls(key, p):
    t = rows[key][p - 1]
    if t not in nl:
        nl[t] = tok.decode([t]).endswith("\n")
    return nl[t]

g = defaultdict(list)
for x in json.load(open("results/penalty.json")):
    if x["group"] != "stop" and is_ls((x["id"], x["sample"]), x["pos"]):
        g[x["group"]].append(x)
for a in json.load(open("results/checks_acts.json"))["approach"]:
    for i, p in enumerate(a["pos"]):
        if p["k"] == 0:
            g["stop (147)"].append(p)
        elif i > 0 and a["pos"][i - 1]["tok"].endswith("\n"):
            g["line start within 24 tokens of a stop"].append(p)
out = {}
for k, xs in g.items():
    gaps = np.array([x["gap"] for x in xs]); pp = np.array([x["p_end_pen"] for x in xs])
    out[k] = {"n": len(xs), "gap_median": float(np.median(gaps)), "gap_p10": float(np.percentile(gaps, 10)), "gap_min": float(gaps.min()),
              "p_end_pen_mean": float(pp.mean()), "share_p_gt_0": float((pp > 0).mean())}
    print(f"{k:42s} n={len(xs):4d} gap median {out[k]['gap_median']:5.1f} p10 {out[k]['gap_p10']:5.1f} min {out[k]['gap_min']:5.1f}  P(stop) mean {out[k]['p_end_pen_mean']:.4f}  share>0 {out[k]['share_p_gt_0']:.1%}")
json.dump(out, open("results/linestart.json", "w"), indent=1)
