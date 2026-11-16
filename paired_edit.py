"""Forced@8k vs the authors' edit (p50c per-problem logs), paired; and the gold-present check by answer type
with a shuffled-gold control.   uv run --with numpy python paired_edit.py"""
import glob, gzip, json, re
from collections import defaultdict
import numpy as np

rng = np.random.default_rng(0)
f8 = defaultdict(list)
for x in json.load(open("results/forced.json"))["rows"]:
    if x["cut"] == "8k" and x["variant"] == "forced":
        f8[x["id"]].append(x)
edit = {}
for f in glob.glob("logs/p50c/parts/*/math500_per_problem.json"):
    for p in json.load(open(f)):
        edit[p["id"]] = np.mean(p["correct"]) if isinstance(p["correct"], list) else float(p["correct"])
ids = [i for i in f8 if i in edit]
a = np.array([np.mean([x["correct"] for x in f8[i]]) for i in ids]); b = np.array([edit[i] for i in ids])
boots = rng.choice(a - b, (10000, len(ids)), replace=True).mean(1)
print(f"matched {len(ids)}; forced {a.mean():.1%}, edit {b.mean():.1%}; paired diff {np.mean(a-b):+.1%} [{np.percentile(boots,2.5):+.1%}, {np.percentile(boots,97.5):+.1%}]; r={np.corrcoef(a,b)[0,1]:.2f}")
print(f"solved at least once: both {np.sum((a>0)&(b>0))}, forced only {np.sum((a>0)&(b==0))}, edit only {np.sum((a==0)&(b>0))}, neither {np.sum((a==0)&(b==0))}")

# gold present before the cut, non-trivial answers only, with shuffled-gold control (needs decoded prefixes)
from transformers import AutoTokenizer
from fullrun import BASE, REV
tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
rows = {(r["id"], r["sample"]): r for r in (json.loads(l) for l in gzip.open("results/base_samples.jsonl.gz", "rt"))}
golds = [r["answer"] for r in rows.values() if r["sample"] == 0]
pat = lambda g: re.compile(r"(?<![\w.])" + re.escape(g.strip()) + r"(?![\w.])")
trivial = lambda g: bool(re.fullmatch(r"-?\d{1,2}", g.strip()))
res = defaultdict(list)
for i in f8:
    for x in f8[i]:
        if not x["applied"]:
            continue
        r = rows[(i, x["sample"])]
        txt = tok.decode(r["token_ids"][:8192])
        g = r["answer"]
        other = golds[rng.integers(len(golds))]
        while trivial(other) != trivial(g) or other == g:
            other = golds[rng.integers(len(golds))]
        key = ("short int" if trivial(g) else "other", "correct" if x["correct"] else "wrong")
        res[key].append((bool(pat(g).search(txt)), bool(pat(other).search(txt))))
for k, v in sorted(res.items()):
    v = np.array(v)
    print(f"{k[0]:9s} forced-{k[1]:7s} n={len(v):4d}  gold present {v[:,0].mean():5.1%}  shuffled gold {v[:,1].mean():5.1%}")
