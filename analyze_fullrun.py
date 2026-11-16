"""Phase 2 analysis: per-cut accuracy with bootstrap CIs over problems, truncation rates, token costs.

Run after pulling results/base_scores.json and results/forced.json from the volume:
    uv run --with numpy python analyze_fullrun.py
"""

import json
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(0)
base = json.load(open("results/base_scores.json"))
forced = json.load(open("results/forced.json"))


def ci(per_problem_means, n_boot=10000):
    a = np.array(per_problem_means)
    boots = rng.choice(a, (n_boot, len(a)), replace=True).mean(1)
    return a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


rows = []
# base, as is
pp = [np.mean(p["correct"]) for p in base["per_problem"]]
m, lo, hi = ci(pp)
cut = np.mean([c for p in base["per_problem"] for c in p["cut"]])
rows.append(("base, 32k, as is", m, lo, hi, cut, base["mean_tokens"]))

# forced / free per cut
by = defaultdict(lambda: defaultdict(list))
for x in forced["rows"]:
    by[(x["cut"], x["variant"])][x["id"]].append(x)
tok_by_id = {p["id"]: p["tokens"] for p in base["per_problem"]}
for (cname, vname), d in sorted(by.items(), key=lambda kv: ({"4k": 0, "8k": 1, "16k": 2, "loop": 3}[kv[0][0]], kv[0][1])):
    pp = [np.mean([x["correct"] for x in xs]) for xs in d.values()]
    m, lo, hi = ci(pp)
    # thinking tokens actually spent: min(trace length, cut) for applied rows, full length otherwise
    spent = []
    for pid, xs in d.items():
        for x in xs:
            full = tok_by_id[pid][x["sample"]]
            spent.append(min(full, x["cut_tok"]) + x["gen_tokens"] if x["applied"] else full)
    applied = np.mean([x["applied"] for xs in d.values() for x in xs])
    label = f"{'cap + forced box' if vname == 'forced' else 'cap only'} @ {cname}"
    rows.append((label, m, lo, hi, applied, np.mean(spent)))

print(f"{'config':26s} {'acc':>6s}  {'95% CI':>14s}  {'cut/applied':>11s}  {'mean tokens':>11s}")
for label, m, lo, hi, c, t in rows:
    print(f"{label:26s} {m:6.1%}  [{lo:5.1%}, {hi:5.1%}]  {c:11.1%}  {t:11,.0f}")

print("\nReference rows (not ours): Astra edit 48.1% (post) / 51.0% (two independent reruns), 33% cut, ~22k tokens;"
      " thinking-off 43-50% (issue #4); 81,920-token budget 22.5% (Carlos C2).")

# paired comparison forced@8k vs base on per-problem means
d8 = by[("8k", "forced")]
diff = [np.mean([x["correct"] for x in d8[p["id"]]]) - np.mean(p["correct"]) for p in base["per_problem"]]
m, lo, hi = ci(diff)
print(f"\nforced@8k minus base, paired over problems: {m:+.1%} [{lo:+.1%}, {hi:+.1%}]")
fixed = sum(1 for x in diff if x > 0); broken = sum(1 for x in diff if x < 0)
print(f"problems improved {fixed}, worsened {broken}, unchanged {500 - fixed - broken}")

# gold already written before the cut (gold_precut.py), split by forced outcome, applied rows only
try:
    gp = {(g["id"], g["sample"]): g for g in json.load(open("results/gold_precut.json"))}
except FileNotFoundError:
    gp = None
if gp:
    print("\ngold present in the first k thinking tokens, among samples where forcing was applied:")
    for cname in ("4k", "8k", "16k"):
        ap = [x for xs in by[(cname, "forced")].values() for x in xs if x["applied"]]
        for crit in ("boxed", "verbatim"):
            ok = [gp[(x["id"], x["sample"])][f"{crit}_{cname}"] for x in ap if x["correct"]]
            ko = [gp[(x["id"], x["sample"])][f"{crit}_{cname}"] for x in ap if not x["correct"]]
            print(f"  {cname:>3s} {crit:8s}: forced-correct {np.mean(ok):5.1%} (n={len(ok)})  forced-wrong {np.mean(ko):5.1%} (n={len(ko)})")
