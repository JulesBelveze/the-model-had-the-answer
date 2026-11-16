"""Offline: build direction variants from B2's saved span means and compare with Astra's u.

Run: uv run --with numpy python analyze_directions.py
Writes our_edit_<variant>.json for the smoke test (pilot.py --stage C --edit ...).
"""

import json
import sys

import numpy as np

D = json.load(open("results/directions_v2.json"))
Z = np.load("u_astra.npz")
bf = lambda b: (b.astype(np.uint32) << 16).view(np.float32)  # noqa: E731
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))  # noqa: E731
unit = lambda v: v / np.linalg.norm(v)  # noqa: E731

variants, edits = {}, {}
for L in ("15", "19"):
    m = {k: np.array(v) for k, v in D["means"][L].items()}
    u_astra, cols_astra = Z[f"u{L}"], set(Z[f"cols{L}"].tolist())
    d_pos = m["pre_late"] - m["pre_early"]
    d = {
        "naive": m["loop"] - m["pre"],
        "vs_late": m["loop"] - m["pre_late"],
        "tight512": m["loop_first512"] - m["pre_last512"],
    }
    d["naive_orth_pos"] = d["naive"] - (d["naive"] @ unit(d_pos)) * unit(d_pos)
    d["tight_orth_pos"] = d["tight512"] - (d["tight512"] @ unit(d_pos)) * unit(d_pos)
    W = bf(np.load(f"data/base_tensors/down_proj_{L}.npy")).astype(np.float32)  # [1024, 3584]
    act = {k: np.array(v) for k, v in D["act_means"][L].items()}
    print(f"\nlayer {L}  (cos(d_pos, u_astra) = {cos(d_pos, u_astra):.3f})")
    print(f"{'variant':16s} {'cos(u_astra)':>12s} {'cos(d_pos)':>10s} {'cols∩astra':>10s} {'Δcols∩astra':>11s}")
    for name, v in d.items():
        u = unit(v)
        writer = act["loop"] * (u @ W)
        delta = (act["loop"] - act["pre"]) * (u @ W)
        top = set(np.argsort(-np.abs(writer))[:64].tolist())
        topd = set(np.argsort(-np.abs(delta))[:64].tolist())
        print(f"{name:16s} {cos(u, u_astra):12.3f} {cos(u, d_pos):10.3f} {len(top & cols_astra):10d} {len(topd & cols_astra):11d}")
        variants.setdefault(name, {})[L] = u
        edits.setdefault(name, {"u_ours": {}, "cols_ours": {}, "tag": name})
        edits[name]["u_ours"][L] = u.tolist()
        edits[name]["cols_ours"][L] = sorted(top)

print("\ncross-layer consistency cos(u15, u19): astra 0.849;",
      {k: round(cos(v["15"], v["19"]), 3) for k, v in variants.items()})
for name, e in edits.items():
    json.dump(e, open(f"our_edit_{name}.json", "w"))
print("wrote", ", ".join(f"our_edit_{n}.json" for n in edits))
