"""Turn B3's directions_v3.json into smoke-test edit files (one per direction variant).

Columns: top-64 neurons by |mean loop activation × (uᵀW_j)| for that variant's own u.
Run: uv run --with numpy python build_v3_edits.py
"""

import json

import numpy as np

D = json.load(open("results/directions_v3.json"))
bf = lambda b: (b.astype(np.uint32) << 16).view(np.float32)  # noqa: E731
W = {L: bf(np.load(f"data/base_tensors/down_proj_{L}.npy")).astype(np.float32) for L in ("15", "19")}
Z = np.load("u_astra.npz")
cols_astra = {L: set(Z[f"cols{L}"].tolist()) for L in ("15", "19")}

print(f"traces used {D['traces_used']}, skipped {D['traces_skipped']}")
for L in ("15", "19"):
    st = D["stats"][L]
    print(f"layer {L}: first tokens {st['n_first_tokens']}, repeat tokens {st['n_repeat_tokens']}, probe acc {st['probe_train_acc']:.3f}")
    print("  proj on u_astra:", {k: round(v, 3) for k, v in st["proj_u_astra"].items()})
    print("  cos with u_astra:", {k: round(v, 3) for k, v in st["cos_with_u_astra"].items()})
print("cross-layer cos:", {k.replace("cos15_19_", ""): round(v, 3) for k, v in D["stats"].items() if k.startswith("cos15_19_")})

for variant in D["u"]["15"]:
    e = {"u_ours": {}, "cols_ours": {}, "tag": f"v3_{variant}"}
    for L in ("15", "19"):
        u = np.array(D["u"][L][variant], dtype=np.float32)
        act = np.array(D["act_means"][L]["loop"])
        top = np.argsort(-np.abs(act * (u @ W[L])))[:64]
        e["u_ours"][L], e["cols_ours"][L] = u.tolist(), sorted(top.tolist())
        print(f"  {variant:13s} L{L}: cols ∩ astra = {len(set(top.tolist()) & cols_astra[L])}/64")
    json.dump(e, open(f"edit_v3_{variant}.json", "w"))
print("wrote edit_v3_<variant>.json for:", ", ".join(D["u"]["15"]))
