"""Recover Astra's loop direction u per layer from the published bit-exact edit.

Fetches only the two edited down_proj tensors from the base checkpoint (HTTP range requests),
rebuilds the edited tensors, sha256-verifies both against the manifest, then SVDs the diff.
Run: uv run --with numpy --with requests python recover_direction.py
"""
import hashlib, json, struct
import numpy as np, requests

REV = "2fc06364715b967f1860aea9cf38778875588b17"
URL = f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{REV}/model.safetensors-00001-of-00001.safetensors"

def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)

def fetch_tensor(name):
    n = struct.unpack("<Q", requests.get(URL, headers={"Range": "bytes=0-7"}).content)[0]
    hdr = json.loads(requests.get(URL, headers={"Range": f"bytes=8-{7+n}"}).content)
    m = hdr[name]; assert m["dtype"] == "BF16", m
    a, b = m["data_offsets"]
    raw = requests.get(URL, headers={"Range": f"bytes={8+n+a}-{8+n+b-1}"}).content
    return np.frombuffer(raw, dtype=np.uint16).reshape(m["shape"])

man = json.load(open("edit/manifest.json")); z = np.load("edit/edit.npz")
out = {}
for i, t in enumerate(man["tensors"]):
    base = fetch_tensor(t["name"])
    assert hashlib.sha256(base.tobytes()).hexdigest() == t["base_sha256"], "base tensor mismatch"
    ed = base.copy().reshape(-1); ed[z[f"t{i}_idx"]] = z[f"t{i}_val"]; ed = ed.reshape(base.shape)
    assert hashlib.sha256(ed.tobytes()).hexdigest() == t["edited_sha256"], "edited tensor mismatch"
    W, Wp = bf16_to_f32(base).astype(np.float64), bf16_to_f32(ed).astype(np.float64)
    D = Wp - W
    cols = np.where(np.abs(D).sum(0) > 0)[0]
    U, s, Vt = np.linalg.svd(D[:, cols], full_matrices=False)
    u = U[:, 0]
    # strength a from W' = W - a u (u^T W) on the edited columns: u^T D_j = -a (u^T W_j)
    a = -(u @ D[:, cols]) / (u @ W[:, cols])
    # were the edited columns the top-64 writers of u in the base weights?
    writers = np.abs(u @ W); top64 = set(np.argsort(-writers)[:64])
    layer = t["name"].split(".layers.")[1].split(".")[0]
    out[layer] = {"u": u, "cols": cols, "sv": s[:4], "a_mean": float(a.mean()), "a_std": float(a.std()),
                  "top64_overlap": len(top64 & set(cols)), "writers_rank_of_cols": np.sort(np.argsort(np.argsort(-writers))[cols])[-5:]}
    print(f"layer {layer}: {len(cols)} cols edited; singular values {np.round(s[:4]/s[0],4)} (rank-1 if 2nd≈0); "
          f"strength a = {a.mean():.3f} ± {a.std():.3f}; edited cols ∩ top-64 writers = {len(top64 & set(cols))}/64; "
          f"worst writer-rank among edited cols = {out[layer]['writers_rank_of_cols'][-1]}")
u15, u19 = out["15"]["u"], out["19"]["u"]
print(f"cos(u15, u19) = {abs(u15 @ u19):.3f}")
np.savez("u_astra.npz", u15=u15, u19=u19, cols15=out["15"]["cols"], cols19=out["19"]["cols"],
         a15=out["15"]["a_mean"], a19=out["19"]["a_mean"])
print("saved u_astra.npz")
