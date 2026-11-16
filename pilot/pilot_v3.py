"""B3: content-matched loop direction. Same token 8-gram, first occurrence vs later repeats.

Also: per-dimension standardized diff-of-means and a logistic probe on the same tokens.
    modal run pilot_v3.py
"""

import json

import modal

from pilot import BASE, LAYERS, REV, grader, image, loop_tok, problems, prompt_ids, save, vol  # noqa: F401

app = modal.App("astra-pilot-v3")
image3 = image.add_local_file("pilot.py", "/root/pilot.py")


@app.function(image=image3, gpu="L4", volumes={"/vol": vol}, timeout=3600)
def directions_v3(ngram: int = 8, span: int = 2048, max_ls: int = 10000, probe_rows: int = 30000):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
    mlps = {L: m for n, m in model.named_modules() for L in LAYERS if n.endswith(f"layers.{L}.mlp")}
    cap = {}
    for L, m in mlps.items():
        m.register_forward_hook(lambda mod, i, o, L=L: cap.__setitem__(("out", L), o[0]))
        m.down_proj.register_forward_pre_hook(lambda mod, i, L=L: cap.__setitem__(("in", L), i[0][0]))
    Z = np.load("/root/u_astra.npz")
    U = {L: torch.tensor(Z[f"u{L}"], dtype=torch.float32, device="cuda") for L in LAYERS}
    cos = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))  # noqa: E731

    probs = problems()
    rows = [r for r in json.load(open("/root/pilot_inputs.json")) if r["loop_start_char"] is not None]
    z = lambda n: torch.zeros(n, dtype=torch.float64, device="cuda")  # noqa: E731
    S = {L: {k: [z(1024), z(1024), 0] for k in ("first", "repeat", "pre", "loop")} for L in LAYERS}  # sum, sumsq, n
    act = {L: {k: z(3584) for k in ("first", "repeat", "pre", "loop")} for L in LAYERS}
    X = {L: {"first": [], "repeat": []} for L in LAYERS}
    used = skipped = 0
    rng = np.random.default_rng(0)
    for r in rows:
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        ls = loop_tok(enc, r["loop_start_char"])
        if ls is None or ls < 256 or ls > max_ls:
            skipped += 1
            continue
        ids = enc.input_ids[: ls + span]
        # label every position in [ls, ls+span) by whether its token n-gram occurred earlier in the trace
        first_pos, seen = {}, {}
        for t in range(ngram - 1, len(ids)):
            key = tuple(ids[t - ngram + 1 : t + 1])
            if key not in seen:
                seen[key] = t
        repeat = [t for t in range(ls, len(ids)) if seen[tuple(ids[t - ngram + 1 : t + 1])] < t]
        first = sorted({seen[tuple(ids[t - ngram + 1 : t + 1])] for t in repeat})
        if len(repeat) < 64 or len(first) < 32:
            skipped += 1
            continue
        p = prompt_ids(tok, probs[r["id"]])
        P = len(p)
        with torch.no_grad():
            model(input_ids=torch.tensor([p + ids], device="cuda"))
        sel = {"first": torch.tensor(first, device="cuda") + P, "repeat": torch.tensor(repeat, device="cuda") + P,
               "pre": torch.arange(P + 128, P + ls, device="cuda"), "loop": torch.arange(P + ls, P + len(ids), device="cuda")}
        for L in LAYERS:
            o, inp = cap[("out", L)].float(), cap[("in", L)].float()
            for k, idx in sel.items():
                S[L][k][0] += o[idx].sum(0).double()
                S[L][k][1] += (o[idx] ** 2).sum(0).double()
                S[L][k][2] += len(idx)
                act[L][k] += inp[idx].sum(0).double()
            for k in ("first", "repeat"):
                take = rng.choice(len(sel[k]), min(len(sel[k]), 100), replace=False)
                X[L][k].append(o[sel[k][torch.tensor(take, device="cuda")]].cpu())
        used += 1
        cap.clear()

    out = {"traces_used": used, "traces_skipped": skipped, "ngram": ngram, "span": span, "u": {}, "act_means": {}, "stats": {}}
    for L in LAYERS:
        mean = {k: S[L][k][0] / S[L][k][2] for k in S[L]}
        var = {k: S[L][k][1] / S[L][k][2] - mean[k] ** 2 for k in S[L]}
        pooled_std = ((var["first"] + var["repeat"]) / 2).clamp_min(1e-8).sqrt()
        d = {"matched": (mean["repeat"] - mean["first"]).float(),
             "naive": (mean["loop"] - mean["pre"]).float()}
        d["matched_std"] = (d["matched"] / pooled_std.float())
        d["matched_std2"] = (d["matched"] / pooled_std.float() ** 2)
        Xf, Xr = torch.cat(X[L]["first"]).numpy(), torch.cat(X[L]["repeat"]).numpy()
        n = min(len(Xf), len(Xr), probe_rows)
        Xa = torch.tensor(np.concatenate([Xf[:n], Xr[:n]]), device="cuda")
        y = torch.cat([torch.zeros(n), torch.ones(n)]).to("cuda")
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        Xs = (Xa - mu) / sd
        w = torch.zeros(1024, device="cuda", requires_grad=True)
        b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + 1e-3 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        probe_acc = float((((Xs @ w + b) > 0).float() == y).float().mean())
        d["probe"] = (w / sd).detach()  # back to activation space
        out["stats"][str(L)] = {
            "n_first_tokens": int(S[L]["first"][2]), "n_repeat_tokens": int(S[L]["repeat"][2]),
            "probe_train_acc": probe_acc,
            "proj_u_astra": {k: float((mean[k].float() @ U[L])) for k in mean},
            "cos_with_u_astra": {k: cos(v, U[L]) for k, v in d.items()},
        }
        out["u"][str(L)] = {k: (v / v.norm()).cpu().tolist() for k, v in d.items()}
        out["act_means"][str(L)] = {k: (act[L][k] / S[L][k][2]).cpu().tolist() for k in act[L]}
    for k in out["u"]["15"]:
        a, b = torch.tensor(out["u"]["15"][k]), torch.tensor(out["u"]["19"][k])
        out["stats"][f"cos15_19_{k}"] = cos(a, b)
    save("directions_v3.json", out)
    return out["stats"] | {"traces_used": used, "traces_skipped": skipped}


@app.local_entrypoint()
def main():
    print(json.dumps(directions_v3.remote(), indent=1))
