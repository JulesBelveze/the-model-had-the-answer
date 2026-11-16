"""Post-review checks.

    modal run --detach checks.py

gen():  close-only at 8k with a 4,096-token answer budget (was 512), and forced commit at 0 / 256 / 1k / 2k tokens.
acts(): (1) rebuild Astra's direction from its own eight texts (fwd_4: residual after blocks 15/19, mean of the
        last 36 tokens, repetitive minus productive) and compare with the released u; (2) the same kind of contrast
        on natural looping traces at the residual stream; (3) stop-token logits at every one of the 24 positions
        before each natural stop, with the text, to see whether stopping is a cliff or a short ramp.
"""

import gzip
import json
import random

import modal

from fullrun import BASE, FORCED_TXT, OUT, REV, chat_prompt, dataset, grader, image, loop_start_tok, make_llm, save_json, vol
from penalty import PEN, sampler_prob

app = modal.App("astra-checks")
img = (image.add_local_file("fullrun.py", "/root/fullrun.py").add_local_file("penalty.py", "/root/penalty.py")
       .add_local_file("u_astra.npz", "/root/u_astra.npz").add_local_file("data/fwd_4.json", "/root/fwd_4.json"))


@app.function(image=img, gpu="H100", volumes={"/vol": vol}, timeout=3 * 3600)
def gen():
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    Q, ds = grader(), dataset()
    probs = {r["unique_id"]: r["problem"] for r in ds}
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    close = tok("\n</think>\n\n", add_special_tokens=False).input_ids
    forced = tok(FORCED_TXT, add_special_tokens=False).input_ids
    rows = [json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt")]
    arms = [("free4096@8k", 8192, [], SamplingParams(max_tokens=4096, temperature=0))] + [
        (f"forced@{c}", c, forced, SamplingParams(max_tokens=32, temperature=0)) for c in (0, 256, 1024, 2048)]
    reqs, params, meta, res, pc = [], [], [], [], {}
    for r in rows:
        ids = r["token_ids"]
        p = pc.setdefault(r["id"], tok(chat_prompt(tok, probs[r["id"]]), add_special_tokens=False).input_ids)
        for name, c, extra, sp in arms:
            if r["finish"] == "stop" and len(ids) <= c:
                res.append({"id": r["id"], "sample": r["sample"], "arm": name, "applied": False, "correct": r["correct"]})
                continue
            reqs.append(TokensPrompt(prompt_token_ids=p + ids[:c] + close + extra))
            params.append(sp)
            meta.append((r["id"], r["answer"], r["sample"], name))
    outs = make_llm().generate(reqs, params)
    for (pid, gold, k, name), o in zip(meta, outs):
        g = o.outputs[0]
        pred, _ = Q.extract(FORCED_TXT + g.text if name.startswith("forced") else g.text)
        res.append({"id": pid, "sample": k, "arm": name, "applied": True, "pred": pred, "correct": bool(Q.is_correct(pred, gold)),
                    "finish": g.finish_reason, "gen_tokens": len(g.token_ids)})
    summary = {}
    for name, *_ in arms:
        g = [x for x in res if x["arm"] == name]
        ap = [x for x in g if x["applied"]]
        summary[name] = {"acc": sum(x["correct"] for x in g) / len(g), "applied": len(ap),
                         "hit_limit": sum(x["finish"] == "length" for x in ap) / len(ap),
                         "no_box": sum(not x["pred"] for x in ap) / len(ap),
                         "mean_gen_tokens": sum(x["gen_tokens"] for x in ap) / len(ap)}
    save_json("checks_gen.json", {"summary": summary, "rows": res})
    return summary


@app.function(image=img, gpu="H100", volumes={"/vol": vol}, timeout=2 * 3600)
def acts():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
    Z = np.load("/root/u_astra.npz")
    u = {16: Z["u15"], 20: Z["u19"]}  # edit on layer l uses the residual after block l, i.e. hidden_states[l+1]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    out = {}

    def hidden(ids):
        hs = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, output_hidden_states=True).hidden_states
        return {l: hs[l][0].float().cpu().numpy() for l in (16, 20)}, hs[-1][0]

    with torch.inference_mode():
        # 1. Astra's recipe on Astra's eight texts
        texts = json.load(open("/root/fwd_4.json"))["texts"]
        means = {16: [], 20: []}
        for t in texts:
            for add in (False,):
                h, _ = hidden(tok(t, add_special_tokens=add).input_ids)
            for l in means:
                means[l].append(h[l][-36:].mean(0))
        d = {l: (np.stack(m)[1::2] - np.stack(m)[::2]).mean(0) for l, m in means.items()}
        out["astra_recipe_cos"] = {l: cos(d[l], u[l]) for l in d}
        out["astra_recipe_per_pair_cos"] = {l: [cos(np.stack(means[l])[2 * i + 1] - np.stack(means[l])[2 * i], u[l]) for i in range(4)] for l in d}

        # 2. natural looping traces, residual stream
        probs = {r["unique_id"]: r["problem"] for r in dataset()}
        rows = [json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt")]
        random.seed(0)
        looped = [r for r in rows if r["finish"] == "length"]
        for r in looped:
            r["ls"] = loop_start_tok(r["token_ids"])
        looped = random.sample([r for r in looped if r["ls"] and 1024 < r["ls"] < 9000], 60)
        acc = {k: {16: [], 20: []} for k in ("all", "last36")}
        for r in looped:
            P = tok(chat_prompt(tok, probs[r["id"]]), add_special_tokens=False).input_ids
            L, n = r["ls"], len(P)
            h, _ = hidden(P + r["token_ids"][: L + 3000])
            for l in (16, 20):
                acc["all"][l].append(h[l][n + L + 256 :].mean(0) - h[l][n + 256 : n + L].mean(0))
                acc["last36"][l].append(h[l][-36:].mean(0) - h[l][n + L - 36 : n + L].mean(0))
        out["natural_residual_cos"] = {k: {l: cos(np.mean(v[l], 0), u[l]) for l in v} for k, v in acc.items()}
        out["natural_vs_astra_recipe_cos"] = {k: {l: cos(np.mean(v[l], 0), d[l]) for l in v} for k, v in acc.items()}

        # 3. the last 24 positions before every natural stop
        end_id = tok.convert_tokens_to_ids("</think>")
        fin = [r for r in rows if r["finish"] == "stop" and end_id in r["token_ids"] and 64 < r["token_ids"].index(end_id) < 16000]
        approach = []
        for r in fin:
            ids = r["token_ids"]
            stop = ids.index(end_id)
            P = tok(chat_prompt(tok, probs[r["id"]]), add_special_tokens=False).input_ids
            _, last = hidden(P + ids[:stop])
            rec = {"id": r["id"], "sample": r["sample"], "stop": stop, "tail": tok.decode(ids[stop - 40 : stop]), "pos": []}
            for k in range(24, -1, -1):
                p = stop - k
                logits = model.lm_head(last[len(P) + p - 1]).float()
                seen = torch.zeros_like(logits, dtype=torch.bool)
                seen[torch.tensor(ids[:p], device="cuda")] = True
                pen = logits - PEN * seen
                rec["pos"].append({"k": k, "tok": tok.decode(ids[p : p + 1]), "gap": float(logits.max() - logits[end_id]),
                                   "p_end": sampler_prob(logits, end_id), "p_end_pen": sampler_prob(pen, end_id)})
            approach.append(rec)
        out["n_stops"] = len(approach)
        out["gap_median_by_k"] = {k: float(np.median([a["pos"][24 - k]["gap"] for a in approach])) for k in range(24, -1, -1)}
        out["p_end_pen_mean_by_k"] = {k: float(np.mean([a["pos"][24 - k]["p_end_pen"] for a in approach])) for k in range(24, -1, -1)}
    save_json("checks_acts.json", {"summary": out, "approach": approach})
    return out


@app.local_entrypoint()
def main():
    a, g = acts.spawn(), gen.spawn()
    for name, h in (("acts", a), ("gen", g)):
        try:
            print(f"=== {name}\n" + json.dumps(h.get(), indent=1))
        except Exception as e:  # noqa: BLE001
            print(f"=== {name} FAILED: {e!r}")
