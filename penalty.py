"""Penalty rung: teacher-force saved base traces and ask, at pre-loop vs loop positions, how far </think> is from
firing, whether presence_penalty 1.5 moves it, and what probability the protocol's sampler (temp .6, top_k 20,
top_p .95) actually gives it. Finished traces give the calibration at the position where the model did stop.

    modal run --detach penalty.py          # L4, ~15 min, saves /vol/full/penalty.json
"""

import gzip
import json
import random

import modal

from fullrun import BASE, OUT, REV, chat_prompt, image, loop_start_tok, vol

app = modal.App("astra-penalty")
img = image.add_local_file("fullrun.py", "/root/fullrun.py")
MAXLEN, NPOS, N_LOOP, N_FIN = 16384, 48, 100, 50  # ponytail: first 16k tokens cover pre-loop and loop for most traces
PEN, TEMP, TOPK, TOPP = 1.5, 0.6, 20, 0.95


def spaced(lo, hi, n):
    return sorted(set(int(lo + (hi - lo) * k / n) for k in range(n))) if hi - lo > n else list(range(lo, hi))


def sampler_prob(logits, tid):
    """vLLM order: penalties already applied; temperature, top_k, top_p, then softmax over what is left."""
    import torch

    z = logits / TEMP
    top_v, top_i = z.topk(TOPK)
    if tid not in top_i:
        return 0.0
    p = torch.softmax(top_v, 0)
    cum = p.cumsum(0)
    keep = cum - p < TOPP  # tokens whose cumulative mass before them is below top_p
    p = p * keep
    j = (top_i == tid).nonzero()[0, 0]
    return float(p[j] / p.sum())


@app.function(image=img, gpu="H100", volumes={"/vol": vol}, timeout=2 * 3600)  # L4 pool had no capacity on 2026-11-10
def run():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    end_id = tok.convert_tokens_to_ids("</think>")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
    from datasets import load_dataset

    probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    rows = [json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt")]
    random.seed(0)
    looped = [r for r in rows if r["finish"] == "length"]
    for r in looped:
        r["ls"] = loop_start_tok(r["token_ids"])
    looped = [r for r in looped if r["ls"] is not None and 512 < r["ls"] < MAXLEN - 1024]
    fin = [r for r in rows if r["finish"] == "stop" and end_id in r["token_ids"] and r["token_ids"].index(end_id) < MAXLEN - 1]
    picks = [(r, "loop") for r in random.sample(looped, N_LOOP)] + [(r, "fin") for r in random.sample(fin, min(N_FIN, len(fin)))]

    out = []
    with torch.inference_mode():
        for r, kind in picks:
            ids = r["token_ids"]
            if kind == "loop":
                groups = {"pre": spaced(256, r["ls"], NPOS), "loop": spaced(r["ls"] + 256, min(len(ids), MAXLEN), NPOS)}
            else:
                stop = ids.index(end_id)
                groups = {"pre_fin": spaced(256, stop, NPOS), "stop": [stop], "last256": spaced(max(256, stop - 256), stop, 16)}
            P = tok(chat_prompt(tok, probs[r["id"]]), add_special_tokens=False).input_ids
            seq = torch.tensor([P + ids[:MAXLEN]], device="cuda")
            h = model.model(input_ids=seq, use_cache=False).last_hidden_state[0]
            for g, poss in groups.items():
                for p in poss:
                    logits = model.lm_head(h[len(P) + p - 1]).float()
                    seen = torch.zeros_like(logits, dtype=torch.bool)
                    seen[torch.tensor(ids[:p], device="cuda")] = True
                    pen = logits - PEN * seen
                    top20 = logits.topk(TOPK).indices
                    out.append({
                        "id": r["id"], "sample": r["sample"], "kind": kind, "group": g, "pos": p,
                        "gap": float(logits.max() - logits[end_id]),
                        "rank": int((logits > logits[end_id]).sum()),
                        "gap_pen": float(pen.max() - pen[end_id]),
                        "rank_pen": int((pen > pen[end_id]).sum()),
                        "seen_top20": float(seen[top20].float().mean()),
                        "p_end": sampler_prob(logits, end_id),
                        "p_end_pen": sampler_prob(pen, end_id),
                        "pen_changes_argmax": bool(logits.argmax() != pen.argmax()),
                    })
    json.dump(out, open(f"{OUT}/penalty.json", "w"))
    vol.commit()
    return summarize(out)


def summarize(out):
    import numpy as np

    s = {}
    for g in ("pre", "loop", "pre_fin", "last256", "stop"):
        rs = [x for x in out if x["group"] == g]
        if not rs:
            continue
        s[g] = {"n": len(rs),
                "gap_median": float(np.median([x["gap"] for x in rs])),
                "gap_pen_median": float(np.median([x["gap_pen"] for x in rs])),
                "rank_median": float(np.median([x["rank"] for x in rs])),
                "rank_pen_median": float(np.median([x["rank_pen"] for x in rs])),
                "seen_top20_mean": float(np.mean([x["seen_top20"] for x in rs])),
                "end_in_top20": float(np.mean([x["rank"] < TOPK for x in rs])),
                "end_in_top20_pen": float(np.mean([x["rank_pen"] < TOPK for x in rs])),
                "p_end_mean": float(np.mean([x["p_end"] for x in rs])),
                "p_end_pen_mean": float(np.mean([x["p_end_pen"] for x in rs])),
                "pen_changes_argmax": float(np.mean([x["pen_changes_argmax"] for x in rs]))}
    return s


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), indent=1))
