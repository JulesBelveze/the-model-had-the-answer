"""Experiments that close the questions the post leaves open.

    modal run --detach answers.py

lines():   every line start in 147 finished and 120 looping traces: stop-token gap, sampler probability, the line
           before it; per-line mean log-prob with how many times the same line already appeared (repeat curve);
           Astra's eight texts read at the MLP output (the missing site/contrast cell).
arms():    at 8k: forced box with the protocol's sampling; "Final Answer:" prefix, sampled and greedy.
           Full 32k runs on 100 problems x 2 with a repetition penalty and a frequency penalty added.
edited():  the released edited checkpoint through this pipeline (500 x 4, 32k), then forced commit at 8k on top.
second():  a second thinking model (Qwen3-1.7B): 300 problems, then the same stop-token scan.
"""

import gzip
import json
import random

import modal

from fullrun import BASE, FORCED_TXT, OUT, REV, THINK35, chat_prompt, dataset, grader, image, loop_start_tok, save_json, vol
from penalty import PEN, sampler_prob

app = modal.App("astra-answers")
img = (image.add_local_file("fullrun.py", "/root/fullrun.py").add_local_file("penalty.py", "/root/penalty.py")
       .add_local_file("u_astra.npz", "/root/u_astra.npz").add_local_file("data/fwd_4.json", "/root/fwd_4.json"))
ASTRA = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
SECOND = "Qwen/Qwen3-1.7B"
GPU = dict(image=img, gpu="H100", volumes={"/vol": vol})


def scan(model, tok, P, ids, end_id, upto):
    """Teacher-force P + ids[:upto]; return one row per line start and one per line (mean log-prob, repeat count)."""
    import torch

    h = model.model(input_ids=torch.tensor([P + ids[:upto]], device="cuda"), use_cache=False).last_hidden_state[0]
    is_nl = [tok.decode([t]).endswith("\n") for t in ids[:upto]]
    starts = [p for p in range(1, upto) if is_nl[p - 1]] + ([upto] if upto < len(ids) and ids[upto] == end_id and is_nl[upto - 1] else [])
    lp_tok = torch.empty(upto)
    keep = {}
    for a in range(0, upto, 512):
        b = min(a + 512, upto)
        logits = model.lm_head(h[len(P) + a - 1 : len(P) + b - 1]).float()
        lp_tok[a:b] = torch.log_softmax(logits, -1).gather(1, torch.tensor(ids[a:b], device="cuda")[:, None])[:, 0].cpu()
        for p in starts:
            if a <= p < b:
                keep[p] = logits[p - a].clone()
    if upto in starts:
        keep[upto] = model.lm_head(h[len(P) + upto - 1]).float()
    seen = torch.zeros(model.lm_head.weight.shape[0], dtype=torch.bool, device="cuda")
    rows, last, counts, prev_start = [], 0, {}, 0
    for p in starts:
        seen[torch.tensor(ids[last:p], device="cuda")] = True
        last = p
        lg = keep[p]
        line = tok.decode(ids[prev_start:p])
        n_prev = counts.get(line, 0)
        counts[line] = n_prev + 1
        rows.append({"p": p, "prev_line": line[-110:], "n_prev": n_prev, "line_lp": float(lp_tok[prev_start:p].mean()) if p > prev_start else None,
                     "line_len": p - prev_start, "gap": float(lg.max() - lg[end_id]), "p_end_pen": sampler_prob(lg - PEN * seen, end_id),
                     "is_stop": p < len(ids) and ids[p] == end_id})
        prev_start = p
    return rows


@app.function(timeout=3 * 3600, **GPU)
def lines():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    end_id = tok.convert_tokens_to_ids("</think>")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
    probs = {r["unique_id"]: r["problem"] for r in dataset()}
    rows = [json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt")]
    random.seed(1)
    fin = [r for r in rows if r["finish"] == "stop" and end_id in r["token_ids"] and 64 < r["token_ids"].index(end_id) < 16000]
    looped = [r for r in rows if r["finish"] == "length"]
    for r in looped:
        r["ls"] = loop_start_tok(r["token_ids"])
    looped = random.sample([r for r in looped if r["ls"] and 1024 < r["ls"] < 10000], 120)
    out = {"traces": []}
    with torch.inference_mode():
        for kind, rs in (("fin", fin), ("loop", looped)):
            for r in rs:
                ids = r["token_ids"]
                upto = ids.index(end_id) if kind == "fin" else r["ls"] + 2000
                P = tok(chat_prompt(tok, probs[r["id"]]), add_special_tokens=False).input_ids
                out["traces"].append({"id": r["id"], "sample": r["sample"], "kind": kind, "ls": r.get("ls"), "rows": scan(model, tok, P, ids, end_id, upto)})
        # Astra's eight texts at the MLP output of layers 15 and 19
        Z = np.load("/root/u_astra.npz")
        grab = {}
        hooks = [model.model.language_model.layers[L].mlp.register_forward_hook(lambda m, i, o, L=L: grab.__setitem__(L, o[0].float().cpu().numpy()))
                 for L in (15, 19)]
        means = {15: [], 19: []}
        for t in json.load(open("/root/fwd_4.json"))["texts"]:
            model.model(input_ids=torch.tensor([tok(t, add_special_tokens=False).input_ids], device="cuda"), use_cache=False)
            for L in means:
                means[L].append(grab[L][-36:].mean(0))
        for hk in hooks:
            hk.remove()
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
        out["astra_texts_mlp_output_cos"] = {L: cos((np.stack(m)[1::2] - np.stack(m)[::2]).mean(0), Z[f"u{L}"]) for L, m in means.items()}
    save_json("answers_lines.json", out)
    return {"n_traces": len(out["traces"]), "astra_texts_mlp_output_cos": out["astra_texts_mlp_output_cos"]}


def lenient(Q, text, gold):
    """Strict box if present, else the first line after 'Final Answer:' stripped of $ and trailing dots."""
    pred, _ = Q.extract(text)
    if pred:
        return bool(Q.is_correct(pred, gold)), True
    first = text.strip().split("\n")[0].strip().strip("$").rstrip(".").strip().strip("$")
    try:
        return bool(first) and bool(Q.is_correct(first, gold)), False
    except Exception:  # noqa: BLE001
        return False, False


@app.function(timeout=4 * 3600, **GPU)
def arms():
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from fullrun import make_llm

    Q, ds = grader(), dataset()
    probs = {r["unique_id"]: r["problem"] for r in ds}
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    close, forced, fa = enc("\n</think>\n\n"), enc(FORCED_TXT), enc("Final Answer:")
    rows = [json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt")]
    cut_arms = [("box_sampled@8k", forced, SamplingParams(max_tokens=32, **THINK35)),
                ("final_answer_sampled@8k", fa, SamplingParams(max_tokens=1024, **THINK35)),
                ("final_answer_greedy@8k", fa, SamplingParams(max_tokens=1024, temperature=0))]
    reqs, params, meta, res = [], [], [], []
    for r in rows:
        p = enc(chat_prompt(tok, probs[r["id"]]))
        for name, extra, sp in cut_arms:
            if r["finish"] == "stop" and len(r["token_ids"]) <= 8192:
                res.append({"id": r["id"], "arm": name, "applied": False, "correct": r["correct"], "boxed": True})
                continue
            reqs.append(TokensPrompt(prompt_token_ids=p + r["token_ids"][:8192] + close + extra))
            params.append(sp)
            meta.append((r["id"], r["answer"], name))
    random.seed(2)
    sub = random.sample(list(ds), 100)
    pen_arms = {"repetition_1.1": dict(repetition_penalty=1.1), "frequency_0.2": dict(frequency_penalty=0.2)}
    for name, extra in pen_arms.items():
        for r in sub:
            reqs.append(chat_prompt(tok, r["problem"]))
            params.append(SamplingParams(n=2, max_tokens=32768, **THINK35, **extra))
            meta.append((r["unique_id"], r["answer"], name))
    outs = make_llm().generate(reqs, params)
    for (pid, gold, name), o in zip(meta, outs):
        for s in o.outputs:
            if name in pen_arms:
                pred, _ = Q.extract(s.text)
                res.append({"id": pid, "arm": name, "applied": True, "correct": bool(Q.is_correct(pred, gold)), "cut": s.finish_reason == "length",
                            "tokens": len(s.token_ids), "tail": s.text[-200:]})
            else:
                ok, boxed = lenient(Q, (FORCED_TXT if name.startswith("box") else "") + s.text, gold)
                res.append({"id": pid, "arm": name, "applied": True, "correct": ok, "boxed": boxed, "finish": s.finish_reason, "gen": s.text[:160]})
    summary = {}
    for name in [a[0] for a in cut_arms] + list(pen_arms):
        g = [x for x in res if x["arm"] == name]
        summary[name] = {"n": len(g), "acc": sum(x["correct"] for x in g) / len(g)}
        if name in pen_arms:
            summary[name] |= {"cut_share": sum(x["cut"] for x in g) / len(g), "mean_tokens": sum(x["tokens"] for x in g) / len(g)}
        else:
            summary[name]["boxed"] = sum(x["boxed"] for x in g) / len(g)
    summary["penalty_subset_ids"] = [r["unique_id"] for r in sub]
    save_json("answers_arms.json", {"summary": summary, "rows": res})
    return {k: v for k, v in summary.items() if k != "penalty_subset_ids"}


@app.function(timeout=5 * 3600, **GPU)
def edited():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    Q, ds = grader(), dataset()
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    prompts = [chat_prompt(tok, r["problem"]) for r in ds]
    llm = LLM(model=ASTRA, dtype="bfloat16", max_model_len=34816, gpu_memory_utilization=0.92, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    outs = llm.generate(prompts, SamplingParams(n=4, max_tokens=32768, **THINK35))
    end_id = tok.convert_tokens_to_ids("</think>")
    close, forced = tok("\n</think>\n\n", add_special_tokens=False).input_ids, tok(FORCED_TXT, add_special_tokens=False).input_ids
    per, reqs, meta = [], [], []
    for r, pr, o in zip(ds, prompts, outs):
        rec = {"id": r["unique_id"], "correct": [], "cut": [], "tokens": [], "boxed": [], "think_tokens": [], "forced8k": [None] * 4}
        p = tok(pr, add_special_tokens=False).input_ids
        for k, s in enumerate(o.outputs):
            ids = list(s.token_ids)
            pred, _ = Q.extract(s.text)
            c = bool(Q.is_correct(pred, r["answer"]))
            rec["correct"].append(int(c)); rec["cut"].append(s.finish_reason == "length"); rec["tokens"].append(len(ids))
            rec["boxed"].append(bool(pred)); rec["think_tokens"].append(ids.index(end_id) if end_id in ids else None)
            if s.finish_reason == "stop" and len(ids) <= 8192:
                rec["forced8k"][k] = int(c)
            else:
                reqs.append(TokensPrompt(prompt_token_ids=p + ids[:8192] + close + forced))
                meta.append((len(per), k, r["answer"]))
        per.append(rec)
    fo = llm.generate(reqs, SamplingParams(max_tokens=32, temperature=0))
    for (i, k, gold), o in zip(meta, fo):
        pred, _ = Q.extract(FORCED_TXT + o.outputs[0].text)
        per[i]["forced8k"][k] = int(bool(Q.is_correct(pred, gold)))
    n = 4 * len(per)
    tot = lambda key: sum(sum(bool(v) for v in p[key]) for p in per) / n  # noqa: E731
    summary = {"acc": tot("correct"), "cut_share": tot("cut"), "boxed": tot("boxed"), "forced8k_acc": tot("forced8k"),
               "mean_tokens": sum(sum(p["tokens"]) for p in per) / n, "forced_applied": len(reqs) / n}
    save_json("answers_edited.json", {"summary": summary, "per_problem": per})
    return summary


@app.function(timeout=3 * 3600, **GPU)
def second():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    Q = grader()
    ds = list(dataset())[:300]
    tok = AutoTokenizer.from_pretrained(SECOND)
    end_id = tok.convert_tokens_to_ids("</think>")
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + Q.INSTRUCTION}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for r in ds]
    llm = LLM(model=SECOND, dtype="bfloat16", max_model_len=18432, gpu_memory_utilization=0.6, seed=0)
    outs = llm.generate(prompts, SamplingParams(max_tokens=16384, temperature=0.6, top_p=0.95, top_k=20, seed=0))
    gens = []
    for r, o in zip(ds, outs):
        s = o.outputs[0]
        pred, _ = Q.extract(s.text)
        gens.append({"id": r["unique_id"], "ids": list(s.token_ids), "finish": s.finish_reason, "correct": bool(Q.is_correct(pred, r["answer"]))})
    del llm
    import gc

    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(SECOND, dtype=torch.bfloat16).to("cuda").eval()
    traces = []
    with torch.inference_mode():
        for g, pr in zip(gens, prompts):
            ids = g["ids"]
            if end_id not in ids or not 64 < ids.index(end_id) < 12000:
                continue
            P = tok(pr, add_special_tokens=False).input_ids
            stop = ids.index(end_id)
            rows = scan(model, tok, P, ids, end_id, stop)
            h = model.model(input_ids=torch.tensor([P + ids[:stop]], device="cuda"), use_cache=False).last_hidden_state[0]
            lg = model.lm_head(h[len(P) + stop - 25 :]).float()
            traces.append({"id": g["id"], "rows": rows, "gap_by_k": {k: float(lg[24 - k].max() - lg[24 - k][end_id]) for k in range(24, -1, -1)},
                           "tail": tok.decode(ids[stop - 40 : stop])})
    summary = {"n": len(gens), "acc": sum(g["correct"] for g in gens) / len(gens), "cut_share": sum(g["finish"] == "length" for g in gens) / len(gens),
               "closed_thinking": sum(end_id in g["ids"] for g in gens) / len(gens), "n_scanned": len(traces),
               "mean_tokens": sum(len(g["ids"]) for g in gens) / len(gens)}
    save_json("answers_second.json", {"summary": summary, "traces": traces})
    return summary


@app.local_entrypoint()
def main(only: str = ""):
    fns = {"lines": lines, "arms": arms, "edited": edited, "second": second}
    calls = {k: f.spawn() for k, f in fns.items() if not only or k in only.split(",")}
    for k, c in calls.items():
        try:
            print(f"=== {k}\n" + json.dumps(c.get(), indent=1))
        except Exception as e:  # noqa: BLE001
            print(f"=== {k} FAILED: {str(e)[:2000]}")
