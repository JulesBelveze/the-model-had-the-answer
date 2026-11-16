"""Phase 2: base Qwen3.5-0.8B on MATH-500 x 4 samples (the post's protocol), all samples saved,
then forced commit ("</think>\n\nThe final answer is $\\boxed{") at 4k / 8k / 16k / loop-start tokens.

    modal run fullrun.py --stage gen      # ~75-90 min H100, saves /vol/full/base_samples.jsonl.gz
    modal run fullrun.py --stage force    # ~15 min, saves /vol/full/forced.json
"""

import gzip
import json

import modal

app = modal.App("astra-fullrun")
vol = modal.Volume.from_name("astra-baseline", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.33.0", "transformers==5.20.2", "math-verify==0.9.0", "datasets", "numpy", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("their_code", "/root/their_code")
)

BASE, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
THINK35 = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=0)
OUT = "/vol/full"
FORCED_TXT = "The final answer is $\\boxed{"


def grader():
    import sys

    sys.path.insert(0, "/root/their_code")
    import mathqa

    return mathqa


def dataset():
    from datasets import load_dataset

    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def chat_prompt(tok, problem):
    msgs = [{"role": "user", "content": problem + "\n" + grader().INSTRUCTION}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def loop_start_tok(ids, n=12, win=400, dup=0.6):
    """First token index from which the trace is a verbatim loop to the end, else None."""
    if len(ids) < win:
        return None

    def looping(i):
        grams = [tuple(ids[j : j + n]) for j in range(i, min(i + win, len(ids)) - n)]
        return bool(grams) and 1 - len(set(grams)) / len(grams) >= dup

    starts = list(range(0, len(ids) - win, win // 2))
    flags = [looping(i) for i in starts]
    for k, i in enumerate(starts):
        if flags[k] and all(flags[k:]):
            return i
    return None


def make_llm():
    from vllm import LLM

    return LLM(model=BASE, revision=REV, dtype="bfloat16", max_model_len=34816, gpu_memory_utilization=0.92,
               enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)


def save_json(name, obj):
    import os

    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(f"{OUT}/{name}", "w"))
    vol.commit()


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=4 * 3600)
def gen():
    import os

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    Q, ds = grader(), dataset()
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    prompts = [chat_prompt(tok, r["problem"]) for r in ds]
    llm = make_llm()
    outs = llm.generate(prompts, SamplingParams(n=4, max_tokens=32768, **THINK35))
    os.makedirs(OUT, exist_ok=True)
    per, n_ok = [], 0
    with gzip.open(f"{OUT}/base_samples.jsonl.gz", "wt") as f:
        for r, o in zip(ds, outs):
            cs = []
            for k, s in enumerate(o.outputs):
                pred, _ = Q.extract(s.text)
                c = bool(Q.is_correct(pred, r["answer"]))
                cs.append(int(c))
                f.write(json.dumps({"id": r["unique_id"], "answer": r["answer"], "sample": k, "token_ids": list(s.token_ids),
                                    "text": s.text, "finish": s.finish_reason, "correct": c}) + "\n")
            per.append({"id": r["unique_id"], "correct": cs, "cut": [s.finish_reason == "length" for s in o.outputs],
                        "tokens": [len(s.token_ids) for s in o.outputs]})
    acc = sum(sum(p["correct"]) for p in per) / (4 * len(per))
    cut = sum(sum(p["cut"]) for p in per) / (4 * len(per))
    summary = {"acc": acc, "cut_share": cut, "mean_tokens": sum(sum(p["tokens"]) for p in per) / (4 * len(per)),
               "protocol": {"n": 4, "max_tokens": 32768, **THINK35}, "vllm": "0.33.0", "per_problem": per}
    save_json("base_scores.json", summary)
    return {k: v for k, v in summary.items() if k != "per_problem"}


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=2 * 3600)
def force():
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    Q, ds = grader(), dataset()
    probs = {r["unique_id"]: r["problem"] for r in ds}
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    close = tok("\n</think>\n\n", add_special_tokens=False).input_ids
    forced = tok(FORCED_TXT, add_special_tokens=False).input_ids
    rows = [json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt")]
    pcache = {}
    reqs, params, meta, passthrough = [], [], [], []
    for r in rows:
        ids = r["token_ids"]
        if r["id"] not in pcache:
            pcache[r["id"]] = tok(chat_prompt(tok, probs[r["id"]]), add_special_tokens=False).input_ids
        p = pcache[r["id"]]
        ls = loop_start_tok(ids)
        cuts = {"4k": 4096, "8k": 8192, "16k": 16384, "loop": ls if ls is not None else len(ids) + 1}
        for cname, c in cuts.items():
            variants = [("forced", forced, SamplingParams(max_tokens=32, temperature=0))]
            if cname == "8k":
                variants.append(("free", [], SamplingParams(max_tokens=512, temperature=0)))
            for vname, extra, sp in variants:
                if r["finish"] == "stop" and len(ids) <= c:  # finished before the budget: untouched
                    passthrough.append({"id": r["id"], "sample": r["sample"], "cut": cname, "variant": vname,
                                        "applied": False, "correct": r["correct"], "pred": None, "gen_tokens": 0})
                    continue
                reqs.append(TokensPrompt(prompt_token_ids=p + ids[:c] + close + extra))
                params.append(sp)
                meta.append((r["id"], r["answer"], r["sample"], cname, vname, c))
    llm = make_llm()
    outs = llm.generate(reqs, params)
    res = list(passthrough)
    for (pid, gold, k, cname, vname, c), o in zip(meta, outs):
        gen_txt = o.outputs[0].text
        pred, _ = Q.extract(FORCED_TXT + gen_txt if vname == "forced" else gen_txt)
        res.append({"id": pid, "sample": k, "cut": cname, "variant": vname, "applied": True, "cut_tok": c,
                    "pred": pred, "correct": bool(Q.is_correct(pred, gold)), "finish": o.outputs[0].finish_reason,
                    "gen_tokens": len(o.outputs[0].token_ids), "gen": gen_txt[:200]})
    summary = {}
    for cname in ("4k", "8k", "16k", "loop"):
        for vname in ("forced", "free"):
            g = [x for x in res if x["cut"] == cname and x["variant"] == vname]
            if g:
                summary[f"{cname}/{vname}"] = {"n": len(g), "acc": sum(x["correct"] for x in g) / len(g),
                                               "applied": sum(x["applied"] for x in g)}
    save_json("forced.json", {"summary": summary, "rows": res})
    return summary


@app.local_entrypoint()
def main(stage: str = "gen"):
    if "gen" in stage:
        print("=== gen\n" + json.dumps(gen.remote(), indent=1))
    if "force" in stage:
        print("=== force\n" + json.dumps(force.remote(), indent=1))
