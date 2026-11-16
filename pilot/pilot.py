"""Phase 1a pilot on Astra's 187 truncated MATH-500 traces.

A. forced_answers: stop the thinking at the loop start / 8k / 16k tokens and force an answer.
B. directions:     diff-of-means loop direction at layers 15/19, compared with Astra's u.
C. smoke_edit:     apply our direction like Astra did, 25 problems, cut rate vs base / Astra.

    modal run pilot.py --stage AB     # A and B in parallel
    modal run pilot.py --stage C      # needs B's output on the volume
"""

import json

import modal

app = modal.App("astra-pilot")
vol = modal.Volume.from_name("astra-baseline", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.33.0", "transformers==5.20.2", "math-verify==0.9.0", "datasets", "numpy", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("their_code", "/root/their_code")
    .add_local_file("data/pilot_inputs.json", "/root/pilot_inputs.json")
    .add_local_file("u_astra.npz", "/root/u_astra.npz")
)

BASE, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
ASTRA = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
THINK35 = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=0)
OUT = "/vol/pilot"
LAYERS = (15, 19)


def grader():
    import sys

    sys.path.insert(0, "/root/their_code")
    import mathqa

    return mathqa


def problems():
    from datasets import load_dataset

    return {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}


def prompt_ids(tok, problem):
    msgs = [{"role": "user", "content": problem + "\n" + grader().INSTRUCTION}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return tok(s, add_special_tokens=False).input_ids


def loop_tok(enc, char):
    return next((i for i, (a, _) in enumerate(enc.offset_mapping) if a >= char), None)


def save(name, obj):
    import os

    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(f"{OUT}/{name}", "w"), indent=1)
    vol.commit()


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, timeout=3600)
def forced_answers():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    Q, probs = grader(), problems()
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    rows = json.load(open("/root/pilot_inputs.json"))
    close = tok("\n</think>\n\n", add_special_tokens=False).input_ids
    forced_txt = "The final answer is $\\boxed{"
    forced = tok(forced_txt, add_special_tokens=False).input_ids
    variants = {
        "free_sampled": ([], SamplingParams(max_tokens=512, **THINK35)),
        "free_greedy": ([], SamplingParams(max_tokens=512, temperature=0)),
        "forced_greedy": (forced, SamplingParams(max_tokens=32, temperature=0)),
    }
    reqs, params, meta = [], [], []
    for r in rows:
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        cuts = {"8k": 8192, "16k": 16384}
        if r["loop_start_char"] is not None:
            cuts["loop"] = loop_tok(enc, r["loop_start_char"])
        p = prompt_ids(tok, probs[r["id"]])
        for cname, c in cuts.items():
            c = min(c, len(enc.input_ids))
            for vname, (extra, sp) in variants.items():
                reqs.append(TokensPrompt(prompt_token_ids=p + enc.input_ids[:c] + close + extra))
                params.append(sp)
                meta.append((r["id"], r["answer"], cname, vname, c))

    llm = LLM(model=BASE, revision=REV, dtype="bfloat16", max_model_len=34816, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    outs = llm.generate(reqs, params)
    res = []
    for (pid, gold, cname, vname, c), o in zip(meta, outs):
        gen = o.outputs[0].text
        pred, _ = Q.extract(forced_txt + gen if vname == "forced_greedy" else gen)
        res.append({"id": pid, "cut": cname, "variant": vname, "cut_tok": c, "pred": pred,
                    "correct": bool(Q.is_correct(pred, gold)), "finish": o.outputs[0].finish_reason,
                    "gen_tokens": len(o.outputs[0].token_ids), "gen": gen[:400]})
    summary = {}
    for cname in ("loop", "8k", "16k"):
        for vname in variants:
            g = [x for x in res if x["cut"] == cname and x["variant"] == vname]
            summary[f"{cname}/{vname}"] = {"n": len(g), "correct": sum(x["correct"] for x in g),
                                           "answered": sum(x["pred"] is not None for x in g),
                                           "hit_limit": sum(x["finish"] == "length" for x in g)}
    save("forced_answers.json", {"summary": summary, "rows": res})
    return summary


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, timeout=3600)
def directions():
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
    cols_astra = {L: set(Z[f"cols{L}"].tolist()) for L in LAYERS}
    uW = {L: U[L] @ mlps[L].down_proj.weight.float() for L in LAYERS}  # [3584] static write of u per neuron

    probs = problems()
    rows = [r for r in json.load(open("/root/pilot_inputs.json")) if r["loop_start_char"] is not None]
    SPAN, MAX_LS = 2048, 10000
    keys = ("pre", "loop", "pre_early", "pre_late", "pre_last512", "loop_first512")
    S = {L: {k: [torch.zeros(1024, dtype=torch.float64, device="cuda"), 0] for k in keys} for L in LAYERS}
    writer = {L: {"loop": torch.zeros(3584, dtype=torch.float64, device="cuda"),
                  "pre": torch.zeros(3584, dtype=torch.float64, device="cuda")} for L in LAYERS}
    per_trace = {L: {"cos": [], "proj_pre": [], "proj_loop": []} for L in LAYERS}
    act = {L: {k: torch.zeros(3584, dtype=torch.float64, device="cuda") for k in ("pre", "loop")} for L in LAYERS}
    used = skipped = 0
    cos = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))  # noqa: E731
    for r in rows:
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        ls = loop_tok(enc, r["loop_start_char"])
        if ls is None or ls < 256 or ls > MAX_LS:
            skipped += 1
            continue
        p = prompt_ids(tok, probs[r["id"]])
        P = len(p)
        ids = torch.tensor([p + enc.input_ids[: ls + SPAN]], device="cuda")
        with torch.no_grad():
            model(input_ids=ids)
        pre, loop = slice(P + 128, P + ls), slice(P + ls, P + ls + SPAN)
        mid = (pre.start + pre.stop) // 2
        spans = {"pre": pre, "loop": loop, "pre_early": slice(pre.start, mid), "pre_late": slice(mid, pre.stop),
                 "pre_last512": slice(max(pre.start, pre.stop - 512), pre.stop), "loop_first512": slice(loop.start, loop.start + 512)}
        for L in LAYERS:
            o, inp = cap[("out", L)].float(), cap[("in", L)].float()
            for k, sl in spans.items():
                S[L][k][0] += o[sl].sum(0).double()
                S[L][k][1] += sl.stop - sl.start
            d = o[loop].mean(0) - o[pre].mean(0)
            per_trace[L]["cos"].append(cos(d, U[L]))
            per_trace[L]["proj_pre"].append(float((o[pre] @ U[L]).mean()))
            per_trace[L]["proj_loop"].append(float((o[loop] @ U[L]).mean()))
            writer[L]["loop"] += (inp[loop].mean(0) * uW[L]).double()
            writer[L]["pre"] += (inp[pre].mean(0) * uW[L]).double()
            act[L]["loop"] += inp[loop].mean(0).double()
            act[L]["pre"] += inp[pre].mean(0).double()
        used += 1
        cap.clear()

    out = {"traces_used": used, "traces_skipped": skipped, "span_tokens": SPAN, "u_ours": {}, "cols_ours": {}}
    mean = lambda L, k: S[L][k][0] / S[L][k][1]  # noqa: E731
    d_ours = {L: (mean(L, "loop") - mean(L, "pre")).float() for L in LAYERS}
    for L in LAYERS:
        d_pos = (mean(L, "pre_late") - mean(L, "pre_early")).float()
        w_loop, w_pre = writer[L]["loop"] / used, writer[L]["pre"] / used
        top64_loop = set(torch.topk(w_loop.abs(), 64).indices.tolist())
        top64_delta = set(torch.topk((w_loop - w_pre).abs(), 64).indices.tolist())
        c = per_trace[L]["cos"]
        out[str(L)] = {
            "cos_ours_vs_astra": cos(d_ours[L], U[L]),
            "cos_position_control_vs_astra": cos(d_pos, U[L]),
            "cos_position_control_vs_ours": cos(d_pos, d_ours[L]),
            "per_trace_cos_median": float(np.median(c)), "per_trace_cos_frac_above_0.3": float(np.mean(np.array(c) > 0.3)),
            "proj_on_u_astra_pre_mean": float(np.mean(per_trace[L]["proj_pre"])),
            "proj_on_u_astra_loop_mean": float(np.mean(per_trace[L]["proj_loop"])),
            "astra_cols_in_top64_loop_writers": len(cols_astra[L] & top64_loop),
            "astra_cols_in_top64_delta_writers": len(cols_astra[L] & top64_delta),
        }
        out["u_ours"][str(L)] = (d_ours[L] / d_ours[L].norm()).cpu().tolist()
        out["cols_ours"][str(L)] = sorted(top64_loop)
    out["cos_ours15_vs_ours19"] = cos(d_ours[15], d_ours[19])
    out["means"] = {str(L): {k: mean(L, k).cpu().tolist() for k in keys} for L in LAYERS}
    out["act_means"] = {str(L): {k: (act[L][k] / used).cpu().tolist() for k in ("pre", "loop")} for L in LAYERS}
    out["per_trace"] = {str(L): per_trace[L] for L in LAYERS}
    out["random_cos_expectation"] = 1 / 1024**0.5
    save("directions_v2.json", out)
    return {k: v for k, v in out.items() if k not in ("u_ours", "cols_ours", "means", "act_means", "per_trace")}


@app.function(image=image, gpu="A10G", volumes={"/vol": vol}, timeout=5400)
def smoke_edit(edit: dict, n_problems: int = 25, strengths=(2.0, 6.0), refs: bool = True):
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    D = edit
    src = snapshot_download(BASE, revision=REV)
    shard = "model.safetensors-00001-of-00001.safetensors"
    models = {"base": BASE, "astra": ASTRA} if refs else {}
    for a in strengths:
        dst = f"{OUT}/ours_a{a:g}"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        sd = load_file(f"{dst}/{shard}")
        for L in LAYERS:
            name = f"model.language_model.layers.{L}.mlp.down_proj.weight"
            W = sd[name].float()
            u = torch.tensor(D["u_ours"][str(L)])
            cols = torch.tensor(D["cols_ours"][str(L)])
            W[:, cols] -= a * torch.outer(u, u @ W[:, cols])
            sd[name] = W.to(torch.bfloat16)
        save_file(sd, f"{dst}/{shard}", metadata={"format": "pt"})
        models[f"ours_a{a:g}"] = dst

    Q, probs = grader(), problems()
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(n_problems))
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    reqs = [TokensPrompt(prompt_token_ids=prompt_ids(tok, r["problem"])) for r in ds]
    sp = SamplingParams(max_tokens=32768, **THINK35)
    res = {}
    for name, path in models.items():
        llm = LLM(model=path, dtype="bfloat16", max_model_len=34816, gpu_memory_utilization=0.9,
                  limit_mm_per_prompt={"image": 0, "video": 0}, seed=0,
                  **({"revision": REV} if path == BASE else {}))
        outs = llm.generate(reqs, sp)
        rows = []
        for r, o in zip(ds, outs):
            pred, _ = Q.extract(o.outputs[0].text)
            rows.append({"id": r["unique_id"], "correct": bool(Q.is_correct(pred, r["answer"])),
                         "cut": o.outputs[0].finish_reason == "length", "tokens": len(o.outputs[0].token_ids)})
        res[name] = {"acc": sum(x["correct"] for x in rows) / len(rows), "cut_share": sum(x["cut"] for x in rows) / len(rows),
                     "mean_tokens": sum(x["tokens"] for x in rows) / len(rows), "rows": rows}
        del llm
        torch.cuda.empty_cache()
    save(f"smoke_edit_{edit.get('tag', 'x')}.json", res)
    return {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res.items()}


@app.local_entrypoint()
def main(stage: str = "AB", edit: str = "our_edit_naive.json", strengths: str = "2,6", refs: bool = True):
    calls = {}
    if "A" in stage:
        calls["A"] = forced_answers.spawn()
    if "B" in stage:
        calls["B"] = directions.spawn()
    if "C" in stage:
        st = tuple(float(x) for x in strengths.split(","))
        for f in edit.split(","):
            calls[f"C:{f}"] = smoke_edit.spawn(json.load(open(f)), strengths=st, refs=refs)
    for k, c in calls.items():
        try:
            print(f"=== {k}\n{json.dumps(c.get(), indent=1)}")
        except Exception as e:  # keep the other jobs alive
            print(f"=== {k} FAILED\n{str(e)[:3000]}")
