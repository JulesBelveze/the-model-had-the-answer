"""One forward pass on the opening trace (algebra/2789, sample 0): stop-token gap and sampler probability at every
line start in the 500 tokens before the loop and the first 80 tokens of it.   modal run line47.py"""
import gzip, json
import modal
from fullrun import BASE, OUT, REV, chat_prompt, dataset, image, loop_start_tok, save_json, vol
from penalty import PEN, sampler_prob

app = modal.App("astra-line47")
img = image.add_local_file("fullrun.py", "/root/fullrun.py").add_local_file("penalty.py", "/root/penalty.py")


@app.function(image=img, gpu=["A10G", "L4", "H100"], volumes={"/vol": vol}, timeout=1800)
def run():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    end_id = tok.convert_tokens_to_ids("</think>")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
    r = next(x for x in (json.loads(l) for l in gzip.open(f"{OUT}/base_samples.jsonl.gz", "rt"))
             if x["id"] == "test/algebra/2789.json" and x["sample"] == 0)
    ids, L = r["token_ids"], loop_start_tok(r["token_ids"])
    prob = next(d["problem"] for d in dataset() if d["unique_id"] == r["id"])
    P = tok(chat_prompt(tok, prob), add_special_tokens=False).input_ids
    out = []
    with torch.inference_mode():
        h = model.model(input_ids=torch.tensor([P + ids[: L + 80]], device="cuda"), use_cache=False).last_hidden_state[0]
        start = L - 500
        for p in range(start, L + 80):
            if not tok.decode(ids[p - 1 : p]).endswith("\n"):
                continue
            logits = model.lm_head(h[len(P) + p - 1]).float()
            seen = torch.zeros_like(logits, dtype=torch.bool)
            seen[torch.tensor(ids[:p], device="cuda")] = True
            pen = logits - PEN * seen
            line = tok.decode(ids[max(start, p - 60) : p]).rstrip("\n").split("\n")[-1]
            out.append({"pos": p, "prev_line": line[-90:], "next": tok.decode(ids[p : p + 6]), "gap": float(logits.max() - logits[end_id]),
                        "rank": int((logits > logits[end_id]).sum()), "p_end": sampler_prob(logits, end_id), "p_end_pen": sampler_prob(pen, end_id)})
    save_json("line47.json", {"loop_start": L, "rows": out})
    return out


@app.local_entrypoint()
def main():
    for x in run.remote():
        print(f"{x['pos']:5d} gap {x['gap']:5.1f} rank {x['rank']:5d} p {x['p_end_pen']:.3f} | {x['prev_line']!r} -> {x['next']!r}")
