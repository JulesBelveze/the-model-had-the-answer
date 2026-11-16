"""Samples that sat in a verbatim loop for >= 4,000 tokens: how many later closed the thinking on their own?
    uv run --with transformers --with modal python escapes.py"""
import gzip, json
from transformers import AutoTokenizer
from fullrun import BASE, REV
END = AutoTokenizer.from_pretrained(BASE, revision=REV).convert_tokens_to_ids("</think>")
N, WIN, DUP, MIN = 12, 400, 0.6, 4000
entered, escaped = 0, []
for l in gzip.open("results/base_samples.jsonl.gz", "rt"):
    r = json.loads(l); ids = r["token_ids"]
    think = ids[: ids.index(END)] if END in ids else ids
    run = best = 0
    for i in range(0, len(think) - WIN, WIN // 2):
        g = [tuple(think[j : j + N]) for j in range(i, i + WIN - N)]
        run = run + WIN // 2 if 1 - len(set(g)) / len(g) >= DUP else 0
        best = max(best, run)
    if best >= MIN:
        entered += 1
        if END in ids:
            escaped.append((r["id"], r["sample"], best, ids.index(END), r["finish"], r["correct"]))
print(f"samples with a verbatim loop of >= {MIN} tokens: {entered}; of those, closed the thinking on their own: {len(escaped)}")
for e in escaped: print("  ", e)
