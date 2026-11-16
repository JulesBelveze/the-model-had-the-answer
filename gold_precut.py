"""Per sample: was the gold answer already written in the first k thinking tokens?
Two criteria per cut: a \\boxed{} in the prefix that math-verify grades equal to gold ("boxed"),
or the gold string appearing with non-word boundaries ("verbatim"; inflated for short numeric golds).

    uv run --with transformers --with math-verify==0.9.0 --with numpy python gold_precut.py
"""

import gzip
import json
import re
import sys

from transformers import AutoTokenizer

sys.path.insert(0, "their_code")
import mathqa  # noqa: E402

REV = "2fc06364715b967f1860aea9cf38778875588b17"
CUTS = {"4k": 4096, "8k": 8192, "16k": 16384}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=REV)


def boxed_all(txt):
    out, i = [], 0
    while (j := txt.find("\\boxed{", i)) != -1:
        d, k = 1, j + 7
        while k < len(txt) and d:
            d += (txt[k] == "{") - (txt[k] == "}")
            k += 1
        out.append(txt[j + 7 : k - 1])
        i = k
    return set(out)


rows = []
for line in gzip.open("results/base_samples.jsonl.gz", "rt"):
    r = json.loads(line)
    gold, ids = r["answer"], r["token_ids"]
    pat = re.compile(r"(?<![\w.])" + re.escape(gold.strip()) + r"(?![\w.])")
    rec = {"id": r["id"], "sample": r["sample"], "finish": r["finish"], "tokens": len(ids)}
    for c, k in CUTS.items():
        txt = tok.decode(ids[:k])
        rec[f"boxed_{c}"] = any(mathqa.is_correct(b, gold) for b in boxed_all(txt))
        rec[f"verbatim_{c}"] = bool(pat.search(txt))
    rows.append(rec)
json.dump(rows, open("results/gold_precut.json", "w"))
cut = [x for x in rows if x["finish"] == "length"]
for c in CUTS:
    print(f"{c}: truncated samples with gold boxed in prefix {sum(x[f'boxed_{c}'] for x in cut)/len(cut):.1%},"
          f" verbatim {sum(x[f'verbatim_{c}'] for x in cut)/len(cut):.1%}  (n={len(cut)})")
