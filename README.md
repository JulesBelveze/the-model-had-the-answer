# This model knows the answer. It just can't stop thinking.

Code, traces and per-sample results for the blog post
[This model knows the answer. It just can't stop thinking.](https://julesbelveze.github.io/the-model-had-the-answer/)

The post follows up on MIT CSAIL's
[Sparks of Weight-Space Programming](https://github.com/RikiyaT/lm_understand-logs), where an
agent edited 131,072 weights of Qwen3.5-0.8B and MATH-500 accuracy went from 7.9% to 48.1%. Here
the base model's thinking is ended at 8k tokens and the answer is started for it
(`The final answer is $\boxed{`), which gets 49.8% with no weights changed.

## Check the numbers without a GPU

Everything in `results/` is the raw output of the runs. These scripts only read it:

```sh
uv run --with numpy python analyze_fullrun.py    # base run, forcing at 4k / 8k / 16k, intervals
uv run --with numpy python analyze_answers.py    # answer prefixes, extra penalties, edited checkpoint, Qwen3-1.7B
uv run --with numpy python analyze_lines.py      # closing lines, repeat curve, chances to stop
uv run --with numpy --with transformers --with modal python escapes.py     # loops that ended on their own
uv run --with matplotlib --with numpy --with modal python make_charts.py    # the figures, into figures/
```

Two more need the authors' files (see below):

```sh
uv run --with numpy --with transformers --with modal python paired_edit.py   # forcing vs the edit, answer already in the scratchpad
uv run --with numpy --with requests python recover_direction.py              # the edit as one direction per layer, writes u_astra.npz
```

## Rerun the experiments

The GPU scripts run on [Modal](https://modal.com) and write to a volume called `astra-baseline`.
Copy the outputs back with `modal volume get astra-baseline full results`.

| Script | What it runs | GPU |
|---|---|---|
| `fullrun.py` | Base model on MATH-500, 4 samples per problem, 32k budget. Then forcing at 4k, 8k, 16k and at the loop start. | H100, about 2 hours |
| `checks.py` | Ending the thinking without starting the answer, forcing at 0 to 2k tokens, Astra's direction rebuilt from its eight texts, the 24 tokens before each stop. | H100 |
| `penalty.py` | How far `</think>` is behind the top token at 12,850 positions, with and without the presence penalty. | H100 |
| `line47.py` | The same measurement on the trace that opens the post. | any |
| `answers.py` | Every line start in 267 traces, other answer prefixes, repetition and frequency penalties, the edited checkpoint, Qwen3-1.7B. | H100 |

`linestart.py` and `gold_precut.py` are CPU steps that turn those outputs into
`results/linestart.json` and `results/gold_precut.json`.

The whole thing cost about $12.

## Files from the authors' repo

The grader, the eight direction texts, the released edit and the edited model's per-problem
results belong to the authors, so they are not copied here. This fetches them into the places
the scripts expect:

```sh
sh fetch_upstream.sh
```

## What is in `results/`

| File | Contents |
|---|---|
| `base_samples.jsonl.gz` | The 2,000 base traces: text, token ids, finish reason, correct or not. Every forcing result is a cut of these. |
| `base_scores.json` | Accuracy and budget use per problem for the base run. |
| `forced.json`, `checks_gen.json` | One row per sample for each forcing variant and budget. |
| `gold_precut.json` | Whether the right answer already appears in the first 4k, 8k, 16k thinking tokens. |
| `loop_starts.json` | Token index where each out-of-budget sample starts looping. |
| `penalty.json`, `linestart.json`, `line47.json` | Stop-token gaps and sampling probabilities. |
| `checks_acts.json` | Direction cosines and the gap at each of the 24 positions before a stop. |
| `answers_lines.json` | Every line start in 147 finished and 120 looping traces. |
| `answers_arms.json` | Sampled answers, `Final Answer:` prefix, repetition and frequency penalties. |
| `answers_edited.json` | The authors' checkpoint through this pipeline, with and without forcing. |
| `answers_second.json` | Qwen3-1.7B. |
| `directions*.json`, `smoke_edit_*.json`, `forced_answers.json` | Outputs of the pilot scripts. |

## `pilot/`

The first exploratory scripts, kept as they were run: forcing on the 187 samples the authors'
base run left unfinished, and my own attempts at finding the loop direction. They produced the
feed-forward and probe numbers in the post's last footnote. They need inputs built from the
authors' traces and two base weight tensors, which are not included, and they expect to sit at
the top level of the repo.

## Setup

Protocol for the thinking part, same as the original evaluation: vLLM 0.33.0, official weights
at revision `2fc0636`, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0,
graded with math-verify 0.9.0 through the authors' script. Intervals are bootstrap intervals
over problems.
