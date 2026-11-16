"""Figures for the post, written to figures/ (or to the directory given as the first argument).

    uv run --with matplotlib --with numpy --with modal python make_charts.py [outdir]
"""

import gzip
import json
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
OUT.mkdir(parents=True, exist_ok=True)
for f in (HERE / "fonts").glob("*.ttf"):  # optional: drop Poppins TTFs in fonts/ to match the post
    font_manager.fontManager.addfont(f)

INK, TEXT, MUTED = "#1d1b26", "#55535f", "#8b8996"
CARD, BORDER, PANEL, GRID = "#ffffff", "#e7e7eb", "#f8f8f9", "#ececef"
LINE, DOT, BAR, ACCENT = "#45434f", "#77758a", "#c9c8d1", "#c43a00"
matplotlib.rcParams.update({
    "font.family": ["Poppins", "Helvetica Neue", "DejaVu Sans"], "font.size": 10.5, "svg.fonttype": "path",
    "axes.facecolor": "none", "axes.edgecolor": "none", "axes.labelcolor": TEXT, "axes.labelpad": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 10, "ytick.major.pad": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "axes.axisbelow": True,
})


def frame(title, sub, note, figsize=(8, 5.2), left=0.135):
    """White rounded card, title block top left, grey rounded plot panel, axis note bottom right."""
    fig, ax = plt.subplots(figsize=figsize)
    w, h = figsize
    fig.patch.set_alpha(0)
    fig.add_artist(FancyBboxPatch((0.012, 0.012 * w / h), 0.976, 1 - 0.024 * w / h, boxstyle="round,pad=0,rounding_size=0.022",
                                  mutation_aspect=w / h, fc=CARD, ec=BORDER, lw=1, transform=fig.transFigure, zorder=-10))
    fig.subplots_adjust(left=left, right=0.945, top=1 - 1.25 / h, bottom=0.95 / h)
    fig.text(0.055, 1 - 0.50 / h, title, fontsize=16, fontweight="medium", color=INK, va="center")
    fig.text(0.055, 1 - 0.82 / h, sub, fontsize=10.5, color=MUTED, va="center")
    fig.text(0.945, 0.36 / h, note, fontsize=10.5, color=TEXT, va="center", ha="right")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", mutation_aspect=(w * 0.835) / (h - 2.2),
                                fc=PANEL, ec="none", transform=ax.transAxes, zorder=-5, clip_on=False))
    return fig, ax


def halo(ax, x, y, s=70):
    ax.scatter([x], [y], s=s * 4.5, color=ACCENT, alpha=0.15, lw=0, zorder=5)
    ax.scatter([x], [y], s=s, color=ACCENT, ec=CARD, lw=1.5, zorder=6)


def pct(ax, ticks, axis="y"):
    getattr(ax, f"set_{axis}ticks")(ticks, [f"{t}%" for t in ticks])


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", transparent=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, transparent=True)
    plt.close(fig)
    print("wrote", name)


# ---- 1. the ladder (cover)
rows = [  # label, value, color
    ("Base model", 8.2, BAR),
    ("Thinking stopped at 8k tokens", 36.7, BAR),
    ("Stopped at 8k, and the\nanswer started for it", 49.8, ACCENT),
    ("Astra's weight edit", 51.8, LINE),  # their checkpoint through this pipeline; they report 48.1
]
fig, ax = frame("Qwen3.5-0.8B on MATH-500", "500 problems, 4 samples each",
                "Accuracy", figsize=(8, 4.7), left=0.365)
y = np.arange(len(rows))[::-1]
for yi, (lab, v, c) in zip(y, rows):
    mine = c == ACCENT
    ax.barh(yi, v, height=0.46, color=c)
    ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=12, fontweight="semibold" if mine else "medium", color=INK if mine else TEXT)
ax.set_yticks(y, [r[0] for r in rows], fontsize=11, color=INK)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlim(0, 62)
pct(ax, [0, 20, 40, 60], "x")
ax.grid(axis="y", visible=False)
save(fig, "ladder")

# ---- 2. where the loop starts
cache = pathlib.Path("results/loop_starts.json")
if not cache.exists():
    from fullrun import loop_start_tok

    ls = [loop_start_tok(r["token_ids"]) for r in (json.loads(l) for l in gzip.open("results/base_samples.jsonl.gz", "rt"))
          if r["finish"] == "length"]
    cache.write_text(json.dumps(ls))
ls = np.array([x for x in json.loads(cache.read_text()) if x is not None])
fig, ax = frame("Where the loops start", "1,696 looping samples, 32k token budget",
                "Thinking tokens before the first repeat")
ax.hist(ls / 1000, bins=np.arange(0, 33, 1), color=BAR, edgecolor=PANEL, linewidth=1.5)
med = np.median(ls) / 1000
ax.set_ylim(0, 250)
ax.plot([med, med], [0, 232], color=ACCENT, lw=2, solid_capstyle="round")
ax.text(med + 0.6, 226, f"median: {med*1000:,.0f} tokens", color=INK, fontsize=11.5, fontweight="medium", va="center")
ax.set_xlim(0, 32.768)
ax.set_xticks([0, 8, 16, 24, 32], ["0", "8k", "16k", "24k", "32k"])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_ylabel("Samples")
ax.grid(axis="x", visible=False)
save(fig, "loop-start")

# ---- 3. the cliff: every position in the 24 tokens before a natural stop
ch = json.load(open("results/checks_acts.json"))
gk = ch["summary"]["gap_median_by_k"]
ks = sorted((int(k) for k in gk), reverse=True)
xs, ys = [-k for k in ks], [gk[str(k)] for k in ks]
fig, ax = frame("The last 24 tokens before the model stops", "Median over 147 traces that stopped on their own",
                "Tokens before the stop")
ax.plot(xs, ys, color=LINE, lw=1.8, zorder=3)
ax.scatter(xs[:-1], ys[:-1], s=34, color=DOT, ec=PANEL, lw=1.2, zorder=4)
halo(ax, xs[-1], ys[-1])
ax.text(-24, 30.5, "mid-line: 21 to 26 behind", fontsize=11, color=TEXT, va="center")
ax.text(-1.2, ys[-1] + 0.3, f"at the stop: {ys[-1]:.1f} behind", fontsize=11.5, fontweight="semibold", color=INK, ha="right", va="center")
ax.set_xlim(-25.2, 1.2)
ax.set_ylim(-2, 35)
ax.set_yticks([0, 10, 20, 30])
ax.set_ylabel("Logits behind the top choice")
ax.set_xticks([-24, -18, -12, -6, 0], ["24", "18", "12", "6", "stop"])
ax.grid(axis="x", ls=(0, (2, 3)))
save(fig, "stop-cliff")

# ---- 4. how much thinking the answer needs
cg = json.load(open("results/checks_gen.json"))["rows"]
fr = json.load(open("results/forced.json"))["rows"]


def acc(rows):
    by = {}
    for x in rows:
        by.setdefault(x["id"], []).append(x["correct"])
    return 100 * np.mean([np.mean(v) for v in by.values()])


pts = [(lab, acc([x for x in cg if x["arm"] == arm])) for lab, arm in
       (("0", "forced@0"), ("256", "forced@256"), ("1k", "forced@1024"), ("2k", "forced@2048"))]
pts += [(lab, acc([x for x in fr if x["cut"] == lab and x["variant"] == "forced"])) for lab in ("4k", "8k", "16k")]
fig, ax = frame("Forced answer after k thinking tokens", "Same 2,000 traces, cut at k",
                "Thinking tokens before the cut (k)")
x = np.arange(len(pts))
m = np.array([p[1] for p in pts])
hi = [p[0] for p in pts].index("8k")
ax.axhline(51.8, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)  # their checkpoint through this pipeline
ax.text(-0.3, 54.9, "Astra's weight edit: 51.8%", fontsize=10, color=MUTED, va="center")
ax.plot(x, m, color=LINE, lw=1.8, zorder=3)
ax.scatter(np.delete(x, hi), np.delete(m, hi), s=44, color=DOT, ec=PANEL, lw=1.4, zorder=4)
halo(ax, x[hi], m[hi])
for xi, v in zip(x, m):
    top = xi == hi
    ax.text(xi + (0 if top else 0.14), v + (-5.5 if top else -3.6), f"{v:.1f}%", fontsize=11.5 if top else 10.5, ha="center" if top else "left",
            va="center", fontweight="semibold" if top else "regular", color=INK if top else TEXT)
ax.set_xticks(x, [p[0] for p in pts])
ax.set_xlim(-0.5, len(pts) - 0.3)
ax.set_ylim(-3, 62)
pct(ax, [0, 20, 40, 60])
ax.set_ylabel("MATH-500 accuracy")
ax.grid(axis="x", ls=(0, (2, 3)))
save(fig, "budget")
print({p[0]: round(float(p[1]), 1) for p in pts})

# ---- 5. the penalty against the gap, by where you are in the trace
pen = json.load(open("results/penalty.json"))
lines = json.load(open("results/answers_lines.json"))["traces"]
g = {"any": [r["gap"] for r in pen if r["group"] in ("pre", "loop", "pre_fin")], "start": [], "loop": [], "stop": []}
for t in lines:
    for r in t["rows"]:
        k = "stop" if r["is_stop"] else "loop" if t["kind"] == "loop" and t["ls"] is not None and r["p"] >= t["ls"] else "start"
        g[k].append(r["gap"])
rows = [("Random position, mostly mid-line", "any", "trillions ×"), ("Start of a line", "start", "4 million ×"),
        ("Start of a line, in a loop", "loop", "700,000 ×"), ("Right after a closing line", "stop", "4 ×")]
fig, ax = frame("How far behind the stop token is",
                "Median gap in logits, and how many times less likely than the top choice that makes it",
                "Logits behind the top choice (dot: median, bar: middle half)", figsize=(8, 4.7), left=0.365)
y = np.arange(len(rows))[::-1]
for yi, (lab, k, odds) in zip(y, rows):
    lo, med, hi = np.percentile(g[k], [25, 50, 75])
    ax.plot([lo, hi], [yi, yi], color=BAR, lw=7, solid_capstyle="round", zorder=3)
    ax.scatter([med], [yi], s=60, color=LINE, ec=PANEL, lw=1.4, zorder=4)
    ax.text(hi + 1.3, yi, f"{med:.1f}", fontsize=12, fontweight="medium", color=INK, va="center")
    ax.text(hi + (5.5 if med >= 10 else 4.0), yi, f"· {odds}", fontsize=10.5, color=MUTED, va="center")
    if k == "loop":
        ax.plot([med - 1.5, med], [yi, yi], color=ACCENT, lw=7, solid_capstyle="butt", zorder=3.5)  # to scale, on the row
        ax.text(med - 1.5, yi - 0.34, "presence penalty: 1.5 logits", fontsize=10.5, fontweight="medium", color=ACCENT, va="center")
ax.set_yticks(y, [r[0] for r in rows], fontsize=11, color=INK)
ax.set_ylim(-0.7, len(rows) - 0.4)
ax.set_xlim(0, 50)
ax.set_xticks([0, 10, 20, 30, 40])
ax.grid(axis="y", visible=False)
save(fig, "penalty-gap")
print({k: (len(v), [round(float(z), 2) for z in np.percentile(v, [25, 50, 75])]) for k, v in g.items()})

# ---- 6. social preview: the post title over three bars, opaque, 1200x630
rows = [("Base model", 8.2, BAR), ("Thinking stopped at 8k,\nanswer started for it", 49.8, ACCENT), ("Astra's weight edit", 51.8, LINE)]
fig, ax = frame("The model had the answer. It couldn't stop thinking.", "Qwen3.5-0.8B on MATH-500",
                "Accuracy", figsize=(8, 4.2), left=0.365)
y = np.arange(len(rows))[::-1]
for yi, (lab, v, c) in zip(y, rows):
    ax.barh(yi, v, height=0.5, color=c)
    ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=13, fontweight="semibold" if c == ACCENT else "medium", color=INK if c == ACCENT else TEXT)
ax.set_yticks(y, [r[0] for r in rows], fontsize=11.5, color=INK)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlim(0, 62)
pct(ax, [0, 20, 40, 60], "x")
ax.grid(axis="y", visible=False)
fig.patch.set(alpha=1, facecolor="white")
fig.savefig(OUT / "teaser.png", dpi=150)
plt.close(fig)
print("wrote teaser")
