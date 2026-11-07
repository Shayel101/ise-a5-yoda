#!/usr/bin/env python3
"""All report/README figures, computed from RAW results only (same functions as the notebook).

Writes report/figures/*.png and report/tables/*.csv.
Palette (validated with the dataviz validator, light mode): PMS = #2a78d6 (blue), CBe = #eb6834
(orange), third series = #1baf7a (aqua, below 3:1 contrast, so it is always direct-labelled). Baseline
vs after is encoded by line style / hatch as well as position, so identity is never carried by colour
alone. Status colours (C2): good #0ca30c, critical #d03b3b, serious #ec835a, always shown with a text label.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from a5 import analysis as A  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, CRIT, SERIOUS = "#0ca30c", "#d03b3b", "#ec835a"
COL = {"PMS": BLUE, "CBe": ORANGE}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False,
})

FIG, TAB = ROOT / "report/figures", ROOT / "report/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

runs = {"baseline": A.load("configs/baseline.yaml"), "after": A.load("configs/after.yaml")}
tabs = {k: A.cell_table(r) for k, r in runs.items()}
decs = {k: A.decisions(r, tabs[k]) for k, r in runs.items()}
fdec = {k: A.flip_decomposition(r) for k, r in runs.items()}
for k in runs:
    tabs[k].to_csv(TAB / f"cells_{k}.csv", index=False)
    decs[k].to_csv(TAB / f"decisions_{k}.csv", index=False)
    fdec[k].to_csv(TAB / f"flip_decomposition_{k}.csv", index=False)
cfg = runs["baseline"]["cfg"]
names = A.cell_names(cfg)
real = [n for n, c in zip(names, cfg["cells"])
        if any(c["fault"] == r["fault"] and np.isclose(c["rate"], r["rate"]) for r in cfg["realistic"])]
FAULTS = ["F1_resample", "F1_noise", "F2_allwise_swap", "F3_iphas_swap"]
TITLE = {"F1_resample": "F1 W3/W4 resampled", "F1_noise": "F1 W3/W4 + N(0,1 mag)",
         "F2_allwise_swap": "F2 AllWISE mismatch", "F3_iphas_swap": "F3 IPHAS/VPHAS+ mismatch"}
LS = {"baseline": "-", "after": "--"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def realistic_rate(f):
    return [c["rate"] for c in cfg["realistic"] if c["fault"] == f]


# ---------------------------------------------------------------- 1. flip curves
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, f in zip(axes, FAULTS):
    for k in runs:
        t = tabs[k]
        sub = pd.concat([t[t.fault == "clean"], t[t.fault == f]]).sort_values("rate")
        for c in ["PMS", "CBe"]:
            y, lo, hi = sub[f"flip_{c}_cand"] * 100, sub[f"flip_{c}_cand_lo"] * 100, sub[f"flip_{c}_cand_hi"] * 100
            ax.errorbar(sub.rate * 100, y, yerr=[y - lo, hi - y], color=COL[c], ls=LS[k], marker="o",
                        ms=4, capsize=2, label=f"{c} {k}")
    for r in realistic_rate(f):
        ax.axvline(r * 100, color=INK2, lw=0.8, ls=":")
        ax.text(r * 100, ax.get_ylim()[1] if False else 23.5, " paper rate", color=INK2, fontsize=7, va="top")
    ax.axhline(5, color=CRIT, lw=0.9, ls=":")
    ax.set_title(TITLE[f]); ax.set_xlabel("fault rate (% of sources)")
axes[0].set_ylabel("flip rate of published candidates (%)")
axes[0].text(1, 5.4, "5 % threshold", color=CRIT, fontsize=7)
axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.93))
axes[0].set_ylim(0, 24)
fig.suptitle("C1: class flips grow with fault rate (95 % bootstrap CI; solid = baseline, dashed = after guard)",
             fontsize=10, color=INK, y=1.04)
save(fig, "flip_curves.png")

# ---------------------------------------------------------------- 2. delta recall
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for ax, f in zip(axes, FAULTS):
    for k, mk in [("baseline", "o"), ("after", "s")]:
        t = tabs[k]
        sub = pd.concat([t[t.fault == "clean"], t[t.fault == f]]).sort_values("rate")
        off = 0 if k == "baseline" else (sub.rate.max() * 100 * 0.015)
        y, lo, hi = sub.d_recall_PMS * 100, sub.d_recall_lo * 100, sub.d_recall_hi * 100
        ax.errorbar(sub.rate * 100 + off, y, yerr=[y - lo, hi - y], color=BLUE, ls=LS[k], marker=mk, ms=4,
                    capsize=2, label=k)
    ax.axhspan(-2.8, 2.8, color=GRID, alpha=0.6, lw=0)
    for r in realistic_rate(f):
        ax.axvline(r * 100, color=INK2, lw=0.8, ls=":")
    ax.set_title(TITLE[f]); ax.set_xlabel("fault rate (%)")
axes[0].set_ylabel("Δ PMS recall on test set (pp)")
axes[0].text(0.5, 3.2, "±2.8 pp band (2× paper σ)", fontsize=7, color=INK2)
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("C1 rule (b): paired change in PMS recall (95 % bootstrap CI; ≈7 pp per test star)", fontsize=10, y=1.04)
save(fig, "delta_recall.png")

# ---------------------------------------------------------------- 3. transition matrices
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

seq = LinearSegmentedColormap.from_list("blue", ["#f0f5fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
lab = ["other", "PMS", "CBe"]
for ax, nm in zip(axes, real):
    m = A.transitions(runs["baseline"], nm).astype(float)
    pct = m / m.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap=seq, vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({int(m[i, j]):,})", ha="center", va="center", fontsize=7,
                    color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(3), [f"→{x}" for x in lab]); ax.set_yticks(range(3), lab)
    ax.grid(False); ax.set_title(nm.split("_", 1)[1].replace("_", " "), fontsize=9)
axes[0].set_ylabel("class without fault")
fig.suptitle("Class transitions of published candidates at paper-realistic rates (baseline, 3 seeds pooled; row %)",
             fontsize=10, y=1.04)
save(fig, "transitions.png")

# ---------------------------------------------------------------- 4. C2 matrix
def c2_rows(run):
    c = run["c2"]
    rows = {x["case"]: x["verdict"] for x in c["cases"]}
    rows[c["bootgather"]["case"]] = c["bootgather"]["verdict"]
    return rows


cb, ca = c2_rows(runs["baseline"]), c2_rows(runs["after"])
cases = list(cb)
sup = json.loads((ROOT / "results/after/supplementary_bootgather_unflagged.json").read_text())
SC = {"PASS": GOOD, "FAIL-silent": CRIT, "FAIL-crash": SERIOUS}
SYM = {"PASS": "✓ PASS", "FAIL-silent": "✗ FAIL-silent", "FAIL-crash": "! FAIL-crash"}
fig, ax = plt.subplots(figsize=(7.2, 6.4))
for i, case in enumerate(cases):
    for j, v in enumerate([cb[case], ca[case]]):
        nominal = case == "bootgather_original_ids" and j == 1
        ax.add_patch(plt.Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, facecolor=SC[v], alpha=0.9, lw=0,
                                   hatch="////" if nominal else None, edgecolor=SURF))
        txt = SYM[v]
        if case == "bootgather_original_ids" and j == 1:
            txt += "*  (crash)"
        ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", color="white", fontsize=7.5, weight="bold")
ax.set_xlim(0, 2); ax.set_ylim(len(cases), 0)
ax.set_xticks([0.5, 1.5], ["baseline (original path)", "after (guard A)"])
ax.set_yticks(np.arange(len(cases)) + 0.5, [c.replace("_", " ") for c in cases], fontsize=8)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("C2 boundary suite: verdict per malformed-input case")
fig.text(0.01, 0.005, f"* nominal PASS: the frozen rule counts Bootgather's crash (on the guard's NaN output) as a loud error.\n"
         f"  A supplementary run on the {sup['n_sources_in']} unflagged rows: {sup['n_ids_altered']} IDs are still silently altered.",
         fontsize=7, color=INK2)
save(fig, "c2_matrix.png")

# ---------------------------------------------------------------- 5. flip decomposition
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, c in zip(axes, ["PMS", "CBe"]):
    xs, labels = [], []
    for i, nm in enumerate(real):
        for j, k in enumerate(["baseline", "after"]):
            r = fdec[k][fdec[k].cell == nm].iloc[0]
            x = i * 3 + j
            ch, rj = r[f"{c}_changed_class"] * 100, r[f"{c}_rejected"] * 100
            ax.bar(x, ch, color=COL[c], width=0.85, edgecolor=SURF, linewidth=2,
                   label="changed class (silent)" if i == j == 0 else None)
            ax.bar(x, rj, bottom=ch, color=AQUA, width=0.85, hatch="///", edgecolor=SURF, linewidth=2,
                   label="rejected by guard (loud)" if i == 0 and j == 1 else None)
            ax.text(x, ch + rj + 0.3, f"{ch + rj:.1f}", ha="center", fontsize=7, color=INK)
            xs.append(x); labels.append(f"{'B' if k == 'baseline' else 'A'}")
        short = {"F1_resample": "F1-resample 44%", "F1_noise": "F1-noise 44%", "F2_allwise_swap": "F2 2.3%",
                 "F3_iphas_swap": "F3 1.3%"}
        key = [k2 for k2 in short if k2 in nm][0]
        ax.text(i * 3 + 0.5, -2.6, short[key], ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xs, labels); ax.axhline(5, color=CRIT, lw=0.9, ls=":")
    ax.set_title(f"{c} candidates"); ax.set_ylim(0, 15); ax.set_axisbelow(True)
axes[0].set_ylabel("flipped published candidates (%)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("What a 'flip' is: silent class change vs guard rejection (B = baseline, A = after guard)", fontsize=10)
fig.subplots_adjust(bottom=0.2)
save(fig, "flip_decomposition.png")

# ---------------------------------------------------------------- 6. reproduction agreement
af = pd.read_csv(ROOT / "results/baseline/agreement_full.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
x = np.arange(len(af))
for k, (c, col) in enumerate([("PMS", BLUE), ("CBe", ORANGE)]):
    axes[0].bar(x * 3 + k, af[f"n_repro_{c}"], color=col, width=0.85, edgecolor=SURF, linewidth=2,
                label=f"{c}: reproduction")
    axes[0].hlines(af[f"n_published_{c}"], x * 3 + k - 0.42, x * 3 + k + 0.42, color=INK, lw=2,
                   label="published count" if k == 0 else None)
    for xi, v in zip(x, af[f"n_repro_{c}"]):
        axes[0].text(xi * 3 + k, v * 1.08, f"{v:,}", ha="center", fontsize=7)
axes[0].set_yscale("log"); axes[0].set_xticks(x * 3 + 0.5, [f"seed {s}" for s in af.seed])
axes[0].set_ylabel("p ≥ 0.5 candidates (log)"); axes[0].legend(fontsize=7, loc="upper left")
axes[0].set_title("Candidate counts: reproduction vs published")
metrics = [("published_recovered_PMS", "PMS recovered", BLUE), ("published_recovered_CBe", "CBe recovered", ORANGE),
           ("spearman_PMS", "Spearman ρ p(PMS)", BLUE), ("spearman_CBe", "Spearman ρ p(CBe)", ORANGE),
           ("jaccard_PMS", "Jaccard PMS", BLUE), ("jaccard_CBe", "Jaccard CBe", ORANGE)]
for i, (m, lbl, col) in enumerate(metrics):
    v = af[m].to_numpy()
    axes[1].barh(i, v.mean(), color=col, height=0.7, alpha=0.85 if "recovered" in m else 0.55,
                 edgecolor=SURF, linewidth=2)
    axes[1].plot(v, [i] * len(v), "o", color=INK, ms=4)
    axes[1].text(1.03, i, f"{v.mean():.2f}", va="center", fontsize=7.5)
axes[1].set_yticks(range(len(metrics)), [m[1] for m in metrics]); axes[1].invert_yaxis()
axes[1].set_xlim(0, 1.12); axes[1].set_title("Agreement (bar = mean, dots = seeds)")
fig.subplots_adjust(wspace=0.45)
fig.suptitle("Route C reproduction vs Vioque et al. (2020) catalogue (full SoSt minus training sources)", fontsize=10, y=1.04)
save(fig, "agreement.png")

# ---------------------------------------------------------------- 7. training curves
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for t, col in zip(runs["baseline"]["training"], [BLUE, ORANGE, AQUA]):
    h = t["history"]["val_precision"]
    ax.plot(h, color=col, lw=1.4)
    best = int(np.argmax(h))
    ax.plot(best, h[best], "o", color=col, ms=6)
    ax.text(len(h), h[-1], f" seed {t['seed']} ({t['epochs_run']} ep)", color=INK, fontsize=7, va="center")
ax.set_xlabel("epoch"); ax.set_ylabel("validation PMS precision (p ≥ 0.5)")
ax.set_title("Training: early stopping on validation PMS precision (patience 50; dot = best)")
save(fig, "training_curves.png")

# ---------------------------------------------------------------- 8. counterexample SEDs
from a5.data import load_sample  # noqa: E402
from a5.pipeline import PASSBANDS  # noqa: E402

sample = load_sample()
picks = [("c03_F1_resample_0.44", 1), ("c10_F2_allwise_swap_0.023", 0), ("c10_F2_allwise_swap_0.023", 2)]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=False)
for ax, (nm, row) in zip(axes, picks):
    ce, nflip = A.counterexamples(runs["baseline"], nm, sample, max_rows=10)
    r = ce.iloc[row]
    before = np.array([r[b] for b in PASSBANDS])
    after = before.copy()
    for part in r["changed"].split("; "):
        band, vals = part.split(": ")
        after[PASSBANDS.index(band)] = float(vals.split("->")[1])
    xx = np.arange(len(PASSBANDS))
    ax.plot(xx, before, "o-", color=BLUE, label="observed")
    ax.plot(xx, after, "s--", color=ORANGE, label="after injected fault")
    ax.invert_yaxis(); ax.set_xticks(xx, [b.replace("mag", "") for b in PASSBANDS], fontsize=7)
    pc, pf = r["p_clean(o,PMS,CBe)"], r["p_fault(o,PMS,CBe)"]
    ax.set_title(f"Gaia DR2 {r.GaiaDR2}\n{nm.split('_', 1)[1].replace('_', ' ')}", fontsize=8.5)
    ax.text(0.02, 0.96, f"{r.clean_class} → {r.fault_class}\np(o,PMS,CBe): {pc[0]:.2f}/{pc[1]:.2f}/{pc[2]:.2f}"
            f" → {pf[0]:.2f}/{pf[1]:.2f}/{pf[2]:.2f}", transform=ax.transAxes, fontsize=7, color=INK, va="top",
            bbox=dict(fc=SURF, ec=GRID))
axes[0].set_ylabel("magnitude"); axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Concrete counterexamples: published candidates whose class flips under a realistic fault (seed 11)",
             fontsize=10, y=1.04)
save(fig, "counterexamples.png")

# ---------------------------------------------------------------- 9. side effects
se = json.loads((ROOT / "results/after/side_effects.json").read_text())
fr = se["false_rejections_clean"]
sets = [("test", "E_test (labelled + other)"), ("rand", "E_rand (100k field)"), ("cand", "E_cand (published candidates)")]
fig, ax = plt.subplots(figsize=(7.5, 3))
for i, (s_, lbl) in enumerate(sets):
    vals = [fr[f"seed{s}_{s_}"]["rate"] * 100 for s in cfg["model_seeds"]]
    ax.barh(i, np.mean(vals), color=[AQUA, BLUE, ORANGE][i], height=0.6, edgecolor=SURF, linewidth=2)
    ax.plot(vals, [i] * 3, "o", color=INK, ms=4)
    ax.text(max(vals) + 0.05, i, f"{np.mean(vals):.2f} % (seeds {', '.join(f'{v:.2f}' for v in vals)})",
            va="center", fontsize=7.5)
ax.set_yticks(range(3), [s[1] for s in sets]); ax.invert_yaxis(); ax.set_xlim(0, 3.5)
ax.set_xlabel("valid sources wrongly rejected by the guard (%)")
ax.set_title("Side effect of improvement A: false rejections on clean data")
save(fig, "side_effects.png")

# ---------------------------------------------------------------- 10. candidate count shifts
fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
for ax, f in zip(axes, FAULTS):
    t = tabs["baseline"]
    sub = pd.concat([t[t.fault == "clean"], t[t.fault == f]]).sort_values("rate")
    for c in ["PMS", "CBe"]:
        ax.plot(sub.rate * 100, sub[f"dN_{c}_rand"], marker="o", ms=4, color=COL[c], label=c)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(TITLE[f]); ax.set_xlabel("fault rate (%)")
axes[0].set_ylabel("Δ candidates per 100k field sources"); axes[0].legend(fontsize=7)
fig.suptitle("Candidate-count drift on random field sources (baseline, mean of 3 seeds)", fontsize=10, y=1.04)
save(fig, "candidate_counts.png")
