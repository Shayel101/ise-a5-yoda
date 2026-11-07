#!/usr/bin/env python3
"""Explanatory diagrams (pure matplotlib, no data): report/figures/diagram_*.png

1. diagram_pipeline.png   YODA pipeline, where each fault class is injected, where the guard sits
2. diagram_workflow.png   study phases, git tags and student decision points
3. diagram_experiment.png experimental design: seeds -> models -> evaluation sets -> oracles -> verdicts
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE_L, BLUE = "#cde2fb", "#2a78d6"
ORANGE_L, ORANGE = "#fbe0d4", "#eb6834"
AQUA_L, AQUA = "#d2f0e4", "#1baf7a"
GREY_L = "#f0efec"
CRIT, GOOD = "#d03b3b", "#0ca30c"


def box(ax, x, y, w, h, text, fc=GREY_L, ec=INK2, fs=8, weight="normal", ls="-", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec,
                                lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color,
            wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK2, ls="-", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ 1. pipeline + faults
fig, ax = canvas(14, 6.2)
ax.text(0.2, 5.9, "YODA pipeline (re-implemented) and where each fault class enters", fontsize=12, weight="bold")
# data sources
src = [("Gaia DR2\nG, BP, RP, e(F), N_obs", 4.5), ("2MASS + AllWISE\nJ H Ks W1 W2 W3 W4", 3.4),
       ("IPHAS / VPHAS+\nr, Hα", 2.3)]
for t, y in src:
    box(ax, 0.2, y, 2.0, 0.8, t, fc=BLUE_L, ec=BLUE, fs=7.5)
box(ax, 2.7, 3.2, 1.5, 1.4, "cross-match\n(1″ IPHAS/VPHAS+,\n2″ AllWISE)\n→ SoSt 4.15 M rows", fs=7)
for _, y in src:
    arrow(ax, 2.2, y + 0.4, 2.7, 3.9)
box(ax, 4.7, 3.2, 1.6, 1.4, "13 base\ncharacteristics\nr−Hα, Gvar, Vhtg\n+ 10 passbands", fs=7.5, weight="bold",
    fc=ORANGE_L, ec=ORANGE)
arrow(ax, 4.2, 3.9, 4.7, 3.9)
box(ax, 6.8, 3.2, 1.4, 1.4, "48 observables\n3 + 45 colours", fs=7.5)
arrow(ax, 6.3, 3.9, 6.8, 3.9)
box(ax, 8.7, 3.2, 1.3, 1.4, "Standard-\nScaler\n→ PCA\n(12 comp.)", fs=7.5)
arrow(ax, 8.2, 3.9, 8.7, 3.9)
box(ax, 10.5, 3.2, 1.5, 1.4, "ANN\n2 × 580 ReLU\ndropout .5, BN\nsoftmax(3)", fs=7.5)
arrow(ax, 10.0, 3.9, 10.5, 3.9)
box(ax, 12.4, 3.2, 1.4, 1.4, "p(other)\np(PMS)\np(CBe)\nclass @ p ≥ 0.5", fs=7.5, fc=BLUE_L, ec=BLUE)
arrow(ax, 12.0, 3.9, 12.4, 3.9)
box(ax, 12.4, 1.3, 1.4, 1.1, "Bootgather.py\nmean of 30\nbootstraps", fs=7.5, ls="--")
arrow(ax, 13.1, 3.2, 13.1, 2.4)
ax.text(13.1, 1.1, "IDs parsed as float\n→ 791/1000 altered", ha="center", va="top", fontsize=7, color=CRIT)
# faults
faults = [("F1  spurious W3/W4\n(resample / +N(0,1))", 5.3, 5.2), ("F2  AllWISE mismatch\n(J…W4 from donor)", 3.0, 5.2),
          ("F3  IPHAS mismatch\n(r, Hα from donor)", 3.0, 1.3), ("F4  inflated Gaia\nflux errors (Gvar, Vhtg × k)", 5.2, 1.3)]
for t, x, y in faults:
    box(ax, x, y, 1.9, 0.6, t, fc="#fde8e8", ec=CRIT, fs=7, color=CRIT)
    arrow(ax, x + 0.95, y + (0 if y > 3 else 0.6), 5.5, 4.6 if y > 3 else 3.2, color=CRIT, ls="--")
box(ax, 0.2, 0.4, 2.6, 0.9, "F5  malformed input file\nNaN/±inf, out-of-range, column swap,\nduplicate IDs, empty, 1 row, dtype",
    fc="#fde8e8", ec=CRIT, fs=7, color=CRIT)
arrow(ax, 2.8, 0.85, 4.9, 3.2, color=CRIT, ls="--", rad=-0.15)
# guard
box(ax, 7.3, 1.0, 3.6, 1.2, "Improvement A — guard (after run only)\nexact int64 IDs · 1-row support\n"
    "non-finite / out-of-training-range / duplicate-ID rows\n→ NaN probabilities + flag", fc=AQUA_L, ec=AQUA, fs=7.2)
arrow(ax, 9.1, 2.2, 7.5, 3.2, color=AQUA, lw=1.6)
ax.text(0.2, 0.05, "Faults are pure seeded functions applied to the 13 base characteristics BEFORE scaling/PCA, "
        "at inference time, on models trained on clean data.", fontsize=7.5, color=INK2)
fig.savefig(OUT / "diagram_pipeline.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 2. workflow
fig, ax = canvas(14, 3.6)
ax.text(0.2, 3.3, "Study workflow: phases, git tags and student decision points", fontsize=12, weight="bold")
phases = [("Phase 0\nRecon +\nroute C", None, True), ("Phase 1\nEnvironment\nfreeze", "a5-env", False),
          ("Phase 2\nThreat /\nfault model", None, False), ("Phase 3\nFrozen\nprotocol", "a5-protocol-frozen", True),
          ("Phase 4\nBaseline\n57 runs + C2", "a5-baseline", False), ("Phase 5\nImprovement A\n+ identical rerun", "a5-improvement", True),
          ("Phase 6\nAnalysis\nnotebook", None, False), ("Phase 7\nResidual risk\n+ report", "a5-final", True)]
w, gap = 1.5, 0.22
for i, (t, tag, stop) in enumerate(phases):
    x = 0.2 + i * (w + gap)
    box(ax, x, 1.4, w, 1.2, t, fc=BLUE_L if i < 4 else ORANGE_L, ec=BLUE if i < 4 else ORANGE, fs=7.8)
    if i < len(phases) - 1:
        arrow(ax, x + w, 2.0, x + w + gap, 2.0)
    if tag:
        ax.text(x + w / 2, 1.2, f"tag: {tag}", ha="center", va="top", fontsize=7, family="monospace", color=INK)
    if stop:
        ax.add_patch(plt.Circle((x + w / 2, 0.55), 0.17, color=CRIT))
        ax.text(x + w / 2, 0.55, "■", ha="center", va="center", color="white", fontsize=7)
        ax.text(x + w / 2, 0.25, "STOP: student decides", ha="center", va="top", fontsize=6.5, color=CRIT)
ax.text(0.2, 2.85, "before any fault injection  ─────────────────────────────────────►", fontsize=7.5, color=BLUE)
ax.text(7.1, 2.85, "after freeze: no edits to protocol_v1; failed runs & negative results kept", fontsize=7.5,
        color=ORANGE)
fig.savefig(OUT / "diagram_workflow.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 3. experiment design
fig, ax = canvas(14, 5.4)
ax.text(0.2, 5.1, "Experimental design (protocol_v1): grid, evaluation sets, oracles and decision rules",
        fontsize=12, weight="bold")
box(ax, 0.2, 2.2, 2.2, 2.2, "Labelled set\n187 SIMBAD stars in SoSt\n(120 PMS, 67 CBe)\n+ 47,026 random\n'other'"
    "\n\nbootstrap per seed\n11 · 22 · 33", fc=BLUE_L, ec=BLUE, fs=7.5)
box(ax, 2.9, 2.2, 1.8, 2.2, "train 3 models\n(90/10 split,\nES patience 50,\n≤ 300 epochs)", fs=7.5)
arrow(ax, 2.4, 3.3, 2.9, 3.3)
sets = [("E_test  held-out labelled + other", 4.0), ("E_cand  10,472 published candidates", 3.1),
        ("E_rand  100k field sources (+ donor pool)", 2.2)]
for t, y in sets:
    box(ax, 5.3, y, 3.0, 0.7, t, fs=7.5, fc=GREY_L)
    arrow(ax, 4.7, 3.3, 5.3, y + 0.35)
box(ax, 5.3, 0.5, 3.0, 1.2, "18 fault cells × 3 seeds\nF1 ×8 · F2 ×3 · F3 ×2 · F4 ×4 · clean\n"
    "same seed → same corruption\non every evaluation set", fc="#fde8e8", ec=CRIT, fs=7.3, color=CRIT)
arrow(ax, 6.8, 1.7, 6.8, 2.2, color=CRIT)
orc = [("flip rate by clean class\n(PMS / CBe / other) + 95 % CI", 3.9), ("ΔRecall / ΔPrecision PMS\n(paired, bootstrap CI)", 2.9),
       ("Δ candidate counts,\ntransition matrices", 1.9)]
for t, y in orc:
    box(ax, 8.9, y, 2.4, 0.8, t, fs=7.3, fc=ORANGE_L, ec=ORANGE)
arrow(ax, 8.3, 3.45, 8.9, 4.3); arrow(ax, 8.3, 4.35, 8.9, 3.3); arrow(ax, 8.3, 2.55, 8.9, 2.3)
box(ax, 11.8, 3.1, 2.0, 1.5, "C1 falsified if, at a\nrealistic rate (F1 44 %,\nF2 2.3 %, F3 1.3 %):\nflip > 5 % & CI-lo > 5 %\nor |ΔR| > 2.8 pp & CI excl. 0",
    fc="#fde8e8", ec=CRIT, fs=7, color=INK)
box(ax, 11.8, 1.2, 2.0, 1.4, "C2 falsified if ANY\nmalformed case returns\nfinite p or altered ID\nwithout error/flag\n(16 cases)", fc="#fde8e8",
    ec=CRIT, fs=7, color=INK)
arrow(ax, 11.3, 4.3, 11.8, 3.9); arrow(ax, 11.3, 3.3, 11.8, 3.7)
ax.text(12.8, 0.95, "verdicts: C1 ✗  C2 ✗ (baseline)", ha="center", va="top", fontsize=7.5, weight="bold", color=CRIT)
fig.savefig(OUT / "diagram_experiment.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote 3 diagrams")
