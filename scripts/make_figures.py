#!/usr/bin/env python3
"""Report figures from RAW results (same functions as the notebook) -> report/figures/*.png and
report/tables/*.csv."""
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

fig_dir, tab_dir = ROOT / "report/figures", ROOT / "report/tables"
fig_dir.mkdir(parents=True, exist_ok=True)
tab_dir.mkdir(parents=True, exist_ok=True)
runs = {"baseline": A.load("configs/baseline.yaml"), "after (guard A)": A.load("configs/after.yaml")}
tabs = {k: A.cell_table(r) for k, r in runs.items()}
tabs["baseline"].to_csv(tab_dir / "cells_baseline.csv", index=False)
tabs["after (guard A)"].to_csv(tab_dir / "cells_after.csv", index=False)
for k, r in runs.items():
    A.decisions(r, tabs[k]).to_csv(tab_dir / f"decisions_{k.split()[0]}.csv", index=False)
    A.flip_decomposition(r).to_csv(tab_dir / f"flip_decomposition_{k.split()[0]}.csv", index=False)

faults = ["F1_resample", "F1_noise", "F2_allwise_swap", "F3_iphas_swap"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for ax, f in zip(axes, faults):
    for k, ls in [("baseline", "-"), ("after (guard A)", "--")]:
        t = tabs[k]
        sub = pd.concat([t[t.fault == "clean"], t[t.fault == f]]).sort_values("rate")
        for c, col in [("PMS", "C3"), ("CBe", "C0")]:
            y, lo, hi = sub[f"flip_{c}_cand"], sub[f"flip_{c}_cand_lo"], sub[f"flip_{c}_cand_hi"]
            ax.errorbar(sub.rate * 100, y * 100, yerr=[(y - lo) * 100, (hi - y) * 100], color=col, ls=ls,
                        marker="o", ms=3, capsize=2, label=f"{c} {k}")
    rr = [c["rate"] for c in runs["baseline"]["cfg"]["realistic"] if c["fault"] == f]
    for r in rr:
        ax.axvline(r * 100, color="grey", lw=0.8, ls=":")
    ax.axhline(5, color="k", lw=0.8, ls=":")
    ax.set_title(f.replace("_", " "))
    ax.set_xlabel("fault rate (%)")
axes[0].set_ylabel("flip rate on published candidates (%)")
axes[0].legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(fig_dir / "flip_curves.png", dpi=160)
print("wrote figures and tables")
