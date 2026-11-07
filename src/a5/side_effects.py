"""Side effects of the improvement (python -m a5.side_effects) -> results/after/side_effects.json

* valid rows wrongly rejected: guard-flagged rows in the CLEAN cell. Every E_test / E_cand / E_rand row
  is a real SoSt source with all observables, so every rejection there is a false rejection.
* determinism check: after-run clean probabilities on unflagged rows vs baseline (same seeds, so they
  should be identical if training is deterministic)
* clean recall / precision / candidate count deltas and wall times are reported by the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis as A

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    b, a = A.load("configs/baseline.yaml"), A.load("configs/after.yaml")
    clean = A.cell_names(b["cfg"])[0]
    res = {"false_rejections_clean": {}, "max_abs_prob_diff_unflagged_vs_baseline": {},
           "class_changes_unflagged_vs_baseline": {}, "wall_seconds": {"baseline": b["meta"]["wall_seconds"],
                                                                       "after": a["meta"]["wall_seconds"]}}
    for s in b["cfg"]["model_seeds"]:
        nb, na = A.npz(b, s, clean), A.npz(a, s, clean)
        for k in ["test", "cand", "rand"]:
            ca, cb = na[f"{k}_cls"], nb[f"{k}_cls"]
            rej = ca == -1
            key = f"seed{s}_{k}"
            res["false_rejections_clean"][key] = {"n": int(rej.sum()), "of": int(len(ca)),
                                                  "rate": float(rej.mean())}
            res["class_changes_unflagged_vs_baseline"][key] = int((ca[~rej] != cb[~rej]).sum())
            if f"{k}_p" in na.files and f"{k}_p" in nb.files:
                d = np.abs(na[f"{k}_p"][~rej] - nb[f"{k}_p"][~rej])
                res["max_abs_prob_diff_unflagged_vs_baseline"][key] = float(np.nanmax(d)) if d.size else 0.0
        # rejected true PMS / CBe in the clean test set
        y = na["test_y"]
        rej = na["test_cls"] == -1
        res["false_rejections_clean"][f"seed{s}_test_by_label"] = {
            int(c): int((rej & (y == c)).sum()) for c in [0, 1, 2]}
    out = ROOT / "results/after/side_effects.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
