"""Reporting-metric correction (logged in results/baseline/rerun_log.md):

protocol_v1 oracle 6 measured the Jaccard overlap on E_rand. But E_rand excludes every published
candidate by construction, so the published p >= 0.5 set on E_rand is empty and the Jaccard index is
trivially 0. This script recomputes the same agreement metrics on the FULL Sample of Study, minus each
model's training sources, using the clean models. It is not a decision rule; C1 and C2 do not depend
on it.

    python -m a5.agreement_full configs/baseline.yaml  ->  <out_dir>/agreement_full.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from .data import base13, load_candidates, load_labels, load_sample
from .pipeline import classify
from .run_boundary import load_model
from .run_experiment import build_sets

ROOT = Path(__file__).resolve().parents[2]


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = ROOT / cfg["out_dir"]
    sample = load_sample()
    cand_ids = np.unique(np.concatenate(list(load_candidates().values())))
    labelled, _ = build_sets(sample, load_labels(), cand_ids)
    pool_other = sample[~sample.GaiaDR2.isin(set(labelled.GaiaDR2) | set(cand_ids))]
    X = base13(sample)
    pub = sample[["pPMS", "pCBe"]].to_numpy()
    rows = []
    for s in cfg["model_seeds"]:
        # regenerate the training split exactly as run_experiment did, to exclude its sources
        lab_boot = resample(labelled, replace=True, n_samples=len(labelled), random_state=s)
        others = pool_other.sample(n=cfg["n_other"], random_state=s).assign(label=0)
        train, _ = train_test_split(pd.concat([others, lab_boot], ignore_index=True), test_size=0.1, random_state=s)
        keep = ~sample.GaiaDR2.isin(set(train.GaiaDR2)).to_numpy()
        m = load_model(out, s)
        p = m.predict_proba(X[keep])
        c = classify(p)
        r = {"seed": s, "n_eval": int(keep.sum())}
        for j, name in [(1, "PMS"), (2, "CBe")]:
            a, b = c == j, pub[keep, j - 1] >= 0.5
            r[f"n_repro_{name}"], r[f"n_published_{name}"] = int(a.sum()), int(b.sum())
            r[f"jaccard_{name}"] = float((a & b).sum() / max(1, (a | b).sum()))
            r[f"published_recovered_{name}"] = float((a & b).sum() / max(1, b.sum()))
            r[f"spearman_{name}"] = float(spearmanr(p[:, j], pub[keep, j - 1]).statistic)
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(out / "agreement_full.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
