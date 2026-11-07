"""Oracles of protocol_v1: flip rates, PMS recall/precision, candidate counts, bootstrap CIs."""
from __future__ import annotations

import numpy as np

CLASS_NAMES = {0: "other", 1: "PMS", 2: "CBe"}


def flip_rates(clean_cls: np.ndarray, fault_cls: np.ndarray) -> dict:
    """Fraction of sources whose class at p>=0.5 changed, conditioned on the CLEAN class.

    ``flip_PMS`` = P(fault class != PMS | clean class == PMS), etc. NaN if the clean class is empty.
    """
    clean_cls = np.asarray(clean_cls)
    fault_cls = np.asarray(fault_cls)
    out = {"flip_all": float(np.mean(clean_cls != fault_cls)) if len(clean_cls) else float("nan")}
    for c, name in CLASS_NAMES.items():
        m = clean_cls == c
        out[f"n_clean_{name}"] = int(m.sum())
        out[f"flip_{name}"] = float(np.mean(fault_cls[m] != c)) if m.any() else float("nan")
    return out


def transition_matrix(clean_cls: np.ndarray, fault_cls: np.ndarray) -> np.ndarray:
    """3x3 counts, rows = clean class (0,1,2), cols = faulted class (0,1,2)."""
    m = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            m[i, j] = int(np.sum((clean_cls == i) & (fault_cls == j)))
    return m


def pms_metrics(p_pms: np.ndarray, y: np.ndarray, thr: float = 0.5) -> dict:
    """Recall and precision of the PMS class at p(PMS) >= thr, exactly as in YODA.py.
    Precision is a lower limit (paper Appx. B.1)."""
    pred = np.asarray(p_pms) >= thr
    true = np.asarray(y) == 1
    tp = int(np.sum(pred & true)); fn = int(np.sum(~pred & true)); fp = int(np.sum(pred & ~true))
    return {"TP": tp, "FN": fn, "FP": fp,
            "recall_PMS": tp / (tp + fn) if tp + fn else float("nan"),
            "precision_PMS": tp / (tp + fp) if tp + fp else float("nan")}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI of the mean of a 1-D sample of 0/1 or real values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1) if len(v) < 20000 else \
        np.array([rng.choice(v, size=len(v)).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)
