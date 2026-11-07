"""Improvement A: input-validation and identifier-integrity guard on the inference path.

Chosen by the student from docs/improvement_candidates.md. It never changes a model or its training.
It only decides which rows may receive a class, and it keeps identifiers exact.

Row flags (bit field):
  1  NON_FINITE   a base characteristic is NaN or +/-inf
  2  OUT_OF_RANGE one of the 48 observables lies outside the training range
                  [q0.01% - margin, q99.99% + margin]
  4  DUPLICATE_ID the source ID occurs more than once in the input file

Flagged rows get NaN probabilities (classified 'undefined', -1), never a silent class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import expand_48

NON_FINITE, OUT_OF_RANGE, DUPLICATE_ID = 1, 2, 4


@dataclass
class Guard:
    lo: np.ndarray
    hi: np.ndarray
    margin: float

    @classmethod
    def fit(cls, x13_train: np.ndarray, q: float = 0.01, margin: float = 1.0) -> "Guard":
        x48 = expand_48(np.asarray(x13_train, float))
        lo = np.percentile(x48, q, axis=0) - margin
        hi = np.percentile(x48, 100 - q, axis=0) + margin
        return cls(lo=lo, hi=hi, margin=margin)

    def flags(self, x13: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
        x13 = np.atleast_2d(np.asarray(x13, float))
        if x13.ndim != 2 or x13.shape[1] != 13:
            raise ValueError(f"guard: expected (n, 13) characteristics, got shape {x13.shape}")
        f = np.zeros(len(x13), dtype=np.int8)
        f[~np.isfinite(x13).all(axis=1)] |= NON_FINITE
        with np.errstate(invalid="ignore"):
            x48 = expand_48(x13)
            oor = ((x48 < self.lo) | (x48 > self.hi)).any(axis=1)
        f[oor & (f == 0)] |= OUT_OF_RANGE
        if ids is not None:
            ids = np.asarray(ids)
            _, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
            f[cnt[inv] > 1] |= DUPLICATE_ID
        return f


def guarded_predict(model, x13: np.ndarray, guard: Guard, ids: np.ndarray | None = None,
                    return_flags: bool = False):
    x13 = np.atleast_2d(np.asarray(x13, float))
    if len(x13) == 0:
        raise ValueError("guard: empty input")
    f = guard.flags(x13, ids)
    p = np.full((len(x13), 3), np.nan, dtype=np.float32)
    ok = f == 0
    if ok.any():
        p[ok] = model.predict_proba(x13[ok])
    return (p, f) if return_flags else p


def guarded_inference(csv_in, csv_out, model, guard: Guard) -> np.ndarray:
    """Replacement for YODA.py's generalisation step: exact int64 IDs, works for one row, header-checked,
    flags written as an extra column, IDs written as integers."""
    import pandas as pd

    df = pd.read_csv(csv_in, dtype={0: "int64"})
    if df.shape[1] < 18:
        raise ValueError(f"guard: expected >= 18 columns (id, ra, dec, 13 characteristics, label, extra), got {df.shape[1]}")
    if df.empty:
        raise ValueError("guard: empty input")
    ids = df.iloc[:, 0].to_numpy(dtype=np.int64)
    x = df.iloc[:, 3:16].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    p, f = guarded_predict(model, x, guard, ids=ids, return_flags=True)
    out = df.iloc[:, :18].copy()
    out[["p_other", "p_pms", "p_cbe"]] = p
    out["guard_flag"] = f
    out.to_csv(csv_out, index=False)
    return out
