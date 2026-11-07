"""Aggregation of RAW run outputs (metrics.csv, per_source/*.npz). Used by notebooks/a5_analysis.ipynb.

Nothing here hard-codes a result; everything is recomputed from files written by run_experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CLS = {0: "other", 1: "PMS", 2: "CBe"}


def cell_names(cfg: dict) -> list[str]:
    from .run_experiment import cell_name
    return [cell_name(i, c) for i, c in enumerate(cfg["cells"])]


def load(cfg_path: str | Path) -> dict:
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    d = ROOT / cfg["out_dir"]
    return {"dir": d, "cfg": cfg, "metrics": pd.read_csv(d / "metrics.csv"),
            "training": json.loads((d / "training.json").read_text()),
            "agreement": pd.read_csv(d / "agreement.csv"),
            "c2": json.loads((d / "c2_boundary.json").read_text()) if (d / "c2_boundary.json").exists() else None,
            "meta": json.loads((d / "run_meta.json").read_text())}


def npz(run: dict, seed: int, name: str):
    return np.load(run["dir"] / "per_source" / f"{seed}_{name}.npz")


def flip_indicators(run: dict, name: str, set_: str, cls: int) -> list[np.ndarray]:
    clean_name = cell_names(run["cfg"])[0]
    out = []
    for s in run["cfg"]["model_seeds"]:
        c0 = npz(run, s, clean_name)[f"{set_}_cls"]
        c1 = npz(run, s, name)[f"{set_}_cls"]
        m = c0 == cls
        out.append((c1[m] != cls).astype(float))
    return out


def boot_ci(per_seed: list[np.ndarray], n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Mean over pooled sources; percentile CI resampling sources within each seed (stratified)."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(per_seed) if per_seed else np.array([])
    if pooled.size == 0:
        return np.nan, np.nan, np.nan
    sizes = [len(v) for v in per_seed]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        tot = 0.0
        for v, n in zip(per_seed, sizes):
            if n:
                tot += v[rng.integers(0, n, n)].sum()
        stats[b] = tot / sum(sizes)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(pooled.mean()), float(lo), float(hi)


def recall_delta_ci(run: dict, name: str, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Paired ΔRecall_PMS on E_test: per seed mean(det_fault - det_clean) over true PMS test sources,
    averaged over seeds; CI by resampling PMS test sources within each seed."""
    clean_name = cell_names(run["cfg"])[0]
    diffs = []
    for s in run["cfg"]["model_seeds"]:
        a, b = npz(run, s, clean_name), npz(run, s, name)
        y = a["test_y"]
        det0 = (a["test_p"][:, 1] >= 0.5)[y == 1].astype(float)
        det1 = (b["test_p"][:, 1] >= 0.5)[y == 1].astype(float)
        diffs.append(det1 - det0)
    point = float(np.mean([d.mean() for d in diffs]))
    rng = np.random.default_rng(seed)
    stats = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs]) for _ in range(n_boot)])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return point, float(lo), float(hi)


def cell_table(run: dict) -> pd.DataFrame:
    """One row per grid cell: flip rates (E_cand, E_test) with CIs, ΔRecall/ΔPrecision, ΔN candidates."""
    cfg, m = run["cfg"], run["metrics"]
    rows = []
    for i, (cell, name) in enumerate(zip(cfg["cells"], cell_names(cfg))):
        r = {"cell": name, "fault": cell["fault"], "rate": cell["rate"], "k": cell.get("k", np.nan)}
        for set_ in ["cand", "test"]:
            for c in [1, 2]:
                mean, lo, hi = boot_ci(flip_indicators(run, name, set_, c), cfg["n_boot"])
                r[f"flip_{CLS[c]}_{set_}"], r[f"flip_{CLS[c]}_{set_}_lo"], r[f"flip_{CLS[c]}_{set_}_hi"] = mean, lo, hi
        sub = m[(m.cell == i) & (m.set == "test")]
        r["d_recall_PMS"], r["d_recall_lo"], r["d_recall_hi"] = recall_delta_ci(run, name, cfg["n_boot"])
        r["d_precision_PMS"] = sub.d_precision_PMS.mean()
        r["recall_PMS"] = sub.recall_PMS.mean()
        r["precision_PMS"] = sub.precision_PMS.mean()
        for set_ in ["rand", "cand"]:
            ss = m[(m.cell == i) & (m.set == set_)]
            r[f"dN_PMS_{set_}"] = ss.dN_PMS.mean()
            r[f"dN_CBe_{set_}"] = ss.dN_CBe.mean()
            r[f"N_PMS_{set_}"] = ss.n_PMS_cand.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def decisions(run: dict, table: pd.DataFrame) -> pd.DataFrame:
    cfg = run["cfg"]
    thr, dr = cfg["thresholds"]["flip_rate"], cfg["thresholds"]["delta_recall_pp"] / 100
    out = []
    for rc in cfg["realistic"]:
        t = table[(table.fault == rc["fault"]) & np.isclose(table.rate, rc["rate"])].iloc[0]
        a_pms = t.flip_PMS_cand > thr and t.flip_PMS_cand_lo > thr
        a_cbe = t.flip_CBe_cand > thr and t.flip_CBe_cand_lo > thr
        b = abs(t.d_recall_PMS) > dr and (t.d_recall_hi < 0 or t.d_recall_lo > 0)
        out.append({"cell": t.cell, "flip_PMS_cand": t.flip_PMS_cand, "CI_lo_PMS": t.flip_PMS_cand_lo,
                    "flip_CBe_cand": t.flip_CBe_cand, "CI_lo_CBe": t.flip_CBe_cand_lo,
                    "d_recall_PMS": t.d_recall_PMS, "d_recall_CI": f"[{t.d_recall_lo:.4f}, {t.d_recall_hi:.4f}]",
                    "rule_a_PMS": a_pms, "rule_a_CBe": a_cbe, "rule_b_recall": b,
                    "falsifies_C1": bool(a_pms or a_cbe or b)})
    return pd.DataFrame(out)


def transitions(run: dict, name: str, set_: str = "cand") -> np.ndarray:
    m = run["metrics"]
    i = cell_names(run["cfg"]).index(name)
    sub = m[(m.cell == i) & (m.set == set_)]
    return np.sum([np.array(json.loads(t)) for t in sub.transition], axis=0)


def counterexamples(run: dict, name: str, sample: pd.DataFrame, max_rows: int = 30) -> pd.DataFrame:
    """E_cand sources (first model seed) whose class flipped under the fault; observables before/after are
    regenerated deterministically with the same fault seed (and checked against the stored indices)."""
    from . import faults as F
    from .data import base13
    from .pipeline import BASE13
    from .run_experiment import apply_fault

    cfg = run["cfg"]
    s = cfg["model_seeds"][0]
    i = cell_names(cfg).index(name)
    clean_name = cell_names(cfg)[0]
    a, b = npz(run, s, clean_name), npz(run, s, name)
    ids = np.load(run["dir"] / "ids" / f"{s}_cand.npy")
    rand_ids = np.load(run["dir"] / "ids" / f"{s}_rand.npy")
    idx_sample = sample.set_index("GaiaDR2")
    X = base13(idx_sample.loc[ids].reset_index())
    Xr = base13(idx_sample.loc[rand_ids].reset_index())
    xf, idx = apply_fault(cfg["cells"][i], X, s * 100000 + i * 10 + 1, Xr, cfg["w34_noise_sigma"])
    assert np.array_equal(np.sort(idx), np.sort(b["cand_idx"])), "fault regeneration mismatch"
    flipped = np.flatnonzero(a["cand_cls"] != b["cand_cls"])
    rows = []
    for j in flipped[:max_rows]:
        r = {"GaiaDR2": str(ids[j]), "clean_class": CLS.get(int(a["cand_cls"][j]), "undef"),
             "fault_class": CLS.get(int(b["cand_cls"][j]), "undef"),
             "p_clean(o,PMS,CBe)": np.round(a["cand_p"][j], 3).tolist(),
             "p_fault(o,PMS,CBe)": np.round(b["cand_p"][j], 3).tolist(), "injected": bool(j in set(idx))}
        changed = [k for k in range(13) if X[j, k] != xf[j, k]]
        r["changed"] = "; ".join(f"{BASE13[k]}: {X[j, k]:.3f}->{xf[j, k]:.3f}" for k in changed)
        r.update({c: X[j, k] for k, c in enumerate(BASE13)})
        rows.append(r)
    return pd.DataFrame(rows), len(flipped)


def flip_decomposition(run: dict, set_: str = "cand") -> pd.DataFrame:
    """Split each cell's flips of clean PMS/CBe sources into (i) rejections by the guard (faulted class -1)
    and (ii) real class changes to another class."""
    names = cell_names(run["cfg"])
    rows = []
    for name in names:
        r = {"cell": name}
        for c in [1, 2]:
            n = rej = chg = 0
            for s in run["cfg"]["model_seeds"]:
                c0 = npz(run, s, names[0])[f"{set_}_cls"]
                c1 = npz(run, s, name)[f"{set_}_cls"]
                m = c0 == c
                n += int(m.sum()); rej += int((c1[m] == -1).sum()); chg += int(((c1[m] != c) & (c1[m] != -1)).sum())
            r[f"{CLS[c]}_n"] = n
            r[f"{CLS[c]}_rejected"] = rej / n if n else np.nan
            r[f"{CLS[c]}_changed_class"] = chg / n if n else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
