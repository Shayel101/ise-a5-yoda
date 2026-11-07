"""Config-driven execution of protocol_v1:  python -m a5.run_experiment configs/baseline.yaml

Writes to <out_dir>/:
  metrics.csv              one row per (seed, cell, eval-set) with all oracles (raw, no aggregation)
  per_source/<seed>_<cell>.npz   per-source probabilities / classes / injected indices (raw outputs)
  training.json            per-seed training info (epochs, n_components, sizes, wall time)
  agreement.csv            per-seed agreement with the published probabilities (clean run)
  ids/<seed>_{test,cand,rand}.npy  Gaia DR2 IDs of every evaluation set (for counterexamples)
  failures/                tracebacks of any crashed cell (kept, never deleted)
  models/                  trained models (git-ignored)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from . import faults as F
from .data import base13, load_candidates, load_labels, load_sample
from .oracles import flip_rates, pms_metrics, transition_matrix
from .pipeline import YodaModel, classify

ROOT = Path(__file__).resolve().parents[2]


def build_sets(sample: pd.DataFrame, labels: pd.DataFrame, cand_ids: np.ndarray):
    lab = labels.drop_duplicates(["GaiaDR2", "label"])
    conflict = lab.GaiaDR2[lab.GaiaDR2.duplicated()].unique()
    lab = lab[~lab.GaiaDR2.isin(conflict)].drop_duplicates("GaiaDR2")
    labelled = sample.merge(lab[["GaiaDR2", "label", "otype"]], on="GaiaDR2", how="inner")
    return labelled, conflict


def make_predict(model: YodaModel, guard: bool):
    if not guard:
        return model.predict_proba
    from .guard import guarded_predict  # improvement (Phase 5)
    return lambda x: guarded_predict(model, x)


def apply_fault(cell: dict, x: np.ndarray, seed: int, rand_x: np.ndarray, sigma: float):
    f = cell["fault"]
    if f == "clean":
        return x.copy(), np.array([], dtype=int)
    if f == "F1_resample":
        return F.f1_w3w4_resample(x, cell["rate"], seed, pool=rand_x[:, F.W34])
    if f == "F1_noise":
        return F.f1_w3w4_noise(x, cell["rate"], seed, sigma=sigma)
    if f in ("F2_allwise_swap", "F3_iphas_swap"):
        return F.FAULTS[f](x, cell["rate"], seed, donors=rand_x)
    if f == "F4_fluxerr":
        return F.f4_inflate_flux_errors(x, cell["rate"], seed, k=cell["k"])
    raise ValueError(f)


def cell_name(i: int, c: dict) -> str:
    return f"c{i:02d}_{c['fault']}_{c['rate']}" + (f"_k{c['k']}" if "k" in c else "")


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = ROOT / cfg["out_dir"]
    for sub in ["per_source", "failures", "models", "ids"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    sample = load_sample()
    cands = load_candidates()
    cand_ids = np.unique(np.concatenate(list(cands.values())))
    labels = load_labels()
    labelled, conflict = build_sets(sample, labels, cand_ids)
    print(f"SoSt rows={len(sample)}  labelled in SoSt={len(labelled)} "
          f"(PMS={int((labelled.label == 1).sum())}, CBe={int((labelled.label == 2).sum())}, "
          f"otype={labelled.otype.value_counts().to_dict()}), conflicting IDs dropped={len(conflict)}", flush=True)
    excl = set(labelled.GaiaDR2) | set(cand_ids)
    pool_other = sample[~sample.GaiaDR2.isin(excl)]

    metrics, training, agreement = [], [], []
    for s in cfg["model_seeds"]:
        t0 = time.time()
        lab_boot = resample(labelled, replace=True, n_samples=len(labelled), random_state=s)
        others = pool_other.sample(n=cfg["n_other"], random_state=s).assign(label=0)
        full = pd.concat([others, lab_boot], ignore_index=True)
        train, test = train_test_split(full, test_size=0.1, random_state=s)
        model = YodaModel(seed=s, max_epochs=cfg["max_epochs"], patience=cfg["patience"])
        model.fit(base13(train), train.label.to_numpy())
        train_ids = set(train.GaiaDR2)
        model.model.save(out / "models" / f"seed_{s}.keras")
        with open(out / "models" / f"seed_{s}_prep.pkl", "wb") as fh:
            pickle.dump({"scaler": model.scaler, "pca": model.pca, "n_components": model.n_components}, fh)
        training.append({"seed": s, "epochs_run": model.epochs_run, "n_components": model.n_components,
                         "n_train": len(train), "n_test": len(test),
                         "n_train_by_label": train.label.value_counts().sort_index().to_dict(),
                         "n_test_by_label": test.label.value_counts().sort_index().to_dict(),
                         "best_val_precision": max(model.history.get("val_precision", [float("nan")])),
                         "train_seconds": round(time.time() - t0, 1), "history": model.history})
        print(f"[seed {s}] trained: epochs={model.epochs_run} ncomp={model.n_components} "
              f"{time.time() - t0:.0f}s", flush=True)

        cand = sample[sample.GaiaDR2.isin(cand_ids) & ~sample.GaiaDR2.isin(train_ids)]
        rand = sample[~sample.GaiaDR2.isin(train_ids | set(cand_ids))].sample(n=cfg["n_rand"],
                                                                                  random_state=1000 + s)
        sets = {"test": test, "cand": cand, "rand": rand}
        X = {k: base13(v) for k, v in sets.items()}
        for k, v in sets.items():
            np.save(out / "ids" / f"{s}_{k}.npy", v.GaiaDR2.to_numpy())
        y_test = test.label.to_numpy()
        predict = make_predict(model, cfg.get("guard", False))

        clean_p = {k: predict(X[k]) for k in sets}
        clean_c = {k: classify(p) for k, p in clean_p.items()}
        # agreement with published probabilities (clean, E_rand)
        pub = rand[["pPMS", "pCBe"]].to_numpy()
        row = {"seed": s}
        for j, name in [(1, "PMS"), (2, "CBe")]:
            a, b = clean_p["rand"][:, j] >= 0.5, pub[:, j - 1] >= 0.5
            row[f"jaccard_{name}"] = float((a & b).sum() / max(1, (a | b).sum()))
            row[f"spearman_{name}"] = float(spearmanr(clean_p["rand"][:, j], pub[:, j - 1]).statistic)
            row[f"n_repro_{name}"] = int(a.sum())
            row[f"n_published_{name}"] = int(b.sum())
        # published-candidate recovery on E_cand
        pc = cand[["pPMS", "pCBe"]].to_numpy()
        row["cand_published_PMS_recovered"] = float(np.mean(clean_c["cand"][pc[:, 0] >= 0.5] == 1))
        row["cand_published_CBe_recovered"] = float(np.mean(clean_c["cand"][pc[:, 1] >= 0.5] == 2))
        agreement.append(row)

        for i, cell in enumerate(cfg["cells"]):
            name = cell_name(i, cell)
            try:
                tc = time.time()
                rec = {}
                for si, k in enumerate(["test", "cand", "rand"]):
                    fseed = s * 100000 + i * 10 + si
                    xf, idx = apply_fault(cell, X[k], fseed, X["rand"], cfg["w34_noise_sigma"])
                    p = predict(xf)
                    c = classify(p)
                    m = {"seed": s, "cell": i, "cell_name": name, "fault": cell["fault"],
                         "rate": cell["rate"], "k": cell.get("k", np.nan), "set": k, "n": len(xf),
                         "n_injected": len(idx),
                         "n_PMS_cand": int((c == 1).sum()), "n_CBe_cand": int((c == 2).sum()),
                         "n_undefined": int((c == -1).sum()),
                         "dN_PMS": int((c == 1).sum() - (clean_c[k] == 1).sum()),
                         "dN_CBe": int((c == 2).sum() - (clean_c[k] == 2).sum())}
                    m.update(flip_rates(clean_c[k], c))
                    m["transition"] = json.dumps(transition_matrix(clean_c[k], c).tolist())
                    if k == "test":
                        pm = pms_metrics(p[:, 1], y_test)
                        pm0 = pms_metrics(clean_p["test"][:, 1], y_test)
                        m.update({f"{a}": b for a, b in pm.items()})
                        m["d_recall_PMS"] = pm["recall_PMS"] - pm0["recall_PMS"]
                        m["d_precision_PMS"] = pm["precision_PMS"] - pm0["precision_PMS"]
                    metrics.append(m)
                    rec[f"{k}_p"] = p.astype(np.float32)
                    rec[f"{k}_cls"] = c.astype(np.int8)
                    rec[f"{k}_idx"] = idx
                    if k == "rand":  # keep only classes for the 100k set (size)
                        del rec["rand_p"]
                rec["test_y"] = y_test
                np.savez_compressed(out / "per_source" / f"{s}_{name}.npz", **rec)
                print(f"[seed {s}] {name} done in {time.time() - tc:.1f}s", flush=True)
            except Exception:
                (out / "failures" / f"{s}_{name}.txt").write_text(traceback.format_exc())
                print(f"[seed {s}] {name} FAILED (traceback kept)", flush=True)
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(agreement).to_csv(out / "agreement.csv", index=False)
        (out / "training.json").write_text(json.dumps(training, indent=1))
    meta = {"config": cfg, "wall_seconds": round(time.time() - t_start, 1), "n_labelled_in_sost": len(labelled),
            "labelled_by_otype": labelled.otype.value_counts().to_dict(), "n_conflicting_ids": len(conflict),
            "n_cand_ids": len(cand_ids)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"total wall time {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
