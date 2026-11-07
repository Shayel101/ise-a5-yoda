"""Run the C2 (F5) boundary suite:  python -m a5.run_boundary configs/baseline.yaml

Uses the trained model of the first model seed from <out_dir>/models. Writes <out_dir>/c2_boundary.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import keras
import numpy as np
import yaml

from . import boundary as B
from .data import base13, load_sample
from .pipeline import YodaModel

ROOT = Path(__file__).resolve().parents[2]


def load_model(out: Path, seed: int) -> YodaModel:
    m = YodaModel(seed=seed)
    m.model = keras.models.load_model(out / "models" / f"seed_{seed}.keras")
    prep = pickle.load(open(out / "models" / f"seed_{seed}_prep.pkl", "rb"))
    m.scaler, m.pca, m.n_components = prep["scaler"], prep["pca"], prep["n_components"]
    return m


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = ROOT / cfg["out_dir"]
    seed = cfg["model_seeds"][0]
    model = load_model(out, seed)
    if cfg.get("guard", False):
        from .guard import guarded_predict
        predict = lambda x: guarded_predict(model, x)  # noqa: E731
    else:
        predict = model.predict_proba
    sample = load_sample()
    # 5 fixed SoSt rows (first 5 by recno) for the malformed-value cases
    rows = sample.iloc[:5]
    work = ROOT / "results" / "c2_work" / cfg["name"]
    orig = ROOT / "data" / "external" / "yoda_original"
    hashes = B.fetch_original(orig)
    big = sample.sample(n=1000, random_state=0)  # identifier-integrity cases: 1000 random SoSt rows
    cases = B.run_cases(predict, base13(rows), rows.GaiaDR2.to_numpy(), work, base13(big), big.GaiaDR2.to_numpy())
    p = predict(base13(big))
    boot = B.run_bootgather(orig, work / "bootgather", [int(v) for v in big.GaiaDR2], p)
    boot["case"] = "bootgather_original_ids"
    boot["kind"] = "identifier integrity (original Bootgather.py, unmodified)"
    script = B.run_original_yoda_script(orig, work / "yoda_script")
    result = {"config": cfg["name"], "model_seed": seed, "original_sha256": hashes, "cases": cases,
              "bootgather": boot, "original_yoda_py_run": script}
    (out / "c2_boundary.json").write_text(json.dumps(result, indent=1, default=str))
    for c in cases + [boot]:
        print(f"{c['case']:28s} {c['verdict']:12s} {c.get('outcome', '')}")
    print("original YODA.py returncode:", script["returncode"], "|", script["stderr_tail"].strip().splitlines()[-1:])


if __name__ == "__main__":
    main(sys.argv[1])
