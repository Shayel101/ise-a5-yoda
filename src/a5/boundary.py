"""C2 boundary / malformed-input tests (fault class F5).

Two harnesses:
1. ``yoda_inference`` reproduces YODA.py's generalisation step line by line (np.loadtxt of an
   Input_sample-style CSV with usecols 0..17, characteristics = columns 3..15, colour expansion,
   scaler, PCA, model.predict, np.append, np.savetxt). It uses a trained reproduction model.
2. The ORIGINAL ``Bootgather.py`` (fetched at the pinned commit) runs unmodified on 30 synthetic
   per-iteration files, written exactly as YODA.py writes them.

Verdicts:
  PASS          explicit error, or NaN/flagged output
  FAIL-silent   finite probabilities (or an altered identifier) returned without any error or flag
  FAIL-crash    a crash on VALID input (availability or correctness defect; reported, but not a C2
                falsifier)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np

from .pipeline import expand_48

YODA_COMMIT = "0c8097df40b8ba35fdb6a6b6f35d93fce054bd1f"
RAW_URL = f"https://raw.githubusercontent.com/MVioque/YODA/{YODA_COMMIT}/"
HEADER = "source_id,ra,dec," + ",".join(f"c{i}" for i in range(13)) + ",label,extra"


def fetch_original(dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in ["YODA.py", "Bootgather.py"]:
        p = dest / f
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(RAW_URL + f, timeout=60).read())
        import hashlib
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def yoda_inference(csv_in: Path, csv_out: Path, predict) -> np.ndarray:
    """YODA.py lines 'GENERALIZE' ... np.savetxt, with ``predict`` = scaler+PCA+ANN.

    The colour expansion is done inside ``predict`` (a5.pipeline.expand_48 is the same loop)."""
    Input_set = np.loadtxt(open(csv_in), delimiter=",", skiprows=1,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
    x_sample = Input_set[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    Final_classes = predict(x_sample)
    Output_final_set = np.append(Input_set, Final_classes, axis=1)
    np.savetxt(csv_out, Output_final_set, delimiter=",")
    return Output_final_set


def _write(path: Path, rows: list[list], header: str = HEADER) -> None:
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")


def _row(gid: int, x13: np.ndarray) -> list:
    return [gid, 10.0, 60.0] + [repr(float(v)) for v in x13] + [0, 0]


def _first_column_ids(path: Path) -> list[int]:
    """Read column 0 exactly (Decimal), whether written as float text by YODA.py or as int by the guard."""
    from decimal import Decimal
    out = []
    for line in open(path):
        tok = line.split(",")[0].strip()
        try:
            out.append(int(Decimal(tok)))
        except Exception:
            continue  # header line
    return out


def run_cases(predict, valid_rows: np.ndarray, valid_ids: np.ndarray, workdir: Path,
              id_rows: np.ndarray, id_ids: np.ndarray, inference=None) -> list[dict]:
    """``valid_rows``: (5, 13) clean SoSt rows; ``valid_ids``: their exact Gaia DR2 IDs.
    ``id_rows``/``id_ids``: a larger random SoSt sample for the identifier round-trip case."""
    workdir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(valid_rows[:5], float)
    ids = [int(v) for v in valid_ids[:5]]
    ref = predict(x)  # batch reference
    inference = inference or yoda_inference
    cases = []

    def case(name, kind, rows=None, raw_text=None, valid=False, check=None):
        f_in, f_out = workdir / f"{name}_in.csv", workdir / f"{name}_out.csv"
        if raw_text is not None:
            f_in.write_text(raw_text)
        else:
            _write(f_in, rows)
        rec = {"case": name, "kind": kind, "valid_input": valid}
        try:
            out = inference(f_in, f_out, predict)
            probs = out[:, -3:] if out.ndim == 2 else np.empty((0, 3))
            finite = bool(np.isfinite(probs).all()) and probs.size > 0
            rec["n_out"] = int(len(probs))
            rec["outcome"] = "finite probabilities" if finite else ("NaN/non-finite output" if probs.size else "empty output")
            extra = check(out, probs) if check else None
            if extra:
                rec.update(extra)
            if valid:
                rec["verdict"] = rec.pop("verdict_override", "PASS")
            elif "verdict_override" in rec:
                rec["verdict"] = rec.pop("verdict_override")
            else:
                rec["verdict"] = "FAIL-silent" if finite else "PASS"
        except Exception as exc:
            rec["outcome"] = f"raised {type(exc).__name__}: {str(exc)[:160]}"
            rec["traceback"] = traceback.format_exc()[-1500:]
            rec["verdict"] = "FAIL-crash" if valid else "PASS"
        cases.append(rec)

    base = [_row(ids[i], x[i]) for i in range(5)]
    # 1-3 non-finite values
    for nm, val in [("nan_W3", "nan"), ("posinf_W4", "inf"), ("neginf_G", "-inf")]:
        r = [list(b) for b in base]
        col = {"nan_W3": 3 + 11, "posinf_W4": 3 + 12, "neginf_G": 3 + 4}[nm]
        r[0][col] = val
        case(nm, "non-finite", r)
    # 4 missing band: empty field
    r = [list(b) for b in base]; r[0][3 + 8] = ""
    case("missing_value_H", "missing band (empty field)", r)
    # 5 missing band: whole column absent (12 characteristics)
    case("missing_column_W1", "missing band (column absent)", [b[:3 + 9] + b[3 + 10:] for b in base])
    # 6-8 out-of-range magnitudes / colours (finite but physically impossible)
    for nm, col, val in [("W3_mag_99", 3 + 11, 99.0), ("G_mag_minus50", 3 + 4, -50.0), ("rHa_colour_50", 3 + 0, 50.0),
                         ("Gvar_negative", 3 + 1, -5.0)]:
        r = [list(b) for b in base]; r[0][col] = val
        case(nm, "out-of-range value", r)
    # 9 wrong column order: J and W4 swapped for all rows
    r = [list(b) for b in base]
    for b in r:
        b[3 + 6], b[3 + 12] = b[3 + 12], b[3 + 6]
    case("column_order_J_W4_swapped", "wrong column order", r)
    # 10 wrong dtype
    r = [list(b) for b in base]; r[0][3 + 5] = "bright"
    case("dtype_text_in_mag", "wrong dtype", r)
    # 11 empty input
    case("empty_input", "empty input", raw_text=HEADER + "\n")
    # 12 single valid row: should equal batch output
    def chk_single(out, probs):
        same = probs.shape == (1, 3) and np.allclose(probs[0], ref[0], atol=1e-5)
        return {"equals_batch": bool(same), "verdict_override": "PASS" if same else "FAIL-crash"}
    case("single_row", "single row (valid)", [base[0]], valid=True, check=chk_single)
    # 13 duplicate source IDs (same Gaia ID, different photometry)
    r = [list(b) for b in base]; r[1][0] = r[0][0]
    def chk_dup(out, probs):
        return {"note": "duplicate ID returned twice with different probabilities, no flag"
                if len(probs) == 5 else "rows dropped"}
    case("duplicate_source_id", "duplicate IDs", r, check=chk_dup)
    # 14 identifier integrity: 19-digit Gaia DR2 IDs through YODA.py's loadtxt/savetxt
    big_ids = [int(v) for v in id_ids]
    def chk_ids(out, probs):
        got = _first_column_ids(workdir / "gaia_id_roundtrip_out.csv")
        altered = [(str(a), str(b)) for a, b in zip(big_ids, got) if a != b]
        return {"n_ids": len(big_ids), "n_altered": len(altered), "examples": altered[:10],
                "verdict_override": "FAIL-silent" if altered else "PASS"}
    case("gaia_id_roundtrip", "identifier integrity (YODA.py loadtxt/savetxt)",
         [_row(big_ids[i], id_rows[i]) for i in range(len(big_ids))], check=chk_ids)
    return cases


def run_bootgather(original_dir: Path, workdir: Path, ids: list[int], probs: np.ndarray) -> dict:
    """Run the ORIGINAL Bootgather.py on 30 files laid out as YODA.py writes them (np.savetxt, no
    header, 21 columns: 18 input columns + 3 probabilities)."""
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(original_dir / "Bootgather.py", workdir / "Bootgather.py")
    n = len(ids)
    for v in range(30):
        rows = np.zeros((n, 21))
        rows[:, 0] = np.array(ids, dtype=float)       # what YODA.py's loadtxt/savetxt produce
        rows[:, 1] = 10.0 + np.arange(n) * 0.01        # distinct ra
        rows[:, 2] = 60.0
        rows[:, 18:21] = probs
        np.savetxt(workdir / f"Output_Input_sample_v{v}.csv", rows, delimiter=",")
    proc = subprocess.run([sys.executable, "Bootgather.py"], cwd=workdir, capture_output=True, text=True,
                          timeout=600)
    rec = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-800:]}
    res = workdir / "Output_Input_sample_mean_average.csv"
    if res.exists():
        import csv
        rows = list(csv.reader(open(res)))[1:]
        got = [int(float(r[0])) for r in rows]
        rec["n_sources_in"] = n
        rec["n_sources_out"] = len(rows)
        idset = set(ids)
        altered = [(r[0], int(float(r[0]))) for r in rows if int(float(r[0])) not in idset]
        rec["n_ids_altered"] = len(altered)
        rec["examples_altered"] = [a for a, _ in altered[:10]]
        # all 30 iterations carry identical probabilities, so a correct average equals the input
        got_p = np.array([float(r[4]) for r in rows])
        rec["max_abs_error_mean_pPMS"] = float(np.max(np.abs(np.sort(got_p) - np.sort(probs[:, 1])))) \
            if len(rows) == n else None
        rec["n_iterations_averaged_note"] = ("v0 has no header but Bootgather skips its first line "
                                             "(next(csvreader0)), so 1 source loses iteration 0")
    rec["verdict"] = "FAIL-silent" if rec.get("n_ids_altered", 0) else ("PASS" if proc.returncode else "PASS")
    return rec


def run_original_yoda_script(original_dir: Path, workdir: Path) -> dict:
    """Execute the unmodified YODA.py (with tiny dummy inputs) to record whether it runs as published."""
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(original_dir / "YODA.py", workdir / "YODA.py")
    proc = subprocess.run([sys.executable, "YODA.py"], cwd=workdir, capture_output=True, text=True, timeout=600)
    return {"returncode": proc.returncode, "stderr_tail": proc.stderr[-600:]}


def save(cases: list[dict], path: Path) -> None:
    path.write_text(json.dumps(cases, indent=1, default=str))
