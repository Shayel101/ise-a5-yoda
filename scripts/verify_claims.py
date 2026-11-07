#!/usr/bin/env python3
"""Independent re-verification of every headline number quoted in README.md, report.md and
submission_summary.md.

The numbers are recomputed straight from the raw files (metrics.csv, per_source/*.npz, *.json) with
plain numpy/pandas. The a5.analysis module is deliberately NOT used, so that this check is
independent of the notebook code. The CLAIMED values below are copied from the documents and are the
things under test. Any mismatch beyond rounding is printed as FAIL.

    python scripts/verify_claims.py  ->  results/verification/claims_check.txt
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
B, AF = ROOT / "results/baseline", ROOT / "results/after"
cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
SEEDS = cfg["model_seeds"]
lines, fails = [], 0


def cell(i, c):
    return f"c{i:02d}_{c['fault']}_{c['rate']}" + (f"_k{c['k']}" if "k" in c else "")


NAMES = [cell(i, c) for i, c in enumerate(cfg["cells"])]


def check(label, got, claimed, tol):
    global fails
    ok = abs(got - claimed) <= tol
    fails += 0 if ok else 1
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {label}: recomputed {got:.4f} vs claimed {claimed} (tol {tol})")


def flip(run_dir, name, cls, set_="cand"):
    num = den = 0
    for s in SEEDS:
        c0 = np.load(run_dir / "per_source" / f"{s}_{NAMES[0]}.npz")[f"{set_}_cls"]
        c1 = np.load(run_dir / "per_source" / f"{s}_{name}.npz")[f"{set_}_cls"]
        m = c0 == cls
        num += int((c1[m] != cls).sum()); den += int(m.sum())
    return num / den


def drecall(run_dir, name):
    d = []
    for s in SEEDS:
        a = np.load(run_dir / "per_source" / f"{s}_{NAMES[0]}.npz")
        b = np.load(run_dir / "per_source" / f"{s}_{name}.npz")
        y = a["test_y"] == 1
        d.append(((b["test_p"][:, 1] >= .5)[y].mean() - (a["test_p"][:, 1] >= .5)[y].mean()))
    return float(np.mean(d))


R = {"F1r": "c03_F1_resample_0.44", "F1n": "c07_F1_noise_0.44", "F2": "c10_F2_allwise_swap_0.023",
     "F3": "c12_F3_iphas_swap_0.013"}
# ---- C1 baseline realistic cells (README 5.1 / report 5)
for key, (pms, cbe, dr) in {"F1r": (10.5, 9.5, -8.9), "F1n": (3.0, 4.7, -2.2), "F2": (1.9, 1.9, -2.4),
                            "F3": (0.4, 0.2, 0.0)}.items():
    check(f"baseline {R[key]} PMS flip %", flip(B, R[key], 1) * 100, pms, 0.05)
    check(f"baseline {R[key]} CBe flip %", flip(B, R[key], 2) * 100, cbe, 0.05)
    check(f"baseline {R[key]} dRecall pp", drecall(B, R[key]) * 100, dr, 0.05)
# ---- after
check("after F1-resample PMS flip %", flip(AF, R["F1r"], 1) * 100, 13.4, 0.05)
check("after F1-resample CBe flip %", flip(AF, R["F1r"], 2) * 100, 13.1, 0.05)
check("after F1-noise CBe flip %", flip(AF, R["F1n"], 2) * 100, 5.6, 0.05)
# ---- dose response quoted in report
check("baseline F1-resample 10% PMS flip %", flip(B, "c01_F1_resample_0.1", 1) * 100, 2.4, 0.05)
check("baseline F1-resample 28% PMS flip %", flip(B, "c02_F1_resample_0.28", 1) * 100, 6.6, 0.05)
check("baseline F1-resample 72% PMS flip %", flip(B, "c04_F1_resample_0.72", 1) * 100, 17.5, 0.05)
check("baseline F2 5% PMS flip %", flip(B, "c11_F2_allwise_swap_0.05", 1) * 100, 4.5, 0.05)
check("baseline F4 k5 10% CBe flip %", flip(B, "c17_F4_fluxerr_0.1_k5.0", 2) * 100, 2.7, 0.05)
# ---- flip decomposition (after, F1-resample 44 %)
rej = chg = den = 0
for s in SEEDS:
    c0 = np.load(AF / "per_source" / f"{s}_{NAMES[0]}.npz")["cand_cls"]
    c1 = np.load(AF / "per_source" / f"{s}_{R['F1r']}.npz")["cand_cls"]
    m = c0 == 1
    rej += int((c1[m] == -1).sum()); chg += int(((c1[m] != 1) & (c1[m] != -1)).sum()); den += int(m.sum())
check("after F1-resample PMS rejected %", rej / den * 100, 5.9, 0.05)
check("after F1-resample PMS changed class %", chg / den * 100, 7.5, 0.05)
# ---- counterexamples: seed 11 flip count, all flips injected
a = np.load(B / "per_source" / f"11_{NAMES[0]}.npz"); b = np.load(B / "per_source" / f"11_{R['F1r']}.npz")
fl = np.flatnonzero(a["cand_cls"] != b["cand_cls"])
check("seed-11 F1-resample flipped candidates", len(fl), 1011, 0)
nonin = 0
for nm in NAMES[1:]:
    for s in SEEDS:
        a_, b_ = np.load(B / "per_source" / f"{s}_{NAMES[0]}.npz"), np.load(B / "per_source" / f"{s}_{nm}.npz")
        for k in ["test", "cand", "rand"]:
            f_ = np.flatnonzero(a_[f"{k}_cls"] != b_[f"{k}_cls"])
            nonin += len(np.setdiff1d(f_, b_[f"{k}_idx"]))
check("flips on NON-injected rows (all baseline cells, seeds, sets)", nonin, 0, 0)
# ---- injected-rate exactness on the real evaluation sets
bad = 0
for nm, c in zip(NAMES[1:], cfg["cells"][1:]):
    for s in SEEDS:
        z = np.load(B / "per_source" / f"{s}_{nm}.npz")
        for k in ["test", "cand", "rand"]:
            n = len(z[f"{k}_cls"])
            bad += int(len(z[f"{k}_idx"]) != round(c["rate"] * n))
check("cells whose injected count != round(rate*n)", bad, 0, 0)
# ---- clean metrics
m = pd.read_csv(B / "metrics.csv"); ma = pd.read_csv(AF / "metrics.csv")
t0 = m[(m.cell == 0) & (m.set == "test")]; ta0 = ma[(ma.cell == 0) & (ma.set == "test")]
check("clean recall PMS %", t0.recall_PMS.mean() * 100, 74.9, 0.05)
check("clean precision PMS %", t0.precision_PMS.mean() * 100, 37.9, 0.05)
check("after clean precision PMS %", ta0.precision_PMS.mean() * 100, 38.1, 0.05)
r0 = m[(m.cell == 0) & (m.set == "rand")].n_PMS_cand.mean(); ra0 = ma[(ma.cell == 0) & (ma.set == "rand")].n_PMS_cand.mean()
check("clean N_PMS per 100k baseline", r0, 528.7, 0.05)
check("clean N_PMS per 100k after", ra0, 518.0, 0.05)
tr = json.loads((B / "training.json").read_text())
check("min PMS per test split", min(t["n_test_by_label"]["1"] for t in tr), 14, 0)
check("max PMS per test split", max(t["n_test_by_label"]["1"] for t in tr), 15, 0)
meta = json.loads((B / "run_meta.json").read_text())
check("labelled in SoSt", meta["n_labelled_in_sost"], 187, 0)
check("baseline wall s", meta["wall_seconds"], 647, 1)
check("after wall s", json.loads((AF / "run_meta.json").read_text())["wall_seconds"], 555, 1)
# ---- C2
c2b, c2a = json.loads((B / "c2_boundary.json").read_text()), json.loads((AF / "c2_boundary.json").read_text())
fs = [c["case"] for c in c2b["cases"] if c["verdict"] == "FAIL-silent"] + \
     (["bootgather"] if c2b["bootgather"]["verdict"] == "FAIL-silent" else [])
check("baseline FAIL-silent cases", len(fs), 8, 0)
check("baseline YODA.py roundtrip altered IDs", [c for c in c2b["cases"] if c["case"] == "gaia_id_roundtrip"][0]["n_altered"], 636, 0)
check("baseline Bootgather altered IDs", c2b["bootgather"]["n_ids_altered"], 791, 0)
check("after FAIL-silent cases", sum(c["verdict"] == "FAIL-silent" for c in c2a["cases"]), 0, 0)
check("after roundtrip altered IDs", [c for c in c2a["cases"] if c["case"] == "gaia_id_roundtrip"][0]["n_altered"], 0, 0)
sup = json.loads((AF / "supplementary_bootgather_unflagged.json").read_text())
check("supplementary Bootgather altered IDs", sup["n_ids_altered"], 790, 0)
check("C2 cases total (incl. Bootgather)", len(c2b["cases"]) + 1, 16, 0)
# ---- side effects / determinism
se = json.loads((AF / "side_effects.json").read_text())
fr = se["false_rejections_clean"]
cand = [fr[f"seed{s}_cand"]["rate"] * 100 for s in SEEDS]
check("false rejection cand min %", min(cand), 1.6, 0.05)
check("false rejection cand max %", max(cand), 2.3, 0.05)
check("labelled PMS/CBe rejected in clean test", sum(fr[f"seed{s}_test_by_label"]["1"] + fr[f"seed{s}_test_by_label"]["2"] for s in SEEDS), 0, 0)
check("max prob diff after vs baseline (unflagged)", max(se["max_abs_prob_diff_unflagged_vs_baseline"].values()), 1.2e-7, 1e-9)
check("class changes after vs baseline (unflagged)", sum(se["class_changes_unflagged_vs_baseline"].values()), 0, 0)
# ---- agreement
ag = pd.read_csv(B / "agreement_full.csv")
for col, v in [("published_recovered_PMS", 0.64), ("published_recovered_CBe", 0.95), ("spearman_PMS", 0.58),
               ("spearman_CBe", 0.39), ("jaccard_PMS", 0.19), ("jaccard_CBe", 0.07)]:
    check(f"agreement mean {col}", ag[col].mean(), v, 0.005)
# ---- catalogue facts recomputed from the raw VizieR files
# NOTE: data/raw/{pms,cbe}.tsv hold only VizieR's default columns (the '-out.all' parameter was ignored,
# see results/verification/verification_report.md), so the W3W4 flag is taken from the sample table.
sys.path.insert(0, str(ROOT / "src"))
from a5.data import load_candidates, load_sample  # noqa: E402
smp = load_sample()[["GaiaDR2", "FW3W4"]]
cands = load_candidates()
for t, claimed, n_claimed in [("pms", 43.5, 8470), ("cbe", 27.8, 693)]:
    sub = smp[smp.GaiaDR2.isin(cands[t])]
    check(f"{t} table rows", len(cands[t]), n_claimed, 0)
    check(f"{t} candidates with W3W4 flag %", (sub.FW3W4.str.strip() == "W3W4").mean() * 100, claimed, 0.05)
# ---- every "x % [lo, hi]" CI printed in README / report tables matches the computed CI (1 dp)
tabs = {k: pd.read_csv(ROOT / f"report/tables/cells_{k}.csv").set_index("cell") for k in ["baseline"]}
rowkey = {"F1-resample 44 %": R["F1r"], "F1-noise 44 %": R["F1n"], "F2 AllWISE 2.3 %": R["F2"], "F2 2.3 %": R["F2"],
          "F3 IPHAS 1.3 %": R["F3"], "F3 1.3 %": R["F3"]}
for doc in ["README.md", "report/report.md"]:
    for line in (ROOT / doc).read_text().splitlines():
        cells_ = [c.strip() for c in line.split("|")]
        if len(cells_) > 4 and cells_[1] in rowkey:
            t = tabs["baseline"].loc[rowkey[cells_[1]]]
            for col, cls in [(2, "PMS"), (3, "CBe")]:
                mm = re.match(r"\**([\d.]+) %\** \[([\d.]+), ([\d.]+)\]", cells_[col])
                if mm:
                    want = [round(100 * t[f"flip_{cls}_cand{x}"], 1) for x in ["", "_lo", "_hi"]]
                    got = [float(v) for v in mm.groups()]
                    ok = all(abs(a - b) < 0.051 for a, b in zip(got, want))
                    fails += 0 if ok else 1
                    lines.append(f"[{'PASS' if ok else 'FAIL'}] {doc} '{cells_[1]}' {cls} printed {got} vs computed {want}")
# ---- summary word count
words = len(re.findall(r"\S+", (ROOT / "submission_summary.md").read_text()))
check("submission summary words (150-300)", words, 248, 0)

out = ROOT / "results/verification/claims_check.txt"
out.write_text("\n".join(lines) + f"\n\n{len(lines) - fails}/{len(lines)} checks passed\n")
print("\n".join(lines)); print(f"\n{len(lines) - fails}/{len(lines)} checks passed")
sys.exit(1 if fails else 0)
