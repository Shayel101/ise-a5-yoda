# ISE-A5: Quality & Security Counterexample Study of the YODA ANN

This study tries to falsify the fault-tolerance claims made for the YODA artificial neural network
(Vioque et al. 2020, A&A 638, A21, doi:10.1051/0004-6361/202037731; code at github.com/MVioque/YODA,
commit `0c8097d`).

- Report: `report/report.pdf`.
- Result summary: `submission_summary.md`.
- AI use: `AI_USE_LOG.md`.

> **The model studied is a faithful re-implementation (Route C), not the authors' trained network.**
> No weights or training labels were published (`docs/reproduction_decision.md`).

## Repository map
| Path | Content |
|---|---|
| `docs/reproduction_decision.md` | Phase 0 inventory, route choice, Revision 1 (quick mode) |
| `docs/threat_model.md` | asset, fault classes F1–F5, out-of-scope items |
| `docs/protocol_v1.md` | frozen protocol (SHA256 in `results/protocol_hash.txt`, tag `a5-protocol-frozen`) |
| `docs/improvement_candidates.md` | 5 scored candidates, the choice (A) and why the others were rejected |
| `docs/residual_risk.md` | risk register, recovery and rollback plan |
| `src/a5/` | pipeline re-implementation, faults, oracles, guard (improvement), runners, analysis |
| `tests/` | pytest suite: exact fault rates, determinism, no input mutation, toy oracle example, guard |
| `configs/` | `baseline.yaml` and `after.yaml` (identical grid; only the `guard` switch differs) |
| `results/` | `env.json`, `protocol_hash.txt`, `baseline/`, `after/` (raw per-cell CSV / NPZ / JSON, logs), `pilot_smoke/` (disclosed pre-freeze debug run), `verification/` |
| `notebooks/a5_analysis.ipynb` | analysis; reads raw results only |
| `report/` | `report.md` → `report.pdf` |

Git tags, in order: `a5-env`, `a5-protocol-frozen`, `a5-baseline`, `a5-improvement`, `a5-final`.

## Reproduce from clone to PDF
Tested on macOS 26 x86_64 (Intel i7-9750H, 16 GB RAM), CPU only. Linux x86_64 is also covered by the
lock file.

```bash
git clone <repo-url> ise-a5-yoda && cd ise-a5-yoda
git checkout a5-final

# 1. Environment: Python 3.12, exact versions from uv.lock
pip install uv==0.13.7
uv sync --frozen

# 2. Data: about 1 GB from a VizieR mirror (CDS Strasbourg may be unreachable). About 15-30 min
#    with 6 workers. Checksums are verified against data/checksums.json.
for i in 0 1 2 3 4 5; do uv run python scripts/download_data.py --shard $i --nshards 6 & done; wait
uv run python scripts/download_data.py --verify
uv run python scripts/build_labels.py   # SIMBAD is a live DB; --verify reports whether it still matches the frozen hash

# 3. Tests
uv run pytest -q

# 4. Experiments: about 11 min each on the machine above
uv run python -m a5.run_experiment configs/baseline.yaml
uv run python -m a5.run_boundary   configs/baseline.yaml
uv run python -m a5.agreement_full configs/baseline.yaml
uv run python -m a5.run_experiment configs/after.yaml
uv run python -m a5.run_boundary   configs/after.yaml
uv run python -m a5.side_effects

# 5. Analysis notebook (runs top to bottom, reading raw results only)
uv run python scripts/make_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/a5_analysis.ipynb

# 6. Report figures + PDF (pandoc 3.6.1 ships with the pinned dev dependency pypandoc-binary;
#    needs a XeLaTeX install, e.g. MacTeX / TeX Live, and the font "Arial Unicode MS")
uv run python scripts/make_figures.py
cd report && $(uv run python -c "import pypandoc;print(pypandoc.get_pandoc_path())") report.md -o report.pdf \
    --pdf-engine=xelatex -V mainfont="Arial Unicode MS" -V monofont="Menlo" && cd ..
```

**Determinism.** Seeds are fixed per run (Python, NumPy, TensorFlow) and TF op determinism is on.
Rerunning the baseline on the same machine reproduces the same models. On different CPUs, floating
point results may differ slightly. The after run retrains with the same seeds, and
`results/after/side_effects.json` compares its clean outputs with the baseline as a determinism check.

**Rollback of the improvement.** `git revert <a5-improvement commit>`, or `git checkout a5-baseline`.
See `docs/residual_risk.md`.

## Licence notes
- YODA has **no licence**. It is cited, not redistributed. The C2 suite downloads the two original
  scripts at the pinned commit into `data/external/` (git-ignored) and runs them unmodified.
- Our code is an independent re-implementation under the MIT licence (see `LICENSE`).
- VizieR and SIMBAD data are used under the CDS terms
  (https://cds.unistra.fr/vizier-org/licences_vizier.html) and are not redistributed. Only the
  checksums are committed.
