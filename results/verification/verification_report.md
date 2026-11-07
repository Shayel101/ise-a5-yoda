# Verification & validation report

Full double-check of the project on 2026-11-07, run after all phases were complete and before tag
`a5-final`. All commands are reproducible, and each check writes or keeps an artefact in this folder.

## A. Verification (did we build and run it right?)

| # | Check | Method | Result |
|---|---|---|---|
| V1 | Unit tests | `pytest -q` | **45 / 45 pass** |
| V2 | Tests can fail | deliberate break (input mutation, self-donor swap) → `deliberate_break_pytest.txt` | 9 tests fail as expected |
| V3 | Raw data integrity | `download_data.py --verify` | **211 / 211 files** match `data/checksums.json` |
| V4 | Label integrity | `build_labels.py --verify` | SIMBAD labels match the frozen SHA256 |
| V5 | Protocol not edited after freeze | `shasum -c results/protocol_hash.txt`; `git diff a5-protocol-frozen HEAD -- docs/protocol_v1.md configs/baseline.yaml` | hashes OK, empty diff; the last commit touching the protocol is the freeze commit |
| V6 | Freeze happened before fault injection | tag/commit times: freeze 12:37, baseline `metrics.csv` written 13:03, `a5-baseline` 13:07 | order correct; the only pre-freeze injection was the disclosed smoke run (`results/pilot_smoke/`) |
| V7 | "Identical" rerun after the improvement | `diff configs/baseline.yaml configs/after.yaml` | only `name`, `out_dir`, `guard` and the header comment differ |
| V8 | **Independent full rerun of the baseline** (11.6 min) with the current code into a scratch directory | compare with the committed `results/baseline/` | **bit-identical**: `metrics.csv` max diff 0.0; all 54 `per_source` files have 0 class/index/label differences and 0.0 probability difference; `agreement.csv` equal |
| V9 | Determinism across the improvement | `results/after/side_effects.json` | the after run retrained the same models: max prob diff ≤ 1.2e-7, 0 class changes on unflagged rows |
| V10 | Injected-rate exactness on the real evaluation sets | `verify_claims.py`: `len(idx) == round(rate·n)` for every cell × seed × set | 0 violations |
| V11 | No spurious flips | `verify_claims.py`: flips on rows that were **not** injected, over all cells, seeds and sets | 0 |
| V12 | **Every number quoted in README, report and summary** | `scripts/verify_claims.py`, which recomputes from raw files with plain numpy (no `a5.analysis`), plus a parser that checks every printed "x % [lo, hi]" CI in the README/report tables | **74 / 74 pass** after the corrections below (`claims_check.txt`) |
| V13 | Fresh clone | `git clone` → `uv sync --frozen` → `pytest` in a new venv | 45 / 45 pass |
| V14 | Notebook runs top to bottom | `jupyter nbconvert --execute` with the fresh-clone venv kernel | 13 / 13 code cells executed, 0 errors |
| V15 | Report renders | pandoc 3.6.1 + XeLaTeX | 11 pages, 0 missing glyphs |
| V16 | Links | every relative link in README.md and report.md | 0 broken |
| V17 | References | doi.org handle API + CSL metadata (`doi_check.txt`) | 13 / 13 resolve; author, year and title match |
| V18 | Secrets / personal data | `git grep` for keys, tokens and private keys | none. Commits carry the student's git identity (name + e-mail), which is a publication choice for the student |
| V19 | Summary length | regex word count | 248 words (150–300 required) |

### Defects found by this double-check (and how they were handled)
1. **Mis-rounded numbers in README/report (fixed).**
   - F3 CBe flip rate: printed as 0.3 %; the true value is 0.246 %, so **0.2 %**.
   - Two CI upper bounds were printed one unit low: F1-noise CBe [4.2, 5.1] → **[4.2, 5.2]**; F2 CBe
     [1.6, 2.1] → **[1.6, 2.2]**.
   - None of these changes a verdict. They are now covered by the automatic CI check in V12.
2. **Downloader defect (documented, data not changed).**
   - `download_data.py` requests the three small candidate tables with `-out.all`, but VizieR ignored
     the empty-valued parameter. So `data/raw/{pms,cbe,either}.tsv` contain only the **default columns**
     (IDs, probabilities, a few names), without the flag columns.
   - The experiments use only the Gaia IDs from these files, so **no result is affected**.
   - The 43.5 % / 27.8 % `W3W4` shares had been computed in Phase 0 from a separate scratch query.
     They are now recomputed reproducibly by joining the candidate IDs to the `sample` table, which
     carries `FW3W4`: **43.52 % and 27.85 %, confirmed**.
   - The data files were not re-downloaded, so the frozen checksums stay valid.
3. **Run script changed after the freeze (disclosed, shown harmless).**
   - The improvement commit modified `run_experiment.py`, `run_boundary.py` and `boundary.py` to add the
     `guard` switch.
   - Protocol §1 allows only logged bug fixes. These were additive changes behind a flag that is off
     in the baseline config.
   - V8 shows the modified code reproduces the frozen baseline **bit for bit**. The earlier C2 harness
     regression check gave identical verdicts with the same 636 altered IDs.

## B. Validation (did we test the right thing?)

| Question | Assessment |
|---|---|
| Does the re-implementation match YODA? | Line-by-line port of YODA.py: 48-observable expansion, eigen-ratio PCA rule giving 12 components (paper: 12), 2×580 network, dropout, BN, L2, Adamax, balanced weights, full batch, early stopping with patience 50. **Known deviations:** (i) early stopping monitors Keras `Precision(class_id=1, threshold=0.5)`, which matches the paper text ("p ≥ 50 %"), whereas the abandoned `keras_metrics` may have used argmax; (ii) Keras 3/TF 2.16 instead of Keras 2. |
| Is the model close to the published one? | Only partly. It recovers 64 % of PMS and 95 % of CBe published candidates, with Spearman ρ 0.58 / 0.39 but Jaccard 0.19 / 0.07, and it produces many more candidates. Conclusions are about **this architecture and feature set under these faults**, not about the exact published probabilities. Stated as a limit everywhere. |
| Are the fault rates the paper's? | Yes for 44 % (recomputed from the catalogue), 2.3 % and 1.3 % (paper Sect. 2.4). The 28 % reading was corrected to "reliable" (≈72 % unreliable). F4 and the noise σ are declared assumptions and cannot falsify C1. |
| Is the fault mechanism realistic? | F2/F3 donors are random field sources, as in real mismatches. F1-resample draws real W3/W4 pairs. They are still synthetic and are stated as such. |
| Data leakage? | E_cand and E_rand exclude every model's training IDs. E_rand also excludes all candidates. The guard is fitted on training data only. |
| Is the flip oracle fair to the improvement? | Under the frozen definition, a guard rejection counts as a flip. That is conservative for the improvement, so the decomposition (rejected vs changed class) is reported alongside it and not substituted for it. |
| Statistical power | 3 models; 14–15 PMS per test split, so recall moves in about 7 pp steps and its CIs are wide. Flip-rate CIs are tight (about 17k pooled PMS-candidate evaluations) but ignore between-model variance; per-seed values are in `metrics.csv`. |
| Can the verdicts be trusted? | C1 at F1-resample 44 % passes both rules with large margins (CI lower bound 10.0 % vs 5 %). The C2 failures are deterministic and reproducible from the committed harness, and include runs of the authors' unmodified scripts. |
