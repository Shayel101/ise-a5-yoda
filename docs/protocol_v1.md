# Protocol v1 (frozen before any fault injection)

## 1. Baseline and environment
- Baseline code: this repository at tag `a5-env` plus the files added by the protocol commit. The run
  scripts are committed with the protocol and are not changed afterwards, except for bug fixes, which
  are logged in `results/baseline/rerun_log.md` with the reason.
- Original system for reference: github.com/MVioque/YODA @ `0c8097df40b8ba35fdb6a6b6f35d93fce054bd1f`.
- Environment: `uv.lock` (Python 3.12.15, TensorFlow 2.16.2 / Keras 3.15.1), CPU only, op determinism
  on (`results/env.json`).
- Data: VizieR J/A+A/638/A21 tables, verified against `data/checksums.json`. Labels come from SIMBAD,
  verified against `data/labels_checksum.json` (7,957 rows, sha256 `ffe875e7…`).

## 2. Model (reproduction, quick mode; see reproduction_decision.md Revision 1)
- Labelled sources = SIMBAD `Ae*`/`TT*` (PMS) and `Be*` (CBe) with a Gaia DR2 ID present in the SoSt
  table. A source matching both classes is dropped.
- Per model seed s:
  - **Labelled set.** Bootstrap-resample it (with replacement, same size, `random_state=s`).
  - **"Other" class.** Draw 47,026 random SoSt sources (`random_state=s`), excluding labelled sources
    and published candidates.
  - **Split.** Split 90/10 into train and test (`random_state=s`). Keras `validation_split=0.1` is the
    CV set.
  - **Training.** Max 300 epochs, early-stopping patience 50 on validation PMS precision (p ≥ 0.5).
- **Model seeds: 11, 22, 33** (3 bootstrap models; the paper used 30). Justification: 1 CPU, same-day
  budget. Training cost was measured at 26 s/epoch at full size and is estimated at ≈ 3 s/epoch at
  1/10 size.

## 3. Evaluation sets (fixed per seed)
- **E_test:** the model's held-out test split (labelled + other). Used for recall and precision.
- **E_cand:** all published candidates (VizieR pms ∪ cbe ∪ either, 10,472 IDs) that are present in SoSt
  and not used in training. Used for the C1(a) flip rates.
- **E_rand:** 100,000 random SoSt sources not used in training (`random_state=1000+s`). Used for the
  candidate-count deltas and as the **donor / resampling pool** for F1-resample, F2 and F3. Mismatched
  photometry comes from ordinary field sources, not from other candidates.

## 4. Grid (fault × rate × seed); every cell runs, none added or removed
| Fault | Parameter | Rates |
|---|---|---|
| clean | — | 0 |
| F1-resample | pool = E_rand (W3, W4) pairs | 0.10, 0.28, 0.44, 0.72 |
| F1-noise | σ = 1.0 mag | 0.10, 0.28, 0.44, 0.72 |
| F2 | donors = E_rand | 0.01, 0.023, 0.05 |
| F3 | donors = E_rand | 0.013, 0.03 |
| F4 | k = 2 | 0.05, 0.10 |
| F4 | k = 5 | 0.05, 0.10 |

There are 18 fault cells × 3 seeds = 54 faulted evaluations plus 3 clean evaluations. Each fault draw
uses the fault seed `model_seed*100000 + cell_index*10 + set_index`, where set_index is 0 for E_test,
1 for E_cand and 2 for E_rand. Python's `hash()` is not used.

**Paper-realistic cells** (used to decide C1): F1-resample 0.44, F1-noise 0.44, F2 0.023, F3 0.013.

## 5. Oracles
1. **Flip rate** on E_cand and on E_test. A source flips if its class at p ≥ 0.5 (PMS / CBe / other;
   `a5.pipeline.classify`) differs between the faulted and the clean run of the **same model**.
   It is reported separately for the clean classes PMS, CBe and other.
2. **ΔRecall_PMS and ΔPrecision_PMS** on E_test (faulted − clean, same model). Precision is a lower
   limit (paper Appx. B.1).
3. **ΔN candidates** (p ≥ 0.5 PMS and CBe) on E_rand and on E_cand.
4. **Transition matrices** (clean class → faulted class), summed over seeds.
5. **C2.** The F5 suite is implemented in `src/a5/boundary.py` and `src/a5/run_boundary.py`, using
   model seed 11. It runs a line-by-line copy of YODA.py's generalisation step: np.loadtxt of an
   Input_sample-style CSV, then predict, then np.savetxt. It also runs the **unmodified original**
   `Bootgather.py` and `YODA.py` fetched at the pinned commit.
   - Malformed cases: NaN W3, +inf W4, −inf G, empty field, missing column, W3 = 99, G = −50,
     r−Hα = 50, Gvar < 0, J/W4 column order swapped, text in a magnitude, empty input, duplicate
     source ID.
   - Identifier-integrity cases: 1,000 random SoSt IDs through YODA.py's loadtxt/savetxt, and the
     same IDs through the original Bootgather.py.
   - Valid case: a single row.
   
   Verdicts:
   - **PASS:** an explicit error, or NaN or flagged output.
   - **FAIL-silent:** finite probabilities, or an altered identifier, returned with no error or flag.
   - **FAIL-crash:** a crash on *valid* input. It is reported as a defect but does not falsify C2.
   
   The pilot run of this suite is disclosed in `results/pilot_smoke/README.md`.
6. **Reproduction agreement.** This is reported and is not a gate for C1. It is measured on E_rand,
   averaged over seeds, against the published probabilities:
   - Jaccard overlap of the p ≥ 0.5 PMS sets and of the CBe sets;
   - Spearman ρ of p(PMS) and of p(CBe).

## 6. Decision rules (thresholds)
**C1 is falsified** if, in **any** paper-realistic cell, either of the following holds:
- (a) the mean over seeds of the PMS or CBe flip rate on E_cand exceeds **5 %** and the lower bound of
  its 95 % bootstrap CI exceeds 5 %; or
- (b) |mean ΔRecall_PMS| on E_test exceeds **2.8 percentage points** (2 × the paper's ±1.4) and its
  95 % bootstrap CI excludes 0.

The CI excluding 0 is added because 3 models give a noisy recall. F4 cells and the non-realistic rates
are reported as dose–response evidence only.

**C2 is falsified** if **any** F5 case is FAIL-silent.

## 7. Statistics
- **Flip rates.** Percentile bootstrap (2,000 resamples, seed 0) over sources, with per-seed indicator
  vectors pooled. This ignores between-model variance, so the per-seed values are reported alongside.
- **ΔRecall.** Paired per seed. CI by bootstrapping the test-set PMS sources within each seed
  (2,000 resamples), averaged over seeds.

## 8. Stopping rule and compute cap
- Every grid cell runs to completion. There is no early stop on favourable results and no cell is
  added or removed after seeing results.
- Compute cap: **3 h wall-clock** for the baseline (training + grid). If the cap is hit, the cells
  not run are reported as not run.
- Crashes are kept with their tracebacks in `results/baseline/failures/`. There are no silent retries,
  and any rerun is logged with its reason.

## 9. Improvement phase
Exactly one improvement is chosen from `docs/improvement_candidates.md` (written from baseline
failures only). The **identical** grid, seeds, data and evaluation sets are rerun into `results/after/`.

## 10. Planned outputs
- **Tables:** per-cell before/after metrics; the C2 pass/fail matrix; the reproduction-agreement
  table; counterexample list (Gaia DR2 IDs that flipped in realistic cells, with their 13 observables
  before and after the fault).
- **Figures:** flip rate vs fault rate (one panel per fault, PMS and CBe, 95 % CI); ΔRecall vs rate;
  transition matrices for the realistic cells.
