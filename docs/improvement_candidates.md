# Improvement candidates (written from the baseline failures only, tag `a5-baseline`)

## Baseline failures that motivate them
1. **C1 falsified, F1-resample at 44 %:**
   - PMS flip rate 10.5 % [10.0, 11.0];
   - CBe flip rate 9.5 % [8.8, 10.2];
   - ΔRecall_PMS −8.9 pp [−17.8, −2.2].
   
   The other realistic cells pass. The F1-noise and F2/F3 flips stay below 5 %.
2. **C2 falsified.** Eight cases are FAIL-silent:
   - W3 = 99, G = −50, r−Hα = 50, Gvar < 0;
   - J/W4 columns swapped;
   - duplicate source ID;
   - Gaia ID integrity: YODA.py's own loadtxt/savetxt altered 636/1000 IDs, and the original
     Bootgather.py altered 791/1000.
3. **FAIL-crash:** a single valid row crashes YODA.py's inference step (1-D array indexing).

## Candidates
| # | Candidate | Addresses | Cost | Risk / side effects | Expected effect |
|---|---|---|---|---|---|
| **A** | **Input-validation and identifier-integrity guard** on the inference path (`src/a5/guard.py`). It: (i) reads IDs as int64 and writes them with `%d`; (ii) uses `np.atleast_2d` so one row works; (iii) rejects non-finite rows; (iv) flags rows whose 48 observables fall outside the [0.01, 99.99] percentile range of the training data plus a 1-mag margin, or that repeat a source ID. Flagged rows get NaN probabilities and a flag code, never a silent class. | C2 (all 7), single-row crash | Low. No retraining, about 100 lines. | Valid but extreme sources can be wrongly rejected (we measure the rate on clean E_cand and E_rand). A J/W4 swap is caught only if it produces out-of-range colours. It does **not** touch C1. | Most C2 FAIL-silent become PASS. ID corruption becomes 0. C1 is unchanged. |
| B | **Train-time W3/W4 corruption augmentation**: add a copy of the training set in which (W3, W4) are corrupted with Gaussian noise (σ = 1 mag) at 44 %. Using the noise model, not resampling, avoids training on the exact test fault. | C1 (F1) | Medium. Retrain 3 models (~10 min). | Circularity: still a synthetic fault, so a gain may not transfer to real spurious W3/W4. It may reduce clean recall or precision, or shift candidate counts. It does not fix C2. | Lower F1 flip rates. Uncertain whether they drop below 5 %. |
| C | **Use the AllWISE ext flag (W3W4) as a feature or sample weight** | C1 (F1) | Medium–high. New feature and retraining; the flag must be available at inference. | Our fault injection does not change flags, so the test cannot show an effect. That makes the result uninformative by construction. | Not measurable with this protocol. |
| D | **Post-hoc probability calibration** (temperature scaling on the CV set) | Probability interpretability | Low | It barely changes the argmax at p ≥ 0.5, so it does not address flips or C2. | ≈ no effect on the failures found. |
| E | **Drop the W3/W4 colours** (W1/W2 only) | C1 (F1) | Medium. Retrain. | The paper (Sect. 5.4) identifies the mid-IR as key to separating PMS from CBe, so clean performance is likely to drop sharply. | Removes F1 sensitivity by construction, at a probable large clean-accuracy cost. |

## Recommendation
**A** is cheap, fully reviewable and addresses the largest set of baseline failures (C2 plus the crash).
Its effect can be measured honestly with the identical protocol.

**B** is the only realistic candidate for C1. However, it trains on a synthetic fault close to the one
tested, so any improvement it shows is partly circular.

## Decision (student, 2026-11-07): **A**
Rejected candidates and the reasons:
- **B:** it would have targeted the C1 failure, but it trains on a synthetic fault close to the tested
  one (circular evidence) and needs retraining. Not chosen. It remains the main recommendation for
  C1 (see residual_risk.md).
- **C:** the effect cannot be measured with this protocol, because the faults do not alter AllWISE flags.
- **D:** it does not change the argmax at p ≥ 0.5, so no failure would be addressed.
- **E:** a likely large loss of clean PMS/CBe separation (paper Sect. 5.4).

Implementation: `src/a5/guard.py`, wired in with the `guard: true` switch (`configs/after.yaml`).
- It does **not** modify the original `Bootgather.py`, which stays out of scope. Its ID corruption
  therefore remains in the after run and is a residual risk.
- Guard parameters were fixed before the after run: range = [0.01, 99.99] percentile of each model's
  training observables ± 1.0, as stated in candidate A.
