# Residual risk and recovery plan

Status after improvement A (input-validation and identifier-integrity guard). The figures below come
from the raw results in `results/baseline` and `results/after`; the notebook recomputes them.

## Risk register
Likelihood and impact are rated for catalogue users selecting follow-up targets.
- **Owner "maintainer":** whoever reruns or redistributes the pipeline. For this study that is the
  student. For the original catalogue it would be the YODA authors.
- **Owner "user":** the astronomer selecting targets.

| # | Remaining failure mode | Evidence | Likelihood | Impact | Detection / monitoring | Owner |
|---|---|---|---|---|---|---|
| R1 | **Spurious W3/W4 flips classes.** At the paper's own 44 % rate, about 10 % of PMS and CBe candidates change class, and PMS recall drops by about 9 pp. **The guard does not fix this**: resampled W3/W4 values are in range, so they pass. | baseline and after, cell `F1_resample 0.44` | High (44 % of PMS candidates carry the W3W4 flag) | High: wrong targets, missed objects | Treat `FW3W4`-flagged candidates as lower confidence; compare p with and without W3/W4; rerun with candidate B | maintainer, user |
| R2 | **Silent ID corruption in the original `Bootgather.py`.** About 79 % of 1,000 IDs were altered. It is out of scope for the guard, which covers only the inference step. | `c2_boundary.json` (before and after) | Certain if the original aggregation is rerun | High: the probability is attached to the wrong star | Assert `int64` ID equality between input and output; never parse IDs with `float()` or the default pandas float parser | maintainer |
| R3 | **Physically implausible but in-range inputs** (e.g. a J/W4 swap that stays inside the training range, or plausible but wrong photometry from a mismatch) still get a confident class. | after C2 table; F2/F3 cells | Medium (F2 at 2.3 % gives about 2 % flips) | Medium | Cross-match residual checks, as the paper already does for `X-mtch` (G vs r, G_RP−J vs J−H) | maintainer |
| R4 | **Guard false rejections**: valid but extreme sources get NaN and no class. | `results/after/side_effects.json` | Low to medium (rate measured) | Medium for rare, extreme objects, which are often the interesting ones | Report the flag counts; review flagged candidates by hand rather than dropping them | maintainer |
| R5 | **Training-label poisoning / misclassification** in SIMBAD, BeSS or literature labels. Not tested. | out of scope (threat model) | Medium | High: systematic bias | Label audits; leave-one-catalogue-out training | maintainer |
| R6 | **Reproduction gap.** Our model is not the authors' model: 187 SIMBAD labels instead of 848 + 775, an "other" class at 1/10 size, 3 models instead of 30. The size of flip effects on the *published* model may differ. | `agreement_full.csv`: Jaccard 0.17–0.24 (PMS), 0.05–0.10 (CBe) | — | Limits external validity | Rerun this protocol with the original labels or weights if the authors release them | maintainer |
| R7 | **Model instability across bootstrap seeds.** Clean PMS candidate counts vary about 4× between seeds. | `agreement_full.csv` | High | Medium | Average over 30 or more models (as the paper does) and report e_pPMS | maintainer |
| R8 | **Original `YODA.py` does not run as published**: `keras_metrics` is missing and `train_test_split` is never imported. | C2 suite | Certain | Low (reproducibility) | Pin dependencies; add a smoke test to CI | maintainer |

## Recovery and rollback plan
1. **Revert the improvement:**
   - `git revert <commit of a5-improvement>` removes the guard while keeping the history;
   - or `git checkout a5-baseline` for the exact pre-improvement state;
   - or set `guard: false` in the config, since the guard is a runtime switch.
   
   The baseline results remain in `results/baseline/` and are never overwritten.
2. **How catalogue users should treat affected candidates:**
   - **`W3W4`-flagged candidates** (44 % of PMS, 27 % of CBe): treat the class as provisional. The
     baseline shows that about 1 in 10 would change class under realistic spurious mid-IR. Prioritise
     unflagged candidates for expensive spectroscopy, or confirm flagged ones with an Hα spectrum first.
   - **`ID AllW`, `ID IPH/VPH` and `X-mtch` candidates:** check the cross-match by eye. A 2.3 %
     mismatch rate flips about 2 % of classes.
   - **IDs:** always join on the exact 19-digit `GaiaDR2` string or int64, never on a float column.
3. **Triggers that should prompt a rerun of this protocol:**
   - new input data (Gaia DR3/DR4, unWISE/CatWISE replacing AllWISE, VPHAS+ DR updates);
   - any change to the model, the features or the guard thresholds;
   - publication of the authors' weights or labelled set;
   - a change in the `W3W4` flag fraction of the candidates by more than ±5 pp;
   - a downstream report of a candidate whose class contradicts its spectrum.
