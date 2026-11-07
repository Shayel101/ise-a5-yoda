# Reproduction decision (DRAFT, Phase 0, 2026-11-07, awaiting confirmation)

## Inventory of the original system

| Item | Status | Evidence |
|---|---|---|
| Code | `YODA.py` (training + inference), `Bootgather.py` (averaging over 30 bootstraps), README. Nothing else. | https://github.com/MVioque/YODA, HEAD `0c8097df40b8ba35fdb6a6b6f35d93fce054bd1f` (2020-05-06), 42 commits, first 2019-09-24 |
| Licence | **None** (no LICENSE file; GitHub API `license: null`). Default copyright applies, so we cite it and fetch it by commit. We do not vendor or redistribute it. | GitHub API, 2026-11-07 |
| Trained weights | **Not published** | repo and VizieR |
| Labelled training set (848 PMS, 775 CBe) | **Not published** as a table. The paper gives only the source catalogues it was built from (Sect. 3) and the special cases (Appx. A). | paper Sect. 3, Table 1 |
| Input data / SoSt | **Published**: VizieR J/A+A/638/A21/sample, 4,150,983 rows. It contains all 13 base observables, e(F) columns, AllWISE flags and the published pPMS/pCBe/pOther. | VizieR metadata, 2026-11-07 |
| Candidate tables | pms 8,470 / cbe 693 / either 1,309 rows. These match the paper's Fig. 3 counts. | VizieR |
| Dependencies | Not declared. The imports show Keras 2.x (standalone `keras`), `keras_metrics` (abandoned), sklearn of about 2019, numpy, matplotlib. | `YODA.py` |
| Tests | None | repo |

The code's 13 input columns are r−Hα, Gvar and Vhtg plus 10 passbands (BP, G, RP, J, H, Ks, W1–W4). With the 45 pairwise colours this gives 48 observables, which matches paper Sect. 2.1.

## Decision: Route C (best faithful reproduction)

Route A is impossible because there are no weights. Route B is impossible because the labelled set is not public. Two checks show that the labelled sources are not recoverable from the SoSt table. Cone searches at 40 Vioque+2018 HAeBe positions returned 0 hits. Cone searches at 30 Raddi+2015 CBe positions returned 1 hit. So the labelled sources' observables have to be rebuilt.

Plan (C-a, preferred):
1. **Labels from public literature, using the same sources as paper Sect. 3:**
   - HAeBe: Vioque+2018 (J/A+A/620/A128).
   - T Tauri: Herbig & Bell 1988 plus SIMBAD `TT*`. SIMBAD now holds 6,354 `TT*` objects against about 3,500 in 2019, so there is catalogue drift.
   - CBe: BeSS (currently returns HTTP 500; we will retry), Raddi+2013/2015 and Shokry+2018. SIMBAD `Be*` is the fallback, which is a declared deviation.
   - We apply the Appx. A overlap and FS CMa decisions.
2. **Observables:**
   - Gaia DR2 from the ESA TAP (reachable).
   - AllWISE with a 2″ cone (IRSA, reachable).
   - IPHAS DR2 / VPHAS+ through the VizieR mirror.
   - We use the same cross-match radii as Sect. 2.4.
3. **Gvar/Vhtg** need a reference "median e(F)√N/F at the same brightness" curve. We calibrate it on SoSt sources: we fetch N_obs from Gaia and check that we reproduce the published Gvar/Vhtg. A pre-declared agreement gate applies.
4. **Not reproduced (limitation):** the Hα-EW → r−Hα synthetic colours (297 T Tauri, 442 CBe). Our labelled set will therefore be smaller than 848/775. We will report the actual counts.
5. **Training:** the architecture is that of Appx. B.2/B.3. The "other" class is 470,263 random SoSt rows. We report agreement with the published probabilities (Jaccard overlap of the p ≥ 0.5 sets, and Spearman ρ of pPMS and pCBe). **C1 is tested on this reproduction, not on the original model.**

Fallback (C-b, only if C-a fails its gate): a surrogate trained on SoSt with the published argmax as pseudo-labels. This is weaker because test-set recall would then be meaningless.

## Code-vs-paper discrepancies found during recon (for later phases)

1. `train_test_split` is never imported. `YODA.py` raises NameError as published.
2. The number of PCA components is chosen by the eigenvalue-ratio rule `var[i+1]/var[i] ≤ 0.01`. The paper text describes a cumulative variance of 99.99%.
3. There is no input validation anywhere. NaN, inf, empty input and wrong column order all flow straight into the scaler, PCA and ANN (C2-relevant).
4. `Bootgather.py` parses `source_id` with `float()`. Gaia DR2 IDs exceed 2^53, so they are silently corrupted. For example, 5781891918842753664 becomes 5781891918842754048. All 20 sampled published IDs do exist in DR2, so the published catalogue looks unaffected, but this check is limited.
5. The 2-layer network in the code puts Dropout before BatchNorm, after each hidden layer. The paper text says BN comes "after every layer". We follow the code.
