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

## Revision 1 (2026-11-07, before any experiment): quick-mode label route and training budget

The user asked for a quick, same-day study. Two findings also forced changes:

- **Observables.** IPHAS DR2 (II/321) and VPHAS+ (II/341) return no rows from the reachable mirrors
  (CfA, IDIA). CDS Strasbourg is unreachable from this network. So the r and Hα of external labelled
  stars cannot be fetched. Plan C-a step 2 was **dropped**.
- **Labels.** These come from SIMBAD (CfA mirror, live database, retrieved 2026-11-07, 7,957 rows):
  - otype `Ae*` (Herbig Ae/Be, 177) and `TT*` (T Tauri, 5,664) are labelled PMS (1);
  - otype `Be*` (2,116) is labelled CBe (2).
  
  Only labelled stars **already present in the published SoSt table** are used. Their 13 base
  observables, including Gvar and Vhtg, are therefore exactly the paper's own values, and no
  Gvar/Vhtg recalibration is needed. Because the paper's own labelled sources appear to have been
  removed from the published table, this labelled set is largely **disjoint** from the authors'. It
  consists of objects classified in SIMBAD but not used for training in 2019/2020. This is a
  deviation, and it is stated as a limitation.
- **Training budget.** A benchmark of the faithful network takes 26 s per full-batch epoch with the
  paper's 470,263 "other" sources on this CPU (i7-9750H). The quick mode therefore uses:
  - "other" = 47,026 random SoSt sources (1/10 of the paper's size);
  - max 300 epochs, with early-stopping patience 50 as in the paper;
  - **3 seeds = 3 bootstrap models** (paper: 30).
  
  A smaller "other" class is known to raise recall and lower precision (paper Fig. 2), so absolute
  metrics are not comparable with 78.8%. The C1 oracles are **paired deltas** against the same-seed
  clean run, so they remain well-defined.
