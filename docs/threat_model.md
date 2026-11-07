# Threat / fault model

**System.** YODA ANN pipeline (Vioque et al. 2020), studied through a faithful re-implementation
(Route C, `docs/reproduction_decision.md`): 13 base characteristics → 48 observables → StandardScaler
→ PCA → 2×580 ReLU ANN → p(other), p(PMS), p(CBe).

**Asset.** The catalogue classes and probabilities (p ≥ 0.5 PMS / CBe candidates) that astronomers use
to select follow-up targets. The paper names spectroscopic follow-up with WEAVE. A wrong class wastes
telescope time (false positive) or hides a real object (false negative). Silent corruption of a source
identifier sends the observer to the wrong star.

**Actors.** There is no adversary. The faults are natural data-quality defects of the input surveys
and malformed input from users who rerun the pipeline. This is a quality/robustness study.

## Claims under test
- **C1 (fault tolerance).** The paper argues that incorrect or duplicated cross-matches "do not affect
  the classification" (Sect. 2.4). It also argues that, thanks to many other observables, "the
  algorithm can deal better" with spurious W3/W4 (Sect. 5.2 point 2).
  
  *Note:* the paper's "do not affect" sentence is literally about the 4.9% / 0.31% *duplicated*
  matches. Our C1 extends it to the 2.3% / 1.3% *incorrect*-match estimates, which is a stronger
  reading. The report says so.
- **C2 (boundary robustness).** The pipeline fails loudly or flags malformed input rather than silently
  returning finite probabilities.

## Fault classes

All faults are pure seeded functions in `src/a5/faults.py`. They are applied to the 13 base
characteristics, which determine all 48 observables, **before** scaling and PCA. They are applied at
inference time to models trained on clean data.

| ID | Mechanism | Rates (fraction of evaluated sources) | Source of rate | Expected effect if C1 holds | Falsifies C1 if |
|---|---|---|---|---|---|
| F1-resample | (W3, W4) of a source are replaced by a (W3, W4) pair drawn from random SoSt sources (spurious mid-IR detections in the Galactic plane) | 0, 10, 28, 44, 72 % | **44 %**: share of PMS candidates with AllWISE ext flag ≠ 0 ("W3W4", paper Sect. 5.2; we recomputed 43.5 % from VizieR). **28 %**: Koenig & Leisawitz (2014) found only ≈28 % *reliable* W3/W4, as cited in the paper. Read correctly, that implies ≈72 % unreliable, so 72 % is an upper stress level and 28 % a mid level. | few flips; other bands carry the class | see protocol thresholds |
| F1-noise | N(0, σ²) added to W3 and W4, σ = 1.0 mag (declared assumption, no literature value) | 0, 10, 28, 44, 72 % | as above | as above | as above |
| F2 | AllWISE mismatch: J, H, Ks, W1–W4 of a source replaced by those of a random SoSt source | 0, 1, **2.3**, 5 % | paper Sect. 2.4 (≈2.3 % bad AllWISE matches; 1/42 ≈ 2.4 % fake IR photometry) | few flips | see protocol |
| F3 | IPHAS/VPHAS+ mismatch: r and Hα (hence r−Hα) replaced by those of a random SoSt source | 0, **1.3**, 3 % | paper Sect. 2.4 (1.3 %) | few flips | see protocol |
| F4 | Inflated Gaia flux errors (extended sources or binaries): e(F_G), e(F_BP), e(F_RP) × k, so Gvar and Vhtg × k exactly (Eqs. 1–2) | k ∈ {2, 5} at 5 % and 10 % | no literature rate; exploratory, **not used to falsify C1** | variability-driven flips | reported only |
| F5 | Malformed or boundary input (C2): NaN, ±inf, missing band, out-of-range magnitude or colour, duplicate IDs, empty input, single row, wrong column order, wrong dtype, 19-digit Gaia IDs through the original `Bootgather.py` | one case each | — | loud error or flag | any silent finite output |

## Out of scope (with reasons)
- **Adversarial gradient attacks.** No adversary controls survey photometry, and the catalogue is
  produced offline once. The threat is unrealistic for this asset.
- **Training-label poisoning.** It is plausible: SIMBAD and BeSS misclassifications, and the paper's
  own Appx. A ambiguities. Testing it would need retraining per poisoning level, which exceeds the
  quick-mode budget. It is listed as a **residual risk** instead.
- **Privilege / authorisation tests.** YODA is an offline research script with no users, roles,
  network service, credentials or persisted state. There is no authorisation boundary to test.
  The integrity risks that remain (tampered input files, mixed-up IDs) are covered by F5.
