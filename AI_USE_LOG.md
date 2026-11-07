# AI use log

**Tool:** Claude Code (Anthropic), model Claude Opus 5.5 (`claude-opus-5-5`), run in the VS Code
extension on the student's Mac.
**Session:** 2026-11-07, Asia/Shanghai.
**Division of work:** the student wrote the task brief: claims, phases, rules, stop points. The AI
executed the phases and stopped at the student's decision points.

## Tasks delegated to the AI
| # | Task | Output |
|---|---|---|
| 1 | Phase 0 reconnaissance: clone YODA, inventory, VizieR metadata, read the paper, choose a route | `docs/reproduction_decision.md` |
| 2 | Data download with checksums | `scripts/download_data.py`, `data/checksums.json` |
| 3 | Rebuilding the labelled set | `scripts/build_labels.py` |
| 4 | Pinned environment | `pyproject.toml`, `uv.lock`, `results/env.json` |
| 5 | Re-implementing YODA, fault functions, oracles, tests | `src/a5/*`, `tests/` |
| 6 | Threat model and protocol drafts | `docs/threat_model.md`, `docs/protocol_v1.md` |
| 7 | Running the baseline and after grids; C2 suite | `results/` |
| 8 | Analysis notebook, report, residual risk, summary | `notebooks/`, `report/`, `docs/` |

## Advice from the AI: accepted / rejected by the student
| Advice | Decision | Reason |
|---|---|---|
| Route C (re-implementation), because there are no public weights or labels | accepted | No alternative exists. Verified by the inventory below. |
| Correct the brief's reading of the 28 % W3/W4 figure. It is K&L's *reliable* fraction, so the unreliable fraction is ≈72 %. Keep 28 % and add 72 % to the grid. | accepted (student replied "continue" / "confirm") | The paper's text (Sect. 5.2 pt 2) supports the AI's reading. |
| The paper's "do not affect the classification" sentence refers to *duplicated* matches, so C1 as briefed is a stronger reading | accepted, disclosed in threat model and report | Text check, paper Sect. 2.4 |
| Rule (b) also requires the ΔRecall CI to exclude 0, because 3 models are noisy | accepted at freeze ("confirm") | Statistical caution |
| Separate C2 "FAIL-crash on valid input" from "FAIL-silent" | accepted at freeze | Keeps C2 faithful to its wording ("fails loudly") |
| Quick mode (student request "use quick mode, finish today"): 3 models, 1/10 "other" class, 300-epoch cap | accepted, initiated by the student | Benchmark: 26 s/epoch at full size |
| Initial plan C-a: rebuild labelled observables from Gaia + AllWISE + IPHAS/VPHAS+ | **rejected** (by the AI, before freezing; reported to the student) | The IPHAS/VPHAS+ tables returned no rows from any reachable mirror, and the student wanted the quick mode. Replaced with SIMBAD-labelled stars already in the SoSt. |
| Fallback C-b: surrogate trained on published labels | **rejected** / not needed | C-a-lite was feasible, and C-b would make recall meaningless |
| The improvement: A (guard), recommended over B because it is cheap, reviewable and measurable; B is circular | accepted (student chose "A") | Student decision |
| Score the guard's NaN outputs as flips under the frozen oracle, even though that made C1 look worse after the improvement | accepted (the AI did not change the oracle after seeing results) | The protocol was frozen. The decomposition into rejections and class changes is added as extra analysis only |
| Bootgather "PASS" after the guard: report it as nominal and add a clearly labelled supplementary run that shows the IDs are still corrupted | accepted | Avoids a misleading PASS |
| Recompute the ill-posed agreement Jaccard on the full table | accepted, logged in `results/baseline/rerun_log.md` | It is a reporting metric only; no decision depends on it |
| Install pandoc through the pinned dev dependency `pypandoc-binary==1.15` (lock file updated after `a5-env`) | accepted | No system pandoc was available. The change affects only report rendering, not the experiment environment |

## Independent verification of AI outputs
Each check produced a stored artefact or a reproducible command.
- **Paper numbers vs catalogue:**
  - recomputed from VizieR: 43.5 % of PMS and 27.8 % of CBe candidates have the W3W4 flag (paper: 44 % / 27 %);
  - 18 / 57 X-mtch flags (paper: 18 / 57);
  - table sizes 8,470 / 693 / 1,309 (paper Fig. 3).
- **Published IDs are real:** 20 of 20 sampled published Gaia DR2 IDs exist in the ESA Gaia archive.
- **Float corruption of IDs:** `int(float(5781891918842753664))` = …754048 (recon log).
- **Fault rates by hand:** `tests/test_faults_oracles.py::test_hand_count_100_rows` counts changed rows
  one by one on 100 rows (1, 2 and 44 for 1.3 %, 2.3 % and 44 %). Rates are checked exactly for every
  fault at 6 rates.
- **Deliberate break:** I made F1-noise mutate its input and allowed a swap to use the row itself as
  donor. 9 tests then failed (`results/verification/deliberate_break_pytest.txt`). With the code
  restored, all pass.
- **Oracles on a hand-built toy example:** hand-computed flip rates, transition matrix and
  recall/precision (`test_classify_and_oracles_toy`).
- **Clean install:** a fresh venv from `uv.lock` passes all tests (`results/verification/clean_install.txt`).
- **DOIs:** all 13 resolve through the doi.org handle API. CSL metadata (author, year, title) matches the
  intended reference (`results/verification/doi_check.txt`).
- **Reference not independently verified:** the ≈28 % figure of Koenig & Leisawitz (2014) is checked
  only as cited by Vioque et al. (2020). The publisher page blocked automated access and the arXiv
  search returned nothing. It is marked as such in the report.
- **Counterexamples:** the fault indices are regenerated from the stored seed in the notebook and
  asserted equal to the stored indices (`analysis.counterexamples`). Every flipped source was an
  injected one (0 flips among untouched rows), which is a sanity check against nondeterminism.
- **End-to-end determinism:** the after run retrained all models with the same seeds. On unflagged
  rows the probabilities match the baseline to ≤ 1.2e-7, with 0 class changes (`results/after/side_effects.json`).
- **Harness regression:** after the C2 harness was extended for the guard, the baseline C2 suite was
  rerun into a scratch directory. The verdicts were identical, with the same 636 altered IDs.
- **Suspicious result investigated:** the after-run Bootgather PASS was traced to a crash caused by NaN
  input (`results/after/rerun_log.md`), not to a fix.
- **Notebook:** executed with `jupyter nbconvert --execute` in the fresh environment installed from
  the lock file. 0 errors (`results/verification/clean_install.txt`).
- **Summary:** the word count was computed with a regex over non-space tokens (248 words).

## Process deviations disclosed
- **Pilot runs:** a 3-epoch smoke run on partial data was used to debug the harness before freezing
  (`results/pilot_smoke/`). After seeing it, the C2 identifier case was changed from the first 5 rows
  to 1,000 random rows, because 5 sky-adjacent IDs are not representative. This was done before the
  freeze and is disclosed.
- **Commit history:** the first commit accidentally included `src/*.egg-info` and partial download
  logs. They were removed in the next commit, and tag `a5-env` was moved to the corrected commit
  before anything was published.

## Student decision points
1. Confirmed the Phase 0 route and the repo location by replying "continue".
2. Requested quick mode ("no need to train so much … finish today").
3. Confirmed the protocol thresholds and grid ("confirm") before the freeze.
4. Chose improvement A (Phase 5), replying "A, and go for next".
5. Approves the final tag.
