# Rerun / deviation log (baseline)

- **2026-11-07, run 1:** `python -m a5.run_experiment configs/baseline.yaml` completed every cell on
  the first attempt. There were no crashes (`failures/` is empty) and no reruns. Wall time was 647 s.
  - The trailing `EXIT` line in stdout.log is empty because `${PIPESTATUS[0]}` was not populated by
    the shell wrapper.
  - The background task itself reported exit code 0.
- **C2 suite:** `python -m a5.run_boundary configs/baseline.yaml` ran once and produced
  `c2_boundary.json` / `c2_stdout.log`. No reruns.
- **Reporting-metric correction (not a decision rule):** oracle 6 (agreement with the published
  catalogue) was specified on E_rand.
  - E_rand excludes all published candidates by design, so the published p ≥ 0.5 set there is empty,
    and the Jaccard index in `agreement.csv` is trivially 0. This is a protocol design error.
  - The Spearman values in `agreement.csv` are still valid.
  - `python -m a5.agreement_full configs/baseline.yaml` recomputes the same metrics on the full SoSt
    minus each model's training sources (`agreement_full.csv`). Both files are kept.
  - C1 and C2 decisions do not use this metric.
