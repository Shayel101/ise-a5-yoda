# Rerun / deviation log (after)

- **2026-11-07, run 1:** `python -m a5.run_experiment configs/after.yaml` completed every cell on the
  first attempt (wall time 555 s, `failures/` empty). `python -m a5.run_boundary configs/after.yaml`
  also ran once. No reruns.
- **Determinism check (`side_effects.json`):** the retrained models match the baseline models. On every
  unflagged row, the maximum absolute probability difference is ≤ 1.2e-7 and there are 0 class changes.
- **C2 case `bootgather_original_ids`:**
  - **What happened:** after the improvement, the unmodified original Bootgather.py **crashed**
    (`ValueError: could not convert string to float: ''`). The cause is the guard's NaN probability
    for 1 flagged row among the 1,000.
  - **Verdict under the frozen rule:** an explicit error counts as PASS, so this is recorded as PASS,
    unchanged.
  - **Why that PASS is misleading:** the crash masks the ID corruption; it does not fix it. A
    supplementary diagnostic (not a protocol cell) reruns Bootgather.py on the 999 unflagged rows:
    790/999 IDs are still silently altered (`supplementary_bootgather_unflagged.json`).
  - **Reporting:** the report gives both results and treats the identifier corruption as unresolved
    (residual risk R2). The guard's NaN outputs breaking the downstream aggregator is reported as a
    side effect.
- **C2 case `missing_column_W1`:** it is PASS after the improvement, but by an indirect route. With
  17 values under an 18-name header, pandas shifts the columns, the label value lands in the W4
  position, and the range check flags the row. The guard's column-count check does not fire in this
  layout.
