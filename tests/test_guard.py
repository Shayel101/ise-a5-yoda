import numpy as np
import pandas as pd

from a5.guard import DUPLICATE_ID, NON_FINITE, OUT_OF_RANGE, Guard, guarded_inference, guarded_predict


class FakeModel:
    def predict_proba(self, x):
        x = np.atleast_2d(x)
        return np.tile([0.2, 0.7, 0.1], (len(x), 1)).astype(np.float32)


def train(n=2000, seed=0):
    return np.random.default_rng(seed).normal(12, 1, size=(n, 13))


def test_flags():
    g = Guard.fit(train())
    x = train(6, 1)
    x[1, 11] = np.nan
    x[2, 4] = -50.0
    x[3, 0] = 50.0
    ids = np.array([1, 2, 3, 4, 5, 5])
    f = g.flags(x, ids)
    assert f[0] == 0 and f[1] & NON_FINITE and f[2] & OUT_OF_RANGE and f[3] & OUT_OF_RANGE
    assert f[4] & DUPLICATE_ID and f[5] & DUPLICATE_ID


def test_valid_rows_untouched_and_flagged_rows_nan():
    g = Guard.fit(train())
    x = train(5, 2)
    x[0, 12] = 99.0
    p, f = guarded_predict(FakeModel(), x, g, return_flags=True)
    assert np.isnan(p[0]).all() and np.allclose(p[1:], [0.2, 0.7, 0.1])


def test_single_row_and_exact_ids(tmp_path):
    g = Guard.fit(train())
    ids = [5781891918842753664, 4293416645328442624, 6917528997577384320]
    rows = train(3, 3)
    df = pd.DataFrame([[i, 10.0, 60.0, *r, 0, 0] for i, r in zip(ids, rows)],
                      columns=["source_id", "ra", "dec"] + [f"c{k}" for k in range(13)] + ["label", "extra"])
    df.to_csv(tmp_path / "in.csv", index=False)
    guarded_inference(tmp_path / "in.csv", tmp_path / "out.csv", FakeModel(), g)
    back = pd.read_csv(tmp_path / "out.csv", dtype={"source_id": "int64"}).source_id.tolist()
    assert back == ids
    df.iloc[:1].to_csv(tmp_path / "one.csv", index=False)
    out = guarded_inference(tmp_path / "one.csv", tmp_path / "one_out.csv", FakeModel(), g)
    assert len(out) == 1 and out.guard_flag.iloc[0] == 0
