import numpy as np
import pytest

from a5 import faults as F
from a5.oracles import flip_rates, pms_metrics, transition_matrix
from a5.pipeline import classify, expand_48


def toy(n=1000, seed=1):
    return np.random.default_rng(seed).normal(10, 2, size=(n, 13))


POOL = np.random.default_rng(9).normal(8, 1, size=(500, 2))
CASES = [
    ("F1_resample", dict(pool=POOL), F.W34),
    ("F1_noise", dict(sigma=1.0), F.W34),
    ("F2_allwise_swap", {}, F.ALLWISE),
    ("F3_iphas_swap", {}, [F.RHA]),
    ("F4_fluxerr", dict(k=5.0), F.VAR),
]


@pytest.mark.parametrize("name,kw,cols", CASES)
@pytest.mark.parametrize("rate", [0.0, 0.013, 0.023, 0.1, 0.44, 1.0])
def test_exact_rate_and_only_target_columns(name, kw, cols, rate):
    x = toy()
    out, idx = F.FAULTS[name](x, rate, seed=3, **kw)
    assert len(idx) == round(rate * len(x)) == len(np.unique(idx))
    changed = np.any(out != x, axis=1)
    # only corrupted rows change (a swap can coincide by chance only with identical donors: impossible here)
    assert set(np.flatnonzero(changed)) <= set(idx)
    assert np.sum(changed) == len(idx)
    other = [c for c in range(13) if c not in cols]
    assert np.array_equal(out[:, other], x[:, other])


@pytest.mark.parametrize("name,kw,cols", CASES)
def test_determinism_and_no_mutation(name, kw, cols):
    x = toy()
    before = x.copy()
    a, ia = F.FAULTS[name](x, 0.3, seed=11, **kw)
    b, ib = F.FAULTS[name](x, 0.3, seed=11, **kw)
    c, ic = F.FAULTS[name](x, 0.3, seed=12, **kw)
    assert np.array_equal(x, before), "input mutated"
    assert np.array_equal(a, b) and np.array_equal(ia, ib)
    assert not np.array_equal(ia, ic)


def test_hand_count_100_rows():
    """Independent check: count changed rows by hand on 100 rows at 2.3% and 44%."""
    x = toy(100)
    for rate, expect in [(0.023, 2), (0.44, 44), (0.013, 1)]:
        out, _ = F.f2_allwise_swap(x, rate, seed=0)
        assert sum(1 for i in range(100) if not np.array_equal(out[i], x[i])) == expect


def test_swap_never_self_donor():
    x = np.arange(13 * 3, dtype=float).reshape(3, 13)
    for s in range(50):
        out, idx = F.f3_iphas_swap(x, 1.0, seed=s)
        assert np.all(out[:, F.RHA] != x[:, F.RHA])


def test_swap_with_external_donors():
    x = toy(200)
    donors = np.full((50, 13), -99.0)
    before = donors.copy()
    out, idx = F.f2_allwise_swap(x, 0.1, seed=4, donors=donors)
    assert len(idx) == 20
    assert np.all(out[np.ix_(idx, F.ALLWISE)] == -99.0)
    assert np.array_equal(donors, before)


def test_f4_scales_variability_exactly():
    x = toy()
    out, idx = F.f4_inflate_flux_errors(x, 0.1, seed=0, k=2.0)
    assert np.allclose(out[np.ix_(idx, F.VAR)], 2 * x[np.ix_(idx, F.VAR)])


def test_expand_48_matches_paper_count_and_order():
    x = np.arange(13, dtype=float)[None, :]
    e = expand_48(x)
    assert e.shape == (1, 48)
    # first colour is BP-G (col3-col4), last is W3-W4 (col11-col12)
    assert e[0, 3] == x[0, 3] - x[0, 4] and e[0, -1] == x[0, 11] - x[0, 12]


def test_classify_and_oracles_toy():
    p = np.array([[0.8, 0.1, 0.1],   # other
                  [0.2, 0.6, 0.2],   # PMS
                  [0.1, 0.2, 0.7],   # CBe
                  [0.4, 0.3, 0.3],   # other (nothing >= .5)
                  [0.2, 0.5, 0.3]])  # PMS (tie rule: >= .5)
    assert classify(p).tolist() == [0, 1, 2, 0, 1]
    clean = np.array([0, 1, 1, 2, 2, 0])
    fault = np.array([0, 1, 0, 1, 2, 2])
    fr = flip_rates(clean, fault)
    assert fr["flip_PMS"] == 0.5 and fr["flip_CBe"] == 0.5 and fr["flip_other"] == 0.5
    assert abs(fr["flip_all"] - 3 / 6) < 1e-12
    tm = transition_matrix(clean, fault)
    assert tm.tolist() == [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
    m = pms_metrics(np.array([0.9, 0.6, 0.1, 0.7, 0.2]), np.array([1, 1, 1, 0, 0]))
    assert (m["TP"], m["FN"], m["FP"]) == (2, 1, 1)
    assert abs(m["recall_PMS"] - 2 / 3) < 1e-12 and m["precision_PMS"] == 2 / 3


def test_nan_probabilities_classified_undefined():
    assert classify(np.array([[np.nan, np.nan, np.nan]])).tolist() == [-1]
