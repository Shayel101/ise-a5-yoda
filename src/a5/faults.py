"""Pure, seeded fault-injection functions on the 13 base characteristics.

All 48 observables are derived from these 13 columns (src/a5/pipeline.expand_48) before scaling and
PCA, so a fault injected here propagates consistently into every affected colour.

Every function:
* takes a (n, 13) float array and returns a NEW array (input is never mutated);
* takes an explicit ``seed`` and uses its own ``numpy.random.Generator``;
* corrupts exactly ``round(rate * n)`` rows, chosen without replacement;
* returns ``(x_new, idx)`` where ``idx`` are the corrupted row indices (sorted).
"""
from __future__ import annotations

import numpy as np

from .pipeline import BASE13

COL = {name: i for i, name in enumerate(BASE13)}
W34 = [COL["W3mag"], COL["W4mag"]]
ALLWISE = [COL[c] for c in ["Jmag", "Hmag", "Ksmag", "W1mag", "W2mag", "W3mag", "W4mag"]]
RHA = COL["rHa"]
VAR = [COL["Gvar"], COL["Vhtg"]]


def n_corrupt(n: int, rate: float) -> int:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    return int(round(rate * n))


def _pick(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    return np.sort(rng.choice(n, size=n_corrupt(n, rate), replace=False))


def f1_w3w4_resample(x: np.ndarray, rate: float, seed: int, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F1 model 1: replace (W3, W4) of a fraction of rows with a (W3, W4) pair drawn from ``pool``
    (an (m, 2) array: the empirical W3/W4 distribution of the Sample of Study)."""
    rng = np.random.default_rng(seed)
    out = np.array(x, dtype=float, copy=True)
    idx = _pick(rng, len(out), rate)
    draw = rng.integers(0, len(pool), size=len(idx))
    out[np.ix_(idx, W34)] = np.asarray(pool, dtype=float)[draw]
    return out, idx


def f1_w3w4_noise(x: np.ndarray, rate: float, seed: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """F1 model 2: add independent N(0, sigma^2) mag noise to W3 and W4 of a fraction of rows."""
    rng = np.random.default_rng(seed)
    out = np.array(x, dtype=float, copy=True)
    idx = _pick(rng, len(out), rate)
    out[np.ix_(idx, W34)] += rng.normal(0.0, sigma, size=(len(idx), 2))
    return out, idx


def _swap(x: np.ndarray, rate: float, seed: int, cols: list[int],
          donors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Give each corrupted row the ``cols`` values of a randomly chosen donor row.

    ``donors`` (an (m, 13) clean array, e.g. random Sample-of-Study sources) is the population a wrong
    cross-match would draw from. If None, donors are other rows of the clean input itself (never the
    row itself). Donors always come from clean data, so corruptions cannot chain.
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(x, dtype=float)
    out = np.array(src, copy=True)
    n = len(out)
    idx = _pick(rng, n, rate)
    if donors is not None:
        d = np.asarray(donors, dtype=float)
        pick = rng.integers(0, len(d), size=len(idx))
        out[np.ix_(idx, cols)] = d[np.ix_(pick, cols)]
        return out, idx
    if n < 2:
        return out, idx
    donor = rng.integers(0, n - 1, size=len(idx))
    donor = donor + (donor >= idx)  # guarantee donor != self
    out[np.ix_(idx, cols)] = src[np.ix_(donor, cols)]
    return out, idx


def f2_allwise_swap(x: np.ndarray, rate: float, seed: int,
                    donors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """F2: wrong AllWISE cross-match -> J, H, Ks, W1..W4 of another source."""
    return _swap(x, rate, seed, ALLWISE, donors)


def f3_iphas_swap(x: np.ndarray, rate: float, seed: int,
                  donors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """F3: wrong IPHAS/VPHAS+ cross-match -> r and H-alpha of another source. YODA only uses r-Halpha
    (the 'rHa' column), so swapping (r, Halpha) together is equivalent to swapping rHa."""
    return _swap(x, rate, seed, [RHA], donors)


def f4_inflate_flux_errors(x: np.ndarray, rate: float, seed: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """F4: multiply e(F_G), e(F_BP), e(F_RP) by k for a fraction of rows.

    Gvar is proportional to e(F_G) (paper Eq. 1) and Vhtg = c_BP*e(F_BP) - c_RP*e(F_RP) (Eq. 2), so
    inflating all three flux errors by k multiplies both Gvar and Vhtg by exactly k.
    """
    rng = np.random.default_rng(seed)
    out = np.array(x, dtype=float, copy=True)
    idx = _pick(rng, len(out), rate)
    out[np.ix_(idx, VAR)] *= k
    return out, idx


FAULTS = {
    "F1_resample": f1_w3w4_resample,
    "F1_noise": f1_w3w4_noise,
    "F2_allwise_swap": f2_allwise_swap,
    "F3_iphas_swap": f3_iphas_swap,
    "F4_fluxerr": f4_inflate_flux_errors,
}
