"""Load the downloaded VizieR tables into pandas (Gaia IDs kept as exact int64, never float)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import BASE13

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "sample.parquet"


def _read_tsv(path: Path) -> pd.DataFrame:
    # canonical VizieR TSV: line0 names, line1 units, line2 dashes, then data
    df = pd.read_csv(path, sep="\t", skiprows=[1, 2], dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


NUMERIC = ["RA_ICRS", "DE_ICRS", "GLON", "GLAT", "pPMS", "e_pPMS", "pCBe", "e_pCBe", "pOther", "e_pOther",
           "Vhtg", "Gvar", "BPmag", "Gmag", "RPmag", "rmag", "e_rmag", "Hamag", "e_Hamag", "Jmag", "Hmag",
           "Ksmag", "W1mag", "W2mag", "W3mag", "e_W3mag", "W4mag", "e_W4mag", "Dist1", "Dist2", "RUWE",
           "e_FG", "e_FBP", "e_FRP", "E(BR/RP)"]


def load_sample(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    files = sorted((RAW / "sample").glob("chunk_*.tsv"))
    df = pd.concat([_read_tsv(f) for f in files], ignore_index=True)
    df["recno"] = df["recno"].astype(np.int64)
    df["GaiaDR2"] = df["GaiaDR2"].astype(np.int64)  # exact: never via float (cf. Bootgather.py)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="raise")
    df["rHa"] = df["rmag"] - df["Hamag"]
    df = df.sort_values("recno").reset_index(drop=True)
    df.to_parquet(CACHE, index=False)
    return df


def load_candidates() -> dict[str, np.ndarray]:
    """Gaia DR2 IDs of the published candidate tables (pms, cbe, either)."""
    return {t: _read_tsv(RAW / f"{t}.tsv")["GaiaDR2"].astype(np.int64).to_numpy() for t in ["pms", "cbe", "either"]}


def load_labels() -> pd.DataFrame:
    lab = pd.read_csv(RAW / "simbad_labels.csv", dtype={"GaiaDR2": np.int64})
    return lab


def base13(df: pd.DataFrame) -> np.ndarray:
    return df[BASE13].to_numpy(dtype=float)
