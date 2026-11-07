#!/usr/bin/env python3
"""Download the VizieR catalogue J/A+A/638/A21 (Vioque et al. 2020) and verify checksums.

Usage:
    python scripts/download_data.py            # download missing chunks, then verify
    python scripts/download_data.py --verify   # verify only (no network)
    python scripts/download_data.py --record   # (maintainer) write data/checksums.json from local files

Design notes
- Data are fetched from a VizieR mirror via the ASU-TSV interface. CDS Strasbourg itself was not
  reachable from the author's network (2026-11-07), so mirrors are tried in order.
- The server prepends comment lines containing the request date; these lines ('#'-prefixed and blank)
  are stripped before hashing, so the checksum covers the column header + data rows only.
- The 4.15M-row `sample` table is fetched in recno chunks so a failed transfer can be resumed.
  Every retry is logged to data/download_log.jsonl (no silent retries).
- Raw data live in data/raw/ and are git-ignored; data/checksums.json is committed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHECKSUMS = ROOT / "data" / "checksums.json"
LOG = ROOT / "data" / "download_log.jsonl"

MIRRORS = [
    "https://vizier.idia.ac.za/viz-bin/asu-tsv",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
]
CATALOGUE = "J/A+A/638/A21"
SAMPLE_ROWS = 4_150_983
CHUNK = 20_000

# Columns needed from the Sample of Study (inputs, flags, published probabilities, identifiers).
SAMPLE_COLS = [
    "recno", "GaiaDR2", "RA_ICRS", "DE_ICRS", "GLON", "GLAT",
    "pPMS", "e_pPMS", "pCBe", "e_pCBe", "pOther", "e_pOther",
    "Vhtg", "Gvar", "BPmag", "Gmag", "RPmag", "rmag", "e_rmag", "Hamag", "e_Hamag",
    "Jmag", "Hmag", "Ksmag", "W1mag", "W2mag", "W3mag", "e_W3mag", "W4mag", "e_W4mag",
    "ccf", "ex", "var", "qph", "FW3W4", "a10p", "Dist1", "Dist2",
    "IVPHAS", "AllWISE", "RUWE", "e_FG", "e_FBP", "e_FRP", "E(BR/RP)",
]
# NOTE (found in final verification): VizieR ignored the empty-valued "-out.all" below, so these
# three files contain only the default columns. Only their Gaia IDs are used. This is kept as is so
# the frozen checksums remain valid (results/verification/verification_report.md).
SMALL_TABLES = ["pms", "cbe", "either"]


def log(event: dict) -> None:
    event["utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def canonical(text: str) -> str:
    """Drop server comment lines (request date etc.) and blank lines."""
    return "".join(l + "\n" for l in text.splitlines() if l.strip() and not l.startswith("#"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def n_data_rows(text: str) -> int:
    # canonical TSV = 3 header lines (names, units, dashes) + data rows
    return max(0, len(text.splitlines()) - 3)


def fetch(params: dict, timeout: int = 300, attempts: int = 4) -> tuple[str, str]:
    q = urllib.parse.urlencode(params, safe=",")
    last = None
    for attempt in range(attempts):
        for mirror in MIRRORS:
            url = f"{mirror}?{q}"
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    body = r.read().decode("utf-8")
                if "#INFO\tQUERY_STATUS=\"ERROR\"" in body or "****" in body[:2000]:
                    raise RuntimeError("VizieR returned an error document")
                return body, mirror
            except Exception as exc:  # logged, then next mirror / attempt
                last = exc
                log({"event": "retry", "mirror": mirror, "attempt": attempt,
                     "params": params.get("recno", params.get("-source")), "error": repr(exc)[:300]})
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"all mirrors failed: {last!r}")


def targets() -> list[tuple[Path, dict, int | None]]:
    out = []
    for t in SMALL_TABLES:
        out.append((RAW / f"{t}.tsv",
                    {"-source": f"{CATALOGUE}/{t}", "-out.all": "", "-out.max": "unlimited"}, None))
    for i, start in enumerate(range(1, SAMPLE_ROWS + 1, CHUNK)):
        end = min(start + CHUNK - 1, SAMPLE_ROWS)
        out.append((RAW / "sample" / f"chunk_{i:04d}.tsv",
                    {"-source": f"{CATALOGUE}/sample", "recno": f"{start}..{end}",
                     "-out": ",".join(SAMPLE_COLS), "-out.max": "unlimited", "-sort": "recno"},
                    end - start + 1))
    return out


def download(shard: int = 0, nshards: int = 1) -> None:
    todo = targets()
    for k, (path, params, expected) in enumerate(todo):
        if k % nshards != shard or path.exists():
            continue
        body, mirror = fetch(params)
        text = canonical(body)
        n = n_data_rows(text)
        if expected is not None and n != expected:
            log({"event": "row_count_mismatch", "file": str(path.relative_to(ROOT)),
                 "expected": expected, "got": n, "mirror": mirror})
            raise RuntimeError(f"{path.name}: expected {expected} rows, got {n} (from {mirror})")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        log({"event": "ok", "file": str(path.relative_to(ROOT)), "rows": n, "mirror": mirror,
             "sha256": sha256(text)})
        print(f"[{k + 1}/{len(todo)}] {path.relative_to(ROOT)} rows={n} via {mirror}", flush=True)


def local_manifest() -> dict:
    files = {}
    for path, _, _ in targets():
        text = path.read_text()
        files[str(path.relative_to(ROOT))] = {"sha256": sha256(text), "rows": n_data_rows(text)}
    return files


def record() -> None:
    manifest = {
        "catalogue": CATALOGUE,
        "doi_dataset": "10.26093/cds/vizier.36380021",
        "paper_doi": "10.1051/0004-6361/202037731",
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hash_scope": "sha256 of file after removing '#' comment lines and blank lines",
        "sample_columns": SAMPLE_COLS,
        "files": local_manifest(),
    }
    combined = hashlib.sha256("".join(v["sha256"] for v in manifest["files"].values()).encode())
    manifest["combined_sha256"] = combined.hexdigest()
    CHECKSUMS.write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", CHECKSUMS, "combined", manifest["combined_sha256"])


def verify() -> bool:
    ref = json.loads(CHECKSUMS.read_text())["files"]
    bad = 0
    for rel, meta in ref.items():
        p = ROOT / rel
        if not p.exists():
            print("MISSING", rel); bad += 1; continue
        text = p.read_text()
        if sha256(text) != meta["sha256"] or n_data_rows(text) != meta["rows"]:
            print("MISMATCH", rel); bad += 1
    print(f"verified {len(ref) - bad}/{len(ref)} files")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="parallel download: this worker's index")
    ap.add_argument("--nshards", type=int, default=1, help="parallel download: number of workers")
    a = ap.parse_args()
    if a.record:
        record()
    elif a.verify:
        sys.exit(0 if verify() else 1)
    else:
        download(a.shard, a.nshards)
        if a.nshards > 1:
            sys.exit(0)
        if CHECKSUMS.exists():
            sys.exit(0 if verify() else 1)
        print("no data/checksums.json yet; run with --record (maintainer only)")
