#!/usr/bin/env python3
"""Fetch literature-labelled PMS / classical Be stars from SIMBAD (by Gaia DR2 identifier).

Route C (see docs/reproduction_decision.md): the paper's labelled set is not public and the
IPHAS/VPHAS+ tables could not be reached from any reachable VizieR mirror, so the labelled set is
rebuilt as: SIMBAD objects with otype Ae* (Herbig Ae/Be) or TT* (T Tauri)  -> label 1 (PMS),
otype Be* (Be star) -> label 2 (CBe), restricted later to sources present in the published
Sample of Study (so all 48 observables come from the paper's own table).

Output: data/raw/simbad_labels.csv (git-ignored) and its SHA256 in data/labels_checksum.json.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "simbad_labels.csv"
REC = ROOT / "data" / "labels_checksum.json"
TAP = "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync"  # CDS Strasbourg not reachable
QUERY = (
    "SELECT b.main_id, b.otype, i.id FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
    "WHERE b.otype IN ('Ae*', 'TT*', 'Be*') AND i.id LIKE 'Gaia DR2 %'"
)
LABEL = {"Ae*": 1, "TT*": 1, "Be*": 2}


def main(verify_only: bool = False) -> None:
    if not verify_only:
        data = urllib.parse.urlencode({"request": "doQuery", "lang": "adql", "format": "csv",
                                       "query": QUERY}).encode()
        raw = urllib.request.urlopen(TAP, data, timeout=300).read().decode()
        rows = list(csv.DictReader(io.StringIO(raw)))
        rows.sort(key=lambda r: int(r["id"].split()[-1]))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["GaiaDR2", "simbad_main_id", "otype", "label"])
        for r in rows:
            w.writerow([int(r["id"].split()[-1]), r["main_id"].strip(), r["otype"], LABEL[r["otype"]]])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(buf.getvalue())
        if not REC.exists():
            REC.write_text(json.dumps({
                "source": TAP, "query": QUERY,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "rows": len(rows), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                "note": "SIMBAD is a live database; a re-query may legitimately differ (see README)."},
                indent=1) + "\n")
    ref = json.loads(REC.read_text())
    got = hashlib.sha256(OUT.read_bytes()).hexdigest()
    status = "OK" if got == ref["sha256"] else "DIFFERS (SIMBAD changed since freeze)"
    print(f"simbad_labels.csv sha256 {got[:16]}... {status}")


if __name__ == "__main__":
    main(verify_only="--verify" in sys.argv)
