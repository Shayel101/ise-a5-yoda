# ISE-A5 — Quality & Security Counterexample Study: fault tolerance of the YODA ANN

Falsification study of fault-tolerance claims made for the YODA artificial neural network
(Vioque et al. 2020, A&A 638, A21, doi:10.1051/0004-6361/202037731; code github.com/MVioque/YODA @ 0c8097d).
See `report/report.pdf` for the full write-up. **The model studied is a re-implementation (Route C), not
the authors' original trained network** — see `docs/reproduction_decision.md`.

## Reproduce (macOS x86_64 or Linux x86_64, CPU only)

```bash
git clone <this repo> && cd ise-a5-yoda
# 1. environment (Python 3.12, exact versions from uv.lock)
pip install uv==0.13.7            # or see https://docs.astral.sh/uv/
uv sync --frozen
# 2. data (~1 GB from a VizieR mirror; checksums verified against data/checksums.json)
uv run python scripts/download_data.py            # resumable; add --shard i --nshards n to parallelise
uv run python scripts/download_data.py --verify
uv run python scripts/build_labels.py              # SIMBAD labels; re-query may differ (live DB)
# 3. tests
uv run pytest -q
# (sections for experiment, notebook and report are added as phases complete)
```

## Licence notes
YODA has **no licence** file; it is therefore not redistributed here, only cited. Our code is an
independent re-implementation (MIT, see LICENSE). VizieR data: CDS terms
(https://cds.unistra.fr/vizier-org/licences_vizier.html), not redistributed.
