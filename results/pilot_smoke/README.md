# Pilot / smoke runs (BEFORE protocol freeze). Disclosed, not used as evidence.

Purpose: debug the harness end to end on partial data. The smoke run used:
- the first 15 of 211 downloaded chunks (300k rows);
- 1 seed, 3 training epochs, 3,000 "other" sources.

The model is untrained and meaningless, so none of these numbers are reported as results.

One protocol-relevant change was made after looking at this pilot:
- In the first smoke run, the C2 identifier case `gaia_id_roundtrip` and the Bootgather case used only
  the first 5 SoSt rows by recno. Five IDs that are adjacent on the sky are not a representative sample
  of identifiers.
- The case therefore now uses 1,000 random SoSt rows (random_state=0).
- The change was made before freezing, and the reason is recorded here and in AI_USE_LOG.md.
- In the second smoke run (1,000 random IDs from the 300k partial rows), YODA.py's
  loadtxt/savetxt altered 0 of 1,000 IDs. The original Bootgather.py altered 247 of 1,000.
- The pilot outcome is reported here for transparency only. The frozen run decides.
