# Release notes for the next version (draft, not released)

Proposed version: **v1.1.0**. No tag yet; the authors decide when to release (planned: at paper submission). On release:
- update `version` and `date-released` in `CITATION.cff`;
- Zenodo mints the new version DOI under the concept DOI 10.5281/zenodo.21382060.

## Changes since v1.0.0
- **PR #2: V6 analysis fixes, envelope drift, mean ellipse area.**
  - Analysis window, coherence test and frequency fields as used in the paper.
  - V6 regression tests (`tests/fixtures/v6_reference_testing_data_B7test.csv`).
- **PR #4: enlarged-window tracking of RainyDay events.**
  - Screen by tile, track on tile + margin.
  - A vectorised tracker identical to V6, about 20× faster.
  - Jump-handling variants v0, v1 and v2, and pluggable trackers.
  - Fix: `mean_direction_weighted` returns `(nan, nan)` for stationary tracks.
- **PR #3: robust linking options** (`stormcatalog_analyzer/tracking/robust.py`, all off by default).
  - Merge partition, the t = 1 label fix, advection, gates, Hungarian assignment, cores.
  - The paper's main version uses the preset `partition_keepsplit`.
- **Tests:** 67 tests pass on main, including the unchanged V6 regression.
