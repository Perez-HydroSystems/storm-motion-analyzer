# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-15

First stable public release.

### Added

- Storm detection from NetCDF gridded rainfall data via rainfall thresholding,
  8-connectivity connected-component labeling, and morphological
  dilation/erosion (`stormcatalog_analyzer/tracking/identification.py`).
- Overlap-based Lagrangian storm tracking adapted from the STEP protocol
  (`stormcatalog_analyzer/tracking/tracking.py`).
- Storm characterization: precipitation-weighted centroids, moment-based
  ellipse fitting, trajectory, translation speed, direction, angular variance,
  intensity (mean/peak/total rainfall), and spatial morphology metrics.
- Catalog-level pipeline (`stormcatalog_analyzer/pipeline.py`) producing a
  per-storm properties CSV, diagnostic figures (trajectory map, wind rose,
  direction vectors, statistics pairplot), and a per-cell direction-probability
  field.
- Command-line interface (`stormcatalog-analyzer diagnostics|event`) and
  `run_diagnostics.py` entry script, driven by a single JSON config.
- Notebooks for catalog-level (`notebooks/storm_motion_catalog.ipynb`) and
  per-event (`notebooks/storm_motion_event.ipynb`) analysis.
- Demo storm catalog (RainyDay output built from NOAA AORC rainfall data) and
  transposition domain under `testing_data/`.
- Conda (`environment.yml`) and pip (`requirements.txt`) dependency
  specifications.

[1.0.0]: https://github.com/Perez-HydroSystems/storm-motion-analyzer/releases/tag/v1.0.0
