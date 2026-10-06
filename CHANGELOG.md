# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Moments ellipse fit (`fit_ellipse_to_rainfall`, the default `ellipse_fit_method = "moments"`)
  used inconsistent moments.** `m00` and the centroid were computed from the thresholded
  rainfall field, but the second central moments (`mu20`, `mu02`, `mu11`) summed the
  unthresholded field, so they were not weighted variances of either field and the ellipse
  was inflated. All moments now use the thresholded field.
  Because the ellipse also masks the rainfall used for `mean_intensity_mmh` (which selects
  the most-intense window), results produced with the moments fit change. On the Iowa and
  Texas 24 h catalogs (median per storm): major axis −13 to −14 %, minor axis −20 %,
  elongation ratio +7 to +8 %, mean intensity and total rainfall +17 %, and the most-intense
  window moves for ~6 % of storms. Area, speed, and direction change only for those storms.
  Outputs generated with v1.0.0 and the moments fit should be regenerated.
- **`tracking.overlap_ratio_threshold`, `tracking.dry_spell_hr`, and `selection.min_area_fraction`
  were ignored.** They were validated in the config but `storm_tracking_event` and
  `continuos_storm` used hardcoded values (`0.2`, `0`, `0.05`). They are now passed through by
  `run_catalog_tracking` and `track_one_event` (new keyword arguments `ratio_threshold`,
  `dry_spell_time`, `area_fraction`, defaulting to the old values). Results are unchanged for
  configs that use the defaults.

### Changed

- **Storm-motion directions are now geodesic bearings relative to true north.**
  Before, directions came from `arctan2(dy, dx)` in the EPSG:2163 plane, so they
  were measured from *grid* north. That differs from true north by the local
  meridian convergence: about 0° at 100°W, about 4–5° in Iowa, and up to about
  20° at the CONUS edges. The result was a rotation that depended on the tile.
  Direction, speed, trajectory length, and angular variance now come from
  WGS84 geodesics (`pyproj.Geod`) between centroid lon/lat.
- `storm_properties_<domain>.csv`: `mean_direction_deg` and
  `mean_direction_vector_deg` keep the counterclockwise-from-east convention
  (0° = E, 90° = N), now relative to true east/north and in [0, 360).
  Figure labels read "° CCW from E".
- The motion grid (`build_storm_motion_grid`, `build_trajectory_direction_grid`)
  assigns geodesic `bearing_deg`/`direction_deg`. `u`/`v` remain
  projected-plane unit vectors, used for drawing on EPSG:2163 maps.
- Direction arrows on the lon/lat maps are drawn along the true bearing.

### Added

- `endpoint_direction_deg` (first-to-last centroid geodesic direction, CCW from E). Legacy
  grid-plane columns `mean_direction_grid_deg`, `angular_variance_grid`,
  `mean_speed_grid_ms`, `trajectory_length_grid_km` are kept for comparison.
- `motion.direction`: `segment_geodesics`, `mean_bearing`, `endpoint_bearing`, `bearing_to_math`,
  `line_length_geodesic`, `mean_speed_geodesic`.

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
