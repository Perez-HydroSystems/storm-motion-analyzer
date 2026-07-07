# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project: Stormcatalog-analyzer

Detect, track, and characterize the **motion of storms** from gridded rainfall
"storm catalogs" produced by the **RainyDay** package. Quantifies storm direction,
speed, spatial extent (fitted ellipses), and intensity via Lagrangian tracking of
storm centroids. Supports the CONUS storm-motion paper (Osorio-Giraldo, Perez,
Wright & Liu — OSU / UW-Madison). MIT licensed.

**Core idea:** the user provides a storm catalog (folder of N `.nc` events) + a JSON
config; the tool produces a modular set of diagnostic plots. Runs as a notebook
**and** headless via a CLI.

## Architecture (installable package)

```
stormcatalog_analyzer/         # the package (import as `stormcatalog_analyzer`)
  config.py     # Config dataclasses + load_config(json) + validation; SINGLE source of truth
  io.py         # discover_events, open_event (xarray), load_vector (geopandas), output dirs
  pipeline.py   # run_catalog_tracking -> summarize_motion -> generate_catalog_diagnostics; run()
  cli.py        # argparse; subcommands `diagnostics`, `event`
  tracking/     # identification.py, tracking.py, catalog.py, dataset.py  (storm tracking core)
  motion/       # direction.py (per-storm dir/speed), grid.py (storm-motion grid, experimental)
  stats/        # circular.py (circular_mean/percentile/trend/permutation)
  plotting/     # helpers.py (_plot_geographic_outline, _annotate_conus_grid),
                #   diagnostics.py (catalog-level figures + pairplot), per_event.py (track/time-steps),
                #   animation.py (animate_storm_tracking -> per-storm tracking GIF)
  geometry.py   # control-area shapefile builders
run_diagnostics.py   # thin root CLI wrapper (defaults to `diagnostics`)
configs/testing_data.json  # default config -> relative paths into testing_data/
storm_tracking_results.ipynb # MAIN notebook (thin: load config -> pipeline.run -> show figures)
notebooks/legacy/    # old exploratory notebooks (unmaintained)
pyproject.toml  requirements.txt  README.md  CLAUDE.md
```

The package was migrated from a flat `functions/` folder; the old orchestration loop
(copy-pasted across notebooks) now lives in `pipeline.py`.

## Pipeline (pipeline.py) — the call path
1. `run_catalog_tracking(cfg)` → loops `io.discover_events`; per event:
   `tracking.dataset.storm_tracking_event` → skip if `longest_duration < min_duration_steps`
   → `continuos_storm` → `storm_tracking_features`. Returns `{event_name: event_dict}`.
2. `summarize_motion(results, cfg)` → per storm: most-intense window
   (`motion.direction.mean_precipitation_within_ellipse` + `get_max_accumulated_value`),
   `mean_velocity`/`mean_direction`/`mean_direction_weighted`; builds `storm_trajectories`,
   `geographic_trajectories` (EPSG:2163→4326), `storm_properties`, `df`. Returns `MotionSummary`.
3. `generate_catalog_diagnostics(summary, cfg)` → saves `storm_properties_<domain>.csv`
   (one row per storm: mean speed/direction, mean & peak intensity, total accumulated rainfall,
   ellipse major/minor/angle, area, duration, trajectory length) + the figures
   (`diagnostic_plot_storm_trajectories`, `diagnostic_plot_storm_direction_vectors`,
   `plot_storm_properties_pairplot`); returns `{"figures": [...], "table": csv}`.
- `run(cfg)` chains all three; returns `{"results","summary","figures","table"}`.
- `storm_properties` schema (also the CSV) uses columns `mean_speed_ms`, `mean_direction_deg`,
  `mean_intensity_mmh`, `peak_intensity_mmh`, `total_rainfall_mm`, `storm_area_km2`,
  `mean_major_axis_km`, `mean_minor_axis_km`, `mean_ellipse_angle_deg`, `angular_variance`, etc.

## Config (config.py)
JSON ⇄ dataclasses, validated once. Parameter names are self-explanatory with a unit suffix.
Sections:
- `io`: catalog_dir, transposition_domain_path, control_area_path?, grid_path?, output_dir,
  domain_name, max_events?
- `tracking`: rainfall_var_name, morph_radius_cells, rainfall_threshold_mmhr,
  ellipse_fit_method∈{moments,contour}, overlap_ratio_threshold, dry_spell_steps
- `selection`: min_duration_steps, min_area_fraction
- `motion`: intensity_window (e.g. "12H"), n_trajectories_plotted
- `figure`: font_size, dpi, smooth_factor, arrow_scale, arrow_width, head_width, head_length
- `direction_grid`: n_sectors, cell_size_km, count_threshold, start_angle_deg — for the
  storm-motion direction-probability diagnostic (`pipeline.build_direction_probability_field`
  + `plotting.motion_grid.plot_direction_probability_summary`; panel (c)/wheel adapt to n_sectors)

Paths resolve relative to `project_root`; **no absolute paths in committed code/configs**.
(overlap_ratio_threshold, dry_spell_steps, min_area_fraction are config-only for now — the
low-level tracking functions still use their internal defaults.)

## Input data
NetCDF per event: dims `(time, latitude, longitude)`, rain var named by `tracking.rainfall_var_name`
(default `"rain"`; the legacy `storm_events/` samples use `"RAINRATE"` — set accordingly).
CRS WGS84. RainyDay catalog filenames `<catalog>_storm_<id>_<date>.nc`.

## Run / verify
- Env: use `/opt/anaconda3/envs/Hydro_process/bin/python` (has the full stack: xarray,
  geopandas, cartopy, opencv, windrose, etc.). Base python only has numpy.
- Headless plotting: set `MPLBACKEND=Agg`.
- CLI: `MPLBACKEND=Agg python run_diagnostics.py --config configs/testing_data.json`
- Subset smoke test: monkeypatch `stormcatalog_analyzer.io.discover_events` to return a slice.
- `testing_data/` (Iowa Domain_22 catalog, 400 storms) is **gitignored** (too big); outputs land there too.

## Conventions / gotchas
- Direction: `direction_deg` math convention (0=E, CCW+); `bearing_deg` compass (0=N, CW+).
- `ellipse_fit_method`: contour method only when exactly `"contour"`; anything else ⇒ moments.
- `mean_direction_weighted` returns 2 values normally but 3 when `len(x)<2` — pipeline guards `len(x)>=2`.
- The CONUS domain-location inset needs a special `grid` shapefile (`row_index`/`col_index`/`left`…);
  it's optional now (`grid_path: null`) and skipped via `_annotate_conus_grid` when absent.
- Diagnostic plot functions hardcode `savefig(dpi=300)` and take `save_path` as a **string**
  (they do `save_path + "/..."`), so pass `str(dir)`.
- Benign warnings: pandas `'H'`→`'h'` deprecation (intensity_window), cartopy/tight_layout notes.
- Grid modules: `motion/grid.py` = compute; `plotting/motion_grid.py` = the grid plot functions.

## Git / workflow
- Remote: `git@github.com:Perez-HydroSystems/storm-motion-analyzer.git` (SSH), default branch `main`.
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Branch before committing on `main`; only commit/push when asked.

## CLI subcommands (cli.py / run_diagnostics.py)
- `diagnostics` → `pipeline.run` (catalog-level figures). Default subcommand of `run_diagnostics.py`.
- `event --storm <name|substring|id>` → `pipeline.generate_event_diagnostics` (per-event track + time-steps).
- `generate_all_event_diagnostics(cfg, results, summary)` → per-event figures for EVERY storm
  (reuses `run()` output; no re-tracking). Notebook cell in `storm_motion_catalog.ipynb`.
- `generate_event_animation(cfg, storm, **kw)` → per-storm tracking GIF via
  `plotting.animation.animate_storm_tracking` (3 phases: storm moving → current-step ellipse +
  growing centroid trail → trajectory + mean-direction vector + stats box). Re-tracks one event;
  the notebook cell (§6) instead reuses `out["results"]`/`summary.storm_trajectories` (no re-tracking).
  Saved to `<output_dir>/<domain>/StormAnimation/`. Needs Pillow (PillowWriter). Reads motion metrics
  (`mean_direction_weighted`/`mean_velocity`/`var_dir_weighted`) off the event_dict — present because
  `compute_storm_trajectory` writes them there.
- `--max-events N` (general flag) / `io.max_events` subsamples the catalog to the first N events.

## Notebooks
- `storm_motion_catalog.ipynb` (root) — catalog diagnostics + a cell for all per-event figures,
  §5 direction-probability summary, §6 storm-tracking animation GIF.
- `notebooks/storm_motion_event.ipynb` — per-event figures for one storm.
- The event notebook starts with a bootstrap cell that chdirs to the repo root, so it runs from anywhere.

## Status
- **Done:** full package restructure; config/io/pipeline/cli; 2 notebooks + 2 CLI subcommands
  (`diagnostics`, `event`); `.gitignore`; fixed the undefined `_plot_geographic_outline` bug; made the
  CONUS `grid` inset optional; added `io.max_events`. Verified end-to-end on `testing_data/`
  (full 400-storm catalog + per-event).
- **Shared helper:** `pipeline.compute_storm_trajectory(storm, interval, storm_id)` computes one storm's
  window/direction/speed AND writes those metrics back onto the event_dict (the per-event plots read them there);
  used by both `summarize_motion` and `track_one_event`.
- **Not committed yet** — all changes are in the working tree.
