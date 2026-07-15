# Storm Motion Analyzer
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
 
Storm detection and tracking toolkit for characterizing storm motion (trajectory, direction, speed, and angular variance) from hourly gridded rainfall data.
 
## Tool description
 
Storm-Motion-Analyzer detects storms in NetCDF gridded rainfall data using rainfall thresholding, connected-component labeling, and morphological operations (dilation/erosion), tracks them through time with overlap-based Lagrangian tracking, and characterizes their motion (trajectory, direction, speed, and angular variance), intensity, and spatial extent via moment-based ellipse fitting.
 
The package is designed primarily to process the output storm catalogs produced by the [RainyDay](https://github.com/HydroclimateExtremesGroup/RainyDay) stochastic storm transposition framework, but it works with any hourly gridded NetCDF rainfall dataset. The toolkit has been tested using NOAA's Analysis of Record for Calibration (AORC) rainfall data.

![Storm tracking demo](storm_tracking_process.gif)
 
## Quick start
 
**Prerequisites:** Python 3.9+ and the scientific + geospatial Python stack (NumPy, SciPy, pandas, xarray, netCDF4, scikit-image, geopandas, shapely, cartopy, pyproj, matplotlib, seaborn, windrose).

> **Install the dependencies before using the repo.** The recommended way is a conda environment from [`environment.yml`](environment.yml), which pulls the geospatial stack (cartopy, geopandas, pyproj) cleanly from conda-forge. A `pip install -r requirements.txt` also works if you already have the system GDAL/GEOS/PROJ libraries.
>
> **Troubleshooting:** if saving a figure raises `GEOSException: ... Points of LinearRing do not form a closed linestring`, your `shapely` is ≥ 2.1 (it tightened ring validation and breaks cartopy's gridline labels) — pin it with `pip install "shapely<2.1"`.
 
```bash
git clone https://github.com/Perez-HydroSystems/storm-motion-analyzer.git
cd storm-motion-analyzer
 
conda env create -f environment.yml      # create the env with all dependencies
conda activate storm-motion-analyzer

pip install -e .                         # optional: enables the `stormcatalog-analyzer` CLI
```
 
You need two things to run an analysis: a storm event or a storm catalog (a folder of NetCDF events) and a JSON config. All inputs, paths, and model parameters live in one config file — see [`configs/testing_data.json`](configs/testing_data.json).
 
```bash
# Catalog-level diagnostic plots (main deliverable)
python run_diagnostics.py --config configs/testing_data.json
 
# Per-event tracking figures for a single storm (by filename, substring, or id)
python run_diagnostics.py event --config configs/testing_data.json --storm 100
```
 
If installed with `pip install -e .`, the same commands are available as `stormcatalog-analyzer diagnostics|event ...`.
 
Prefer a notebook? Open [`notebooks/storm_motion_catalog.ipynb`](notebooks/storm_motion_catalog.ipynb) — it loads the config and runs the same pipeline.
 
## Usage
 
### Python API
 
```python
from stormcatalog_analyzer.config import load_config
from stormcatalog_analyzer import pipeline
 
cfg = load_config("configs/testing_data.json")
cfg.tracking.rainfall_threshold_mmhr = 2   # optional inline overrides
 
out = pipeline.run(cfg)                # track -> summarize -> diagnostics
out["summary"].storm_properties.head()
print(out["figures"])                  # saved figure paths
print(out["table"])                    # per-storm properties CSV
```
 
### Outputs
 
Written to `<output_dir>/<domain_name>/`:
 
- `storm_properties_<domain>.csv` — one row per storm: mean speed & direction, mean/peak intensity, total accumulated rainfall, ellipse major/minor axis + angle, area, duration, trajectory length.
- Diagnostic figures: trajectory map + wind rose, direction vectors, statistics pairplot.
### Notebooks
 
| Notebook | Purpose |
|---|---|
| [`notebooks/storm_motion_catalog.ipynb`](notebooks/storm_motion_catalog.ipynb) | Process all the events in the storm catalog, generating the summary and individual diagnostic plots. |
| [`notebooks/storm_motion_event.ipynb`](notebooks/storm_motion_event.ipynb) | Per-event spatiotemporal plots and tracking results. |
 
## What it extracts
 
- **Motion** — trajectory, translation speed, direction, spatial probability of direction motion.
- **Intensity** — peak and mean rainfall intensity for each storms.
- **Spatial morphology** — storm area coverage, ellipse area and dimensions.


 

## Authors
 
- Diego F. Osorio-Giraldo — School of Civil and Environmental Engineering, Oklahoma State University
- Gabriel Perez — School of Civil and Environmental Engineering, Oklahoma State University

## Acknowledgements
- Yuan Liu — provided seminal functions for the storm tracking algorithm.

## How to cite

If you use this software, please cite it — GitHub's "Cite this repository" button (generated from [`CITATION.cff`](CITATION.cff)) gives ready-made BibTeX/APA entries:

> Osorio-Giraldo, D. F., & Perez, G. (2026). *Storm Motion Analyzer* (Version 1.0.0) [Computer software]. https://github.com/Perez-HydroSystems/storm-motion-analyzer

<!-- Zenodo DOI badge and DOI-based citation will be added here once the v1.0.0 release is archived on Zenodo. -->

## License
 
MIT — see [LICENSE](LICENSE).