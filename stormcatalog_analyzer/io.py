"""Input/output helpers: catalog discovery, optional vector layers, output dirs.

Heavy geospatial imports (xarray, geopandas) are deferred to call time so that
``import stormcatalog_analyzer.io`` stays light.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import Config


def _storm_id(path: Path) -> int:
    """Storm id from a RainyDay filename (``..._storm_<id>_...``); -1 if absent."""
    try:
        return int(path.name.split("_storm_")[1].split("_")[0])
    except (IndexError, ValueError):
        return -1


def discover_events(cfg: Config) -> List[Path]:
    """Return the ``.nc`` storm-event files in the catalog dir.

    When ``io.max_events`` is set, keep only the N **most intense** events, ranked
    by the storm id in the filename (``..._storm_<id>_...``): a higher id means a
    more intense event, so e.g. ``..._storm_400_...`` outranks ``..._storm_399_...``.
    Files without a parseable id (id = -1) rank lowest. The selection is returned
    sorted by name, matching the full (untruncated) ordering.
    """
    catalog = cfg.catalog_dir
    events = sorted(p for p in catalog.glob("*.nc") if p.is_file())
    if not events:
        raise FileNotFoundError(f"No .nc storm events found in {catalog}")
    if cfg.io.max_events is not None:
        # keep the N highest storm ids (most intense), then restore name order
        most_intense = sorted(events, key=_storm_id, reverse=True)[: cfg.io.max_events]
        events = sorted(most_intense)
    return events


def resolve_event(cfg: Config, storm: str) -> Path:
    """Resolve a storm identifier to a catalog file path.

    Accepts an exact filename, a unique substring, or a numeric storm id
    (matched against the ``_storm_<id>_`` filename pattern).
    """
    events = sorted(cfg.catalog_dir.glob("*.nc"))
    by_name = {p.name: p for p in events}
    if storm in by_name:
        return by_name[storm]

    if storm.isdigit():
        token = f"_storm_{storm}_"
        matches = [p for p in events if token in p.name]
    else:
        matches = [p for p in events if storm in p.name]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No storm event matching {storm!r} in {cfg.catalog_dir}")
    raise ValueError(
        f"Ambiguous storm {storm!r}; matches: {[p.name for p in matches[:8]]}"
    )


def open_event(path: Path, var_name: str = "rain"):
    """Open a storm-event NetCDF with xarray (lazy import)."""
    import xarray as xr

    ds = xr.open_dataset(path)
    if var_name not in ds.variables:
        raise KeyError(
            f"Variable {var_name!r} not in {path.name}. "
            f"Available: {list(ds.data_vars)}. Set tracking.rainfall_var_name accordingly."
        )
    return ds


def load_vector(path: Optional[Path]):
    """Read a shapefile/vector layer with geopandas if it exists, else None."""
    if path is None:
        return None
    if not Path(path).exists():
        return None
    import geopandas as gpd

    return gpd.read_file(path)


def ensure_output_dirs(cfg: Config) -> dict:
    """Create the output directory tree and return the key paths."""
    base = cfg.output_dir / cfg.io.domain_name
    paths = {
        "base": base,
        "storm_track": base / "StormTrack",
        "storm_time_steps": base / "StormTimeSteps",
        "storm_animation": base / "StormAnimation",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
