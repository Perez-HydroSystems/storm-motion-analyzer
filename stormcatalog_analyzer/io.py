"""Input/output helpers: catalog discovery, optional vector layers, output dirs.

Heavy geospatial imports (xarray, geopandas) are deferred to call time so that
``import stormcatalog_analyzer.io`` stays light.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import Config


def discover_events(cfg: Config) -> List[Path]:
    """Return the sorted list of ``.nc`` storm-event files in the catalog dir.

    Truncated to ``io.max_events`` when set (the user's "number of events" input).
    """
    catalog = cfg.catalog_dir
    events = sorted(p for p in catalog.glob("*.nc") if p.is_file())
    if not events:
        raise FileNotFoundError(f"No .nc storm events found in {catalog}")
    if cfg.io.max_events is not None:
        events = events[: cfg.io.max_events]
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
            f"Available: {list(ds.data_vars)}. Set tracking.var_name accordingly."
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
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
