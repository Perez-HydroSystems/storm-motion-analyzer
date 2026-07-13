"""Configuration for the storm-catalog diagnostics pipeline.

A single JSON file is the source of truth for model parameters and input/output
paths. The same ``Config`` object is consumed by the notebooks and the CLI.

JSON layout (see ``configs/testing_data.json``)::

    {
      "project_root": ".",            # paths below resolve relative to this
      "io":        { "catalog_dir": ..., "transposition_domain_path": ..., ... },
      "tracking":  { "rainfall_var_name": "rain", "morph_radius_cells": 2, ... },
      "selection": { "min_duration_hr": 6, "min_area_fraction": 0.05 },
      "motion":    { "intensity_window_hr": 12, "n_trajectories_plotted": 100 },
      "figure":    { "font_size": 0, "dpi": 300, "smooth_factor": 0.05 }
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class IOConfig:
    catalog_dir: str
    transposition_domain_path: Optional[str] = None
    control_area_path: Optional[str] = None
    grid_path: Optional[str] = None  # CONUS 5x5 reference grid (optional inset)
    output_dir: str = "outputs"
    domain_name: str = "Domain"
    max_events: Optional[int] = None  # process only the N most intense events by storm id (None = all)


@dataclass
class TrackingConfig:
    rainfall_var_name: str = "rain"          # name of the rain variable in the NetCDF
    morph_radius_cells: int = 2              # morphological merge radius (grid cells)
    rainfall_threshold_mmhr: float = 0.5     # rain-rate threshold for identification (mm/hr)
    ellipse_fit_method: str = "moments"      # "moments" | "contour"
    overlap_ratio_threshold: float = 0.2     # storm-overlap ratio to link across time (fraction)
    dry_spell_hr: int = 0                     # allowed gap when matching storms (hours)


@dataclass
class SelectionConfig:
    min_duration_hr: int = 6                 # drop storms shorter than this (hours)
    min_area_fraction: float = 0.05          # storm must cover >= this fraction of the domain


@dataclass
class MotionConfig:
    intensity_window_hr: int = 12            # window for the most-intense period (hours)
    n_trajectories_plotted: int = 100        # number of trajectories drawn on the map


@dataclass
class FigureConfig:
    font_size: int = 0
    dpi: int = 300
    smooth_factor: float = 0.05
    # Arrow styling for the storm_direction_vector_plot:
    arrow_scale: float = 1.0    # arrow length = trajectory_length / arrow_scale (LARGER => SHORTER)
    arrow_width: float = 0.005  # shaft width as a fraction of axis width (smaller => thinner)
    head_width: float = 2.0     # arrowhead width, in shaft-width units
    head_length: float = 1.5    # arrowhead length, in shaft-width units


@dataclass
class DirectionGridConfig:
    n_sectors: int = 4                        # number of direction sectors/bins (e.g. 4, 8)
    cell_size_km: float = 50.0                # storm-motion grid cell size (km)
    count_threshold: int = 100                # counts colormap split / min counts to show probability
    start_angle_deg: Optional[float] = None   # None -> -(360/n_sectors)/2 (centers sectors on primary dirs)


@dataclass
class Config:
    io: IOConfig
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    direction_grid: DirectionGridConfig = field(default_factory=DirectionGridConfig)
    project_root: str = "."

    # --- path resolution -------------------------------------------------
    def resolve(self, path: Optional[str]) -> Optional[Path]:
        """Resolve a config path against ``project_root`` (absolute paths kept)."""
        if path is None:
            return None
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        return (Path(self.project_root).expanduser() / p).resolve()

    @property
    def catalog_dir(self) -> Path:
        return self.resolve(self.io.catalog_dir)

    @property
    def transposition_domain_path(self) -> Optional[Path]:
        return self.resolve(self.io.transposition_domain_path)

    @property
    def control_area_path(self) -> Optional[Path]:
        return self.resolve(self.io.control_area_path)

    @property
    def grid_path(self) -> Optional[Path]:
        return self.resolve(self.io.grid_path)

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.io.output_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    # --- validation ------------------------------------------------------
    def validate(self) -> "Config":
        errors = []
        if self.tracking.ellipse_fit_method not in {"moments", "contour"}:
            errors.append(
                f"tracking.ellipse_fit_method must be 'moments' or 'contour', "
                f"got {self.tracking.ellipse_fit_method!r}"
            )
        if self.tracking.morph_radius_cells <= 0:
            errors.append("tracking.morph_radius_cells must be > 0")
        if not (0 < self.selection.min_area_fraction < 1):
            errors.append("selection.min_area_fraction must be in (0, 1)")
        if self.motion.n_trajectories_plotted <= 0:
            errors.append("motion.n_trajectories_plotted must be > 0")
        if self.motion.intensity_window_hr <= 0:
            errors.append("motion.intensity_window_hr must be > 0")
        if self.selection.min_duration_hr <= 0:
            errors.append("selection.min_duration_hr must be > 0")
        if self.tracking.dry_spell_hr < 0:
            errors.append("tracking.dry_spell_hr must be >= 0")
        if self.direction_grid.n_sectors < 2:
            errors.append("direction_grid.n_sectors must be >= 2")
        if self.direction_grid.cell_size_km <= 0:
            errors.append("direction_grid.cell_size_km must be > 0")
        if not self.catalog_dir.exists():
            errors.append(f"io.catalog_dir does not exist: {self.catalog_dir}")
        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))
        return self


def _section(cls, data: dict):
    """Build a dataclass section, ignoring unknown keys (with a warning-friendly error)."""
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def from_dict(data: dict, project_root: Optional[str] = None) -> Config:
    """Build a validated ``Config`` from a plain dict."""
    data = dict(data)
    io_data = data.get("io")
    if not io_data or "catalog_dir" not in io_data:
        raise ValueError("config must contain an 'io' section with 'catalog_dir'")
    cfg = Config(
        io=_section(IOConfig, io_data),
        tracking=_section(TrackingConfig, data.get("tracking", {})),
        selection=_section(SelectionConfig, data.get("selection", {})),
        motion=_section(MotionConfig, data.get("motion", {})),
        figure=_section(FigureConfig, data.get("figure", {})),
        direction_grid=_section(DirectionGridConfig, data.get("direction_grid", {})),
        project_root=project_root or data.get("project_root", "."),
    )
    return cfg.validate()


def load_config(path: str | Path) -> Config:
    """Load and validate a config from a JSON file.

    Relative paths in the file resolve against ``project_root`` if given, else the
    directory containing the config file.
    """
    path = Path(path).expanduser().resolve()
    with open(path) as fh:
        data = json.load(fh)
    project_root = data.get("project_root")
    if project_root is None:
        project_root = str(path.parent.parent if path.parent.name == "configs" else path.parent)
    return from_dict(data, project_root=project_root)
