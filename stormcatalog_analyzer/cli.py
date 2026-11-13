"""Command-line interface for stormcatalog-analyzer.

Usage::

    stormcatalog-analyzer diagnostics --config configs/testing_data.json
    stormcatalog-analyzer event       --config configs/testing_data.json --storm 100
    stormcatalog-analyzer drift       --config configs/testing_data.json --axis 156
    stormcatalog-analyzer enlarged    --catalog-dir <tile>/StormCatalog --aorc-dir <AORC> --tile B7 --margin 5 \
                                      --out B7.parquet [--workers 3]
    stormcatalog-analyzer systems     --footprints *_footprints.parquet --out systems.csv

The root helper ``run_diagnostics.py`` defaults to the ``diagnostics`` subcommand,
so ``python run_diagnostics.py --config configs/testing_data.json`` also works.
"""
from __future__ import annotations

import argparse

from .config import load_config

SUBCOMMANDS = ("diagnostics", "event", "drift", "enlarged", "systems")


def _apply_overrides(cfg, args):
    if getattr(args, "output_dir", None):
        cfg.io.output_dir = args.output_dir
    if getattr(args, "domain", None):
        cfg.io.domain_name = args.domain
    if getattr(args, "max_events", None) is not None:
        cfg.io.max_events = args.max_events
    return cfg


def cmd_diagnostics(args) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    from .pipeline import run

    run(cfg, verbose=not args.quiet)
    return 0


def cmd_event(args) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    from .pipeline import generate_event_diagnostics

    generate_event_diagnostics(cfg, args.storm, verbose=not args.quiet)
    return 0


def cmd_drift(args) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    if getattr(args, "axis", None) is not None:
        cfg.drift.axis_azimuth_deg = args.axis
    from .pipeline import run_drift

    run_drift(cfg, verbose=not args.quiet)
    return 0


def cmd_enlarged(args) -> int:
    from pathlib import Path

    from .enlarged import EnlargedConfig, run_catalog

    cfg = EnlargedConfig(margin_deg=args.margin, selection=args.selection)
    rows, fps = run_catalog(args.catalog_dir, args.aorc_dir, cfg, tile=args.tile, workers=args.workers,
                            with_tile=not args.no_tile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out)
    fps.to_parquet(out.with_name(out.stem + "_footprints.parquet"))
    if not args.quiet:
        n_ok = int(rows["retained"].fillna(False).astype(bool).sum()) if "retained" in rows else 0
        print(f"{args.tile}: {len(rows)} events, {n_ok} retained -> {out}")
    return 0


def cmd_systems(args) -> int:
    import pandas as pd

    from .enlarged import assign_system_ids

    fp = pd.concat([pd.read_parquet(f) for f in args.footprints], ignore_index=True)
    ids, pairs = assign_system_ids(fp, time_overlap=args.time_overlap, footprint_overlap=args.footprint_overlap)
    ids.to_csv(args.out, index=False)
    pairs.to_csv(str(args.out).replace(".csv", "_pairs.csv"), index=False)
    if not args.quiet:
        print(f"{len(ids)} events -> {ids['system_id'].nunique()} systems ({len(pairs)} linked pairs)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stormcatalog-analyzer",
        description="Storm-motion diagnostics from a RainyDay storm catalog.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p):
        p.add_argument("--config", required=True, help="Path to a JSON config file.")
        p.add_argument("--output-dir", help="Override io.output_dir.")
        p.add_argument("--domain", help="Override io.domain_name.")
        p.add_argument("--max-events", type=int, help="Process only the first N catalog events.")
        p.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    p = sub.add_parser("diagnostics", help="Catalog-level diagnostic plots.")
    _common(p)
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser("event", help="Per-event tracking figures for a single storm.")
    _common(p)
    p.add_argument("--storm", required=True,
                   help="Storm filename, unique substring, or numeric id (e.g. 100).")
    p.set_defaults(func=cmd_event)

    p = sub.add_parser("drift", help="Envelope drift of the rain field for every catalog event.")
    _common(p)
    p.add_argument("--axis", type=float, help="Reference axis azimuth (compass deg) for along-axis drift.")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("enlarged", help="Track every catalog event on the tile enlarged by a margin (AORC).")
    p.add_argument("--catalog-dir", required=True, help="RainyDay StormCatalog folder of one tile.")
    p.add_argument("--aorc-dir", required=True, help="Folder of daily AORC.<YYYYMMDD>.preciptemp.nc files.")
    p.add_argument("--tile", required=True, help="Tile name (also the coherence seed prefix).")
    p.add_argument("--margin", type=float, default=5.0, help="Margin added on every side (degrees).")
    p.add_argument("--selection", default="circle", choices=("circle", "v6"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-tile", action="store_true", help="Skip the tile-only V6 comparison.")
    p.add_argument("--out", required=True, help="Output parquet (footprints next to it).")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_enlarged)

    p = sub.add_parser("systems", help="Cross-tile duplicate systems from enlarged-tracking footprints.")
    p.add_argument("--footprints", nargs="+", required=True)
    p.add_argument("--time-overlap", type=float, default=0.5)
    p.add_argument("--footprint-overlap", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_systems)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
