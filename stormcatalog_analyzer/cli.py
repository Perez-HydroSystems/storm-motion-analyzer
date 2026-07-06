"""Command-line interface for stormcatalog-analyzer.

Usage::

    stormcatalog-analyzer diagnostics --config configs/testing_data.json
    stormcatalog-analyzer event       --config configs/testing_data.json --storm 100

The root helper ``run_diagnostics.py`` defaults to the ``diagnostics`` subcommand,
so ``python run_diagnostics.py --config configs/testing_data.json`` also works.
"""
from __future__ import annotations

import argparse

from .config import load_config

SUBCOMMANDS = ("diagnostics", "event")


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

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
