#!/usr/bin/env python
"""Headless entry point for the catalog diagnostics.

Run from the repository root::

    python run_diagnostics.py --config configs/testing_data.json
    python run_diagnostics.py diagnostics --config configs/testing_data.json

Defaults to the ``diagnostics`` subcommand when none is given.
"""
import sys

from stormcatalog_analyzer.cli import SUBCOMMANDS, main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["diagnostics"] + argv
    sys.exit(main(argv))
