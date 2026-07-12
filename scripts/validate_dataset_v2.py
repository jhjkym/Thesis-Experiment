#!/usr/bin/env python
"""Validate a saved dataset-v2 directory without regenerating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thesis_experiment.data.validation_v2 import validate_dataset_directory


def _arguments(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a saved dataset v2.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    """Run validation and print a deterministic JSON summary."""

    args = _arguments(argv)
    directory = (
        args.dataset_dir
        if args.dataset_dir.is_absolute()
        else PROJECT_ROOT / args.dataset_dir
    )
    summary = validate_dataset_directory(directory)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
