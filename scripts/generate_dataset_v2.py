#!/usr/bin/env python
"""Generate leakage-free multi-scene dataset-v2 splits and saved-data figures."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thesis_experiment.config_v2 import SPLIT_NAMES, load_dataset_v2_config
from thesis_experiment.data.dataset_v2 import (
    generate_dataset_v2,
    save_generated_dataset_v2,
)


def _arguments(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-scene, multi-motion dataset version 2."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("dataset_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(str(path), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main(argv: Sequence[str] = None) -> int:
    """Generate, save, then visualize dataset-v2 using only saved NPZ inputs."""

    args = _arguments(argv)
    config_path = _project_path(args.config)
    config = load_dataset_v2_config(config_path)
    output_directory = _project_path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_directory / "run.log")
    logger.info("Starting dataset v2 generation with config %s", config_path)
    logger.info("Split seeds: %s", config.split_seeds)
    logger.info(
        "Scene counts: %s | episodes per scene: %d",
        config.scene_counts,
        config.episodes_per_scene,
    )

    generated = generate_dataset_v2(config)
    save_generated_dataset_v2(generated, output_directory)
    for split_name in SPLIT_NAMES:
        split = generated.manifest["splits"][split_name]
        logger.info(
            "%s: scenes=%d episodes=%d windows=%d visible=%.4f "
            "occluded=%.4f dropout=%.4f",
            split_name,
            split["scene_count"],
            split["episode_count"],
            split["window_count"],
            split["visible_ratio"],
            split["geometric_occlusion_ratio"],
            split["random_dropout_ratio"],
        )

    os.environ.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib"))
    from thesis_experiment.visualization.dataset_v2 import (
        generate_dataset_v2_figures,
    )

    figure_paths = generate_dataset_v2_figures(output_directory)
    logger.info("Generated %d figures from saved NPZ data", len(figure_paths))
    for figure_path in figure_paths:
        logger.info("Figure: %s", figure_path)
    logger.info("Dataset v2 generation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
