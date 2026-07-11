#!/usr/bin/env python
"""Run experiment 01 synthetic occlusion data generation."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thesis_experiment.config import load_experiment_config
from thesis_experiment.data.dataset import (
    calculate_sample_statistics,
    create_dataset_windows,
    create_observations,
    local_to_world,
    save_dataset,
    save_statistics,
)
from thesis_experiment.data.trajectory import ConstantVelocityTrajectory, sample_times
from thesis_experiment.geometry.forest import generate_tree_trunks


def _arguments(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and visualize experiment 01 occlusion data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration path, relative to the current directory if needed.",
    )
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    """Resolve configured relative paths from the repository root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def _logger(log_path: Path) -> logging.Logger:
    """Create a logger writing both to stdout and the required run.log."""

    logger = logging.getLogger("experiment_01")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _visualization_sample(
    dataset: Dict[str, np.ndarray], sample_index: int
) -> Dict[str, Any]:
    """Restore one saved dataset row to world coordinates for plotting.

    Keeping this conversion separate ensures every example image can be traced
    to one exact ``dataset.npz`` row instead of to an unsaved raw sequence.
    """

    sample_count = int(dataset["history_position"].shape[0])
    index = int(sample_index)
    if not 0 <= index < sample_count:
        raise IndexError(
            "sample_index {} is outside dataset with {} rows".format(
                index, sample_count
            )
        )
    origin = dataset["coordinate_origin"][index]
    return {
        "sample_index": index,
        "scene_id": int(dataset["scene_id"][index]),
        "start_index": int(dataset["sample_start_index"][index]),
        "history_position": local_to_world(
            dataset["history_position"][index], origin
        ),
        "history_true_position": local_to_world(
            dataset["history_true_position"][index], origin
        ),
        "future_position": local_to_world(dataset["future_position"][index], origin),
        "history_mask": dataset["history_mask"][index].astype(bool),
        "history_occluded": dataset["history_occluded"][index].astype(bool),
        "history_random_dropout": dataset["history_random_dropout"][index].astype(
            bool
        ),
        "sensor_position": local_to_world(dataset["sensor_position"][index], origin),
        "tree_centers": local_to_world(dataset["tree_centers"][index], origin),
        "tree_radii": dataset["tree_radii"][index].copy(),
    }


def main(argv: Sequence[str] = None) -> int:
    """Generate the configured scene, dataset, plots, statistics, and run log."""

    args = _arguments(argv)
    config_path = _project_path(args.config)
    config = load_experiment_config(config_path)
    output_directory = _project_path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_directory / "run.log")
    logger.info("Starting experiment 01 with config %s", config_path)
    logger.info("Random seed: %d", config.seed)

    seed_sequence = np.random.SeedSequence(config.seed)
    tree_seed, observation_seed = seed_sequence.spawn(2)
    tree_rng = np.random.default_rng(tree_seed)
    observation_rng = np.random.default_rng(observation_seed)

    scene_size = (config.scene.width, config.scene.height)
    sensor_position = np.asarray(config.scene.sensor_position, dtype=float)
    tree_centers, tree_radii = generate_tree_trunks(
        scene_size=scene_size,
        tree_count=config.trees.count,
        radius_range=config.trees.radius_range,
        min_spacing=config.trees.min_spacing,
        sensor_position=sensor_position,
        rng=tree_rng,
        max_attempts=config.trees.max_attempts,
    )
    logger.info("Generated %d non-overlapping tree trunks", tree_centers.shape[0])

    times = sample_times(
        config.trajectory.sample_rate_hz, config.trajectory.duration_seconds
    )
    trajectory = ConstantVelocityTrajectory(
        np.asarray(config.trajectory.initial_position, dtype=float),
        np.asarray(config.trajectory.velocity, dtype=float),
    )
    true_positions = trajectory.position_at(times)
    observations = create_observations(
        true_positions,
        sensor_position,
        tree_centers,
        tree_radii,
        config.observation.position_noise_std,
        config.observation.random_dropout_probability,
        observation_rng,
    )
    dataset = create_dataset_windows(
        true_positions,
        observations,
        sensor_position,
        tree_centers,
        tree_radii,
        history_steps=config.dataset.history_steps,
        future_steps=config.dataset.future_steps,
        num_samples=config.dataset.num_samples,
        dt=1.0 / config.trajectory.sample_rate_hz,
        scene_id=0,
    )
    statistics = calculate_sample_statistics(dataset)
    example_sample_index = config.dataset.num_samples // 2
    example = _visualization_sample(dataset, example_sample_index)
    history_start = int(example["start_index"])
    history_end = history_start + config.dataset.history_steps - 1
    future_start = history_end + 1
    future_end = future_start + config.dataset.future_steps - 1
    visible_count = int(np.sum(example["history_mask"]))
    occluded_count = int(np.sum(example["history_occluded"]))
    dropout_count = int(np.sum(example["history_random_dropout"]))
    statistics["visualization_example"] = {
        "scene_id": int(example["scene_id"]),
        "sample_index": int(example["sample_index"]),
        "history_start_index": history_start,
        "history_end_index": history_end,
        "future_start_index": future_start,
        "future_end_index": future_end,
        "visible_history_points": visible_count,
        "occluded_history_points": occluded_count,
        "random_dropout_history_points": dropout_count,
    }
    save_dataset(dataset, output_directory / "dataset.npz")
    save_statistics(statistics, output_directory / "sample_statistics.json")

    os.environ.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib"))
    from thesis_experiment.visualization import (
        plot_mask_timeline,
        plot_scene,
        plot_trajectory,
    )

    example_label = "scene {} | sample {} | history {}-{}".format(
        example["scene_id"], example["sample_index"], history_start, history_end
    )
    sample_path = np.concatenate(
        (example["history_true_position"], example["future_position"]), axis=0
    )
    plot_scene(
        output_directory / "scene_example.png",
        scene_size,
        example["sensor_position"],
        example["tree_centers"],
        example["tree_radii"],
        sample_path,
        title="Forest scene | " + example_label,
    )
    plot_trajectory(
        output_directory / "trajectory_example.png",
        example["history_true_position"],
        example["history_position"],
        example["history_mask"],
        example["sensor_position"],
        example["tree_centers"],
        example["tree_radii"],
        scene_size,
        title="History observations | " + example_label,
    )
    plot_mask_timeline(
        output_directory / "mask_timeline.png",
        times[history_start : history_end + 1],
        example["history_mask"],
        example["history_occluded"],
        example["history_random_dropout"],
        title="Observation masks | " + example_label,
    )

    logger.info("Saved %d samples to %s", statistics["total_samples"], output_directory)
    logger.info(
        "Visibility %.4f | occlusion %.4f | random dropout %.4f",
        statistics["visible_observation_ratio"],
        statistics["occlusion_ratio"],
        statistics["random_dropout_ratio"],
    )
    logger.info(
        "Visualization example: scene_id=%d sample_index=%d history=%d-%d "
        "future=%d-%d",
        example["scene_id"],
        example["sample_index"],
        history_start,
        history_end,
        future_start,
        future_end,
    )
    logger.info(
        "Visualization history counts: visible=%d occluded=%d random_dropout=%d",
        visible_count,
        occluded_count,
        dropout_count,
    )
    logger.info("Dataset array shapes: %s", statistics["array_shapes"])
    logger.info("Experiment 01 completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
