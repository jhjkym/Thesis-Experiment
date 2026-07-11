"""Configuration loading and validation for experiment 01."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml


@dataclass(frozen=True)
class SceneConfig:
    """Rectangular scene and fixed sensor configuration."""

    width: float
    height: float
    sensor_position: Tuple[float, float]


@dataclass(frozen=True)
class TreeConfig:
    """Tree-trunk sampling configuration."""

    count: int
    radius_range: Tuple[float, float]
    min_spacing: float
    max_attempts: int


@dataclass(frozen=True)
class TrajectoryConfig:
    """Constant-velocity trajectory configuration."""

    sample_rate_hz: float
    duration_seconds: float
    initial_position: Tuple[float, float]
    velocity: Tuple[float, float]


@dataclass(frozen=True)
class DatasetConfig:
    """History/future window configuration."""

    history_steps: int
    future_steps: int
    num_samples: int


@dataclass(frozen=True)
class ObservationConfig:
    """Observation noise and frame-loss configuration."""

    position_noise_std: float
    random_dropout_probability: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully validated configuration for experiment 01."""

    seed: int
    scene: SceneConfig
    trees: TreeConfig
    trajectory: TrajectoryConfig
    dataset: DatasetConfig
    observation: ObservationConfig
    output_directory: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Configuration section '{}' must be a mapping".format(name))
    return value


def _pair(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Configuration field '{}' must contain two numbers".format(name))
    return float(value[0]), float(value[1])


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: YAML configuration path.

    Returns:
        A validated, immutable :class:`ExperimentConfig`.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "root")
    scene_raw = _mapping(root.get("scene"), "scene")
    trees_raw = _mapping(root.get("trees"), "trees")
    trajectory_raw = _mapping(root.get("trajectory"), "trajectory")
    dataset_raw = _mapping(root.get("dataset"), "dataset")
    observation_raw = _mapping(root.get("observation"), "observation")

    config = ExperimentConfig(
        seed=int(root["seed"]),
        scene=SceneConfig(
            width=float(scene_raw["width"]),
            height=float(scene_raw["height"]),
            sensor_position=_pair(scene_raw["sensor_position"], "scene.sensor_position"),
        ),
        trees=TreeConfig(
            count=int(trees_raw["count"]),
            radius_range=_pair(trees_raw["radius_range"], "trees.radius_range"),
            min_spacing=float(trees_raw["min_spacing"]),
            max_attempts=int(trees_raw.get("max_attempts", 10000)),
        ),
        trajectory=TrajectoryConfig(
            sample_rate_hz=float(trajectory_raw["sample_rate_hz"]),
            duration_seconds=float(trajectory_raw["duration_seconds"]),
            initial_position=_pair(
                trajectory_raw["initial_position"], "trajectory.initial_position"
            ),
            velocity=_pair(trajectory_raw["velocity"], "trajectory.velocity"),
        ),
        dataset=DatasetConfig(
            history_steps=int(dataset_raw["history_steps"]),
            future_steps=int(dataset_raw["future_steps"]),
            num_samples=int(dataset_raw["num_samples"]),
        ),
        observation=ObservationConfig(
            position_noise_std=float(observation_raw["position_noise_std"]),
            random_dropout_probability=float(
                observation_raw["random_dropout_probability"]
            ),
        ),
        output_directory=Path(str(root.get("output_directory", "outputs/experiment_01"))),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    """Raise ``ValueError`` when a configuration is internally inconsistent."""

    if config.scene.width <= 0.0 or config.scene.height <= 0.0:
        raise ValueError("Scene width and height must be positive")
    sensor_x, sensor_y = config.scene.sensor_position
    if not (0.0 <= sensor_x <= config.scene.width and 0.0 <= sensor_y <= config.scene.height):
        raise ValueError("Sensor position must lie inside the scene")
    radius_min, radius_max = config.trees.radius_range
    if config.trees.count < 0 or radius_min <= 0.0 or radius_max < radius_min:
        raise ValueError("Tree count and radius range must be valid")
    if config.trees.min_spacing < 0.0 or config.trees.max_attempts <= 0:
        raise ValueError("Tree spacing must be non-negative and max_attempts positive")
    if config.trajectory.sample_rate_hz <= 0.0 or config.trajectory.duration_seconds <= 0.0:
        raise ValueError("Trajectory sample rate and duration must be positive")
    if min(config.dataset.history_steps, config.dataset.future_steps, config.dataset.num_samples) <= 0:
        raise ValueError("Dataset sizes must be positive")
    if config.observation.position_noise_std < 0.0:
        raise ValueError("Position noise standard deviation must be non-negative")
    if not 0.0 <= config.observation.random_dropout_probability <= 1.0:
        raise ValueError("Random dropout probability must be in [0, 1]")

    start = config.trajectory.initial_position
    end = (
        start[0] + config.trajectory.velocity[0] * config.trajectory.duration_seconds,
        start[1] + config.trajectory.velocity[1] * config.trajectory.duration_seconds,
    )
    for name, position in (("initial", start), ("final", end)):
        if not (
            0.0 <= position[0] <= config.scene.width
            and 0.0 <= position[1] <= config.scene.height
        ):
            raise ValueError("Trajectory {} position must lie inside the scene".format(name))


def config_as_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Return a JSON-serializable dictionary used by the run log if needed."""

    return {
        "seed": config.seed,
        "scene": {
            "width": config.scene.width,
            "height": config.scene.height,
            "sensor_position": list(config.scene.sensor_position),
        },
        "trees": {
            "count": config.trees.count,
            "radius_range": list(config.trees.radius_range),
            "min_spacing": config.trees.min_spacing,
            "max_attempts": config.trees.max_attempts,
        },
        "trajectory": {
            "sample_rate_hz": config.trajectory.sample_rate_hz,
            "duration_seconds": config.trajectory.duration_seconds,
            "initial_position": list(config.trajectory.initial_position),
            "velocity": list(config.trajectory.velocity),
        },
        "dataset": {
            "history_steps": config.dataset.history_steps,
            "future_steps": config.dataset.future_steps,
            "num_samples": config.dataset.num_samples,
        },
        "observation": {
            "position_noise_std": config.observation.position_noise_std,
            "random_dropout_probability": config.observation.random_dropout_probability,
        },
        "output_directory": str(config.output_directory),
    }
