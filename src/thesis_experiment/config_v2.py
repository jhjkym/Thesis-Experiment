"""Configuration loading and validation for the multi-scene dataset v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import math
import yaml

from thesis_experiment.data.trajectory import sample_times


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SceneV2Config:
    """Rectangular scene and sensor-sampling configuration."""

    width: float
    height: float
    sensor_margin: float
    sensor_tree_clearance: float


@dataclass(frozen=True)
class TreeV2Config:
    """Circular tree-trunk sampling configuration."""

    count: int
    radius_range: Tuple[float, float]
    min_spacing: float
    max_attempts: int


@dataclass(frozen=True)
class TrajectoryV2Config:
    """Motion-parameter sampling and physical validity limits."""

    sample_rate_hz: float
    duration_seconds: float
    initial_speed_range: Tuple[float, float]
    acceleration_magnitude_range: Tuple[float, float]
    turn_rate_magnitude_range: Tuple[float, float]
    stop_start_time_range: Tuple[float, float]
    stop_duration_range: Tuple[float, float]
    piecewise_turn_time_range: Tuple[float, float]
    piecewise_turn_angle_magnitude_range: Tuple[float, float]
    transition_duration_seconds: float
    boundary_margin: float
    min_tree_clearance: float
    max_speed: float
    max_acceleration: float
    max_attempts: int


@dataclass(frozen=True)
class WindowV2Config:
    """History/future window dimensions and episode-local stride."""

    history_steps: int
    future_steps: int
    window_stride: int
    minimum_visible_history_steps: int
    minimum_consecutive_visible_steps: int
    minimum_windows_per_episode: int


@dataclass(frozen=True)
class ObservationV2Config:
    """Noisy observation and random loss configuration."""

    position_noise_std: float
    random_dropout_probability: float


@dataclass(frozen=True)
class DatasetV2Config:
    """Fully validated multi-scene dataset configuration."""

    split_seeds: Dict[str, int]
    scene_counts: Dict[str, int]
    episodes_per_scene: int
    scene: SceneV2Config
    trees: TreeV2Config
    trajectory: TrajectoryV2Config
    window: WindowV2Config
    observation: ObservationV2Config
    output_directory: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Configuration section '{}' must be a mapping".format(name))
    return value


def _pair(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Configuration field '{}' must contain two values".format(name))
    pair = float(value[0]), float(value[1])
    if not all(math.isfinite(component) for component in pair):
        raise ValueError("Configuration field '{}' must be finite".format(name))
    if pair[1] < pair[0]:
        raise ValueError("Configuration field '{}' must be ordered".format(name))
    return pair


def load_dataset_v2_config(path: Path) -> DatasetV2Config:
    """Load and validate a dataset-v2 YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "root")
    seeds_raw = _mapping(raw.get("split_seeds"), "split_seeds")
    counts_raw = _mapping(raw.get("scene_counts"), "scene_counts")
    scene_raw = _mapping(raw.get("scene"), "scene")
    trees_raw = _mapping(raw.get("trees"), "trees")
    trajectory_raw = _mapping(raw.get("trajectory"), "trajectory")
    window_raw = _mapping(raw.get("window"), "window")
    observation_raw = _mapping(raw.get("observation"), "observation")

    config = DatasetV2Config(
        split_seeds={name: int(seeds_raw[name]) for name in SPLIT_NAMES},
        scene_counts={name: int(counts_raw[name]) for name in SPLIT_NAMES},
        episodes_per_scene=int(raw["episodes_per_scene"]),
        scene=SceneV2Config(
            width=float(scene_raw["width"]),
            height=float(scene_raw["height"]),
            sensor_margin=float(scene_raw["sensor_margin"]),
            sensor_tree_clearance=float(scene_raw.get("sensor_tree_clearance", 0.0)),
        ),
        trees=TreeV2Config(
            count=int(trees_raw["count"]),
            radius_range=_pair(trees_raw["radius_range"], "trees.radius_range"),
            min_spacing=float(trees_raw["min_spacing"]),
            max_attempts=int(trees_raw["max_attempts"]),
        ),
        trajectory=TrajectoryV2Config(
            sample_rate_hz=float(trajectory_raw["sample_rate_hz"]),
            duration_seconds=float(trajectory_raw["duration_seconds"]),
            initial_speed_range=_pair(
                trajectory_raw["initial_speed_range"],
                "trajectory.initial_speed_range",
            ),
            acceleration_magnitude_range=_pair(
                trajectory_raw["acceleration_magnitude_range"],
                "trajectory.acceleration_magnitude_range",
            ),
            turn_rate_magnitude_range=_pair(
                trajectory_raw["turn_rate_magnitude_range"],
                "trajectory.turn_rate_magnitude_range",
            ),
            stop_start_time_range=_pair(
                trajectory_raw["stop_start_time_range"],
                "trajectory.stop_start_time_range",
            ),
            stop_duration_range=_pair(
                trajectory_raw["stop_duration_range"],
                "trajectory.stop_duration_range",
            ),
            piecewise_turn_time_range=_pair(
                trajectory_raw["piecewise_turn_time_range"],
                "trajectory.piecewise_turn_time_range",
            ),
            piecewise_turn_angle_magnitude_range=_pair(
                trajectory_raw["piecewise_turn_angle_magnitude_range"],
                "trajectory.piecewise_turn_angle_magnitude_range",
            ),
            transition_duration_seconds=float(
                trajectory_raw["transition_duration_seconds"]
            ),
            boundary_margin=float(trajectory_raw["boundary_margin"]),
            min_tree_clearance=float(trajectory_raw["min_tree_clearance"]),
            max_speed=float(trajectory_raw["max_speed"]),
            max_acceleration=float(trajectory_raw["max_acceleration"]),
            max_attempts=int(trajectory_raw["max_attempts"]),
        ),
        window=WindowV2Config(
            history_steps=int(window_raw["history_steps"]),
            future_steps=int(window_raw["future_steps"]),
            window_stride=int(window_raw["window_stride"]),
            minimum_visible_history_steps=int(
                window_raw.get("minimum_visible_history_steps", 1)
            ),
            minimum_consecutive_visible_steps=int(
                window_raw.get("minimum_consecutive_visible_steps", 1)
            ),
            minimum_windows_per_episode=int(
                window_raw.get("minimum_windows_per_episode", 1)
            ),
        ),
        observation=ObservationV2Config(
            position_noise_std=float(observation_raw["position_noise_std"]),
            random_dropout_probability=float(
                observation_raw["random_dropout_probability"]
            ),
        ),
        output_directory=Path(str(raw.get("output_directory", "outputs/dataset_v2"))),
    )
    validate_dataset_v2_config(config)
    return config


def validate_dataset_v2_config(config: DatasetV2Config) -> None:
    """Raise ``ValueError`` when dataset-v2 settings are inconsistent."""

    finite_scalars = {
        "scene.width": config.scene.width,
        "scene.height": config.scene.height,
        "scene.sensor_margin": config.scene.sensor_margin,
        "scene.sensor_tree_clearance": config.scene.sensor_tree_clearance,
        "trees.min_spacing": config.trees.min_spacing,
        "trajectory.sample_rate_hz": config.trajectory.sample_rate_hz,
        "trajectory.duration_seconds": config.trajectory.duration_seconds,
        "trajectory.transition_duration_seconds": (
            config.trajectory.transition_duration_seconds
        ),
        "trajectory.boundary_margin": config.trajectory.boundary_margin,
        "trajectory.min_tree_clearance": config.trajectory.min_tree_clearance,
        "trajectory.max_speed": config.trajectory.max_speed,
        "trajectory.max_acceleration": config.trajectory.max_acceleration,
        "observation.position_noise_std": config.observation.position_noise_std,
        "observation.random_dropout_probability": (
            config.observation.random_dropout_probability
        ),
    }
    nonfinite = [
        name for name, value in finite_scalars.items() if not math.isfinite(value)
    ]
    if nonfinite:
        raise ValueError(
            "Configuration values must be finite: {}".format(", ".join(nonfinite))
        )

    if set(config.split_seeds) != set(SPLIT_NAMES):
        raise ValueError("split_seeds must define train, validation, and test")
    if len(set(config.split_seeds.values())) != len(SPLIT_NAMES):
        raise ValueError("train, validation, and test seeds must be distinct")
    if set(config.scene_counts) != set(SPLIT_NAMES):
        raise ValueError("scene_counts must define train, validation, and test")
    if any(config.scene_counts[name] <= 0 for name in SPLIT_NAMES):
        raise ValueError("Every split must contain at least one scene")
    if config.episodes_per_scene < 3:
        raise ValueError("episodes_per_scene must be at least 3")
    if any(
        config.scene_counts[name] * config.episodes_per_scene < 5
        for name in SPLIT_NAMES
    ):
        raise ValueError("Every split must contain at least five episodes")
    if config.scene.width <= 0.0 or config.scene.height <= 0.0:
        raise ValueError("Scene dimensions must be positive")
    if config.scene.sensor_margin < 0.0 or 2.0 * config.scene.sensor_margin >= min(
        config.scene.width, config.scene.height
    ):
        raise ValueError("sensor_margin leaves no valid sensor sampling area")
    if config.scene.sensor_tree_clearance < 0.0:
        raise ValueError("sensor_tree_clearance must be non-negative")
    if config.trees.count <= 0 or config.trees.radius_range[0] <= 0.0:
        raise ValueError("Tree count and radii must be positive")
    if config.trees.min_spacing < 0.0 or config.trees.max_attempts <= 0:
        raise ValueError("Tree spacing/attempt limit is invalid")

    trajectory = config.trajectory
    if trajectory.sample_rate_hz <= 0.0 or trajectory.duration_seconds <= 0.0:
        raise ValueError("Trajectory rate and duration must be positive")
    positive_ranges = (
        trajectory.initial_speed_range,
        trajectory.acceleration_magnitude_range,
        trajectory.turn_rate_magnitude_range,
        trajectory.stop_duration_range,
        trajectory.piecewise_turn_angle_magnitude_range,
    )
    if any(pair[0] < 0.0 for pair in positive_ranges):
        raise ValueError("Motion magnitude ranges must be non-negative")
    if trajectory.transition_duration_seconds <= 0.0:
        raise ValueError("transition_duration_seconds must be positive")
    if min(
        trajectory.boundary_margin,
        trajectory.min_tree_clearance,
        trajectory.max_speed,
        trajectory.max_acceleration,
    ) < 0.0:
        raise ValueError("Trajectory limits must be non-negative")
    if trajectory.max_speed <= 0.0 or trajectory.max_acceleration <= 0.0:
        raise ValueError("Speed and acceleration limits must be positive")
    if 2.0 * trajectory.boundary_margin >= min(
        config.scene.width, config.scene.height
    ):
        raise ValueError("boundary_margin leaves no valid trajectory area")
    if trajectory.max_attempts <= 0:
        raise ValueError("trajectory.max_attempts must be positive")
    if (
        trajectory.stop_start_time_range[0] < 0.0
        or trajectory.piecewise_turn_time_range[0] < 0.0
    ):
        raise ValueError("Motion event times must be non-negative")
    if trajectory.stop_start_time_range[1] + 2.0 * trajectory.transition_duration_seconds + trajectory.stop_duration_range[1] > trajectory.duration_seconds:
        raise ValueError("Stop-and-go timing does not fit inside the episode")
    if trajectory.piecewise_turn_time_range[1] + trajectory.transition_duration_seconds > trajectory.duration_seconds:
        raise ValueError("Piecewise turn timing does not fit inside the episode")

    window = config.window
    if min(window.history_steps, window.future_steps, window.window_stride) <= 0:
        raise ValueError("Window sizes and stride must be positive")
    if min(
        window.minimum_visible_history_steps,
        window.minimum_consecutive_visible_steps,
        window.minimum_windows_per_episode,
    ) <= 0:
        raise ValueError("Window visibility requirements must be positive")
    if window.minimum_visible_history_steps > window.history_steps:
        raise ValueError("minimum_visible_history_steps exceeds history_steps")
    if window.minimum_consecutive_visible_steps > window.history_steps:
        raise ValueError("minimum_consecutive_visible_steps exceeds history_steps")
    trajectory_steps = int(
        sample_times(
            trajectory.sample_rate_hz, trajectory.duration_seconds
        ).size
    )
    if trajectory_steps < window.history_steps + window.future_steps:
        raise ValueError("Episode is shorter than one history/future window")
    maximum_windows = 1 + (
        trajectory_steps - window.history_steps - window.future_steps
    ) // window.window_stride
    if window.minimum_windows_per_episode > maximum_windows:
        raise ValueError("minimum_windows_per_episode exceeds available windows")
    observation = config.observation
    if observation.position_noise_std < 0.0:
        raise ValueError("Position noise must be non-negative")
    if not 0.0 <= observation.random_dropout_probability <= 1.0:
        raise ValueError("Dropout probability must be in [0, 1]")
