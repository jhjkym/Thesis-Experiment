"""Leakage-free multi-scene, multi-motion dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import json
import numpy as np

from thesis_experiment.config_v2 import DatasetV2Config, SPLIT_NAMES, TrajectoryV2Config
from thesis_experiment.data.dataset import (
    ObservationSequence,
    create_observations,
    world_to_local,
)
from thesis_experiment.data.trajectory import sample_times
from thesis_experiment.data.trajectory_v2 import (
    TRAJECTORY_TYPE_TO_CODE,
    TrajectoryParameters,
    TrajectoryResult,
    generate_trajectory,
)
from thesis_experiment.geometry.forest import generate_tree_trunks
from thesis_experiment.geometry.occlusion import segment_intersects_circles


OCCLUSION_LENGTH_BIN_TO_CODE = {
    "0": 0,
    "1-5": 1,
    "6-10": 2,
    "11-15": 3,
    "16-20": 4,
}


@dataclass(frozen=True)
class GeneratedDatasetV2:
    """In-memory split arrays and their deterministic manifest."""

    splits: Dict[str, Dict[str, np.ndarray]]
    manifest: Dict[str, Any]


def validate_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    scene_size: Sequence[float],
    tree_centers: np.ndarray,
    tree_radii: np.ndarray,
    min_tree_clearance: float,
    max_speed: float,
    max_acceleration: float,
    boundary_margin: float = 0.0,
) -> Dict[str, float]:
    """Validate one complete trajectory and return physical audit metrics.

    Raises:
        ValueError: If a shape, finite-value, boundary, collision, speed, or
            acceleration constraint is violated.
    """

    position_array = np.asarray(positions, dtype=float)
    velocity_array = np.asarray(velocities, dtype=float)
    acceleration_array = np.asarray(accelerations, dtype=float)
    if position_array.ndim != 2 or position_array.shape[1] != 2:
        raise ValueError("positions must have shape (T, 2)")
    if velocity_array.shape != position_array.shape:
        raise ValueError("velocities must match positions")
    if acceleration_array.shape != position_array.shape:
        raise ValueError("accelerations must match positions")
    if not (
        np.all(np.isfinite(position_array))
        and np.all(np.isfinite(velocity_array))
        and np.all(np.isfinite(acceleration_array))
    ):
        raise ValueError("trajectory contains non-finite values")

    size = np.asarray(scene_size, dtype=float)
    centers = np.asarray(tree_centers, dtype=float)
    radii = np.asarray(tree_radii, dtype=float)
    if size.shape != (2,) or np.any(size <= 0.0):
        raise ValueError("scene_size must contain positive width and height")
    if centers.size == 0:
        centers = np.empty((0, 2), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("tree_centers must have shape (K, 2)")
    if radii.shape != (centers.shape[0],):
        raise ValueError("tree_radii must have shape (K,)")
    if min(min_tree_clearance, max_speed, max_acceleration, boundary_margin) < 0.0:
        raise ValueError("trajectory constraints must be non-negative")

    lower_clearance = np.min(position_array, axis=0)
    upper_clearance = np.min(size - position_array, axis=0)
    minimum_boundary_clearance = float(
        min(np.min(lower_clearance), np.min(upper_clearance))
    )
    tolerance = 1e-10
    if minimum_boundary_clearance < boundary_margin - tolerance:
        raise ValueError("trajectory leaves the configured scene boundary margin")

    minimum_tree_surface_distance = float("inf")
    if centers.shape[0]:
        distances = np.linalg.norm(
            position_array[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        surface_distances = distances - radii[np.newaxis, :]
        minimum_tree_surface_distance = float(np.min(surface_distances))
        if minimum_tree_surface_distance < min_tree_clearance - tolerance:
            raise ValueError("trajectory violates tree surface clearance")
        expanded_radii = radii + float(min_tree_clearance)
        for start, end in zip(position_array[:-1], position_array[1:]):
            if np.any(
                segment_intersects_circles(
                    start, end, centers, expanded_radii, atol=0.0
                )
            ):
                raise ValueError(
                    "trajectory segment intersects a tree clearance circle"
                )

    speed = np.linalg.norm(velocity_array, axis=1)
    acceleration = np.linalg.norm(acceleration_array, axis=1)
    maximum_speed = float(np.max(speed))
    maximum_acceleration = float(np.max(acceleration))
    if maximum_speed > max_speed + tolerance:
        raise ValueError("trajectory exceeds max_speed")
    if maximum_acceleration > max_acceleration + tolerance:
        raise ValueError("trajectory exceeds max_acceleration")
    return {
        "minimum_boundary_clearance": minimum_boundary_clearance,
        "minimum_tree_surface_distance": minimum_tree_surface_distance,
        "minimum_speed": float(np.min(speed)),
        "maximum_speed": maximum_speed,
        "minimum_acceleration": float(np.min(acceleration)),
        "maximum_acceleration": maximum_acceleration,
    }


def _signed_magnitude(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    magnitude = float(rng.uniform(bounds[0], bounds[1]))
    return magnitude if bool(rng.integers(0, 2)) else -magnitude


def _sample_parameters(
    scene_size: Tuple[float, float],
    config: TrajectoryV2Config,
    rng: np.random.Generator,
) -> TrajectoryParameters:
    """Randomly sample every episode motion parameter, including unused modes."""

    margin = config.boundary_margin
    initial_position = rng.uniform(
        np.asarray([margin, margin], dtype=float),
        np.asarray([scene_size[0] - margin, scene_size[1] - margin], dtype=float),
    )
    initial_speed = float(
        rng.uniform(config.initial_speed_range[0], config.initial_speed_range[1])
    )
    initial_direction = float(rng.uniform(-np.pi, np.pi))
    initial_velocity = initial_speed * np.asarray(
        [np.cos(initial_direction), np.sin(initial_direction)], dtype=float
    )
    acceleration_magnitude = float(
        rng.uniform(
            config.acceleration_magnitude_range[0],
            config.acceleration_magnitude_range[1],
        )
    )
    acceleration_direction = float(rng.uniform(-np.pi, np.pi))
    acceleration = acceleration_magnitude * np.asarray(
        [np.cos(acceleration_direction), np.sin(acceleration_direction)], dtype=float
    )
    return TrajectoryParameters(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        acceleration=acceleration,
        turn_rate=_signed_magnitude(rng, config.turn_rate_magnitude_range),
        stop_start_time=float(
            rng.uniform(config.stop_start_time_range[0], config.stop_start_time_range[1])
        ),
        stop_duration=float(
            rng.uniform(config.stop_duration_range[0], config.stop_duration_range[1])
        ),
        piecewise_turn_time=float(
            rng.uniform(
                config.piecewise_turn_time_range[0],
                config.piecewise_turn_time_range[1],
            )
        ),
        piecewise_turn_angle=_signed_magnitude(
            rng, config.piecewise_turn_angle_magnitude_range
        ),
        transition_duration=config.transition_duration_seconds,
    )


def sample_valid_episode(
    trajectory_type: str,
    times: np.ndarray,
    scene_size: Tuple[float, float],
    tree_centers: np.ndarray,
    tree_radii: np.ndarray,
    config: TrajectoryV2Config,
    rng: np.random.Generator,
) -> Tuple[TrajectoryResult, TrajectoryParameters, Dict[str, float]]:
    """Rejection-sample one valid episode or fail after ``max_attempts``."""

    last_error = "no candidate was evaluated"
    for _ in range(config.max_attempts):
        parameters = _sample_parameters(scene_size, config, rng)
        try:
            result = generate_trajectory(trajectory_type, times, parameters)
            metrics = validate_trajectory(
                result.positions,
                result.velocities,
                result.accelerations,
                scene_size,
                tree_centers,
                tree_radii,
                config.min_tree_clearance,
                config.max_speed,
                config.max_acceleration,
                config.boundary_margin,
            )
            return result, parameters, metrics
        except ValueError as error:
            last_error = str(error)
    raise RuntimeError(
        "could not generate a valid {} trajectory after {} attempts; last reason: {}".format(
            trajectory_type, config.max_attempts, last_error
        )
    )


def _history_velocity(
    observed_positions: np.ndarray, visible_mask: np.ndarray, dt: float
) -> np.ndarray:
    """Compute backward velocities from history observations only."""

    velocity = np.full(observed_positions.shape, np.nan, dtype=float)
    adjacent = visible_mask[1:] & visible_mask[:-1]
    indices = np.flatnonzero(adjacent) + 1
    differences = np.diff(observed_positions, axis=0) / dt
    velocity[indices] = differences[adjacent]
    return velocity


def _maximum_consecutive_true(mask: np.ndarray) -> int:
    """Return the longest contiguous true run in one Boolean vector."""

    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(
        np.int8
    )
    differences = np.diff(padded)
    starts = np.flatnonzero(differences == 1)
    ends = np.flatnonzero(differences == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def _occlusion_length_bin(maximum_length: int) -> int:
    """Encode a maximum history occlusion length into a stable interval."""

    if maximum_length <= 0:
        return OCCLUSION_LENGTH_BIN_TO_CODE["0"]
    if maximum_length <= 5:
        return OCCLUSION_LENGTH_BIN_TO_CODE["1-5"]
    if maximum_length <= 10:
        return OCCLUSION_LENGTH_BIN_TO_CODE["6-10"]
    if maximum_length <= 15:
        return OCCLUSION_LENGTH_BIN_TO_CODE["11-15"]
    return OCCLUSION_LENGTH_BIN_TO_CODE["16-20"]


def create_episode_windows(
    true_positions: np.ndarray,
    observations: ObservationSequence,
    times: np.ndarray,
    sensor_position: np.ndarray,
    tree_centers: np.ndarray,
    tree_radii: np.ndarray,
    scene_id: int,
    episode_id: int,
    trajectory_type_code: int,
    history_steps: int,
    future_steps: int,
    window_stride: int,
    minimum_visible_history_steps: int = 1,
    minimum_consecutive_visible_steps: int = 1,
) -> Dict[str, np.ndarray]:
    """Create episode-local windows without ever crossing an episode boundary."""

    positions = np.asarray(true_positions, dtype=float)
    time_values = np.asarray(times, dtype=float)
    dt = float(time_values[1] - time_values[0])
    last_start = positions.shape[0] - history_steps - future_steps
    starts = list(range(0, last_start + 1, window_stride))
    records: Dict[str, List[Any]] = {
        "history_position": [],
        "history_velocity": [],
        "history_mask": [],
        "history_occluded": [],
        "history_random_dropout": [],
        "history_true_position": [],
        "future_position": [],
        "coordinate_origin": [],
        "sensor_position": [],
        "tree_centers": [],
        "tree_radii": [],
        "scene_id": [],
        "episode_id": [],
        "trajectory_type": [],
        "sample_start_index": [],
        "history_start_time": [],
        "future_start_time": [],
        "time_step_seconds": [],
        "history_visible_count": [],
        "last_valid_observation_age_steps": [],
        "valid_velocity_count": [],
        "history_max_consecutive_occlusion_steps": [],
        "occlusion_length_bin": [],
    }
    visible_all = np.asarray(observations.visible_mask, dtype=bool)
    occluded_all = np.asarray(observations.occluded_mask, dtype=bool)
    dropout_all = np.asarray(observations.dropout_mask, dtype=bool)
    for start in starts:
        history_slice = slice(start, start + history_steps)
        future_slice = slice(start + history_steps, start + history_steps + future_steps)
        history_visible = visible_all[history_slice]
        visible_count = int(np.sum(history_visible))
        maximum_visible_run = _maximum_consecutive_true(history_visible)
        if (
            visible_count < minimum_visible_history_steps
            or maximum_visible_run < minimum_consecutive_visible_steps
        ):
            continue
        observed_world = observations.observed_positions[history_slice]
        last_valid = int(np.flatnonzero(history_visible)[-1])
        origin = observed_world[last_valid].copy()
        local_observed = world_to_local(observed_world, origin)
        history_velocity = _history_velocity(observed_world, history_visible, dt)
        maximum_occlusion = _maximum_consecutive_true(occluded_all[history_slice])
        records["history_position"].append(local_observed)
        records["history_velocity"].append(history_velocity)
        records["history_mask"].append(history_visible.astype(np.uint8))
        records["history_occluded"].append(
            occluded_all[history_slice].astype(np.uint8)
        )
        records["history_random_dropout"].append(
            dropout_all[history_slice].astype(np.uint8)
        )
        records["history_true_position"].append(
            world_to_local(positions[history_slice], origin)
        )
        records["future_position"].append(
            world_to_local(positions[future_slice], origin)
        )
        records["coordinate_origin"].append(origin)
        records["sensor_position"].append(world_to_local(sensor_position, origin))
        records["tree_centers"].append(world_to_local(tree_centers, origin))
        records["tree_radii"].append(np.asarray(tree_radii, dtype=float).copy())
        records["scene_id"].append(int(scene_id))
        records["episode_id"].append(int(episode_id))
        records["trajectory_type"].append(int(trajectory_type_code))
        records["sample_start_index"].append(int(start))
        records["history_start_time"].append(float(time_values[start]))
        records["future_start_time"].append(float(time_values[start + history_steps]))
        records["time_step_seconds"].append(dt)
        records["history_visible_count"].append(visible_count)
        records["last_valid_observation_age_steps"].append(
            history_steps - 1 - last_valid
        )
        records["valid_velocity_count"].append(
            int(np.sum(np.all(np.isfinite(history_velocity), axis=1)))
        )
        records["history_max_consecutive_occlusion_steps"].append(
            maximum_occlusion
        )
        records["occlusion_length_bin"].append(
            _occlusion_length_bin(maximum_occlusion)
        )

    result: Dict[str, np.ndarray] = {}
    integer_fields = {
        "scene_id": np.int64,
        "episode_id": np.int64,
        "trajectory_type": np.int8,
        "sample_start_index": np.int64,
        "history_visible_count": np.int16,
        "last_valid_observation_age_steps": np.int16,
        "valid_velocity_count": np.int16,
        "history_max_consecutive_occlusion_steps": np.int16,
        "occlusion_length_bin": np.int8,
    }
    for name, values in records.items():
        if name in integer_fields:
            result[name] = np.asarray(values, dtype=integer_fields[name])
        elif name in ("history_start_time", "future_start_time", "time_step_seconds"):
            result[name] = np.asarray(values, dtype=float)
        else:
            result[name] = np.stack(values, axis=0) if values else np.empty((0,))
    return result


def _merge_window_batches(batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    nonempty = [batch for batch in batches if batch["scene_id"].size]
    if not nonempty:
        raise RuntimeError("split contains no windows with a valid history observation")
    return {
        name: np.concatenate([batch[name] for batch in nonempty], axis=0)
        for name in nonempty[0]
    }


def _occlusion_runs(mask_rows: np.ndarray) -> List[int]:
    lengths: List[int] = []
    for row in np.asarray(mask_rows, dtype=bool):
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        differences = np.diff(padded)
        starts = np.flatnonzero(differences == 1)
        ends = np.flatnonzero(differences == -1)
        lengths.extend((ends - starts).astype(int).tolist())
    return lengths


def _counts_by_type(codes: np.ndarray) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts = {
        name: int(np.sum(codes == code))
        for name, code in TRAJECTORY_TYPE_TO_CODE.items()
    }
    total = int(codes.size)
    ratios = {
        name: float(count / total) if total else 0.0 for name, count in counts.items()
    }
    return counts, ratios


def _split_manifest(
    arrays: Dict[str, np.ndarray], seed: int, rejection_statistics: Mapping[str, Any]
) -> Dict[str, Any]:
    episode_counts, episode_ratios = _counts_by_type(
        arrays["episode_trajectory_types"]
    )
    window_counts, window_ratios = _counts_by_type(arrays["trajectory_type"])
    visible = arrays["history_mask"].astype(bool)
    occluded = arrays["history_occluded"].astype(bool)
    dropout = arrays["history_random_dropout"].astype(bool)
    speed = np.linalg.norm(arrays["episode_velocity_world"], axis=2)
    acceleration = np.linalg.norm(arrays["episode_acceleration_world"], axis=2)
    runs = _occlusion_runs(arrays["episode_occluded_mask"])
    histogram: Dict[str, int] = {}
    for length in runs:
        key = str(int(length))
        histogram[key] = histogram.get(key, 0) + 1

    overlap_count = 0
    adjacent_pair_count = 0
    window_span = int(
        arrays["history_position"].shape[1] + arrays["future_position"].shape[1]
    )
    for episode_id in arrays["episode_ids"]:
        starts = np.sort(
            arrays["sample_start_index"][arrays["episode_id"] == episode_id]
        )
        if starts.size > 1:
            adjacent_pair_count += int(starts.size - 1)
            overlap_count += int(np.sum(np.diff(starts) < window_span))
    return {
        "seed": int(seed),
        "scene_count": int(arrays["scene_ids"].size),
        "episode_count": int(arrays["episode_ids"].size),
        "window_count": int(arrays["scene_id"].size),
        "scene_ids": arrays["scene_ids"].astype(int).tolist(),
        "episode_ids": arrays["episode_ids"].astype(int).tolist(),
        "episode_trajectory_type_counts": episode_counts,
        "episode_trajectory_type_ratios": episode_ratios,
        "window_trajectory_type_counts": window_counts,
        "window_trajectory_type_ratios": window_ratios,
        "visible_ratio": float(np.mean(visible)),
        "geometric_occlusion_ratio": float(np.mean(occluded)),
        "random_dropout_ratio": float(np.mean(dropout)),
        "occlusion_dropout_overlap_count": int(np.sum(occluded & dropout)),
        "rejected_fully_unobservable_episode_count": int(
            rejection_statistics["rejected_fully_unobservable_episode_count"]
        ),
        "rejected_insufficient_history_episode_count": int(
            rejection_statistics["rejected_insufficient_history_episode_count"]
        ),
        "rejected_physical_episode_candidate_count": int(
            rejection_statistics["rejected_physical_episode_candidate_count"]
        ),
        "rejected_episode_counts_by_trajectory_type": dict(
            rejection_statistics["by_trajectory_type"]
        ),
        "rejected_episode_count_scope": (
            "fully_unobservable_or_insufficient_history_candidates; "
            "physical candidate failures are reported separately"
        ),
        "speed_range": [float(np.min(speed)), float(np.max(speed))],
        "acceleration_range": [
            float(np.min(acceleration)),
            float(np.max(acceleration)),
        ],
        "consecutive_occlusion_lengths": runs,
        "consecutive_occlusion_length_histogram": histogram,
        "consecutive_occlusion_scope": "full_episode_geometric_occlusion",
        "overlapping_adjacent_window_pairs": overlap_count,
        "adjacent_window_pair_count": adjacent_pair_count,
        "overlapping_adjacent_window_ratio": float(
            overlap_count / adjacent_pair_count if adjacent_pair_count else 0.0
        ),
        "history_visible_count_distribution": _integer_histogram(
            arrays["history_visible_count"]
        ),
        "last_valid_observation_age_steps_distribution": _integer_histogram(
            arrays["last_valid_observation_age_steps"]
        ),
        "valid_velocity_count_distribution": _integer_histogram(
            arrays["valid_velocity_count"]
        ),
        "windows_without_valid_velocity_count": int(
            np.sum(arrays["valid_velocity_count"] == 0)
        ),
        "window_occlusion_length_bin_counts": _occlusion_bin_counts(
            arrays["occlusion_length_bin"]
        ),
        "trajectory_type_window_audit": _trajectory_type_window_audit(
            arrays, rejection_statistics
        ),
        "fields": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }


def _integer_histogram(values: np.ndarray) -> Dict[str, int]:
    """Return a stable JSON histogram for an integer array."""

    array = np.asarray(values, dtype=int)
    return {
        str(int(value)): int(np.sum(array == value)) for value in np.unique(array)
    }


def _occlusion_bin_counts(codes: np.ndarray) -> Dict[str, int]:
    """Count window maximum-occlusion intervals using named bins."""

    code_array = np.asarray(codes, dtype=int)
    return {
        name: int(np.sum(code_array == code))
        for name, code in OCCLUSION_LENGTH_BIN_TO_CODE.items()
    }


def _trajectory_type_window_audit(
    arrays: Dict[str, np.ndarray], rejection_statistics: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return accepted episode/window and rejection audit statistics by type."""

    audit: Dict[str, Any] = {}
    for name, code in TRAJECTORY_TYPE_TO_CODE.items():
        selected = arrays["trajectory_type"] == code
        audit[name] = {
            "episode_count": int(np.sum(arrays["episode_trajectory_types"] == code)),
            "valid_window_count": int(np.sum(selected)),
            "rejected_episode_count": int(
                rejection_statistics["by_trajectory_type"][name]["total"]
            ),
            "rejected_physical_candidate_count": int(
                rejection_statistics["by_trajectory_type"][name]["physical"]
            ),
            "rejected_total_candidate_count": int(
                rejection_statistics["by_trajectory_type"][name]["total"]
                + rejection_statistics["by_trajectory_type"][name]["physical"]
            ),
            "rejected_fully_unobservable_episode_count": int(
                rejection_statistics["by_trajectory_type"][name][
                    "fully_unobservable"
                ]
            ),
            "rejected_insufficient_history_episode_count": int(
                rejection_statistics["by_trajectory_type"][name][
                    "insufficient_history"
                ]
            ),
            "occlusion_length_bin_counts": _occlusion_bin_counts(
                arrays["occlusion_length_bin"][selected]
            ),
        }
    return audit


def _empty_rejection_statistics() -> Dict[str, Any]:
    return {
        "rejected_fully_unobservable_episode_count": 0,
        "rejected_insufficient_history_episode_count": 0,
        "rejected_physical_episode_candidate_count": 0,
        "by_trajectory_type": {
            name: {
                "fully_unobservable": 0,
                "insufficient_history": 0,
                "physical": 0,
                "total": 0,
            }
            for name in TRAJECTORY_TYPE_TO_CODE
        },
    }


def _sample_observable_episode(
    trajectory_type: str,
    trajectory_type_code: int,
    times: np.ndarray,
    scene_size: Tuple[float, float],
    sensor: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    scene_id: int,
    episode_id: int,
    config: DatasetV2Config,
    rng: np.random.Generator,
) -> Tuple[
    TrajectoryResult,
    TrajectoryParameters,
    ObservationSequence,
    Dict[str, np.ndarray],
    Dict[str, int],
]:
    """Sample until physical and minimum-history constraints both pass."""

    rejected = {"fully_unobservable": 0, "insufficient_history": 0, "physical": 0}
    last_reason = "no candidate evaluated"
    for _ in range(config.trajectory.max_attempts):
        parameters = _sample_parameters(scene_size, config.trajectory, rng)
        try:
            result = generate_trajectory(trajectory_type, times, parameters)
            validate_trajectory(
                result.positions,
                result.velocities,
                result.accelerations,
                scene_size,
                centers,
                radii,
                config.trajectory.min_tree_clearance,
                config.trajectory.max_speed,
                config.trajectory.max_acceleration,
                config.trajectory.boundary_margin,
            )
        except ValueError as error:
            rejected["physical"] += 1
            last_reason = str(error)
            continue
        observations = create_observations(
            result.positions,
            sensor,
            centers,
            radii,
            config.observation.position_noise_std,
            config.observation.random_dropout_probability,
            rng,
        )
        batch = create_episode_windows(
            result.positions,
            observations,
            times,
            sensor,
            centers,
            radii,
            scene_id,
            episode_id,
            trajectory_type_code,
            config.window.history_steps,
            config.window.future_steps,
            config.window.window_stride,
            config.window.minimum_visible_history_steps,
            config.window.minimum_consecutive_visible_steps,
        )
        if int(batch["scene_id"].size) >= config.window.minimum_windows_per_episode:
            return result, parameters, observations, batch, rejected
        if not np.any(observations.visible_mask):
            rejected["fully_unobservable"] += 1
            last_reason = "episode is fully unobservable"
        else:
            rejected["insufficient_history"] += 1
            last_reason = "episode has too few valid history windows"
    raise RuntimeError(
        "could not generate an observable {} episode after {} attempts; last reason: {}".format(
            trajectory_type, config.trajectory.max_attempts, last_reason
        )
    )


def _generate_split(
    split_name: str,
    config: DatasetV2Config,
    scene_id_offset: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    seed = config.split_seeds[split_name]
    scene_count = config.scene_counts[split_name]
    episode_count = scene_count * config.episodes_per_scene
    root_sequence = np.random.SeedSequence(seed)
    children = root_sequence.spawn(scene_count + 1)
    type_rng = np.random.default_rng(children[0])
    type_codes = np.resize(
        np.arange(len(TRAJECTORY_TYPE_TO_CODE), dtype=np.int8), episode_count
    )
    type_rng.shuffle(type_codes)
    code_to_name = {code: name for name, code in TRAJECTORY_TYPE_TO_CODE.items()}

    times = sample_times(
        config.trajectory.sample_rate_hz, config.trajectory.duration_seconds
    )
    scene_ids: List[int] = []
    scene_sensors: List[np.ndarray] = []
    scene_centers: List[np.ndarray] = []
    scene_radii: List[np.ndarray] = []
    episode_ids: List[int] = []
    episode_scene_ids: List[int] = []
    episode_types: List[int] = []
    episode_truth: List[np.ndarray] = []
    episode_observed: List[np.ndarray] = []
    episode_visible: List[np.ndarray] = []
    episode_occluded: List[np.ndarray] = []
    episode_dropout: List[np.ndarray] = []
    episode_velocity: List[np.ndarray] = []
    episode_acceleration: List[np.ndarray] = []
    episode_times: List[np.ndarray] = []
    episode_initial_positions: List[np.ndarray] = []
    episode_initial_velocities: List[np.ndarray] = []
    episode_acceleration_parameters: List[np.ndarray] = []
    episode_turn_rates: List[float] = []
    episode_stop_starts: List[float] = []
    episode_stop_durations: List[float] = []
    episode_piecewise_times: List[float] = []
    episode_piecewise_angles: List[float] = []
    window_batches: List[Dict[str, np.ndarray]] = []
    rejection_statistics = _empty_rejection_statistics()

    scene_size = (config.scene.width, config.scene.height)
    global_type_index = 0
    for local_scene_index in range(scene_count):
        scene_id = scene_id_offset + local_scene_index
        component_sequences = children[local_scene_index + 1].spawn(
            config.episodes_per_scene + 2
        )
        sensor_rng = np.random.default_rng(component_sequences[0])
        tree_rng = np.random.default_rng(component_sequences[1])
        sensor = sensor_rng.uniform(
            np.asarray(
                [config.scene.sensor_margin, config.scene.sensor_margin], dtype=float
            ),
            np.asarray(
                [
                    config.scene.width - config.scene.sensor_margin,
                    config.scene.height - config.scene.sensor_margin,
                ],
                dtype=float,
            ),
        )
        centers, radii = generate_tree_trunks(
            scene_size,
            config.trees.count,
            config.trees.radius_range,
            config.trees.min_spacing,
            sensor,
            rng=tree_rng,
            max_attempts=config.trees.max_attempts,
            sensor_clearance=config.scene.sensor_tree_clearance,
        )
        scene_ids.append(scene_id)
        scene_sensors.append(sensor)
        scene_centers.append(centers)
        scene_radii.append(radii)

        for local_episode_index in range(config.episodes_per_scene):
            episode_id = scene_id * config.episodes_per_scene + local_episode_index
            type_code = int(type_codes[global_type_index])
            global_type_index += 1
            trajectory_type = code_to_name[type_code]
            episode_rng = np.random.default_rng(
                component_sequences[local_episode_index + 2]
            )
            result, parameters, observations, batch, rejected = _sample_observable_episode(
                trajectory_type,
                type_code,
                times,
                scene_size,
                sensor,
                centers,
                radii,
                scene_id,
                episode_id,
                config,
                episode_rng,
            )
            rejection_statistics["rejected_fully_unobservable_episode_count"] += rejected[
                "fully_unobservable"
            ]
            rejection_statistics["rejected_insufficient_history_episode_count"] += rejected[
                "insufficient_history"
            ]
            rejection_statistics["rejected_physical_episode_candidate_count"] += rejected[
                "physical"
            ]
            type_rejections = rejection_statistics["by_trajectory_type"][trajectory_type]
            for reason in ("fully_unobservable", "insufficient_history", "physical"):
                type_rejections[reason] += rejected[reason]
            type_rejections["total"] += (
                rejected["fully_unobservable"] + rejected["insufficient_history"]
            )
            window_batches.append(batch)
            episode_ids.append(episode_id)
            episode_scene_ids.append(scene_id)
            episode_types.append(type_code)
            episode_truth.append(result.positions)
            episode_observed.append(observations.observed_positions)
            episode_visible.append(observations.visible_mask.astype(np.uint8))
            episode_occluded.append(observations.occluded_mask.astype(np.uint8))
            episode_dropout.append(observations.dropout_mask.astype(np.uint8))
            episode_velocity.append(result.velocities)
            episode_acceleration.append(result.accelerations)
            episode_times.append(times)
            episode_initial_positions.append(parameters.initial_position)
            episode_initial_velocities.append(parameters.initial_velocity)
            episode_acceleration_parameters.append(parameters.acceleration)
            episode_turn_rates.append(parameters.turn_rate)
            episode_stop_starts.append(parameters.stop_start_time)
            episode_stop_durations.append(parameters.stop_duration)
            episode_piecewise_times.append(parameters.piecewise_turn_time)
            episode_piecewise_angles.append(parameters.piecewise_turn_angle)

    arrays = _merge_window_batches(window_batches)
    arrays.update(
        {
            "scene_ids": np.asarray(scene_ids, dtype=np.int64),
            "scene_sensor_position_world": np.stack(scene_sensors, axis=0),
            "scene_tree_centers_world": np.stack(scene_centers, axis=0),
            "scene_tree_radii": np.stack(scene_radii, axis=0),
            "episode_ids": np.asarray(episode_ids, dtype=np.int64),
            "episode_scene_ids": np.asarray(episode_scene_ids, dtype=np.int64),
            "episode_trajectory_types": np.asarray(episode_types, dtype=np.int8),
            "episode_true_position_world": np.stack(episode_truth, axis=0),
            "episode_observed_position_world": np.stack(episode_observed, axis=0),
            "episode_visible_mask": np.stack(episode_visible, axis=0),
            "episode_occluded_mask": np.stack(episode_occluded, axis=0),
            "episode_random_dropout_mask": np.stack(episode_dropout, axis=0),
            "episode_velocity_world": np.stack(episode_velocity, axis=0),
            "episode_acceleration_world": np.stack(episode_acceleration, axis=0),
            "episode_times": np.stack(episode_times, axis=0),
            "episode_initial_position": np.stack(episode_initial_positions, axis=0),
            "episode_initial_velocity": np.stack(episode_initial_velocities, axis=0),
            "episode_acceleration_parameter": np.stack(
                episode_acceleration_parameters, axis=0
            ),
            "episode_turn_rate": np.asarray(episode_turn_rates, dtype=float),
            "episode_stop_start_time": np.asarray(episode_stop_starts, dtype=float),
            "episode_stop_duration": np.asarray(episode_stop_durations, dtype=float),
            "episode_piecewise_turn_time": np.asarray(
                episode_piecewise_times, dtype=float
            ),
            "episode_piecewise_turn_angle": np.asarray(
                episode_piecewise_angles, dtype=float
            ),
        }
    )
    for name, code in TRAJECTORY_TYPE_TO_CODE.items():
        if not np.any(arrays["trajectory_type"] == code):
            raise RuntimeError("split {} has no valid windows for {}".format(split_name, name))
    return arrays, rejection_statistics


def generate_dataset_v2(config: DatasetV2Config) -> GeneratedDatasetV2:
    """Generate deterministic scene-level train/validation/test splits in memory."""

    splits: Dict[str, Dict[str, np.ndarray]] = {}
    rejections: Dict[str, Dict[str, Any]] = {}
    scene_offset = 0
    for split_name in SPLIT_NAMES:
        split_arrays, split_rejections = _generate_split(
            split_name, config, scene_offset
        )
        splits[split_name] = split_arrays
        rejections[split_name] = split_rejections
        scene_offset += config.scene_counts[split_name]

    scene_sets = {name: set(splits[name]["scene_ids"].tolist()) for name in SPLIT_NAMES}
    episode_sets = {
        name: set(splits[name]["episode_ids"].tolist()) for name in SPLIT_NAMES
    }
    for first_index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[first_index + 1 :]:
            if scene_sets[first] & scene_sets[second]:
                raise RuntimeError("scene_id leakage between {} and {}".format(first, second))
            if episode_sets[first] & episode_sets[second]:
                raise RuntimeError(
                    "episode_id leakage between {} and {}".format(first, second)
                )

    manifest: Dict[str, Any] = {
        "dataset_version": "2.0",
        "split_policy": "scene_level_before_episode_and_window_generation",
        "history_true_position_usage": "audit_and_visualization_only",
        "future_position_usage": "supervision_label_only",
        "episodes_per_scene": int(config.episodes_per_scene),
        "scene_counts": {
            name: int(config.scene_counts[name]) for name in SPLIT_NAMES
        },
        "split_seeds": {name: int(config.split_seeds[name]) for name in SPLIT_NAMES},
        "trajectory_type_encoding": dict(TRAJECTORY_TYPE_TO_CODE),
        "occlusion_length_bin_encoding": dict(OCCLUSION_LENGTH_BIN_TO_CODE),
        "field_roles": {
            "model_input_fields": [
                "history_position",
                "history_velocity",
                "history_mask",
                "history_velocity_mask",
                "time_step_seconds",
            ],
            "supervision_label_fields": ["future_position"],
            "sample_index_metadata_fields": [
                "scene_id",
                "episode_id",
                "sample_start_index",
            ],
            "audit_only_fields": ["history_true_position"],
            "trajectory_type_usage": "metadata_only",
            "future_motion_parameter_fields": [
                "episode_acceleration_parameter",
                "episode_turn_rate",
                "episode_stop_start_time",
                "episode_stop_duration",
                "episode_piecewise_turn_time",
                "episode_piecewise_turn_angle",
            ],
        },
        "scene": {
            "width": config.scene.width,
            "height": config.scene.height,
            "sensor_margin": config.scene.sensor_margin,
            "sensor_tree_clearance": config.scene.sensor_tree_clearance,
        },
        "trees": {
            "count": config.trees.count,
            "radius_range": list(config.trees.radius_range),
            "min_spacing": config.trees.min_spacing,
        },
        "trajectory": {
            "sample_rate_hz": config.trajectory.sample_rate_hz,
            "duration_seconds": config.trajectory.duration_seconds,
            "steps": int(next(iter(splits.values()))["episode_times"].shape[1]),
            "boundary_margin": config.trajectory.boundary_margin,
            "min_tree_clearance": config.trajectory.min_tree_clearance,
            "max_speed": config.trajectory.max_speed,
            "max_acceleration": config.trajectory.max_acceleration,
            "max_attempts": config.trajectory.max_attempts,
            "initial_speed_range": list(config.trajectory.initial_speed_range),
            "acceleration_magnitude_range": list(
                config.trajectory.acceleration_magnitude_range
            ),
            "turn_rate_magnitude_range": list(
                config.trajectory.turn_rate_magnitude_range
            ),
            "stop_start_time_range": list(
                config.trajectory.stop_start_time_range
            ),
            "stop_duration_range": list(config.trajectory.stop_duration_range),
            "piecewise_turn_time_range": list(
                config.trajectory.piecewise_turn_time_range
            ),
            "piecewise_turn_angle_magnitude_range": list(
                config.trajectory.piecewise_turn_angle_magnitude_range
            ),
            "transition_duration_seconds": config.trajectory.transition_duration_seconds,
        },
        "window": {
            "history_steps": config.window.history_steps,
            "future_steps": config.window.future_steps,
            "window_stride": config.window.window_stride,
            "minimum_visible_history_steps": config.window.minimum_visible_history_steps,
            "minimum_consecutive_visible_steps": config.window.minimum_consecutive_visible_steps,
            "minimum_windows_per_episode": config.window.minimum_windows_per_episode,
        },
        "observation": {
            "position_noise_std": config.observation.position_noise_std,
            "random_dropout_probability": config.observation.random_dropout_probability,
        },
        "splits": {
            name: _split_manifest(
                splits[name], config.split_seeds[name], rejections[name]
            )
            for name in SPLIT_NAMES
        },
    }
    saved_fields = set(next(iter(splits.values())).keys())
    allowed_input_fields = set(manifest["field_roles"]["model_input_fields"])
    manifest["field_roles"]["forbidden_model_input_fields"] = sorted(
        saved_fields - allowed_input_fields
    )
    return GeneratedDatasetV2(splits=splits, manifest=manifest)


def save_generated_dataset_v2(
    generated: GeneratedDatasetV2, output_directory: Path
) -> None:
    """Save split NPZ archives and a deterministic JSON manifest."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        np.savez_compressed(
            str(output_path / (split_name + ".npz")), **generated.splits[split_name]
        )
    with (output_path / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(generated.manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
