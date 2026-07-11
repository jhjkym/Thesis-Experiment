"""Observation simulation, local coordinates, and dataset windowing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np

from thesis_experiment.geometry.occlusion import is_occluded


@dataclass(frozen=True)
class ObservationSequence:
    """Noisy observations and mutually exclusive missing-frame causes."""

    observed_positions: np.ndarray
    visible_mask: np.ndarray
    occluded_mask: np.ndarray
    dropout_mask: np.ndarray


def _positions(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("{} must have shape (N, 2)".format(name))
    return array


def create_observations(
    true_positions: np.ndarray,
    sensor_position: np.ndarray,
    tree_centers: np.ndarray,
    tree_radii: np.ndarray,
    noise_std: float,
    dropout_probability: float,
    rng: np.random.Generator,
) -> ObservationSequence:
    """Simulate occlusion, Gaussian position noise, and random frame loss.

    Random dropout is sampled only for geometrically visible frames, so the
    occlusion and dropout causes remain disjoint and their reported ratios are
    directly interpretable.

    Args:
        true_positions: World-frame target positions with shape ``(T, 2)``.
        sensor_position: Fixed world-frame sensor position.
        tree_centers: World-frame tree centers with shape ``(K, 2)``.
        tree_radii: Tree radii with shape ``(K,)``.
        noise_std: Standard deviation of zero-mean Gaussian position noise.
        dropout_probability: Bernoulli loss probability for visible frames.
        rng: NumPy random generator controlling noise and frame loss.

    Returns:
        Observation arrays; missing positions are represented by ``NaN``.
    """

    positions = _positions("true_positions", true_positions)
    sensor = np.asarray(sensor_position, dtype=float)
    centers = _positions("tree_centers", tree_centers)
    radii = np.asarray(tree_radii, dtype=float)
    if sensor.shape != (2,):
        raise ValueError("sensor_position must have shape (2,)")
    if radii.shape != (centers.shape[0],):
        raise ValueError("tree_radii must have shape (K,)")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout_probability must be in [0, 1]")

    occluded = np.fromiter(
        (is_occluded(sensor, target, centers, radii) for target in positions),
        dtype=bool,
        count=positions.shape[0],
    )
    dropout = (~occluded) & (rng.random(positions.shape[0]) < dropout_probability)
    visible = ~(occluded | dropout)
    observed = np.full(positions.shape, np.nan, dtype=float)
    if np.any(visible):
        noise = rng.normal(0.0, noise_std, size=(int(np.sum(visible)), 2))
        observed[visible] = positions[visible] + noise
    return ObservationSequence(observed, visible, occluded, dropout)


def world_to_local(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Translate world-frame points into a broadcast-compatible local frame."""

    point_array = np.asarray(points, dtype=float)
    origin_array = np.asarray(origin, dtype=float)
    if (
        point_array.ndim == 0
        or origin_array.ndim == 0
        or point_array.shape[-1] != 2
        or origin_array.shape[-1] != 2
    ):
        raise ValueError("Points and origin must end in coordinate dimension 2")
    return point_array - origin_array


def local_to_world(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Restore local-frame points using a broadcast-compatible world origin."""

    point_array = np.asarray(points, dtype=float)
    origin_array = np.asarray(origin, dtype=float)
    if (
        point_array.ndim == 0
        or origin_array.ndim == 0
        or point_array.shape[-1] != 2
        or origin_array.shape[-1] != 2
    ):
        raise ValueError("Points and origin must end in coordinate dimension 2")
    return point_array + origin_array


def _observed_velocity(
    observed_positions: np.ndarray, visible_mask: np.ndarray, dt: float
) -> np.ndarray:
    """Compute one-step velocities only when both adjacent frames are visible."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    velocity = np.full(observed_positions.shape, np.nan, dtype=float)
    adjacent_visible = visible_mask[1:] & visible_mask[:-1]
    differences = np.diff(observed_positions, axis=0) / dt
    valid_indices = np.flatnonzero(adjacent_visible) + 1
    velocity[valid_indices] = differences[adjacent_visible]
    return velocity


def _select_window_starts(
    visible_mask: np.ndarray,
    total_steps: int,
    history_steps: int,
    future_steps: int,
    num_samples: int,
) -> np.ndarray:
    """Choose deterministic, evenly distributed windows with a valid origin."""

    last_start = total_steps - history_steps - future_steps
    if last_start < 0:
        raise ValueError("Trajectory is shorter than one history/future window")
    candidates = np.asarray(
        [
            start
            for start in range(last_start + 1)
            if np.any(visible_mask[start : start + history_steps])
        ],
        dtype=int,
    )
    if candidates.size < num_samples:
        raise ValueError(
            "Only {} windows contain a valid observation; {} requested".format(
                candidates.size, num_samples
            )
        )
    selected_indices = np.linspace(0, candidates.size - 1, num_samples, dtype=int)
    return candidates[selected_indices]


def create_dataset_windows(
    true_positions: np.ndarray,
    observations: ObservationSequence,
    sensor_position: np.ndarray,
    tree_centers: np.ndarray,
    tree_radii: np.ndarray,
    *,
    history_steps: int,
    future_steps: int,
    num_samples: int,
    dt: float,
    scene_id: int = 0,
) -> Dict[str, np.ndarray]:
    """Split a trajectory into fixed-size local-coordinate training samples.

    The local origin for each sample is its final valid noisy history
    observation. Observed history positions, history truth, future truth,
    sensor position, and tree centers are all stored in that same local frame.
    ``coordinate_origin`` retains the world-frame translation needed for exact
    inverse conversion.
    """

    positions = _positions("true_positions", true_positions)
    observed = _positions("observed_positions", observations.observed_positions)
    centers = _positions("tree_centers", tree_centers)
    radii = np.asarray(tree_radii, dtype=float)
    sensor = np.asarray(sensor_position, dtype=float)
    if observed.shape != positions.shape:
        raise ValueError("Observed and true position arrays must have identical shapes")
    for name, mask in (
        ("visible_mask", observations.visible_mask),
        ("occluded_mask", observations.occluded_mask),
        ("dropout_mask", observations.dropout_mask),
    ):
        if np.asarray(mask).shape != (positions.shape[0],):
            raise ValueError("{} must have shape (T,)".format(name))
    if radii.shape != (centers.shape[0],) or sensor.shape != (2,):
        raise ValueError("Tree radii or sensor position has an invalid shape")
    if min(history_steps, future_steps, num_samples) <= 0:
        raise ValueError("Window sizes and sample count must be positive")

    starts = _select_window_starts(
        np.asarray(observations.visible_mask, dtype=bool),
        positions.shape[0],
        history_steps,
        future_steps,
        num_samples,
    )
    samples: Dict[str, List[np.ndarray]] = {
        "history_position": [],
        "history_true_position": [],
        "history_velocity": [],
        "history_mask": [],
        "history_occluded": [],
        "history_random_dropout": [],
        "future_position": [],
        "sensor_position": [],
        "tree_centers": [],
        "tree_radii": [],
        "coordinate_origin": [],
    }

    visible_all = np.asarray(observations.visible_mask, dtype=bool)
    occluded_all = np.asarray(observations.occluded_mask, dtype=bool)
    dropout_all = np.asarray(observations.dropout_mask, dtype=bool)
    for start in starts:
        history_slice = slice(start, start + history_steps)
        future_slice = slice(start + history_steps, start + history_steps + future_steps)
        history_visible = visible_all[history_slice]
        last_valid_index = int(np.flatnonzero(history_visible)[-1])
        history_observed_world = observed[history_slice]
        origin = history_observed_world[last_valid_index].copy()

        samples["history_position"].append(world_to_local(history_observed_world, origin))
        samples["history_true_position"].append(
            world_to_local(positions[history_slice], origin)
        )
        samples["history_velocity"].append(
            _observed_velocity(history_observed_world, history_visible, dt)
        )
        samples["history_mask"].append(history_visible.astype(np.uint8))
        samples["history_occluded"].append(
            occluded_all[history_slice].astype(np.uint8)
        )
        samples["history_random_dropout"].append(
            dropout_all[history_slice].astype(np.uint8)
        )
        samples["future_position"].append(world_to_local(positions[future_slice], origin))
        samples["sensor_position"].append(world_to_local(sensor, origin))
        samples["tree_centers"].append(world_to_local(centers, origin))
        samples["tree_radii"].append(radii.copy())
        samples["coordinate_origin"].append(origin)

    dataset = {name: np.stack(values, axis=0) for name, values in samples.items()}
    dataset["scene_id"] = np.full((num_samples,), int(scene_id), dtype=np.int64)
    dataset["sample_start_index"] = starts.astype(np.int64)
    dataset["time_step_seconds"] = np.asarray(dt, dtype=float)
    return dataset


def _run_lengths(mask_rows: np.ndarray) -> List[int]:
    """Return lengths of all true runs, treating every row independently."""

    lengths: List[int] = []
    for row in np.asarray(mask_rows, dtype=bool):
        current = 0
        for value in row:
            if value:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        if current:
            lengths.append(current)
    return lengths


def calculate_sample_statistics(dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Calculate JSON-ready visibility, occlusion, run, and shape statistics."""

    visible = np.asarray(dataset["history_mask"], dtype=bool)
    occluded = np.asarray(dataset["history_occluded"], dtype=bool)
    dropout = np.asarray(dataset["history_random_dropout"], dtype=bool)
    runs = _run_lengths(occluded)
    return {
        "total_samples": int(visible.shape[0]),
        "visible_observation_ratio": float(np.mean(visible)),
        "occlusion_ratio": float(np.mean(occluded)),
        "random_dropout_ratio": float(np.mean(dropout)),
        "longest_consecutive_occlusion_steps": int(max(runs) if runs else 0),
        "mean_consecutive_occlusion_steps": float(np.mean(runs) if runs else 0.0),
        "array_shapes": {name: list(array.shape) for name, array in dataset.items()},
    }


def save_dataset(dataset: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save all dataset arrays to a compressed NumPy archive."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **dataset)


def save_statistics(statistics: Dict[str, Any], output_path: Path) -> None:
    """Write dataset statistics as indented UTF-8 JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(statistics, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
