"""Visualizations generated exclusively from a saved dataset-v2 artifact.

The public entry point in this module deliberately accepts only a dataset
directory.  It never imports or calls scene/trajectory generators, which keeps
the figures auditable against the saved NPZ files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


SPLIT_NAMES = ("train", "validation", "test")
TRAJECTORY_TYPES = (
    "constant_velocity",
    "constant_acceleration",
    "constant_turn",
    "stop_and_go",
    "piecewise_direction",
)

_REQUIRED_SPLIT_FIELDS = (
    "episode_ids",
    "episode_scene_ids",
    "episode_trajectory_types",
    "episode_true_position_world",
    "episode_observed_position_world",
    "episode_visible_mask",
    "episode_occluded_mask",
    "episode_random_dropout_mask",
    "episode_velocity_world",
    "episode_acceleration_world",
    "episode_times",
    "scene_ids",
    "scene_sensor_position_world",
    "scene_tree_centers_world",
    "scene_tree_radii",
    "episode_id",
    "sample_start_index",
    "history_position",
    "history_true_position",
    "future_position",
    "coordinate_origin",
)

_SPLIT_COLORS = {
    "train": "#1565c0",
    "validation": "#ef6c00",
    "test": "#2e7d32",
}


def _load_manifest(path: Path) -> Mapping[str, Any]:
    """Load and minimally validate the dataset manifest."""
    if not path.is_file():
        raise FileNotFoundError("Required dataset manifest is missing: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("Could not read dataset manifest {}: {}".format(path, error))
    if not isinstance(manifest, dict):
        raise ValueError("dataset_manifest.json must contain a JSON object")
    if "trajectory_type_encoding" not in manifest:
        raise ValueError(
            "dataset_manifest.json is missing required key "
            "'trajectory_type_encoding'"
        )
    return manifest


def _load_split(path: Path, split_name: str) -> Dict[str, np.ndarray]:
    """Load all required arrays from one saved split and validate its layout."""
    if not path.is_file():
        raise FileNotFoundError(
            "Required '{}' dataset split is missing: {}".format(split_name, path)
        )
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            missing = [name for name in _REQUIRED_SPLIT_FIELDS if name not in archive]
            if missing:
                raise ValueError(
                    "{} is missing required fields: {}".format(
                        path, ", ".join(sorted(missing))
                    )
                )
            arrays = {
                name: np.asarray(archive[name]) for name in _REQUIRED_SPLIT_FIELDS
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "missing required fields" in str(error):
            raise
        raise ValueError("Could not read dataset split {}: {}".format(path, error))

    _validate_split_layout(arrays, split_name)
    return arrays


def _require_shape(
    array: np.ndarray, expected: Tuple[int, ...], field_name: str, split_name: str
) -> None:
    """Raise a descriptive error when an array has an unexpected exact shape."""
    if array.shape != expected:
        raise ValueError(
            "{} field '{}' has shape {}, expected {}".format(
                split_name, field_name, array.shape, expected
            )
        )


def _validate_split_layout(
    arrays: Mapping[str, np.ndarray], split_name: str
) -> None:
    """Validate dimensions and cross-array cardinalities needed for plotting."""
    episode_ids = arrays["episode_ids"]
    if episode_ids.ndim != 1 or episode_ids.size == 0:
        raise ValueError("{} episode_ids must be a non-empty 1-D array".format(split_name))
    episode_count = int(episode_ids.size)

    truth = arrays["episode_true_position_world"]
    if truth.ndim != 3 or truth.shape[0] != episode_count or truth.shape[2] != 2:
        raise ValueError(
            "{} episode_true_position_world must have shape (episodes, time, 2)".format(
                split_name
            )
        )
    if truth.shape[1] == 0:
        raise ValueError("{} episode trajectories must not be empty".format(split_name))
    time_steps = int(truth.shape[1])

    for name in ("episode_scene_ids", "episode_trajectory_types"):
        _require_shape(arrays[name], (episode_count,), name, split_name)
    for name in (
        "episode_observed_position_world",
        "episode_velocity_world",
        "episode_acceleration_world",
    ):
        _require_shape(arrays[name], (episode_count, time_steps, 2), name, split_name)
    for name in (
        "episode_visible_mask",
        "episode_occluded_mask",
        "episode_random_dropout_mask",
    ):
        _require_shape(arrays[name], (episode_count, time_steps), name, split_name)

    times = arrays["episode_times"]
    if times.shape not in ((time_steps,), (episode_count, time_steps)):
        raise ValueError(
            "{} episode_times must have shape (time,) or (episodes, time)".format(
                split_name
            )
        )

    if not np.all(np.isfinite(truth)):
        raise ValueError(
            "{} episode_true_position_world contains NaN or Inf".format(split_name)
        )
    for name in ("episode_velocity_world", "episode_acceleration_world", "episode_times"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError("{} field '{}' contains NaN or Inf".format(split_name, name))

    visible = np.asarray(arrays["episode_visible_mask"], dtype=bool)
    observations = arrays["episode_observed_position_world"]
    if np.any(visible & ~np.all(np.isfinite(observations), axis=2)):
        raise ValueError(
            "{} visible observations must contain finite coordinates".format(split_name)
        )

    scene_ids = arrays["scene_ids"]
    if scene_ids.ndim != 1 or scene_ids.size == 0:
        raise ValueError("{} scene_ids must be a non-empty 1-D array".format(split_name))
    scene_count = int(scene_ids.size)
    _require_shape(
        arrays["scene_sensor_position_world"],
        (scene_count, 2),
        "scene_sensor_position_world",
        split_name,
    )
    centers = arrays["scene_tree_centers_world"]
    radii = arrays["scene_tree_radii"]
    if centers.ndim != 3 or centers.shape[0] != scene_count or centers.shape[2] != 2:
        raise ValueError(
            "{} scene_tree_centers_world must have shape (scenes, trees, 2)".format(
                split_name
            )
        )
    _require_shape(
        radii,
        (scene_count, centers.shape[1]),
        "scene_tree_radii",
        split_name,
    )
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        raise ValueError("{} scene tree geometry contains NaN or Inf".format(split_name))
    if np.any(radii <= 0.0):
        raise ValueError("{} scene_tree_radii must be positive".format(split_name))

    window_episode_ids = arrays["episode_id"]
    starts = arrays["sample_start_index"]
    history = arrays["history_position"]
    history_truth = arrays["history_true_position"]
    future = arrays["future_position"]
    origins = arrays["coordinate_origin"]
    if window_episode_ids.ndim != 1 or window_episode_ids.size == 0:
        raise ValueError("{} episode_id must be a non-empty 1-D array".format(split_name))
    window_count = int(window_episode_ids.size)
    _require_shape(starts, (window_count,), "sample_start_index", split_name)
    if history.ndim != 3 or history.shape[0] != window_count or history.shape[2] != 2:
        raise ValueError(
            "{} history_position must have shape (windows, history, 2)".format(
                split_name
            )
        )
    _require_shape(
        history_truth,
        (window_count, history.shape[1], 2),
        "history_true_position",
        split_name,
    )
    if future.ndim != 3 or future.shape[0] != window_count or future.shape[2] != 2:
        raise ValueError(
            "{} future_position must have shape (windows, future, 2)".format(split_name)
        )
    if history.shape[1] == 0 or future.shape[1] == 0:
        raise ValueError("{} history and future windows must not be empty".format(split_name))
    _require_shape(origins, (window_count, 2), "coordinate_origin", split_name)
    for name in ("history_true_position", "future_position", "coordinate_origin"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError("{} field '{}' contains NaN or Inf".format(split_name, name))


def _canonical_value(value: Any) -> str:
    """Convert a NumPy/JSON scalar to a stable encoding lookup key."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _trajectory_decoder(manifest: Mapping[str, Any]) -> Dict[str, str]:
    """Normalize either name-to-code or code-to-name manifest encodings."""
    encoding = manifest["trajectory_type_encoding"]
    if not isinstance(encoding, dict):
        raise ValueError("trajectory_type_encoding must be a JSON object")

    decoder: Dict[str, str] = {}
    for raw_key, raw_value in encoding.items():
        key = _canonical_value(raw_key)
        value = _canonical_value(raw_value)
        if key in TRAJECTORY_TYPES:
            decoder[value] = key
            decoder[key] = key
        elif value in TRAJECTORY_TYPES:
            decoder[key] = value
            decoder[value] = value

    missing = [name for name in TRAJECTORY_TYPES if name not in decoder]
    if missing:
        raise ValueError(
            "trajectory_type_encoding does not define: {}".format(
                ", ".join(missing)
            )
        )
    return decoder


def _decoded_trajectory_types(
    encoded: np.ndarray, decoder: Mapping[str, str], split_name: str
) -> np.ndarray:
    """Decode one split's episode trajectory-type vector to names."""
    decoded = []
    unknown = []
    for value in encoded:
        key = _canonical_value(value)
        if key not in decoder:
            unknown.append(key)
        else:
            decoded.append(decoder[key])
    if unknown:
        raise ValueError(
            "{} contains unknown trajectory type encoding(s): {}".format(
                split_name, ", ".join(sorted(set(unknown)))
            )
        )
    return np.asarray(decoded, dtype="U32")


def _identifier_key(value: Any) -> Any:
    """Return a hashable Python scalar for an identifier."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _find_single_index(values: np.ndarray, target: Any, description: str) -> int:
    """Find exactly one identifier occurrence and report invalid mappings."""
    indices = np.flatnonzero(values == target)
    if indices.size != 1:
        raise ValueError(
            "Expected exactly one {} for {!r}, found {}".format(
                description, _identifier_key(target), int(indices.size)
            )
        )
    return int(indices[0])


def _first_window_bounds(
    arrays: Mapping[str, np.ndarray], episode_id: Any, episode_length: int
) -> Tuple[int, int, int, int]:
    """Return first-window index, start, history length, and future length."""
    matches = np.flatnonzero(arrays["episode_id"] == episode_id)
    if matches.size == 0:
        raise ValueError(
            "Episode {!r} has no saved window".format(_identifier_key(episode_id))
        )
    match_starts = arrays["sample_start_index"][matches]
    window_index = int(matches[int(np.argmin(match_starts))])
    raw_start = arrays["sample_start_index"][window_index]
    start = int(raw_start)
    if float(raw_start) != float(start):
        raise ValueError("sample_start_index must contain integer-valued indices")
    history_steps = int(arrays["history_position"].shape[1])
    future_steps = int(arrays["future_position"].shape[1])
    if start < 0 or start + history_steps + future_steps > episode_length:
        raise ValueError(
            "Episode {!r} first window [{}, {}) exceeds trajectory length {}".format(
                _identifier_key(episode_id),
                start,
                start + history_steps + future_steps,
                episode_length,
            )
        )
    return window_index, start, history_steps, future_steps


def _add_tree_patches(
    axes: Any, centers: np.ndarray, radii: np.ndarray
) -> None:
    """Draw world-coordinate tree trunks with their saved radii."""
    for index, (center, radius) in enumerate(zip(centers, radii)):
        axes.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                float(radius),
                facecolor="#795548",
                edgecolor="#3e2723",
                linewidth=0.9,
                alpha=0.68,
                label="Tree trunk" if index == 0 else None,
                zorder=1,
            )
        )


def _set_world_limits(
    axes: Any,
    truth: np.ndarray,
    sensor: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> None:
    """Set undistorted world-coordinate limits around all plotted geometry."""
    minimum = np.minimum(np.min(truth, axis=0), sensor)
    maximum = np.maximum(np.max(truth, axis=0), sensor)
    if centers.shape[0] > 0:
        minimum = np.minimum(minimum, np.min(centers - radii[:, None], axis=0))
        maximum = np.maximum(maximum, np.max(centers + radii[:, None], axis=0))
    span = np.maximum(maximum - minimum, np.array([1.0, 1.0]))
    margin = 0.05 * span
    axes.set_xlim(float(minimum[0] - margin[0]), float(maximum[0] + margin[0]))
    axes.set_ylim(float(minimum[1] - margin[1]), float(maximum[1] + margin[1]))
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x position (m)")
    axes.set_ylabel("y position (m)")
    axes.grid(True, color="#cfd8dc", linewidth=0.6, alpha=0.7)


def _plot_trajectory_type(
    output_path: Path,
    split_name: str,
    arrays: Mapping[str, np.ndarray],
    episode_index: int,
    trajectory_type: str,
) -> None:
    """Plot one saved full episode and its first saved history/future window."""
    episode_id = arrays["episode_ids"][episode_index]
    scene_id = arrays["episode_scene_ids"][episode_index]
    scene_index = _find_single_index(arrays["scene_ids"], scene_id, "saved scene")

    truth = arrays["episode_true_position_world"][episode_index]
    observations = arrays["episode_observed_position_world"][episode_index]
    visible = np.asarray(arrays["episode_visible_mask"][episode_index], dtype=bool)
    occluded = np.asarray(arrays["episode_occluded_mask"][episode_index], dtype=bool)
    dropout = np.asarray(
        arrays["episode_random_dropout_mask"][episode_index], dtype=bool
    )
    sensor = arrays["scene_sensor_position_world"][scene_index]
    centers = arrays["scene_tree_centers_world"][scene_index]
    radii = arrays["scene_tree_radii"][scene_index]

    window_index, start, history_steps, future_steps = _first_window_bounds(
        arrays, episode_id, int(truth.shape[0])
    )
    history_slice = slice(start, start + history_steps)
    future_slice = slice(
        start + history_steps, start + history_steps + future_steps
    )
    origin = arrays["coordinate_origin"][window_index]
    history_world = arrays["history_true_position"][window_index] + origin
    future_world = arrays["future_position"][window_index] + origin
    if not np.allclose(history_world, truth[history_slice], rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(
            "Episode {!r} saved history window is inconsistent with full truth".format(
                _identifier_key(episode_id)
            )
        )
    if not np.allclose(future_world, truth[future_slice], rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(
            "Episode {!r} saved future window is inconsistent with full truth".format(
                _identifier_key(episode_id)
            )
        )
    history_end_world = history_world[-1]
    future_start_world = future_world[0]

    figure, axes = plt.subplots(figsize=(10.8, 7.0))
    try:
        _add_tree_patches(axes, centers, radii)
        axes.plot(
            truth[:, 0],
            truth[:, 1],
            color="#424242",
            linewidth=1.5,
            label="Full ground truth",
            zorder=2,
        )
        axes.plot(
            history_world[:, 0],
            history_world[:, 1],
            color="#1565c0",
            linewidth=3.0,
            label="First-window history",
            zorder=3,
        )
        axes.plot(
            future_world[:, 0],
            future_world[:, 1],
            color="#8e24aa",
            linewidth=3.0,
            label="First-window future",
            zorder=3,
        )

        finite_observations = np.all(np.isfinite(observations), axis=1)
        visible_points = visible & finite_observations
        if np.any(visible_points):
            axes.scatter(
                observations[visible_points, 0],
                observations[visible_points, 1],
                marker="o",
                s=24,
                color="#43a047",
                edgecolor="white",
                linewidth=0.35,
                label="Visible observation",
                zorder=6,
            )
        if np.any(occluded):
            axes.scatter(
                truth[occluded, 0],
                truth[occluded, 1],
                marker="x",
                s=38,
                color="#c62828",
                linewidth=1.3,
                label="Geometric occlusion",
                zorder=7,
            )
        if np.any(dropout):
            axes.scatter(
                truth[dropout, 0],
                truth[dropout, 1],
                marker="^",
                s=46,
                facecolors="none",
                edgecolors="#ef6c00",
                linewidth=1.3,
                label="Random dropout",
                zorder=8,
            )
        axes.scatter(
            float(sensor[0]),
            float(sensor[1]),
            marker="*",
            s=190,
            color="#fdd835",
            edgecolor="#0d47a1",
            linewidth=1.0,
            label="Sensor",
            zorder=9,
        )
        axes.plot(
            [history_end_world[0], future_start_world[0]],
            [history_end_world[1], future_start_world[1]],
            color="#212121",
            linestyle="--",
            linewidth=1.2,
            zorder=9,
        )
        axes.scatter(
            history_end_world[0],
            history_end_world[1],
            marker="s",
            s=62,
            color="#1565c0",
            edgecolor="white",
            linewidth=0.6,
            label="History end",
            zorder=10,
        )
        axes.scatter(
            future_start_world[0],
            future_start_world[1],
            marker="D",
            s=58,
            color="#8e24aa",
            edgecolor="white",
            linewidth=0.6,
            label="Future start",
            zorder=10,
        )
        boundary_midpoint = 0.5 * (history_end_world + future_start_world)
        axes.annotate(
            "history / future boundary",
            xy=(float(boundary_midpoint[0]), float(boundary_midpoint[1])),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            color="#212121",
        )

        _set_world_limits(axes, truth, sensor, centers, radii)
        axes.set_title(
            "{} | split={} | scene={} | episode={}".format(
                trajectory_type,
                split_name,
                _identifier_key(scene_id),
                _identifier_key(episode_id),
            )
        )
        axes.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            framealpha=0.93,
            fontsize=8,
        )
        figure.tight_layout()
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _select_trajectory_episode(
    splits: Mapping[str, Mapping[str, np.ndarray]],
    decoded_types: Mapping[str, np.ndarray],
    trajectory_type: str,
) -> Tuple[str, Mapping[str, np.ndarray], int]:
    """Select a saved, windowed episode that best exposes observation states.

    Selection is deterministic and based only on saved arrays.  Episodes with
    visible, geometrically occluded, and randomly dropped points are preferred;
    among those, a balanced visible/occluded count improves plot readability.
    """
    best: Any = None
    best_score: Any = None
    for split_index, split_name in enumerate(SPLIT_NAMES):
        arrays = splits[split_name]
        matches = np.flatnonzero(decoded_types[split_name] == trajectory_type)
        for episode_index_raw in matches:
            episode_index = int(episode_index_raw)
            episode_id = arrays["episode_ids"][episode_index]
            if not np.any(arrays["episode_id"] == episode_id):
                continue
            visible_count = int(
                np.count_nonzero(arrays["episode_visible_mask"][episode_index])
            )
            occluded_count = int(
                np.count_nonzero(arrays["episode_occluded_mask"][episode_index])
            )
            dropout_count = int(
                np.count_nonzero(
                    arrays["episode_random_dropout_mask"][episode_index]
                )
            )
            all_states_present = int(
                visible_count > 0 and occluded_count > 0 and dropout_count > 0
            )
            state_count = sum(
                count > 0
                for count in (visible_count, occluded_count, dropout_count)
            )
            score = (
                all_states_present,
                state_count,
                min(visible_count, occluded_count),
                dropout_count,
                -split_index,
                -episode_index,
            )
            if best_score is None or score > best_score:
                best_score = score
                best = (split_name, arrays, episode_index)
    if best is None:
        raise ValueError(
            "Saved NPZ files contain no windowed episode for required trajectory "
            "type '{}'".format(trajectory_type)
        )
    return best


def _contiguous_true_run_lengths(mask: np.ndarray) -> List[int]:
    """Return positive lengths of contiguous true runs across all mask rows."""
    boolean_mask = np.asarray(mask, dtype=bool)
    if boolean_mask.ndim == 1:
        boolean_mask = boolean_mask[None, :]
    lengths: List[int] = []
    for row in boolean_mask:
        padded = np.concatenate(
            (np.array([False]), row, np.array([False]))
        ).astype(np.int8)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        lengths.extend((ends - starts).astype(int).tolist())
    return lengths


def _plot_trajectory_type_distribution(
    output_path: Path,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    decoded_types: Mapping[str, np.ndarray],
) -> None:
    """Plot saved episode counts by trajectory type and split."""
    figure, axes = plt.subplots(figsize=(10.0, 5.4))
    try:
        x_values = np.arange(len(TRAJECTORY_TYPES), dtype=float)
        width = 0.24
        for split_index, split_name in enumerate(SPLIT_NAMES):
            counts = np.asarray(
                [
                    np.count_nonzero(decoded_types[split_name] == trajectory_type)
                    for trajectory_type in TRAJECTORY_TYPES
                ],
                dtype=int,
            )
            offsets = x_values + (split_index - 1) * width
            bars = axes.bar(
                offsets,
                counts,
                width=width,
                color=_SPLIT_COLORS[split_name],
                label=split_name,
            )
            for bar, count in zip(bars, counts):
                axes.text(
                    bar.get_x() + bar.get_width() * 0.5,
                    float(bar.get_height()),
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axes.set_xticks(x_values)
        axes.set_xticklabels(TRAJECTORY_TYPES, rotation=20, ha="right")
        axes.set_ylabel("Saved episode count")
        axes.set_title("Trajectory-type distribution by split")
        axes.grid(True, axis="y", color="#cfd8dc", linewidth=0.6, alpha=0.7)
        axes.legend(loc="best")
        figure.tight_layout()
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_occlusion_length_distribution(
    output_path: Path, splits: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    """Plot contiguous geometric-occlusion lengths from saved episode masks."""
    lengths_by_split = {
        split_name: _contiguous_true_run_lengths(
            splits[split_name]["episode_occluded_mask"]
        )
        for split_name in SPLIT_NAMES
    }
    all_lengths = [
        length
        for split_name in SPLIT_NAMES
        for length in lengths_by_split[split_name]
    ]

    figure, axes = plt.subplots(figsize=(8.6, 5.2))
    try:
        if all_lengths:
            maximum = max(all_lengths)
            bins = np.arange(0.5, maximum + 1.5, 1.0)
            for split_name in SPLIT_NAMES:
                if lengths_by_split[split_name]:
                    axes.hist(
                        lengths_by_split[split_name],
                        bins=bins,
                        histtype="step",
                        linewidth=2.0,
                        color=_SPLIT_COLORS[split_name],
                        label=split_name,
                    )
            axes.set_xlim(0.5, maximum + 0.5)
            axes.legend(loc="best")
        else:
            axes.text(
                0.5,
                0.5,
                "No geometric-occlusion runs in saved episodes",
                transform=axes.transAxes,
                ha="center",
                va="center",
            )
        axes.set_xlabel("Contiguous geometric-occlusion length (steps)")
        axes.set_ylabel("Run count")
        axes.set_title("Geometric-occlusion length distribution")
        axes.grid(True, color="#cfd8dc", linewidth=0.6, alpha=0.7)
        figure.tight_layout()
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_speed_distribution(
    output_path: Path, splits: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    """Plot speed magnitudes computed from saved episode velocities."""
    speeds = {
        split_name: np.linalg.norm(
            splits[split_name]["episode_velocity_world"], axis=2
        ).ravel()
        for split_name in SPLIT_NAMES
    }
    all_speeds = np.concatenate([speeds[name] for name in SPLIT_NAMES])
    minimum = float(np.min(all_speeds))
    maximum = float(np.max(all_speeds))
    if np.isclose(minimum, maximum):
        padding = max(0.05, abs(minimum) * 0.05)
        bins = np.linspace(minimum - padding, maximum + padding, 21)
    else:
        bins = np.linspace(minimum, maximum, 31)

    figure, axes = plt.subplots(figsize=(8.6, 5.2))
    try:
        for split_name in SPLIT_NAMES:
            axes.hist(
                speeds[split_name],
                bins=bins,
                histtype="step",
                density=False,
                linewidth=2.0,
                color=_SPLIT_COLORS[split_name],
                label=split_name,
            )
        axes.set_xlabel("Speed (m/s)")
        axes.set_ylabel("Saved time-step count")
        axes.set_title("Speed distribution by split")
        axes.grid(True, color="#cfd8dc", linewidth=0.6, alpha=0.7)
        axes.legend(loc="best")
        figure.tight_layout()
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_split_statistics_comparison(
    output_path: Path, splits: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    """Compare saved scene/episode/window counts and observation-state rates."""
    count_labels = ("scenes", "episodes", "windows")
    count_values = {
        split_name: np.asarray(
            [
                splits[split_name]["scene_ids"].size,
                splits[split_name]["episode_ids"].size,
                splits[split_name]["episode_id"].size,
            ],
            dtype=int,
        )
        for split_name in SPLIT_NAMES
    }
    rate_labels = ("visible", "occluded", "random dropout")
    rate_fields = (
        "episode_visible_mask",
        "episode_occluded_mask",
        "episode_random_dropout_mask",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    try:
        width = 0.24
        count_x = np.arange(len(count_labels), dtype=float)
        rate_x = np.arange(len(rate_labels), dtype=float)
        for split_index, split_name in enumerate(SPLIT_NAMES):
            offset = (split_index - 1) * width
            count_bars = axes[0].bar(
                count_x + offset,
                count_values[split_name],
                width=width,
                color=_SPLIT_COLORS[split_name],
                label=split_name,
            )
            for bar, value in zip(count_bars, count_values[split_name]):
                axes[0].text(
                    bar.get_x() + bar.get_width() * 0.5,
                    float(bar.get_height()),
                    str(int(value)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            rates = np.asarray(
                [
                    np.mean(np.asarray(splits[split_name][field], dtype=bool))
                    for field in rate_fields
                ],
                dtype=float,
            )
            axes[1].bar(
                rate_x + offset,
                rates,
                width=width,
                color=_SPLIT_COLORS[split_name],
                label=split_name,
            )

        axes[0].set_xticks(count_x)
        axes[0].set_xticklabels(count_labels)
        axes[0].set_ylabel("Saved item count")
        axes[0].set_title("Dataset hierarchy")
        axes[0].grid(True, axis="y", color="#cfd8dc", linewidth=0.6, alpha=0.7)
        axes[0].legend(loc="best")

        axes[1].set_xticks(rate_x)
        axes[1].set_xticklabels(rate_labels, rotation=12, ha="right")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("Fraction of saved episode time steps")
        axes[1].set_title("Observation-state rates")
        axes[1].grid(True, axis="y", color="#cfd8dc", linewidth=0.6, alpha=0.7)
        axes[1].legend(loc="best")

        figure.suptitle("Train / validation / test statistics comparison")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def generate_dataset_v2_figures(dataset_dir: Path) -> List[Path]:
    """Generate dataset-v2 figures from already saved NPZ and manifest files.

    Args:
        dataset_dir: Directory containing ``train.npz``, ``validation.npz``,
            ``test.npz``, and ``dataset_manifest.json``.

    Returns:
        Paths to the five trajectory-type figures followed by the four summary
        figures.  Every returned file is created below ``dataset_dir/figures``.

    Raises:
        FileNotFoundError: If a required saved artifact does not exist.
        ValueError: If a required field, mapping, shape, or episode/window link
            is invalid.  No trajectory or scene is generated as a fallback.
    """
    root = Path(dataset_dir)
    manifest = _load_manifest(root / "dataset_manifest.json")
    splits = {
        split_name: _load_split(root / "{}.npz".format(split_name), split_name)
        for split_name in SPLIT_NAMES
    }
    decoder = _trajectory_decoder(manifest)
    decoded_types = {
        split_name: _decoded_trajectory_types(
            splits[split_name]["episode_trajectory_types"], decoder, split_name
        )
        for split_name in SPLIT_NAMES
    }

    locations = {
        trajectory_type: _select_trajectory_episode(
            splits, decoded_types, trajectory_type
        )
        for trajectory_type in TRAJECTORY_TYPES
    }

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    for trajectory_type in TRAJECTORY_TYPES:
        output_path = figure_dir / "trajectory_{}.png".format(trajectory_type)
        split_name, arrays, episode_index = locations[trajectory_type]
        _plot_trajectory_type(
            output_path,
            split_name,
            arrays,
            episode_index,
            trajectory_type,
        )
        output_paths.append(output_path)

    summary_plots = (
        (
            "trajectory_type_distribution.png",
            lambda path: _plot_trajectory_type_distribution(
                path, splits, decoded_types
            ),
        ),
        (
            "occlusion_length_distribution.png",
            lambda path: _plot_occlusion_length_distribution(path, splits),
        ),
        (
            "speed_distribution.png",
            lambda path: _plot_speed_distribution(path, splits),
        ),
        (
            "split_statistics_comparison.png",
            lambda path: _plot_split_statistics_comparison(path, splits),
        ),
    )
    for filename, plot_function in summary_plots:
        output_path = figure_dir / filename
        plot_function(output_path)
        output_paths.append(output_path)

    return output_paths


__all__ = ["generate_dataset_v2_figures"]
