"""Deterministic trajectory-prediction metrics and reporting records.

The functions in this module operate only on predictions, supervision labels, and
post-inference audit metadata.  ``trajectory_type`` and ``occlusion_length_bin``
are accepted exclusively for reporting groups; they are never model features.

All distance metrics use the same position unit as the input trajectories.  The
per-horizon RMSE is defined as ``sqrt(mean(dx**2 + dy**2))`` over windows.  This is
the root mean squared *Euclidean displacement*, rather than an RMSE averaged over
the two coordinate components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .aggregation import aggregate_window_metrics


def seed_mean_and_sample_std(values: Sequence[float]) -> Tuple[float, float]:
    """Return the mean and across-seed sample standard deviation.

    Independent random seeds are treated as a sample from the experiment's
    run-to-run distribution, so two or more seeds use Bessel's correction
    (``ddof=1``).  A deterministic or single-seed method has no observed
    across-seed dispersion and reports exactly zero.  This helper is not for
    repeated runtime measurements or within-window variation.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("seed values must be a non-empty one-dimensional array")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("seed values must contain only finite values")
    deviation = 0.0 if array.size == 1 else float(np.std(array, ddof=1))
    return float(np.mean(array)), deviation


def _validated_trajectories(
    prediction: np.ndarray, future_position: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return matching finite ``(windows, horizons, 2)`` float64 arrays."""

    prediction_array = np.asarray(prediction)
    target_array = np.asarray(future_position)
    for name, array in (
        ("prediction", prediction_array),
        ("future_position", target_array),
    ):
        if array.dtype.kind not in "biuf":
            raise ValueError("{} must contain real numeric values".format(name))
        if array.ndim != 3 or array.shape[2] != 2:
            raise ValueError("{} must have shape (N, T, 2)".format(name))
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("{} must contain at least one window and horizon".format(name))
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("{} must contain only finite values".format(name))
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction shape {} does not match future_position shape {}".format(
                prediction_array.shape, target_array.shape
            )
        )
    return (
        prediction_array.astype(np.float64, copy=False),
        target_array.astype(np.float64, copy=False),
    )


def displacement_errors(
    prediction: np.ndarray, future_position: np.ndarray
) -> np.ndarray:
    """Return Euclidean position error for every window and future horizon.

    Parameters
    ----------
    prediction, future_position:
        Finite arrays with identical shape ``(N, T, 2)``.

    Returns
    -------
    numpy.ndarray
        A float64 array with shape ``(N, T)``.
    """

    prediction_array, target_array = _validated_trajectories(
        prediction, future_position
    )
    return np.linalg.norm(prediction_array - target_array, axis=2)


def ade_per_window(
    prediction: np.ndarray, future_position: np.ndarray
) -> np.ndarray:
    """Return Average Displacement Error (ADE) for each prediction window."""

    return np.mean(displacement_errors(prediction, future_position), axis=1)


def fde_per_window(
    prediction: np.ndarray, future_position: np.ndarray
) -> np.ndarray:
    """Return Final Displacement Error (FDE) for each prediction window."""

    return displacement_errors(prediction, future_position)[:, -1]


def per_horizon_metrics(
    prediction: np.ndarray, future_position: np.ndarray
) -> Dict[str, np.ndarray]:
    """Return mean Euclidean error and Euclidean RMSE at each future horizon.

    ``mean_euclidean_distance[k]`` is the arithmetic mean of Euclidean position
    errors across windows at horizon ``k + 1``. ``rmse[k]`` is
    ``sqrt(mean(dx**2 + dy**2))`` across those same windows.
    """

    errors = displacement_errors(prediction, future_position)
    return {
        "horizon_step": np.arange(1, errors.shape[1] + 1, dtype=np.int64),
        "mean_euclidean_distance": np.mean(errors, axis=0),
        "rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
    }


def compute_prediction_metrics(
    prediction: np.ndarray, future_position: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute all deterministic metrics without repeating input validation.

    The returned ``ade`` and ``fde`` arrays contain one value per window;
    ``displacement_error`` contains one value per window and horizon.  The three
    ``per_horizon_*`` arrays contain one value per future horizon.
    """

    errors = displacement_errors(prediction, future_position)
    return {
        "displacement_error": errors,
        "ade": np.mean(errors, axis=1),
        "fde": errors[:, -1],
        "per_horizon_step": np.arange(1, errors.shape[1] + 1, dtype=np.int64),
        "per_horizon_mean_euclidean_distance": np.mean(errors, axis=0),
        "per_horizon_rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
    }


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values to JSON/CSV-friendly Python scalars."""

    return value.item() if isinstance(value, np.generic) else value


def _validated_labels(
    values: Sequence[Any], name: str, expected_length: int
) -> np.ndarray:
    """Return a one-dimensional metadata array with the required length."""

    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != expected_length:
        raise ValueError(
            "{} must be one-dimensional with length {}".format(
                name, expected_length
            )
        )
    return array


def _validated_metric_mapping(
    metric_values: Mapping[str, Sequence[float]]
) -> Tuple[Dict[str, np.ndarray], int]:
    """Validate named finite per-window metric vectors."""

    if not metric_values:
        raise ValueError("metric_values must contain at least one named metric")
    result: Dict[str, np.ndarray] = {}
    expected_length: Optional[int] = None
    for name, values in metric_values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        array = np.asarray(values)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("metric {!r} must be a non-empty 1D array".format(name))
        if array.dtype.kind not in "biuf":
            raise ValueError("metric {!r} must contain real numeric values".format(name))
        array = array.astype(np.float64, copy=False)
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("metric {!r} must contain only finite values".format(name))
        if expected_length is None:
            expected_length = int(array.size)
        elif array.size != expected_length:
            raise ValueError("all per-window metrics must have the same length")
        result[name] = array
    assert expected_length is not None
    return result, expected_length


def _add_model_name(record: Dict[str, Any], model_name: Optional[str]) -> None:
    """Add an optional model identifier to a reporting record in place."""

    if model_name is not None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string when provided")
        record["model_name"] = model_name


def prediction_metric_records(
    metric_values: Mapping[str, Sequence[float]],
    scene_id: Sequence[Any],
    episode_id: Sequence[Any],
    trajectory_type: Sequence[Any],
    occlusion_length_bin: Sequence[Any],
    sample_start_index: Optional[Sequence[Any]] = None,
    model_name: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate per-window metrics and return CSV-ready records.

    The return value has six record tables: ``window``, ``summary``,
    ``episode``, ``scene``, ``trajectory_type``, and ``occlusion_length_bin``.
    Every record contains Python scalars and can be passed directly to
    ``pandas.DataFrame``.  Summary records contain the descriptive window mean
    and the equal-weight episode and scene means.

    Motion type and occlusion labels are post-inference metadata used only for
    grouping.  Overlapping windows are not treated as independent paper samples;
    formal reporting should use the episode or scene summary row.
    """

    metrics, window_count = _validated_metric_mapping(metric_values)
    scene_ids = _validated_labels(scene_id, "scene_id", window_count)
    episode_ids = _validated_labels(episode_id, "episode_id", window_count)
    trajectory_types = _validated_labels(
        trajectory_type, "trajectory_type", window_count
    )
    occlusion_bins = _validated_labels(
        occlusion_length_bin, "occlusion_length_bin", window_count
    )
    if sample_start_index is None:
        start_indices: Optional[np.ndarray] = None
    else:
        start_indices = _validated_labels(
            sample_start_index, "sample_start_index", window_count
        )

    tables: Dict[str, List[Dict[str, Any]]] = {
        "window": [],
        "summary": [],
        "episode": [],
        "scene": [],
        "trajectory_type": [],
        "occlusion_length_bin": [],
    }

    for index in range(window_count):
        window_record: Dict[str, Any] = {
            "window_index": index,
            "scene_id": _python_scalar(scene_ids[index]),
            "episode_id": _python_scalar(episode_ids[index]),
            "trajectory_type": _python_scalar(trajectory_types[index]),
            "occlusion_length_bin": _python_scalar(occlusion_bins[index]),
        }
        if start_indices is not None:
            window_record["sample_start_index"] = _python_scalar(
                start_indices[index]
            )
        for metric_name, values in metrics.items():
            window_record[metric_name] = float(values[index])
        _add_model_name(window_record, model_name)
        tables["window"].append(window_record)

    # First-seen metadata mappings are used only to annotate aggregate rows.  The
    # called aggregation utility independently verifies episode mapping integrity.
    episode_metadata: Dict[Any, Tuple[Any, Any]] = {}
    scene_episode_ids: Dict[Any, set] = {}
    for scene_value, episode_value, type_value in zip(
        scene_ids, episode_ids, trajectory_types
    ):
        scene_key = _python_scalar(scene_value)
        episode_key = _python_scalar(episode_value)
        type_key = _python_scalar(type_value)
        episode_metadata.setdefault(episode_key, (scene_key, type_key))
        scene_episode_ids.setdefault(scene_key, set()).add(episode_key)

    for metric_name, values in metrics.items():
        aggregate = aggregate_window_metrics(
            values,
            scene_id=scene_ids,
            episode_id=episode_ids,
            trajectory_type=trajectory_types,
            occlusion_group=occlusion_bins,
        )
        for level in ("window", "episode", "scene"):
            summary_record: Dict[str, Any] = {
                "metric": metric_name,
                "aggregation_level": level,
                "unit_count": int(aggregate["{}_count".format(level)]),
                "mean": float(aggregate["{}_mean".format(level)]),
            }
            _add_model_name(summary_record, model_name)
            tables["summary"].append(summary_record)

        for episode_key, statistics in aggregate["by_episode"].items():
            scene_key, type_key = episode_metadata[episode_key]
            episode_record = {
                "metric": metric_name,
                "episode_id": episode_key,
                "scene_id": scene_key,
                "trajectory_type": type_key,
                "window_count": int(statistics["count"]),
                "mean": float(statistics["window_mean"]),
            }
            _add_model_name(episode_record, model_name)
            tables["episode"].append(episode_record)

        for scene_key, statistics in aggregate["by_scene"].items():
            scene_record = {
                "metric": metric_name,
                "scene_id": scene_key,
                "episode_count": len(scene_episode_ids[scene_key]),
                "window_count": int(statistics["count"]),
                "mean": float(statistics["window_mean"]),
            }
            _add_model_name(scene_record, model_name)
            tables["scene"].append(scene_record)

        for type_key, statistics in aggregate["by_trajectory_type"].items():
            type_record = {
                "metric": metric_name,
                "trajectory_type": type_key,
                "window_count": int(statistics["count"]),
                "window_mean": float(statistics["window_mean"]),
            }
            _add_model_name(type_record, model_name)
            tables["trajectory_type"].append(type_record)

        for bin_key, statistics in aggregate["by_occlusion_group"].items():
            occlusion_record = {
                "metric": metric_name,
                "occlusion_length_bin": bin_key,
                "window_count": int(statistics["count"]),
                "window_mean": float(statistics["window_mean"]),
            }
            _add_model_name(occlusion_record, model_name)
            tables["occlusion_length_bin"].append(occlusion_record)

    return tables


__all__ = [
    "ade_per_window",
    "compute_prediction_metrics",
    "displacement_errors",
    "fde_per_window",
    "per_horizon_metrics",
    "prediction_metric_records",
]
