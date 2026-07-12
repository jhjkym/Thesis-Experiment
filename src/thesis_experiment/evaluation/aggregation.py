"""Aggregate per-window metrics without treating windows as independent trials.

Windows cut from the same episode can overlap heavily.  Their direct mean is useful
as a descriptive diagnostic, but it must not be reported as though every window were
an independent paper sample.  Use the equal-weight episode or scene mean for formal
reporting, according to the evaluation unit declared by the experiment.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Mapping, Sequence, Tuple

import numpy as np


INDEPENDENCE_WARNING = (
    "Overlapping windows are not independent samples; use the equal-weight "
    "episode_mean or scene_mean for formal reporting."
)


def _as_metric_array(values: Sequence[float]) -> np.ndarray:
    """Return finite, one-dimensional metric values as float64."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("metric_values must be a one-dimensional array")
    if array.size == 0:
        raise ValueError("metric_values must contain at least one window")
    if array.dtype.kind not in "biuf":
        raise ValueError("metric_values must contain real numeric values")
    result = array.astype(np.float64, copy=False)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("metric_values must contain only finite values")
    return result


def _python_scalar(value: Any, name: str) -> Hashable:
    """Normalize a NumPy scalar and validate that it is a finite hashable ID."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError("{} must not contain None".format(name))
    if isinstance(value, (float, complex)) and not bool(np.isfinite(value)):
        raise ValueError("{} must contain only finite values".format(name))
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("{} values must be hashable scalars".format(name)) from error
    return value


def _as_label_tuple(
    values: Sequence[Any], name: str, expected_length: int
) -> Tuple[Hashable, ...]:
    """Return validated one-dimensional grouping labels as Python scalars."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional array".format(name))
    if array.shape[0] != expected_length:
        raise ValueError(
            "{} length {} does not match metric_values length {}".format(
                name, array.shape[0], expected_length
            )
        )
    return tuple(_python_scalar(value, name) for value in array)


def _group_statistics(
    metric_values: np.ndarray, labels: Tuple[Hashable, ...]
) -> Dict[Hashable, Dict[str, Any]]:
    """Compute count and direct window mean for each label in first-seen order."""

    grouped_values: Dict[Hashable, list] = {}
    for label, value in zip(labels, metric_values):
        grouped_values.setdefault(label, []).append(float(value))
    return {
        label: {
            "count": len(values),
            "window_mean": float(np.mean(np.asarray(values, dtype=np.float64))),
        }
        for label, values in grouped_values.items()
    }


def _validate_episode_mapping(
    episode_ids: Tuple[Hashable, ...],
    scene_ids: Tuple[Hashable, ...],
    trajectory_types: Tuple[Hashable, ...],
) -> None:
    """Ensure each episode maps to exactly one scene and trajectory type."""

    mappings: Dict[Hashable, Tuple[Hashable, Hashable]] = {}
    for episode_id, scene_id, trajectory_type in zip(
        episode_ids, scene_ids, trajectory_types
    ):
        current = (scene_id, trajectory_type)
        previous = mappings.setdefault(episode_id, current)
        if previous != current:
            raise ValueError(
                "episode_id {!r} maps inconsistently: {!r} and {!r}".format(
                    episode_id, previous, current
                )
            )


def aggregate_window_metrics(
    metric_values: Sequence[float],
    scene_id: Sequence[Any],
    episode_id: Sequence[Any],
    trajectory_type: Sequence[Any],
    occlusion_group: Sequence[Any],
) -> Dict[str, Any]:
    """Aggregate one finite metric value per prediction window.

    Parameters
    ----------
    metric_values:
        One-dimensional finite metric values, with one value per window.
    scene_id, episode_id, trajectory_type, occlusion_group:
        One-dimensional labels of the same length.  An episode ID must map to
        exactly one scene ID and one trajectory type.  Occlusion groups may vary
        between windows from the same episode.

    Returns
    -------
    dict
        ``window_mean`` is the direct mean over windows. ``episode_mean`` first
        averages windows within each episode and then gives every episode equal
        weight. ``scene_mean`` similarly first averages within each scene and then
        gives every scene equal weight. ``by_episode``, ``by_scene``,
        ``by_trajectory_type``, and ``by_occlusion_group`` contain per-group window
        counts and means.

    Notes
    -----
    Overlapping windows are not independent experimental samples.  The direct
    ``window_mean`` is descriptive only; formal paper results should declare and
    use ``episode_mean`` or ``scene_mean`` as their statistical unit.

    Raises
    ------
    ValueError
        If an input is empty, not one-dimensional, has a mismatched length,
        contains a non-finite value, or maps one episode to multiple scenes or
        trajectory types.
    """

    metrics = _as_metric_array(metric_values)
    count = int(metrics.size)
    scene_ids = _as_label_tuple(scene_id, "scene_id", count)
    episode_ids = _as_label_tuple(episode_id, "episode_id", count)
    trajectory_types = _as_label_tuple(
        trajectory_type, "trajectory_type", count
    )
    occlusion_groups = _as_label_tuple(
        occlusion_group, "occlusion_group", count
    )
    _validate_episode_mapping(episode_ids, scene_ids, trajectory_types)

    by_episode = _group_statistics(metrics, episode_ids)
    by_scene = _group_statistics(metrics, scene_ids)
    by_trajectory_type = _group_statistics(metrics, trajectory_types)
    by_occlusion_group = _group_statistics(metrics, occlusion_groups)

    episode_means = np.asarray(
        [statistics["window_mean"] for statistics in by_episode.values()],
        dtype=np.float64,
    )
    scene_means = np.asarray(
        [statistics["window_mean"] for statistics in by_scene.values()],
        dtype=np.float64,
    )
    return {
        "window_count": count,
        "window_mean": float(np.mean(metrics)),
        "episode_count": len(by_episode),
        "episode_mean": float(np.mean(episode_means)),
        "scene_count": len(by_scene),
        "scene_mean": float(np.mean(scene_means)),
        "by_episode": by_episode,
        "by_scene": by_scene,
        "by_trajectory_type": by_trajectory_type,
        "by_occlusion_group": by_occlusion_group,
    }
