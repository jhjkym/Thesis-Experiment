"""Matplotlib visualizations for forest scenes and target observations."""

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Patch


PointArray = Union[np.ndarray, Sequence[Sequence[float]]]
Vector = Union[np.ndarray, Sequence[float]]


def _as_vector(values: Vector, name: str) -> np.ndarray:
    """Return a finite two-dimensional vector after validating its shape."""
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError("{} must contain exactly two finite values".format(name))
    return array


def _as_points(values: PointArray, name: str) -> np.ndarray:
    """Return an ``(N, 2)`` point array, accepting an empty input."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("{} must have shape (N, 2)".format(name))
    return array


def _as_scene_size(scene_size: Vector) -> Tuple[float, float]:
    """Validate and return scene width and height."""
    size = _as_vector(scene_size, "scene_size")
    if np.any(size <= 0.0):
        raise ValueError("scene_size values must be positive")
    return float(size[0]), float(size[1])


def _as_tree_data(
    tree_centers: PointArray, tree_radii: Vector
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate tree centers and radii for plotting."""
    centers = _as_points(tree_centers, "tree_centers")
    radii = np.asarray(tree_radii, dtype=float)
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("tree_radii must have shape (N,) matching tree_centers")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("tree_radii must contain finite positive values")
    return centers, radii


def _add_trees(
    axes: Any, tree_centers: np.ndarray, tree_radii: np.ndarray
) -> None:
    """Draw tree-trunk circles on an existing axes."""
    for index, (center, radius) in enumerate(zip(tree_centers, tree_radii)):
        circle = Circle(
            (float(center[0]), float(center[1])),
            float(radius),
            facecolor="#795548",
            edgecolor="#3e2723",
            linewidth=1.0,
            alpha=0.72,
            label="Tree trunk" if index == 0 else None,
            zorder=2,
        )
        axes.add_patch(circle)


def _prepare_output(output_path: Path) -> Path:
    """Create the output directory and return a normalized path object."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _set_scene_axes(axes: Any, scene_size: Tuple[float, float]) -> None:
    """Apply consistent scene bounds, labels, and aspect ratio."""
    width, height = scene_size
    axes.set_xlim(0.0, width)
    axes.set_ylim(0.0, height)
    axes.set_xlabel("x position")
    axes.set_ylabel("y position")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, color="#cfd8dc", linewidth=0.6, alpha=0.7)


def plot_scene(
    output_path: Path,
    scene_size: Vector,
    sensor_position: Vector,
    tree_centers: PointArray,
    tree_radii: Vector,
    target_positions: Optional[PointArray] = None,
    title: str = "Forest scene",
) -> None:
    """Plot a top-down forest scene and optionally a target trajectory.

    Args:
        output_path: Image file to create. Its parent directory is created.
        scene_size: Scene width and height.
        sensor_position: Fixed sensor ``(x, y)`` position.
        tree_centers: Tree center coordinates with shape ``(N, 2)``.
        tree_radii: Tree radii with shape ``(N,)``.
        target_positions: Optional target coordinates with shape ``(T, 2)``.
        title: Plot title, including sample provenance when available.
    """
    path = _prepare_output(output_path)
    size = _as_scene_size(scene_size)
    sensor = _as_vector(sensor_position, "sensor_position")
    centers, radii = _as_tree_data(tree_centers, tree_radii)
    targets = None
    if target_positions is not None:
        targets = _as_points(target_positions, "target_positions")

    figure, axes = plt.subplots(figsize=(8.0, 6.0))
    try:
        _add_trees(axes, centers, radii)
        axes.scatter(
            sensor[0],
            sensor[1],
            marker="*",
            s=180,
            color="#1565c0",
            edgecolor="white",
            linewidth=0.8,
            label="Sensor",
            zorder=5,
        )
        if targets is not None and targets.shape[0] > 0:
            axes.plot(
                targets[:, 0],
                targets[:, 1],
                color="#7b1fa2",
                linewidth=1.8,
                label="Target path",
                zorder=3,
            )
            axes.scatter(
                targets[0, 0],
                targets[0, 1],
                marker="o",
                s=45,
                color="#43a047",
                label="Target start",
                zorder=4,
            )
            axes.scatter(
                targets[-1, 0],
                targets[-1, 1],
                marker="X",
                s=55,
                color="#e53935",
                label="Target end",
                zorder=4,
            )
        _set_scene_axes(axes, size)
        axes.set_title(title)
        axes.legend(loc="best", framealpha=0.9)
        figure.tight_layout()
        figure.savefig(str(path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_trajectory(
    output_path: Path,
    true_positions: PointArray,
    observed_positions: PointArray,
    visible_mask: Vector,
    sensor_position: Optional[Vector] = None,
    tree_centers: Optional[PointArray] = None,
    tree_radii: Optional[Vector] = None,
    scene_size: Optional[Vector] = None,
    title: str = "True and observed target trajectory",
) -> None:
    """Plot true and observed trajectories, highlighting missing observations.

    Invisible samples are shown at their true positions with orange crosses. The
    dedicated mask-timeline plot distinguishes geometric occlusion from dropout.

    Args:
        output_path: Image file to create. Its parent directory is created.
        true_positions: Ground-truth trajectory with shape ``(T, 2)``.
        observed_positions: Noisy observations with shape ``(T, 2)``; missing
            observations may contain NaNs.
        visible_mask: Boolean-like vector with shape ``(T,)``.
        sensor_position: Optional fixed sensor position.
        tree_centers: Optional tree center coordinates.
        tree_radii: Optional tree radii; required when centers are supplied.
        scene_size: Optional scene width and height used as plot bounds.
        title: Plot title, including sample provenance when available.
    """
    path = _prepare_output(output_path)
    truth = _as_points(true_positions, "true_positions")
    observations = _as_points(observed_positions, "observed_positions")
    visible = np.asarray(visible_mask, dtype=bool)
    if truth.shape[0] == 0:
        raise ValueError("true_positions must contain at least one point")
    if observations.shape != truth.shape:
        raise ValueError("observed_positions must match true_positions")
    if visible.ndim != 1 or visible.shape[0] != truth.shape[0]:
        raise ValueError("visible_mask must have shape (T,)")
    if not np.all(np.isfinite(truth)):
        raise ValueError("true_positions must contain finite values")

    centers = np.empty((0, 2), dtype=float)
    radii = np.empty((0,), dtype=float)
    if tree_centers is not None or tree_radii is not None:
        if tree_centers is None or tree_radii is None:
            raise ValueError("tree_centers and tree_radii must be provided together")
        centers, radii = _as_tree_data(tree_centers, tree_radii)

    sensor = None
    if sensor_position is not None:
        sensor = _as_vector(sensor_position, "sensor_position")
    size = None
    if scene_size is not None:
        size = _as_scene_size(scene_size)

    finite_observation = np.all(np.isfinite(observations), axis=1)
    visible_observation = visible & finite_observation
    missing = ~visible

    figure, axes = plt.subplots(figsize=(8.0, 6.0))
    try:
        _add_trees(axes, centers, radii)
        axes.plot(
            truth[:, 0],
            truth[:, 1],
            color="#263238",
            linewidth=2.0,
            label="Ground truth",
            zorder=3,
        )
        if np.any(visible_observation):
            axes.scatter(
                observations[visible_observation, 0],
                observations[visible_observation, 1],
                marker="o",
                s=28,
                color="#2e7d32",
                edgecolor="white",
                linewidth=0.4,
                label="Visible observation",
                zorder=5,
            )
        if np.any(missing):
            axes.scatter(
                truth[missing, 0],
                truth[missing, 1],
                marker="x",
                s=36,
                color="#ef6c00",
                linewidth=1.4,
                label="Missing (occlusion/dropout)",
                zorder=5,
            )
        if sensor is not None:
            axes.scatter(
                sensor[0],
                sensor[1],
                marker="*",
                s=180,
                color="#1565c0",
                edgecolor="white",
                linewidth=0.8,
                label="Sensor",
                zorder=6,
            )
        if size is not None:
            _set_scene_axes(axes, size)
        else:
            axes.set_xlabel("x position")
            axes.set_ylabel("y position")
            axes.set_aspect("equal", adjustable="datalim")
            axes.margins(0.08)
            axes.grid(True, color="#cfd8dc", linewidth=0.6, alpha=0.7)
        axes.set_title(title)
        axes.legend(loc="best", framealpha=0.9)
        figure.tight_layout()
        figure.savefig(str(path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_mask_timeline(
    output_path: Path,
    times: Vector,
    visible_mask: Vector,
    occluded_mask: Vector,
    dropout_mask: Vector,
    title: str = "Observation mask timeline",
) -> None:
    """Plot visible, occluded, and random-dropout states as discrete bands.

    Each row has its own active color, so simultaneous condition flags remain
    visible rather than being hidden by categorical-state precedence.

    Args:
        output_path: Image file to create. Its parent directory is created.
        times: Strictly increasing sample times with shape ``(T,)``.
        visible_mask: Boolean-like visible-state vector.
        occluded_mask: Boolean-like geometric-occlusion vector.
        dropout_mask: Boolean-like random-dropout vector.
        title: Plot title, including sample provenance when available.
    """
    path = _prepare_output(output_path)
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("times must contain finite values")
    if time_values.size > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing")

    masks = []
    for name, values in (
        ("visible_mask", visible_mask),
        ("occluded_mask", occluded_mask),
        ("dropout_mask", dropout_mask),
    ):
        mask = np.asarray(values, dtype=bool)
        if mask.ndim != 1 or mask.shape != time_values.shape:
            raise ValueError("{} must match times with shape (T,)".format(name))
        masks.append(mask)

    if time_values.size == 1:
        edges = np.array([time_values[0] - 0.5, time_values[0] + 0.5])
    else:
        midpoints = (time_values[:-1] + time_values[1:]) * 0.5
        edges = np.concatenate(
            (
                [time_values[0] - (time_values[1] - time_values[0]) * 0.5],
                midpoints,
                [time_values[-1] + (time_values[-1] - time_values[-2]) * 0.5],
            )
        )

    states = np.zeros((3, time_values.size), dtype=int)
    states[0, masks[0]] = 1
    states[1, masks[1]] = 2
    states[2, masks[2]] = 3
    colors = ["#eceff1", "#2e7d32", "#c62828", "#ef6c00"]
    color_map = ListedColormap(colors)
    normalization = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], color_map.N)

    figure, axes = plt.subplots(figsize=(9.0, 3.2))
    try:
        axes.pcolormesh(
            edges,
            np.arange(4, dtype=float),
            states,
            cmap=color_map,
            norm=normalization,
            shading="flat",
            edgecolors="white",
            linewidth=0.12,
        )
        axes.set_xlim(edges[0], edges[-1])
        axes.set_ylim(0.0, 3.0)
        axes.set_yticks([0.5, 1.5, 2.5])
        axes.set_yticklabels(["Visible", "Occluded", "Random dropout"])
        axes.set_xlabel("Time")
        axes.set_title(title)
        axes.legend(
            handles=[
                Patch(facecolor=colors[1], label="Visible"),
                Patch(facecolor=colors[2], label="Geometric occlusion"),
                Patch(facecolor=colors[3], label="Random dropout"),
                Patch(facecolor=colors[0], label="Inactive"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=4,
            frameon=False,
        )
        figure.tight_layout()
        figure.savefig(str(path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
