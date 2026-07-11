"""Reproducible generation of non-overlapping circular tree trunks."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np


Seed = Optional[Union[int, np.integer]]


def _as_pair(value: Sequence[float], name: str) -> np.ndarray:
    """Convert *value* to a finite floating-point pair."""

    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,):
        raise ValueError("{} must have shape (2,), got {}".format(name, pair.shape))
    if not np.all(np.isfinite(pair)):
        raise ValueError("{} must contain only finite values".format(name))
    return pair


def generate_tree_trunks(
    scene_size: Sequence[float],
    tree_count: int,
    radius_range: Sequence[float],
    min_spacing: float,
    sensor_position: Sequence[float],
    *,
    seed: Seed = None,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate circular trunks inside a rectangular two-dimensional scene.

    Scene coordinates span ``[0, width] x [0, height]``.  Every generated tree
    lies fully within the boundary, and the edge-to-edge distance between any
    pair is at least ``min_spacing``.  The sensor may not lie inside or touch a
    trunk.  Rejection sampling is deterministic for a fixed ``seed``.

    Args:
        scene_size: ``(width, height)`` of the scene.
        tree_count: Number of trunks to generate.
        radius_range: Inclusive lower/upper radius bounds.
        min_spacing: Required minimum edge-to-edge distance between trunks.
        sensor_position: Fixed sensor location as ``(x, y)``.
        seed: Optional seed used to create a local random generator.
        rng: Optional existing NumPy ``Generator``.  Mutually exclusive with
            ``seed``.
        max_attempts: Maximum total number of rejection-sampling attempts.

    Returns:
        ``(centers, radii)`` where centers has shape ``(tree_count, 2)`` and
        radii has shape ``(tree_count,)``.

    Raises:
        ValueError: If configuration values are invalid or both ``seed`` and
            ``rng`` are supplied.
        RuntimeError: If the requested forest cannot be placed within
            ``max_attempts`` attempts.
    """

    size = _as_pair(scene_size, "scene_size")
    if np.any(size <= 0.0):
        raise ValueError("scene_size values must be positive")

    if isinstance(tree_count, (bool, np.bool_)) or not isinstance(
        tree_count, (int, np.integer)
    ):
        raise ValueError("tree_count must be a non-negative integer")
    count = int(tree_count)
    if count < 0:
        raise ValueError("tree_count must be a non-negative integer")

    radius_bounds = _as_pair(radius_range, "radius_range")
    minimum_radius, maximum_radius = radius_bounds
    if minimum_radius <= 0.0 or maximum_radius < minimum_radius:
        raise ValueError(
            "radius_range must satisfy 0 < minimum_radius <= maximum_radius"
        )
    if 2.0 * maximum_radius > float(np.min(size)):
        raise ValueError("the largest radius cannot fit within the scene")

    spacing = float(min_spacing)
    if not np.isfinite(spacing) or spacing < 0.0:
        raise ValueError("min_spacing must be finite and non-negative")

    sensor = _as_pair(sensor_position, "sensor_position")
    if np.any(sensor < 0.0) or np.any(sensor > size):
        raise ValueError("sensor_position must lie within the scene boundary")

    if isinstance(max_attempts, (bool, np.bool_)) or not isinstance(
        max_attempts, (int, np.integer)
    ):
        raise ValueError("max_attempts must be a positive integer")
    attempt_limit = int(max_attempts)
    if attempt_limit <= 0:
        raise ValueError("max_attempts must be a positive integer")

    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")
    generator = rng if rng is not None else np.random.default_rng(seed)

    centers = np.empty((count, 2), dtype=float)
    radii = np.empty((count,), dtype=float)
    placed = 0

    for _ in range(attempt_limit):
        if placed == count:
            break

        radius = float(generator.uniform(minimum_radius, maximum_radius))
        center = np.array(
            [
                generator.uniform(radius, size[0] - radius),
                generator.uniform(radius, size[1] - radius),
            ],
            dtype=float,
        )

        # Touching the sensor is forbidden, hence the strict acceptance test.
        if float(np.linalg.norm(center - sensor)) <= radius:
            continue

        if placed:
            center_distances = np.linalg.norm(centers[:placed] - center, axis=1)
            required_distances = radii[:placed] + radius + spacing
            if np.any(center_distances < required_distances):
                continue

        centers[placed] = center
        radii[placed] = radius
        placed += 1

    if placed != count:
        raise RuntimeError(
            "could place only {} of {} trunks after {} attempts; reduce tree_count, "
            "radii, or min_spacing, or increase max_attempts".format(
                placed, count, attempt_limit
            )
        )

    return centers, radii


__all__ = ["generate_tree_trunks"]
