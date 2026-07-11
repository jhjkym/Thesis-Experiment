"""Geometric predicates used to determine line-of-sight occlusion."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _as_point(value: Sequence[float], name: str) -> np.ndarray:
    """Convert *value* to a finite two-dimensional point."""

    point = np.asarray(value, dtype=float)
    if point.shape != (2,):
        raise ValueError("{} must have shape (2,), got {}".format(name, point.shape))
    if not np.all(np.isfinite(point)):
        raise ValueError("{} must contain only finite values".format(name))
    return point


def _as_circles(
    centers: Sequence[Sequence[float]], radii: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return a collection of circle centers and radii."""

    centers_array = np.asarray(centers, dtype=float)
    radii_array = np.asarray(radii, dtype=float)

    if centers_array.size == 0:
        centers_array = np.empty((0, 2), dtype=float)
    if centers_array.ndim != 2 or centers_array.shape[1] != 2:
        raise ValueError(
            "tree_centers must have shape (num_trees, 2), got {}".format(
                centers_array.shape
            )
        )
    if radii_array.ndim != 1 or radii_array.shape[0] != centers_array.shape[0]:
        raise ValueError(
            "tree_radii must have shape (num_trees,) matching tree_centers"
        )
    if not np.all(np.isfinite(centers_array)):
        raise ValueError("tree_centers must contain only finite values")
    if not np.all(np.isfinite(radii_array)) or np.any(radii_array < 0.0):
        raise ValueError("tree_radii must contain finite, non-negative values")
    return centers_array, radii_array


def segment_intersects_circle(
    segment_start: Sequence[float],
    segment_end: Sequence[float],
    circle_center: Sequence[float],
    radius: float,
    *,
    atol: float = 1e-12
) -> bool:
    """Return whether a closed line segment intersects a closed circle.

    Intersection includes tangency and the case where either endpoint lies inside
    the circle.  A degenerate segment is treated as a single point.  ``atol`` is
    an absolute distance tolerance, so distances no greater than
    ``radius + atol`` count as intersection.

    Args:
        segment_start: First segment endpoint as ``(x, y)``.
        segment_end: Second segment endpoint as ``(x, y)``.
        circle_center: Circle center as ``(x, y)``.
        radius: Non-negative circle radius.
        atol: Non-negative absolute distance tolerance.

    Raises:
        ValueError: If an input has an invalid shape or non-finite value, or if
            ``radius`` or ``atol`` is negative.
    """

    start = _as_point(segment_start, "segment_start")
    end = _as_point(segment_end, "segment_end")
    center = _as_point(circle_center, "circle_center")
    radius_value = float(radius)
    atol_value = float(atol)
    if not np.isfinite(radius_value) or radius_value < 0.0:
        raise ValueError("radius must be finite and non-negative")
    if not np.isfinite(atol_value) or atol_value < 0.0:
        raise ValueError("atol must be finite and non-negative")

    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        closest = start
    else:
        projection = float(np.dot(center - start, direction) / length_squared)
        projection = float(np.clip(projection, 0.0, 1.0))
        closest = start + projection * direction

    distance = float(np.linalg.norm(closest - center))
    return bool(distance <= radius_value + atol_value)


def segment_intersects_circles(
    segment_start: Sequence[float],
    segment_end: Sequence[float],
    circle_centers: Sequence[Sequence[float]],
    circle_radii: Sequence[float],
    *,
    atol: float = 1e-12
) -> np.ndarray:
    """Return an intersection flag for every circle against one segment.

    Args:
        segment_start: First segment endpoint as ``(x, y)``.
        segment_end: Second segment endpoint as ``(x, y)``.
        circle_centers: Circle centers with shape ``(num_circles, 2)``.
        circle_radii: Non-negative radii with shape ``(num_circles,)``.
        atol: Non-negative absolute distance tolerance.

    Returns:
        A Boolean array with shape ``(num_circles,)``.
    """

    start = _as_point(segment_start, "segment_start")
    end = _as_point(segment_end, "segment_end")
    centers, radii = _as_circles(circle_centers, circle_radii)
    atol_value = float(atol)
    if not np.isfinite(atol_value) or atol_value < 0.0:
        raise ValueError("atol must be finite and non-negative")
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        closest = np.broadcast_to(start, centers.shape)
    else:
        projections = np.einsum("ij,j->i", centers - start, direction)
        projections = np.clip(projections / length_squared, 0.0, 1.0)
        closest = start + projections[:, np.newaxis] * direction

    distances = np.linalg.norm(closest - centers, axis=1)
    return distances <= radii + atol_value


def is_occluded(
    sensor_position: Sequence[float],
    target_position: Sequence[float],
    tree_centers: Sequence[Sequence[float]],
    tree_radii: Sequence[float],
    *,
    atol: float = 1e-12
) -> bool:
    """Return whether any tree intersects the sensor-to-target segment."""

    intersections = segment_intersects_circles(
        sensor_position,
        target_position,
        tree_centers,
        tree_radii,
        atol=atol,
    )
    return bool(np.any(intersections))


def occlusion_mask(
    sensor_position: Sequence[float],
    target_positions: Sequence[Sequence[float]],
    tree_centers: Sequence[Sequence[float]],
    tree_radii: Sequence[float],
    *,
    atol: float = 1e-12
) -> np.ndarray:
    """Return one occlusion flag for each target position.

    Args:
        sensor_position: Fixed sensor location as ``(x, y)``.
        target_positions: Target positions with shape ``(num_steps, 2)``.
        tree_centers: Tree centers with shape ``(num_trees, 2)``.
        tree_radii: Tree radii with shape ``(num_trees,)``.
        atol: Non-negative absolute distance tolerance.

    Returns:
        Boolean array with shape ``(num_steps,)``; ``True`` means occluded.
    """

    sensor = _as_point(sensor_position, "sensor_position")
    targets = np.asarray(target_positions, dtype=float)
    if targets.size == 0:
        targets = np.empty((0, 2), dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(
            "target_positions must have shape (num_steps, 2), got {}".format(
                targets.shape
            )
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("target_positions must contain only finite values")
    centers, radii = _as_circles(tree_centers, tree_radii)
    atol_value = float(atol)
    if not np.isfinite(atol_value) or atol_value < 0.0:
        raise ValueError("atol must be finite and non-negative")

    result = np.zeros((targets.shape[0],), dtype=bool)
    for index, target in enumerate(targets):
        result[index] = bool(
            np.any(
                segment_intersects_circles(
                    sensor, target, centers, radii, atol=atol_value
                )
            )
        )
    return result


__all__ = [
    "is_occluded",
    "occlusion_mask",
    "segment_intersects_circle",
    "segment_intersects_circles",
]
