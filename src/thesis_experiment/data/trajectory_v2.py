"""Deterministic two-dimensional motion models for dataset version 2.

The models in this module generate one complete episode at a time.  They do
not sample random parameters or enforce scene constraints; those concerns
belong to the dataset builder.  Every model uses the first requested time as
the episode origin and returns positions integrated from ``initial_position``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


TRAJECTORY_TYPE_TO_CODE: Dict[str, int] = {
    "constant_velocity": 0,
    "constant_acceleration": 1,
    "constant_turn": 2,
    "stop_and_go": 3,
    "piecewise_direction": 4,
}
"""Stable integer encoding used when trajectory types are saved to NPZ."""


def _finite_vector(value: Sequence[float], name: str) -> np.ndarray:
    """Return *value* as a defensive finite ``(2,)`` floating-point copy."""

    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ValueError("{} must have shape (2,), got {}".format(name, vector.shape))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(name))
    return vector.copy()


def _finite_scalar(value: float, name: str) -> float:
    """Return *value* as a finite float."""

    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("{} must be finite".format(name))
    return scalar


def _validated_times(times: Sequence[float]) -> np.ndarray:
    """Return a validated one-dimensional array of increasing sample times."""

    values = np.asarray(times, dtype=float)
    if values.ndim != 1:
        raise ValueError("times must be one-dimensional, got shape {}".format(values.shape))
    if values.size == 0:
        raise ValueError("times must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("times must be non-negative")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return values.copy()


@dataclass(frozen=True)
class TrajectoryParameters:
    """Parameters shared by the five version-2 motion models.

    Event times are measured from the first entry passed to
    :func:`generate_trajectory`, irrespective of the absolute value of that
    entry.  ``initial_position`` and ``initial_velocity`` therefore describe
    the state at the first returned sample.

    Attributes:
        initial_position: Initial world position with shape ``(2,)``.
        initial_velocity: Initial world velocity with shape ``(2,)``.
        acceleration: Constant acceleration used by ``constant_acceleration``.
        turn_rate: Angular velocity in radians per second for ``constant_turn``.
        stop_start_time: Time at which the fully stopped interval begins.
        stop_duration: Duration of the fully stopped interval in seconds.
        piecewise_turn_time: Time at which a piecewise direction change begins.
        piecewise_turn_angle: Signed total direction change in radians.
        transition_duration: Duration of smooth stop/start and turn transitions.
    """

    initial_position: np.ndarray
    initial_velocity: np.ndarray
    acceleration: np.ndarray
    turn_rate: float
    stop_start_time: float
    stop_duration: float
    piecewise_turn_time: float
    piecewise_turn_angle: float
    transition_duration: float

    def __post_init__(self) -> None:
        """Validate values and defensively copy vector parameters."""

        object.__setattr__(
            self,
            "initial_position",
            _finite_vector(self.initial_position, "initial_position"),
        )
        object.__setattr__(
            self,
            "initial_velocity",
            _finite_vector(self.initial_velocity, "initial_velocity"),
        )
        object.__setattr__(
            self,
            "acceleration",
            _finite_vector(self.acceleration, "acceleration"),
        )
        scalar_names = (
            "turn_rate",
            "stop_start_time",
            "stop_duration",
            "piecewise_turn_time",
            "piecewise_turn_angle",
            "transition_duration",
        )
        for name in scalar_names:
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))

        if self.stop_start_time < 0.0:
            raise ValueError("stop_start_time must be non-negative")
        if self.stop_duration < 0.0:
            raise ValueError("stop_duration must be non-negative")
        if self.piecewise_turn_time < 0.0:
            raise ValueError("piecewise_turn_time must be non-negative")
        if self.transition_duration < 0.0:
            raise ValueError("transition_duration must be non-negative")


@dataclass(frozen=True)
class TrajectoryResult:
    """Positions, velocities, and audit accelerations for one episode.

    Each field has shape ``(T, 2)``.  Accelerations are deliberately computed
    with a causal backward difference: the first row is zero because no prior
    sample exists, and row ``i`` uses only velocity rows ``i`` and ``i - 1``.
    """

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    def __post_init__(self) -> None:
        """Ensure all result arrays have matching finite ``(T, 2)`` shapes."""

        arrays: Dict[str, np.ndarray] = {}
        for name in ("positions", "velocities", "accelerations"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
                raise ValueError("{} must have shape (T, 2), got {}".format(name, array.shape))
            if not np.all(np.isfinite(array)):
                raise ValueError("{} must contain only finite values".format(name))
            arrays[name] = array.copy()

        expected_shape = arrays["positions"].shape
        if arrays["velocities"].shape != expected_shape:
            raise ValueError("velocities shape must match positions shape")
        if arrays["accelerations"].shape != expected_shape:
            raise ValueError("accelerations shape must match positions shape")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


def _smoothstep(progress: np.ndarray) -> np.ndarray:
    """Evaluate cubic smoothstep after clipping progress to ``[0, 1]``."""

    clipped = np.clip(progress, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _rotate_vectors(vector: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate a two-dimensional vector by each angle in ``angles``."""

    cosine = np.cos(angles)
    sine = np.sin(angles)
    velocities = np.empty((angles.size, 2), dtype=float)
    velocities[:, 0] = cosine * vector[0] - sine * vector[1]
    velocities[:, 1] = sine * vector[0] + cosine * vector[1]
    return velocities


def _integrate_positions(
    initial_position: np.ndarray,
    times: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Integrate sampled velocities with the trapezoidal rule."""

    positions = np.empty_like(velocities, dtype=float)
    positions[0] = initial_position
    if times.size == 1:
        return positions
    intervals = np.diff(times)[:, np.newaxis]
    increments = 0.5 * (velocities[:-1] + velocities[1:]) * intervals
    positions[1:] = initial_position + np.cumsum(increments, axis=0)
    return positions


def _backward_accelerations(times: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Compute causal accelerations from current and previous velocities."""

    accelerations = np.zeros_like(velocities, dtype=float)
    if times.size > 1:
        intervals = np.diff(times)[:, np.newaxis]
        accelerations[1:] = np.diff(velocities, axis=0) / intervals
    return accelerations


def _constant_velocity_values(
    elapsed: np.ndarray,
    parameters: TrajectoryParameters,
) -> np.ndarray:
    """Return constant-velocity samples for an episode."""

    return np.repeat(parameters.initial_velocity[np.newaxis, :], elapsed.size, axis=0)


def _constant_acceleration_values(
    elapsed: np.ndarray,
    parameters: TrajectoryParameters,
) -> np.ndarray:
    """Return velocity samples under a constant acceleration."""

    return (
        parameters.initial_velocity[np.newaxis, :]
        + elapsed[:, np.newaxis] * parameters.acceleration[np.newaxis, :]
    )


def _constant_turn_values(
    elapsed: np.ndarray,
    parameters: TrajectoryParameters,
) -> np.ndarray:
    """Return constant-speed velocity samples under a constant turn rate."""

    return _rotate_vectors(parameters.initial_velocity, parameters.turn_rate * elapsed)


def _stop_and_go_values(
    elapsed: np.ndarray,
    parameters: TrajectoryParameters,
) -> np.ndarray:
    """Return velocities for a smooth decelerate-stop-accelerate episode."""

    transition = parameters.transition_duration
    if transition <= 0.0:
        raise ValueError("stop_and_go requires transition_duration > 0")
    if parameters.stop_start_time < transition:
        raise ValueError(
            "stop_and_go requires stop_start_time >= transition_duration "
            "so initial_velocity is preserved at the first sample"
        )

    stop_start = parameters.stop_start_time
    stop_end = stop_start + parameters.stop_duration
    deceleration_start = stop_start - transition

    speed_scale = np.ones_like(elapsed, dtype=float)
    decelerating = (elapsed >= deceleration_start) & (elapsed < stop_start)
    deceleration_progress = (elapsed[decelerating] - deceleration_start) / transition
    speed_scale[decelerating] = 1.0 - _smoothstep(deceleration_progress)

    stopped = (elapsed >= stop_start) & (elapsed <= stop_end)
    speed_scale[stopped] = 0.0

    accelerating = (elapsed > stop_end) & (elapsed < stop_end + transition)
    acceleration_progress = (elapsed[accelerating] - stop_end) / transition
    speed_scale[accelerating] = _smoothstep(acceleration_progress)
    return speed_scale[:, np.newaxis] * parameters.initial_velocity[np.newaxis, :]


def _piecewise_direction_values(
    elapsed: np.ndarray,
    parameters: TrajectoryParameters,
) -> np.ndarray:
    """Return constant-speed velocities with one smooth direction change."""

    transition = parameters.transition_duration
    if transition <= 0.0:
        raise ValueError("piecewise_direction requires transition_duration > 0")
    progress = (elapsed - parameters.piecewise_turn_time) / transition
    angles = parameters.piecewise_turn_angle * _smoothstep(progress)
    return _rotate_vectors(parameters.initial_velocity, angles)


def generate_trajectory(
    trajectory_type: str,
    times: Sequence[float],
    parameters: TrajectoryParameters,
) -> TrajectoryResult:
    """Generate one deterministic two-dimensional motion episode.

    Args:
        trajectory_type: One of the keys in :data:`TRAJECTORY_TYPE_TO_CODE`.
        times: Non-empty, finite, non-negative, strictly increasing sample
            times.  Event parameters are interpreted relative to ``times[0]``.
        parameters: Validated model and event parameters.

    Returns:
        A :class:`TrajectoryResult` whose three arrays have shape ``(T, 2)``.
        Positions start exactly at ``parameters.initial_position`` and are
        obtained by trapezoidal integration of the returned velocities.

    Raises:
        TypeError: If ``parameters`` is not a :class:`TrajectoryParameters`.
        ValueError: If the trajectory type, times, or type-specific transition
            parameters are invalid.
    """

    if not isinstance(parameters, TrajectoryParameters):
        raise TypeError("parameters must be a TrajectoryParameters instance")
    if trajectory_type not in TRAJECTORY_TYPE_TO_CODE:
        choices = ", ".join(TRAJECTORY_TYPE_TO_CODE.keys())
        raise ValueError(
            "Unknown trajectory_type {!r}; expected one of: {}".format(
                trajectory_type, choices
            )
        )

    time_values = _validated_times(times)
    elapsed = time_values - time_values[0]

    if trajectory_type == "constant_velocity":
        velocities = _constant_velocity_values(elapsed, parameters)
    elif trajectory_type == "constant_acceleration":
        velocities = _constant_acceleration_values(elapsed, parameters)
    elif trajectory_type == "constant_turn":
        velocities = _constant_turn_values(elapsed, parameters)
    elif trajectory_type == "stop_and_go":
        velocities = _stop_and_go_values(elapsed, parameters)
    else:
        velocities = _piecewise_direction_values(elapsed, parameters)

    positions = _integrate_positions(
        parameters.initial_position,
        time_values,
        velocities,
    )
    accelerations = _backward_accelerations(time_values, velocities)
    return TrajectoryResult(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
    )


__all__ = [
    "TRAJECTORY_TYPE_TO_CODE",
    "TrajectoryParameters",
    "TrajectoryResult",
    "generate_trajectory",
]
