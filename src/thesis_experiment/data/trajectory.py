"""Target-trajectory interfaces and a constant-velocity implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_vector(value: Sequence[float], name: str) -> np.ndarray:
    """Convert *value* to a finite two-dimensional vector."""

    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ValueError("{} must have shape (2,), got {}".format(name, vector.shape))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(name))
    return vector.copy()


def _as_times(times: Sequence[float]) -> np.ndarray:
    """Convert times to a finite, non-negative floating-point array."""

    values = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("times must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("times must be non-negative")
    return values


class Trajectory(ABC):
    """Abstract interface for a two-dimensional time-parameterized trajectory."""

    @abstractmethod
    def position_at(self, times: Sequence[float]) -> np.ndarray:
        """Return world positions at the requested times.

        The returned shape is ``times.shape + (2,)``.  Implementations may use
        any motion model, allowing later acceleration, turning, or stop models
        to share the same interface.
        """

        raise NotImplementedError

    def sample(self, times: Sequence[float]) -> np.ndarray:
        """Return positions at ``times`` as an alias for :meth:`position_at`."""

        return self.position_at(times)


@dataclass(frozen=True)
class ConstantVelocityTrajectory(Trajectory):
    """Two-dimensional trajectory following ``p(t) = p0 + v * t``."""

    initial_position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        """Validate and defensively copy the initial state."""

        object.__setattr__(
            self,
            "initial_position",
            _as_vector(self.initial_position, "initial_position"),
        )
        object.__setattr__(self, "velocity", _as_vector(self.velocity, "velocity"))

    def position_at(self, times: Sequence[float]) -> np.ndarray:
        """Evaluate ``p(t) = p0 + v * t`` at one or more non-negative times."""

        time_values = _as_times(times)
        return self.initial_position + time_values[..., np.newaxis] * self.velocity


def sample_times(sample_rate_hz: float, duration_seconds: float) -> np.ndarray:
    """Create sample times containing both zero and the exact endpoint.

    Samples are spaced by ``1 / sample_rate_hz``.  If the duration is not an
    integer multiple of that interval, a final shorter interval is appended so
    ``duration_seconds`` is always included.

    Args:
        sample_rate_hz: Positive sampling frequency in hertz.
        duration_seconds: Finite, non-negative duration in seconds.

    Returns:
        One-dimensional sample times beginning at zero and ending exactly at
        ``duration_seconds``.
    """

    rate = float(sample_rate_hz)
    duration = float(duration_seconds)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_seconds must be finite and non-negative")
    if duration == 0.0:
        return np.array([0.0], dtype=float)

    interval_count = int(np.floor(duration * rate))
    times = np.arange(interval_count + 1, dtype=float) / rate
    tolerance = 1e-12 * max(1.0, duration)
    if abs(float(times[-1]) - duration) <= tolerance:
        times[-1] = duration
    elif times[-1] < duration:
        times = np.concatenate((times, np.array([duration], dtype=float)))
    else:
        times = times[times < duration]
        times = np.concatenate((times, np.array([duration], dtype=float)))
    return times


def generate_constant_velocity_trajectory(
    initial_position: Sequence[float],
    velocity: Sequence[float],
    times: Sequence[float]
) -> np.ndarray:
    """Convenience function returning constant-velocity positions at ``times``."""

    trajectory = ConstantVelocityTrajectory(
        initial_position=np.asarray(initial_position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
    )
    return trajectory.position_at(times)


__all__ = [
    "ConstantVelocityTrajectory",
    "Trajectory",
    "generate_constant_velocity_trajectory",
    "sample_times",
]
