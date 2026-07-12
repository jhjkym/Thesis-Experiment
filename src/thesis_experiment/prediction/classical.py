"""Leakage-safe classical trajectory-prediction baselines.

The public predictors in this module accept only the inner ``inputs`` mapping
returned by :class:`thesis_experiment.data.prediction_dataset.PredictionDataset`.
The exact five-field whitelist is enforced at runtime so that supervision,
audit truth, trajectory type, and future motion parameters cannot accidentally
enter a predictor.

All predictors accept either one sample (history arrays without a leading
batch dimension) or a batch and always return ``(N, future_steps, 2)``.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Mapping, Optional, Tuple

import numpy as np

from thesis_experiment.data.prediction_dataset import MODEL_INPUT_FIELDS


PredictionInputs = Mapping[str, object]


def _as_binary_mask(values: object, name: str, batched: bool) -> np.ndarray:
    """Return a two-dimensional boolean mask with validated binary values."""

    mask = np.asarray(values)
    expected_ndim = 2 if batched else 1
    if mask.ndim != expected_ndim:
        raise ValueError(
            "{} must have {} dimension(s) for this input".format(
                name, expected_ndim
            )
        )
    if not bool(np.all((mask == 0) | (mask == 1))):
        raise ValueError("{} must contain only 0 and 1".format(name))
    if not batched:
        mask = mask[np.newaxis, :]
    return mask.astype(bool, copy=False)


def _validated_batched_inputs(
    inputs: PredictionInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the strict loader boundary and add a batch dimension if needed.

    Requiring the exact whitelist, instead of accepting arbitrary mappings,
    makes accidental use of a target or audit field a hard error.
    """

    if not isinstance(inputs, MappingABC):
        raise TypeError(
            "inputs must be the sample['inputs'] mapping from PredictionDataset"
        )
    expected = set(MODEL_INPUT_FIELDS)
    actual = set(inputs.keys())
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append("missing={}".format(missing))
        if extra:
            details.append("forbidden_or_unknown={}".format(extra))
        raise ValueError(
            "prediction inputs must contain exactly the strict five-field "
            "whitelist ({})".format(", ".join(details))
        )

    position = np.asarray(inputs["history_position"], dtype=np.float64)
    velocity = np.asarray(inputs["history_velocity"], dtype=np.float64)
    if position.ndim not in (2, 3) or position.shape[-1] != 2:
        raise ValueError("history_position must have shape (H, 2) or (N, H, 2)")
    batched = position.ndim == 3
    expected_velocity_ndim = 3 if batched else 2
    if velocity.ndim != expected_velocity_ndim or velocity.shape[-1] != 2:
        raise ValueError("history_velocity must match history_position shape")
    if velocity.shape != position.shape:
        raise ValueError("history_velocity must match history_position shape")
    if not bool(np.all(np.isfinite(position))):
        raise ValueError(
            "history_position contains non-finite values; use PredictionDataset "
            "to fill missing observations"
        )
    if not bool(np.all(np.isfinite(velocity))):
        raise ValueError(
            "history_velocity contains non-finite values; use PredictionDataset "
            "to fill missing observations"
        )

    position_mask = _as_binary_mask(
        inputs["history_mask"], "history_mask", batched
    )
    velocity_mask = _as_binary_mask(
        inputs["history_velocity_mask"], "history_velocity_mask", batched
    )
    if not batched:
        position = position[np.newaxis, :, :]
        velocity = velocity[np.newaxis, :, :]
    expected_mask_shape = position.shape[:2]
    if position_mask.shape != expected_mask_shape:
        raise ValueError("history_mask must match the history dimensions")
    if velocity_mask.shape != expected_mask_shape:
        raise ValueError("history_velocity_mask must match the history dimensions")

    time_step = np.asarray(inputs["time_step_seconds"], dtype=np.float64)
    batch_size = position.shape[0]
    if batched:
        if time_step.ndim != 1 or time_step.shape[0] != batch_size:
            raise ValueError(
                "batched time_step_seconds must have shape (N,)"
            )
    else:
        if time_step.ndim != 0:
            raise ValueError("single-sample time_step_seconds must be a scalar")
        time_step = time_step.reshape(1)
    if not bool(np.all(np.isfinite(time_step))) or not bool(
        np.all(time_step > 0.0)
    ):
        raise ValueError("time_step_seconds must contain finite positive values")

    return position, velocity, position_mask, velocity_mask, time_step


def _validate_future_steps(future_steps: int) -> int:
    """Return a validated positive forecast length."""

    if isinstance(future_steps, bool) or not isinstance(
        future_steps, (int, np.integer)
    ):
        raise TypeError("future_steps must be an integer")
    value = int(future_steps)
    if value <= 0:
        raise ValueError("future_steps must be positive")
    return value


def constant_position(
    inputs: PredictionInputs, future_steps: int = 20
) -> np.ndarray:
    """Repeat the last valid historical position over the forecast horizon.

    Missing entries are selected using ``history_mask``; their finite fill
    values are never interpreted as observations.  A sample without a valid
    historical position raises ``ValueError``.
    """

    steps = _validate_future_steps(future_steps)
    position, _, position_mask, _, _ = _validated_batched_inputs(inputs)
    prediction = np.empty((position.shape[0], steps, 2), dtype=np.float64)
    for sample_index in range(position.shape[0]):
        valid = np.flatnonzero(position_mask[sample_index])
        if valid.size == 0:
            raise ValueError(
                "constant-position baseline requires at least one valid "
                "historical position"
            )
        prediction[sample_index, :, :] = position[sample_index, valid[-1], :]
    return prediction


def constant_velocity(
    inputs: PredictionInputs, future_steps: int = 20
) -> np.ndarray:
    """Extrapolate the last valid historical position at constant velocity.

    The last velocity with ``history_velocity_mask == 1`` is preferred.  If
    none exists, velocity is estimated from the last two valid historical
    positions and their actual time separation.  Only historical whitelist
    fields are available to this function.

    When the final observation precedes the end of the history window, its
    elapsed missing-history interval is included before forecasting the first
    future time step.
    """

    steps = _validate_future_steps(future_steps)
    position, velocity, position_mask, velocity_mask, time_step = (
        _validated_batched_inputs(inputs)
    )
    batch_size, history_steps, _ = position.shape
    prediction = np.empty((batch_size, steps, 2), dtype=np.float64)

    for sample_index in range(batch_size):
        valid_positions = np.flatnonzero(position_mask[sample_index])
        if valid_positions.size == 0:
            raise ValueError(
                "constant-velocity baseline requires at least one valid "
                "historical position"
            )
        last_position_index = int(valid_positions[-1])
        last_position = position[sample_index, last_position_index]

        valid_velocities = np.flatnonzero(velocity_mask[sample_index])
        if valid_velocities.size:
            velocity_estimate = velocity[sample_index, valid_velocities[-1]]
        elif valid_positions.size >= 2:
            previous_index = int(valid_positions[-2])
            elapsed_steps = last_position_index - previous_index
            velocity_estimate = (
                position[sample_index, last_position_index]
                - position[sample_index, previous_index]
            ) / (elapsed_steps * time_step[sample_index])
        else:
            raise ValueError(
                "constant-velocity baseline requires a valid velocity or two "
                "valid historical positions"
            )

        missing_tail_steps = history_steps - 1 - last_position_index
        elapsed_future_steps = missing_tail_steps + np.arange(
            1, steps + 1, dtype=np.float64
        )
        prediction[sample_index] = last_position + (
            elapsed_future_steps[:, np.newaxis]
            * time_step[sample_index]
            * velocity_estimate[np.newaxis, :]
        )
    return prediction


class CVKalmanFilter:
    """Two-dimensional constant-velocity Kalman-filter predictor.

    The state is ``[px, py, vx, vy]``.  Historical time steps always execute
    the process prediction; a position measurement update is executed only
    when ``history_mask`` is true.  Filled values at missing steps are ignored.

    Args:
        process_noise: Non-negative continuous white-acceleration variance
            used to construct the discrete process covariance.
        observation_noise: Positive isotropic position-observation variance.
        initial_covariance: Positive isotropic diagonal state covariance at the
            first valid historical observation.  Used for both position and
            velocity states unless the component-specific variances below are
            supplied.
        initial_position_variance: Optional positive initial variance for the
            two position states.  Defaults to ``initial_covariance``.
        initial_velocity_variance: Optional positive initial variance for the
            two velocity states.  Defaults to ``initial_covariance``.
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        observation_noise: float = 0.1,
        initial_covariance: float = 1.0,
        initial_position_variance: Optional[float] = None,
        initial_velocity_variance: Optional[float] = None,
    ) -> None:
        self.process_noise = self._finite_scalar(
            process_noise, "process_noise", allow_zero=True
        )
        self.observation_noise = self._finite_scalar(
            observation_noise, "observation_noise", allow_zero=False
        )
        self.initial_covariance = self._finite_scalar(
            initial_covariance, "initial_covariance", allow_zero=False
        )
        self.initial_position_variance = self._finite_scalar(
            initial_covariance
            if initial_position_variance is None
            else initial_position_variance,
            "initial_position_variance",
            allow_zero=False,
        )
        self.initial_velocity_variance = self._finite_scalar(
            initial_covariance
            if initial_velocity_variance is None
            else initial_velocity_variance,
            "initial_velocity_variance",
            allow_zero=False,
        )

    @staticmethod
    def _finite_scalar(value: float, name: str, allow_zero: bool) -> float:
        """Validate a scalar covariance parameter."""

        result = float(value)
        if not np.isfinite(result):
            raise ValueError("{} must be finite".format(name))
        if result < 0.0 or (result == 0.0 and not allow_zero):
            comparator = "non-negative" if allow_zero else "positive"
            raise ValueError("{} must be {}".format(name, comparator))
        return result

    def _system_matrices(
        self, time_step_seconds: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return state transition and process covariance matrices."""

        dt = float(time_step_seconds)
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process_covariance = self.process_noise * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        return transition, process_covariance

    def _predict_one(
        self,
        position: np.ndarray,
        position_mask: np.ndarray,
        time_step_seconds: float,
        future_steps: int,
    ) -> np.ndarray:
        """Filter one history and return its future position predictions."""

        first_valid_candidates = np.flatnonzero(position_mask)
        if first_valid_candidates.size == 0:
            raise ValueError(
                "CV Kalman filter requires at least one valid historical position"
            )
        first_valid = int(first_valid_candidates[0])
        state = np.array(
            [position[first_valid, 0], position[first_valid, 1], 0.0, 0.0],
            dtype=np.float64,
        )
        covariance = np.diag(
            np.array(
                [
                    self.initial_position_variance,
                    self.initial_position_variance,
                    self.initial_velocity_variance,
                    self.initial_velocity_variance,
                ],
                dtype=np.float64,
            )
        )
        transition, process_covariance = self._system_matrices(
            time_step_seconds
        )
        observation_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        observation_covariance = (
            np.eye(2, dtype=np.float64) * self.observation_noise
        )
        identity = np.eye(4, dtype=np.float64)

        for history_index in range(first_valid + 1, position.shape[0]):
            state = np.dot(transition, state)
            covariance = (
                np.dot(np.dot(transition, covariance), transition.T)
                + process_covariance
            )
            if position_mask[history_index]:
                innovation = position[history_index] - np.dot(
                    observation_matrix, state
                )
                innovation_covariance = (
                    np.dot(
                        np.dot(observation_matrix, covariance),
                        observation_matrix.T,
                    )
                    + observation_covariance
                )
                kalman_gain = np.linalg.solve(
                    innovation_covariance,
                    np.dot(observation_matrix, covariance),
                ).T
                state = state + np.dot(kalman_gain, innovation)
                residual_transform = identity - np.dot(
                    kalman_gain, observation_matrix
                )
                # Joseph form preserves covariance symmetry and positivity.
                covariance = (
                    np.dot(
                        np.dot(residual_transform, covariance),
                        residual_transform.T,
                    )
                    + np.dot(
                        np.dot(kalman_gain, observation_covariance),
                        kalman_gain.T,
                    )
                )

        prediction = np.empty((future_steps, 2), dtype=np.float64)
        for future_index in range(future_steps):
            state = np.dot(transition, state)
            covariance = (
                np.dot(np.dot(transition, covariance), transition.T)
                + process_covariance
            )
            prediction[future_index] = state[:2]
        return prediction

    def predict(
        self, inputs: PredictionInputs, future_steps: int = 20
    ) -> np.ndarray:
        """Return ``(N, future_steps, 2)`` forecasts from historical inputs."""

        steps = _validate_future_steps(future_steps)
        position, _, position_mask, _, time_step = _validated_batched_inputs(
            inputs
        )
        prediction = np.empty(
            (position.shape[0], steps, 2), dtype=np.float64
        )
        for sample_index in range(position.shape[0]):
            prediction[sample_index] = self._predict_one(
                position[sample_index],
                position_mask[sample_index],
                time_step[sample_index],
                steps,
            )
        if not bool(np.all(np.isfinite(prediction))):
            raise FloatingPointError("CV Kalman filter produced non-finite output")
        return prediction


def cv_kalman_filter(
    inputs: PredictionInputs,
    future_steps: int = 20,
    process_noise: float = 0.1,
    observation_noise: float = 0.1,
    initial_covariance: float = 1.0,
    initial_position_variance: Optional[float] = None,
    initial_velocity_variance: Optional[float] = None,
) -> np.ndarray:
    """Convenience functional interface for :class:`CVKalmanFilter`."""

    return CVKalmanFilter(
        process_noise=process_noise,
        observation_noise=observation_noise,
        initial_covariance=initial_covariance,
        initial_position_variance=initial_position_variance,
        initial_velocity_variance=initial_velocity_variance,
    ).predict(inputs, future_steps=future_steps)


__all__ = [
    "CVKalmanFilter",
    "PredictionInputs",
    "constant_position",
    "constant_velocity",
    "cv_kalman_filter",
]
