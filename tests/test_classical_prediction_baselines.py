"""Tests for leakage-safe classical trajectory-prediction baselines."""

from __future__ import annotations

import inspect
from typing import Dict

import numpy as np
import pytest

from thesis_experiment.data.prediction_dataset import MODEL_INPUT_FIELDS
from thesis_experiment.prediction.classical import (
    CVKalmanFilter,
    constant_position,
    constant_velocity,
    cv_kalman_filter,
)


def _single_inputs() -> Dict[str, object]:
    """Return one finite, loader-shaped history input mapping."""

    return {
        "history_position": np.array(
            [[-1.5, 0.75], [-1.0, 0.5], [-0.5, 0.25], [0.0, 0.0]],
            dtype=np.float64,
        ),
        "history_velocity": np.array(
            [[0.0, 0.0], [1.0, -0.5], [1.0, -0.5], [2.0, -1.0]],
            dtype=np.float64,
        ),
        "history_mask": np.ones(4, dtype=np.uint8),
        "history_velocity_mask": np.array([0, 1, 1, 1], dtype=np.uint8),
        "time_step_seconds": 0.5,
    }


def _batch_inputs() -> Dict[str, object]:
    """Stack two strict single-sample mappings into a loader-shaped batch."""

    first = _single_inputs()
    second = _single_inputs()
    second["history_position"] = np.asarray(second["history_position"]) + np.array(
        [2.0, 3.0]
    )
    return {
        "history_position": np.stack(
            [first["history_position"], second["history_position"]]
        ),
        "history_velocity": np.stack(
            [first["history_velocity"], second["history_velocity"]]
        ),
        "history_mask": np.stack(
            [first["history_mask"], second["history_mask"]]
        ),
        "history_velocity_mask": np.stack(
            [first["history_velocity_mask"], second["history_velocity_mask"]]
        ),
        "time_step_seconds": np.array([0.5, 0.5], dtype=np.float64),
    }


def test_constant_position_single_output_shape_and_value() -> None:
    """The last valid position is repeated and a batch axis is always kept."""

    inputs = _single_inputs()
    inputs["history_position"][-1] = 99.0  # type: ignore[index]
    inputs["history_mask"][-1] = 0  # type: ignore[index]
    prediction = constant_position(inputs, future_steps=3)
    assert prediction.shape == (1, 3, 2)
    np.testing.assert_allclose(
        prediction[0], np.tile(np.array([-0.5, 0.25]), (3, 1))
    )


def test_constant_position_batch_output() -> None:
    """Batched inputs preserve sample order and output shape."""

    prediction = constant_position(_batch_inputs(), future_steps=2)
    assert prediction.shape == (2, 2, 2)
    np.testing.assert_allclose(prediction[0], 0.0)
    np.testing.assert_allclose(prediction[1], np.array([[2.0, 3.0]] * 2))


def test_constant_position_rejects_history_without_observation() -> None:
    """A fully invisible history is an explicit invalid baseline input."""

    inputs = _single_inputs()
    inputs["history_mask"] = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one valid"):
        constant_position(inputs)


def test_constant_velocity_uses_last_masked_historical_velocity() -> None:
    """The forecast comes from velocity history, not position-derived truth."""

    inputs = _single_inputs()
    # Earlier position differences deliberately disagree with the selected
    # final historical velocity [2, -1].
    inputs["history_position"] = np.array(
        [[-100.0, 50.0], [-20.0, 10.0], [-0.01, 0.01], [0.0, 0.0]]
    )
    prediction = constant_velocity(inputs, future_steps=3)
    expected = np.array([[1.0, -0.5], [2.0, -1.0], [3.0, -1.5]])
    np.testing.assert_allclose(prediction[0], expected)


def test_constant_velocity_ignores_velocity_at_masked_steps() -> None:
    """Finite fill values at invalid velocity steps never affect prediction."""

    inputs = _single_inputs()
    inputs["history_velocity_mask"] = np.array([0, 1, 0, 0], dtype=np.uint8)
    inputs["history_velocity"] = np.array(
        [[999.0, 999.0], [1.0, -0.5], [999.0, 999.0], [-999.0, -999.0]]
    )
    prediction = constant_velocity(inputs, future_steps=2)
    np.testing.assert_allclose(prediction[0], [[0.5, -0.25], [1.0, -0.5]])


def test_constant_velocity_falls_back_to_historical_positions() -> None:
    """Absent valid velocity entries use two observed historical positions."""

    inputs = _single_inputs()
    inputs["history_velocity_mask"] = np.zeros(4, dtype=np.uint8)
    inputs["history_mask"] = np.array([1, 0, 1, 0], dtype=np.uint8)
    inputs["history_position"] = np.array(
        [[-2.0, 0.0], [50.0, 50.0], [0.0, 0.0], [50.0, 50.0]]
    )
    # Two visible points are 2 * 0.5 s apart, hence v=[2,0].  The final
    # history step is missing, so first future target is two steps later.
    prediction = constant_velocity(inputs, future_steps=2)
    np.testing.assert_allclose(prediction[0], [[2.0, 0.0], [3.0, 0.0]])


@pytest.mark.parametrize(
    "predictor",
    [constant_position, constant_velocity, cv_kalman_filter],
)
def test_classical_predictors_enforce_exact_loader_whitelist(predictor) -> None:
    """Targets, audit truth, and arbitrary fields are rejected at the boundary."""

    assert set(_single_inputs()) == set(MODEL_INPUT_FIELDS)
    for forbidden_name in (
        "future_position",
        "history_true_position",
        "trajectory_type",
        "episode_turn_rate",
    ):
        inputs = _single_inputs()
        inputs[forbidden_name] = np.zeros((4, 2))
        with pytest.raises(ValueError, match="forbidden_or_unknown"):
            predictor(inputs, future_steps=2)


def test_classical_predictor_signatures_have_no_future_or_truth_argument() -> None:
    """The APIs cannot receive future labels or audit truth as parameters."""

    for callable_object in (
        constant_position,
        constant_velocity,
        CVKalmanFilter.predict,
        cv_kalman_filter,
    ):
        names = set(inspect.signature(callable_object).parameters)
        assert "future_position" not in names
        assert "history_true_position" not in names


def test_kalman_filter_missing_measurement_only_predicts() -> None:
    """Changing a filled missing position cannot alter the filter result."""

    first = _single_inputs()
    first["history_mask"] = np.array([1, 1, 0, 1], dtype=np.uint8)
    second = _single_inputs()
    second["history_mask"] = np.array([1, 1, 0, 1], dtype=np.uint8)
    first["history_position"][2] = [1.0e9, -1.0e9]  # type: ignore[index]
    second["history_position"][2] = [-4.0e8, 7.0e8]  # type: ignore[index]

    model = CVKalmanFilter(
        process_noise=0.02,
        observation_noise=0.1,
        initial_covariance=1.0,
    )
    np.testing.assert_allclose(
        model.predict(first, future_steps=4),
        model.predict(second, future_steps=4),
        rtol=0.0,
        atol=0.0,
    )


def test_kalman_filter_batch_shape_and_finite_output() -> None:
    """The two-dimensional CV filter produces finite batched trajectories."""

    prediction = CVKalmanFilter().predict(_batch_inputs(), future_steps=7)
    assert prediction.shape == (2, 7, 2)
    assert np.all(np.isfinite(prediction))


def test_kalman_filter_ignores_history_velocity_values() -> None:
    """The position-observation Kalman update never reads filled velocity data."""

    first = _single_inputs()
    second = _single_inputs()
    second["history_velocity"] = np.full((4, 2), 1.0e12)
    second["history_velocity_mask"] = np.ones(4, dtype=np.uint8)
    filter_model = CVKalmanFilter()
    np.testing.assert_allclose(
        filter_model.predict(first, future_steps=2),
        filter_model.predict(second, future_steps=2),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"process_noise": -1.0},
        {"observation_noise": 0.0},
        {"initial_covariance": 0.0},
        {"process_noise": np.nan},
    ],
)
def test_kalman_filter_rejects_invalid_noise_parameters(kwargs) -> None:
    """Noise and initial-covariance parameters fail explicitly when invalid."""

    with pytest.raises(ValueError):
        CVKalmanFilter(**kwargs)


def test_classical_predictors_reject_nonfinite_filled_inputs() -> None:
    """Model code detects bypassing PredictionDataset NaN filling."""

    inputs = _single_inputs()
    inputs["history_position"][0, 0] = np.nan  # type: ignore[index]
    for predictor in (constant_position, constant_velocity, cv_kalman_filter):
        with pytest.raises(ValueError, match="non-finite"):
            predictor(inputs, future_steps=2)

