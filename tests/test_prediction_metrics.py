"""Tests for deterministic trajectory-prediction metrics and records."""

from __future__ import annotations

import numpy as np
import pytest

from thesis_experiment.evaluation.prediction_metrics import (
    ade_per_window,
    compute_prediction_metrics,
    displacement_errors,
    fde_per_window,
    per_horizon_metrics,
    prediction_metric_records,
    seed_mean_and_sample_std,
)


def _metric_example() -> tuple:
    target = np.zeros((2, 3, 2), dtype=np.float64)
    prediction = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]],
        ],
        dtype=np.float64,
    )
    return prediction, target


def test_seed_standard_deviation_uses_bessel_correction() -> None:
    """Multiple independent seeds use sample std while one seed reports zero."""

    single_mean, single_std = seed_mean_and_sample_std([4.5])
    multiple_mean, multiple_std = seed_mean_and_sample_std([1.0, 2.0, 4.0])

    assert single_mean == 4.5
    assert single_std == 0.0
    assert multiple_mean == pytest.approx(7.0 / 3.0)
    assert multiple_std == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))


@pytest.mark.parametrize("values", [[], [1.0, np.nan], [[1.0, 2.0]]])
def test_seed_standard_deviation_rejects_invalid_values(values) -> None:
    """Seed dispersion never silently accepts empty, nonfinite, or 2D input."""

    with pytest.raises(ValueError):
        seed_mean_and_sample_std(values)


def test_ade_and_fde_are_computed_per_window() -> None:
    """ADE averages horizons while FDE uses only the final horizon."""

    prediction, target = _metric_example()

    np.testing.assert_allclose(ade_per_window(prediction, target), [1.0, 7.0 / 3.0])
    np.testing.assert_allclose(fde_per_window(prediction, target), [2.0, 4.0])


def test_displacement_and_per_horizon_metrics_are_correct() -> None:
    """Horizon means and Euclidean RMSE must follow their documented definitions."""

    prediction, target = _metric_example()
    np.testing.assert_allclose(
        displacement_errors(prediction, target), [[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]]
    )

    metrics = per_horizon_metrics(prediction, target)
    np.testing.assert_array_equal(metrics["horizon_step"], [1, 2, 3])
    np.testing.assert_allclose(metrics["mean_euclidean_distance"], [0.0, 2.0, 3.0])
    np.testing.assert_allclose(metrics["rmse"], [0.0, np.sqrt(5.0), np.sqrt(10.0)])


def test_combined_prediction_metrics_have_expected_shapes() -> None:
    """The combined helper must retain per-window and per-horizon dimensions."""

    prediction, target = _metric_example()
    result = compute_prediction_metrics(prediction, target)

    assert result["displacement_error"].shape == (2, 3)
    assert result["ade"].shape == (2,)
    assert result["fde"].shape == (2,)
    assert result["per_horizon_rmse"].shape == (3,)


@pytest.mark.parametrize(
    "prediction,target,message",
    [
        (np.zeros((2, 3)), np.zeros((2, 3)), "shape"),
        (np.zeros((2, 3, 2)), np.zeros((1, 3, 2)), "does not match"),
        (np.zeros((0, 3, 2)), np.zeros((0, 3, 2)), "at least one"),
        (
            np.array([[[np.nan, 0.0]]]),
            np.zeros((1, 1, 2)),
            "finite",
        ),
        (
            np.array([[['bad', 'value']]]),
            np.zeros((1, 1, 2)),
            "numeric",
        ),
    ],
)
def test_metrics_reject_invalid_trajectory_arrays(
    prediction: np.ndarray, target: np.ndarray, message: str
) -> None:
    """Bad shapes, nonnumeric values, and nonfinite values fail explicitly."""

    with pytest.raises(ValueError, match=message):
        displacement_errors(prediction, target)


def test_records_use_equal_weight_episode_and_scene_aggregation() -> None:
    """CSV records distinguish window means from equal-weight formal units."""

    values = np.array([0.0, 2.0, 10.0, 20.0, 20.0, 20.0, 20.0])
    result = prediction_metric_records(
        {"ade": values, "fde": values + 1.0},
        scene_id=[1, 1, 1, 2, 2, 2, 2],
        episode_id=[10, 10, 11, 20, 20, 20, 20],
        trajectory_type=["cv", "cv", "turn", "stop", "stop", "stop", "stop"],
        occlusion_length_bin=[0, 0, 1, 0, 1, 1, 1],
        sample_start_index=[0, 5, 0, 0, 5, 10, 15],
        model_name="constant_velocity",
    )

    ade_summary = {
        row["aggregation_level"]: row
        for row in result["summary"]
        if row["metric"] == "ade"
    }
    assert ade_summary["window"]["mean"] == pytest.approx(92.0 / 7.0)
    assert ade_summary["episode"]["mean"] == pytest.approx(31.0 / 3.0)
    assert ade_summary["scene"]["mean"] == pytest.approx(12.0)
    assert ade_summary["episode"]["unit_count"] == 3
    assert ade_summary["scene"]["unit_count"] == 2

    episode_rows = [row for row in result["episode"] if row["metric"] == "ade"]
    assert [row["episode_id"] for row in episode_rows] == [10, 11, 20]
    assert episode_rows[0]["mean"] == pytest.approx(1.0)
    assert result["window"][1]["sample_start_index"] == 5
    assert result["window"][0]["model_name"] == "constant_velocity"


def test_records_report_post_inference_groups() -> None:
    """Motion and occlusion metadata produce explicit reporting groups."""

    result = prediction_metric_records(
        {"ade": [1.0, 3.0, 5.0, 7.0]},
        scene_id=[1, 1, 2, 2],
        episode_id=[10, 10, 20, 20],
        trajectory_type=["cv", "cv", "turn", "turn"],
        occlusion_length_bin=[0, 1, 1, 2],
    )

    assert result["trajectory_type"] == [
        {"metric": "ade", "trajectory_type": "cv", "window_count": 2, "window_mean": 2.0},
        {"metric": "ade", "trajectory_type": "turn", "window_count": 2, "window_mean": 6.0},
    ]
    assert result["occlusion_length_bin"][1]["window_mean"] == pytest.approx(4.0)


def test_records_reject_inconsistent_episode_mapping() -> None:
    """Record generation retains the underlying episode integrity check."""

    with pytest.raises(ValueError, match="maps inconsistently"):
        prediction_metric_records(
            {"ade": [1.0, 2.0]},
            scene_id=[1, 2],
            episode_id=[10, 10],
            trajectory_type=["cv", "cv"],
            occlusion_length_bin=[0, 0],
        )


def test_records_reject_mismatched_metric_and_metadata_lengths() -> None:
    """Every metric and grouping label must describe the same windows."""

    with pytest.raises(ValueError, match="same length"):
        prediction_metric_records(
            {"ade": [1.0, 2.0], "fde": [3.0]},
            scene_id=[1, 1],
            episode_id=[10, 10],
            trajectory_type=["cv", "cv"],
            occlusion_length_bin=[0, 0],
        )
