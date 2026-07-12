"""Regression tests for experiment-02 multi-seed evaluation artefacts."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("torch")
SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import evaluate_prediction_baselines as evaluation_script
from experiment_02_figures import metric_comparison_values


def _metadata(window_count: int, future_steps: int) -> dict:
    return {
        "future_position": np.zeros((window_count, future_steps, 2), dtype=np.float64),
        "scene_id": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "episode_id": np.asarray([10, 10, 20, 20], dtype=np.int64),
        "sample_start_index": np.asarray([0, 5, 0, 5], dtype=np.int64),
        "trajectory_type": np.asarray([0, 0, 1, 1], dtype=np.int8),
        "occlusion_length_bin": np.asarray([0, 1, 0, 1], dtype=np.int8),
    }


def _read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_evaluation_loader_applies_nondefault_fill_value(tmp_path: Path) -> None:
    """Evaluation passes its configured fill value into PredictionDataset."""

    archive_path = tmp_path / "custom_test.npz"
    np.savez_compressed(
        str(archive_path),
        history_position=np.asarray([[[1.0, 2.0], [np.nan, np.nan]]]),
        history_velocity=np.asarray([[[np.nan, np.nan], [np.nan, np.nan]]]),
        history_mask=np.asarray([[1, 0]], dtype=np.uint8),
        future_position=np.zeros((1, 2, 2), dtype=np.float64),
        time_step_seconds=np.asarray([0.1]),
        scene_id=np.asarray([1]),
        episode_id=np.asarray([10]),
        sample_start_index=np.asarray([0]),
    )
    config = SimpleNamespace(
        fill_value=-23.5,
        dataset=SimpleNamespace(test_split=archive_path.name),
    )
    dataset, loaded_path = evaluation_script._load_evaluation_dataset(
        config, tmp_path
    )
    inputs = dataset.get_batch(slice(None))["inputs"]

    assert loaded_path == archive_path
    assert dataset.fill_value == -23.5
    assert np.all(inputs["history_position"][0, 1] == -23.5)
    assert np.all(inputs["history_velocity"][0] == -23.5)


def test_multiseed_tables_use_real_gru_replicates_only(tmp_path: Path) -> None:
    """GRU rows retain seed identity while a classical result appears once."""

    metadata = _metadata(window_count=4, future_steps=3)
    predictions = {
        ("constant_position", None): np.zeros((4, 3, 2), dtype=np.float64),
        ("deterministic_gru", 101): np.ones((4, 3, 2), dtype=np.float64),
        ("deterministic_gru", 202): np.full((4, 3, 2), 2.0, dtype=np.float64),
    }
    result = evaluation_script._compute_and_write_metrics(
        predictions, metadata, tmp_path, logging.getLogger("multiseed-test")
    )

    required = (
        "per_window_metrics_by_seed.csv",
        "per_episode_metrics_by_seed.csv",
        "per_scene_metrics_by_seed.csv",
        "per_horizon_metrics_by_seed.csv",
        "summary_metrics_by_seed.csv",
        "summary_metrics_mean_std.csv",
    )
    assert all((tmp_path / name).is_file() for name in required)

    window_rows = _read(tmp_path / "per_window_metrics_by_seed.csv")
    classical = [row for row in window_rows if row["model_name"] == "constant_position"]
    gru = [row for row in window_rows if row["model_name"] == "deterministic_gru"]
    assert len(classical) == 4
    assert {row["seed"] for row in classical} == {""}
    assert len(gru) == 8
    assert {row["seed"] for row in gru} == {"101", "202"}

    summary = result["mean_std_rows"]
    classic_ade = next(
        row for row in summary
        if row["model_name"] == "constant_position"
        and row["metric"] == "ade"
        and row["aggregation_level"] == "episode"
    )
    gru_ade = next(
        row for row in summary
        if row["model_name"] == "deterministic_gru"
        and row["metric"] == "ade"
        and row["aggregation_level"] == "episode"
    )
    assert classic_ade["seed_count"] == 1
    assert classic_ade["std"] == 0.0
    assert gru_ade["seed_count"] == 2
    np.testing.assert_allclose(gru_ade["mean"], 1.5 * np.sqrt(2.0))
    np.testing.assert_allclose(gru_ade["std"], 1.0)


def test_saved_multiseed_predictions_keep_seed_and_source_ids(tmp_path: Path) -> None:
    metadata = _metadata(window_count=4, future_steps=2)
    predictions = {
        ("constant_velocity", None): np.zeros((4, 2, 2)),
        ("deterministic_gru", 7): np.ones((4, 2, 2)),
        ("deterministic_gru", 8): np.full((4, 2, 2), 2.0),
    }
    evaluation_script._save_predictions(predictions, metadata, tmp_path)
    with np.load(str(tmp_path / "predictions.npz"), allow_pickle=False) as archive:
        assert archive["model_name"].tolist() == [
            "constant_velocity", "deterministic_gru", "deterministic_gru"
        ]
        assert archive["seed"].tolist() == [-1, 7, 8]
        assert archive["prediction"].shape == (3, 4, 2, 2)
        np.testing.assert_array_equal(archive["scene_id"], metadata["scene_id"])
        np.testing.assert_array_equal(archive["episode_id"], metadata["episode_id"])
        np.testing.assert_array_equal(
            archive["sample_start_index"], metadata["sample_start_index"]
        )


def test_default_statistical_unit_changes_primary_plot_values() -> None:
    rows = []
    for index, model_name in enumerate(
        ("constant_position", "constant_velocity", "cv_kalman_filter", "deterministic_gru")
    ):
        rows.extend(
            [
                {
                    "model_name": model_name,
                    "metric": "ade",
                    "aggregation_level": "episode",
                    "mean": str(index + 1.0),
                    "std": "0.1",
                },
                {
                    "model_name": model_name,
                    "metric": "ade",
                    "aggregation_level": "scene",
                    "mean": str(index + 11.0),
                    "std": "0.2",
                },
            ]
        )

    episode_mean, episode_std = metric_comparison_values(rows, "ade", "episode")
    scene_mean, scene_std = metric_comparison_values(rows, "ade", "scene")
    np.testing.assert_array_equal(episode_mean, np.arange(1.0, 5.0))
    np.testing.assert_array_equal(scene_mean, np.arange(11.0, 15.0))
    np.testing.assert_allclose(episode_std, 0.1)
    np.testing.assert_allclose(scene_std, 0.2)
    with pytest.raises(ValueError, match="episode or scene"):
        metric_comparison_values(rows, "ade", "window")
