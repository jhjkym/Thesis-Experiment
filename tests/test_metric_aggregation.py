"""Tests for model-independent per-window metric aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from thesis_experiment.evaluation import aggregate_window_metrics


def test_aggregation_uses_equal_weight_episodes_and_scenes() -> None:
    """Unequal window counts must produce distinct, correct aggregation levels."""

    metrics = np.array([0.0, 2.0, 10.0, 20.0, 20.0, 20.0, 20.0])
    scene_ids = np.array([1, 1, 1, 2, 2, 2, 2])
    episode_ids = np.array([10, 10, 11, 20, 20, 20, 20])
    trajectory_types = np.array(["cv", "cv", "turn", "stop", "stop", "stop", "stop"])
    occlusion_groups = np.array(["short", "short", "long", "short", "long", "long", "long"])

    result = aggregate_window_metrics(
        metrics,
        scene_ids,
        episode_ids,
        trajectory_types,
        occlusion_groups,
    )

    assert result["window_count"] == 7
    assert result["episode_count"] == 3
    assert result["scene_count"] == 2
    assert result["window_mean"] == pytest.approx(92.0 / 7.0)
    assert result["episode_mean"] == pytest.approx((1.0 + 10.0 + 20.0) / 3.0)
    assert result["scene_mean"] == pytest.approx((4.0 + 20.0) / 2.0)

    assert result["by_episode"][10] == {"count": 2, "window_mean": 1.0}
    assert result["by_episode"][11] == {"count": 1, "window_mean": 10.0}
    assert result["by_scene"][1] == {"count": 3, "window_mean": 4.0}


def test_aggregation_reports_trajectory_and_occlusion_groups() -> None:
    """Metadata group summaries must retain counts and direct window means."""

    result = aggregate_window_metrics(
        [1.0, 3.0, 5.0, 7.0],
        scene_id=[1, 1, 2, 2],
        episode_id=[10, 10, 20, 20],
        trajectory_type=["constant_velocity", "constant_velocity", "constant_turn", "constant_turn"],
        occlusion_group=["none", "short", "short", "long"],
    )

    assert result["by_trajectory_type"] == {
        "constant_velocity": {"count": 2, "window_mean": 2.0},
        "constant_turn": {"count": 2, "window_mean": 6.0},
    }
    assert result["by_occlusion_group"] == {
        "none": {"count": 1, "window_mean": 1.0},
        "short": {"count": 2, "window_mean": 4.0},
        "long": {"count": 1, "window_mean": 7.0},
    }


@pytest.mark.parametrize(
    "metric_values, scene_id, episode_id, trajectory_type, occlusion_group, message",
    [
        ([], [], [], [], [], "at least one"),
        ([[1.0]], [1], [1], [0], ["none"], "one-dimensional"),
        ([1.0, np.nan], [1, 1], [1, 1], [0, 0], ["none", "none"], "finite"),
        ([1.0], [1, 2], [1], [0], ["none"], "length"),
        ([1.0], [np.inf], [1], [0], ["none"], "finite"),
    ],
)
def test_aggregation_rejects_invalid_inputs(
    metric_values: object,
    scene_id: object,
    episode_id: object,
    trajectory_type: object,
    occlusion_group: object,
    message: str,
) -> None:
    """Bad shapes, lengths, and non-finite inputs must fail explicitly."""

    with pytest.raises(ValueError, match=message):
        aggregate_window_metrics(
            metric_values,
            scene_id,
            episode_id,
            trajectory_type,
            occlusion_group,
        )


@pytest.mark.parametrize(
    "scene_id, trajectory_type",
    [
        ([1, 2], ["constant_velocity", "constant_velocity"]),
        ([1, 1], ["constant_velocity", "constant_turn"]),
    ],
)
def test_aggregation_rejects_inconsistent_episode_mapping(
    scene_id: object, trajectory_type: object
) -> None:
    """One episode cannot belong to multiple scenes or motion types."""

    with pytest.raises(ValueError, match="maps inconsistently"):
        aggregate_window_metrics(
            [1.0, 2.0],
            scene_id=scene_id,
            episode_id=[42, 42],
            trajectory_type=trajectory_type,
            occlusion_group=["none", "short"],
        )
