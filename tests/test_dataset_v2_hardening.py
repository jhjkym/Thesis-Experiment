"""Focused tests for dataset-v2 training-readiness hardening.

These checks use either a hand-crafted observation mask or a small in-memory
dataset.  They never read or write the repository's generated ``outputs``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from thesis_experiment.config_v2 import (
    DatasetV2Config,
    load_dataset_v2_config,
    validate_dataset_v2_config,
)
from thesis_experiment.data.dataset import ObservationSequence
from thesis_experiment.data.dataset_v2 import (
    OCCLUSION_LENGTH_BIN_TO_CODE,
    GeneratedDatasetV2,
    create_episode_windows,
    generate_dataset_v2,
)
from thesis_experiment.data.trajectory_v2 import TRAJECTORY_TYPE_TO_CODE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "validation", "test")
MODEL_INPUT_FIELDS = [
    "history_position",
    "history_velocity",
    "history_mask",
    "history_velocity_mask",
    "time_step_seconds",
]
SUPERVISION_FIELDS = ["future_position"]
SAMPLE_METADATA_FIELDS = ["scene_id", "episode_id", "sample_start_index"]
EXPLICITLY_FORBIDDEN_INPUT_FIELDS = {
    "history_true_position",
    "trajectory_type",
    "episode_true_position_world",
    "episode_turn_rate",
    "episode_stop_start_time",
    "episode_stop_duration",
    "episode_piecewise_turn_time",
    "episode_piecewise_turn_angle",
    "episode_acceleration_parameter",
}


@pytest.fixture(scope="module")
def hardened_config() -> DatasetV2Config:
    """Return a fast config retaining all hardening constraints and modes."""

    base = load_dataset_v2_config(
        PROJECT_ROOT / "configs" / "dataset_v2_smoke.yaml"
    )
    return replace(
        base,
        scene_counts={name: 1 for name in SPLIT_NAMES},
        episodes_per_scene=5,
        trees=replace(base.trees, count=4),
    )


@pytest.fixture(scope="module")
def hardened_dataset(hardened_config: DatasetV2Config) -> GeneratedDatasetV2:
    """Generate one deterministic, observable episode of each type per split."""

    return generate_dataset_v2(hardened_config)


def _integer_histogram(values: np.ndarray) -> Dict[str, int]:
    """Independently count integer values using manifest-compatible keys."""

    array = np.asarray(values, dtype=int)
    return {
        str(int(value)): int(np.sum(array == value)) for value in np.unique(array)
    }


def _maximum_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive true run in one Boolean vector."""

    maximum = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def test_window_filter_enforces_visible_count_and_consecutive_visibility() -> None:
    """A window meeting only the total-visible rule must still be rejected."""

    true_positions = np.column_stack(
        (np.arange(10, dtype=float), np.zeros(10, dtype=float))
    )
    visible = np.array(
        [True, False, True, False, True, True, False, False, False, False]
    )
    observed = true_positions.copy()
    observed[~visible] = np.nan
    observations = ObservationSequence(
        observed_positions=observed,
        visible_mask=visible,
        occluded_mask=~visible,
        dropout_mask=np.zeros(visible.shape, dtype=bool),
    )

    windows = create_episode_windows(
        true_positions=true_positions,
        observations=observations,
        times=np.arange(10, dtype=float) * 0.1,
        sensor_position=np.array([0.0, 1.0]),
        tree_centers=np.empty((0, 2), dtype=float),
        tree_radii=np.empty((0,), dtype=float),
        scene_id=7,
        episode_id=11,
        trajectory_type_code=0,
        history_steps=4,
        future_steps=2,
        window_stride=2,
        minimum_visible_history_steps=2,
        minimum_consecutive_visible_steps=2,
    )

    # start=0 has two visible frames but no consecutive pair; starts 2 and 4 pass.
    np.testing.assert_array_equal(windows["sample_start_index"], [2, 4])
    np.testing.assert_array_equal(windows["history_visible_count"], [3, 2])
    np.testing.assert_array_equal(
        windows["last_valid_observation_age_steps"], [0, 2]
    )
    np.testing.assert_array_equal(windows["valid_velocity_count"], [1, 1])


def test_formal_configuration_loads_without_generating_formal_data() -> None:
    """The formal YAML must load with the requested scale and safeguards."""

    config = load_dataset_v2_config(
        PROJECT_ROOT / "configs" / "dataset_v2_formal.yaml"
    )

    assert config.scene_counts["train"] >= 300
    assert config.scene_counts["validation"] >= 50
    assert config.scene_counts["test"] >= 100
    assert config.episodes_per_scene >= 5
    assert len(set(config.split_seeds.values())) == len(SPLIT_NAMES)
    assert config.scene.sensor_tree_clearance == pytest.approx(0.5)
    assert config.window.window_stride == 5
    assert config.window.minimum_visible_history_steps == 2
    assert config.window.minimum_consecutive_visible_steps == 2
    assert config.window.minimum_windows_per_episode == 3
    assert config.output_directory == Path("outputs/dataset_v2_formal")


@pytest.mark.parametrize(
    "field,value",
    [
        ("sensor_tree_clearance", np.nan),
        ("sensor_tree_clearance", np.inf),
    ],
)
def test_nonfinite_scene_configuration_is_rejected(
    hardened_config: DatasetV2Config, field: str, value: float
) -> None:
    """NaN/Inf must not bypass comparison-based configuration checks."""

    scene = replace(hardened_config.scene, **{field: value})
    with pytest.raises(ValueError, match="finite"):
        validate_dataset_v2_config(replace(hardened_config, scene=scene))


def test_each_split_must_have_enough_episodes_for_all_motion_types(
    hardened_config: DatasetV2Config,
) -> None:
    """Five-mode balance must fail during config validation, not generation."""

    with pytest.raises(ValueError, match="at least five episodes"):
        validate_dataset_v2_config(
            replace(
                hardened_config,
                scene_counts={name: 1 for name in SPLIT_NAMES},
                episodes_per_scene=3,
            )
        )


def test_fractional_duration_uses_the_actual_sample_time_count(
    hardened_config: DatasetV2Config,
) -> None:
    """Endpoint-appended sampling and window feasibility must use one rule."""

    trajectory = replace(hardened_config.trajectory, duration_seconds=7.81)
    window = replace(hardened_config.window, minimum_windows_per_episode=9)
    validate_dataset_v2_config(
        replace(hardened_config, trajectory=trajectory, window=window)
    )


def test_every_accepted_window_meets_history_validity_requirements(
    hardened_dataset: GeneratedDatasetV2,
    hardened_config: DatasetV2Config,
) -> None:
    """Saved windows must retain enough total and consecutive observations."""

    window = hardened_config.window
    for split_name in SPLIT_NAMES:
        arrays = hardened_dataset.splits[split_name]
        masks = arrays["history_mask"].astype(bool)
        independently_counted = np.sum(masks, axis=1)
        independently_measured_runs = np.asarray(
            [_maximum_true_run(mask) for mask in masks], dtype=int
        )

        assert np.all(independently_counted >= window.minimum_visible_history_steps)
        assert np.all(
            independently_measured_runs >= window.minimum_consecutive_visible_steps
        )
        np.testing.assert_array_equal(
            arrays["history_visible_count"], independently_counted
        )


def test_every_episode_has_enough_windows_and_none_is_fully_unobservable(
    hardened_dataset: GeneratedDatasetV2,
    hardened_config: DatasetV2Config,
) -> None:
    """Accepted episodes must all contribute prediction windows and observations."""

    expected_type_codes = set(TRAJECTORY_TYPE_TO_CODE.values())
    for split_name in SPLIT_NAMES:
        arrays = hardened_dataset.splits[split_name]
        assert set(arrays["episode_id"].tolist()) == set(
            arrays["episode_ids"].tolist()
        )
        assert set(arrays["trajectory_type"].tolist()) == expected_type_codes
        assert np.all(np.sum(arrays["episode_visible_mask"], axis=1) > 0)
        for episode_id in arrays["episode_ids"]:
            window_count = int(np.sum(arrays["episode_id"] == episode_id))
            assert window_count >= hardened_config.window.minimum_windows_per_episode


def test_manifest_declares_strict_training_field_roles(
    hardened_dataset: GeneratedDatasetV2,
) -> None:
    """The manifest must not authorize audit truth or future parameters as input."""

    roles = hardened_dataset.manifest["field_roles"]

    assert roles["model_input_fields"] == MODEL_INPUT_FIELDS
    assert roles["supervision_label_fields"] == SUPERVISION_FIELDS
    assert roles["sample_index_metadata_fields"] == SAMPLE_METADATA_FIELDS
    assert roles["trajectory_type_usage"] == "metadata_only"
    assert set(roles["future_motion_parameter_fields"]).isdisjoint(
        MODEL_INPUT_FIELDS
    )
    assert EXPLICITLY_FORBIDDEN_INPUT_FIELDS <= set(
        roles["forbidden_model_input_fields"]
    )
    assert set(roles["forbidden_model_input_fields"]).isdisjoint(
        MODEL_INPUT_FIELDS
    )
    for arrays in hardened_dataset.splits.values():
        assert set(roles["forbidden_model_input_fields"]) == (
            set(arrays) - set(MODEL_INPUT_FIELDS)
        )


def test_manifest_hardening_statistics_match_independent_calculations(
    hardened_dataset: GeneratedDatasetV2,
) -> None:
    """Visibility, velocity, bins, and rejection audits must be reproducible."""

    for split_name in SPLIT_NAMES:
        arrays = hardened_dataset.splits[split_name]
        statistics = hardened_dataset.manifest["splits"][split_name]

        assert statistics["history_visible_count_distribution"] == (
            _integer_histogram(arrays["history_visible_count"])
        )
        assert statistics["last_valid_observation_age_steps_distribution"] == (
            _integer_histogram(arrays["last_valid_observation_age_steps"])
        )
        assert statistics["valid_velocity_count_distribution"] == (
            _integer_histogram(arrays["valid_velocity_count"])
        )
        assert statistics["windows_without_valid_velocity_count"] == int(
            np.sum(arrays["valid_velocity_count"] == 0)
        )

        expected_bin_counts = {
            name: int(np.sum(arrays["occlusion_length_bin"] == code))
            for name, code in OCCLUSION_LENGTH_BIN_TO_CODE.items()
        }
        assert statistics["window_occlusion_length_bin_counts"] == (
            expected_bin_counts
        )

        rejection_by_type = statistics[
            "rejected_episode_counts_by_trajectory_type"
        ]
        assert statistics["rejected_fully_unobservable_episode_count"] == sum(
            values["fully_unobservable"] for values in rejection_by_type.values()
        )
        assert statistics["rejected_insufficient_history_episode_count"] == sum(
            values["insufficient_history"] for values in rejection_by_type.values()
        )
        assert statistics["rejected_physical_episode_candidate_count"] == sum(
            values["physical"] for values in rejection_by_type.values()
        )

        for trajectory_type, code in TRAJECTORY_TYPE_TO_CODE.items():
            selected = arrays["trajectory_type"] == code
            audit = statistics["trajectory_type_window_audit"][trajectory_type]
            rejection = rejection_by_type[trajectory_type]
            assert audit["episode_count"] == int(
                np.sum(arrays["episode_trajectory_types"] == code)
            )
            assert audit["valid_window_count"] == int(np.sum(selected))
            assert audit["rejected_episode_count"] == rejection["total"]
            assert audit["occlusion_length_bin_counts"] == {
                name: int(
                    np.sum(arrays["occlusion_length_bin"][selected] == bin_code)
                )
                for name, bin_code in OCCLUSION_LENGTH_BIN_TO_CODE.items()
            }
