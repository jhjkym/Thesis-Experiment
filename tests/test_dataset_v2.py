"""Unit and integration tests for the multi-scene dataset-v2 pipeline.

The checks in this module intentionally generate a small dataset in memory.
They do not read from or write to the repository's ``outputs`` directory.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pytest

from thesis_experiment.config_v2 import DatasetV2Config, load_dataset_v2_config
from thesis_experiment.data.dataset_v2 import (
    GeneratedDatasetV2,
    generate_dataset_v2,
    save_generated_dataset_v2,
    validate_trajectory,
)
from thesis_experiment.data.validation_v2 import validate_dataset_directory
from thesis_experiment.data.trajectory_v2 import (
    TRAJECTORY_TYPE_TO_CODE,
    TrajectoryParameters,
    generate_trajectory,
)
from thesis_experiment.geometry.occlusion import segment_intersects_circle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "validation", "test")
TRAJECTORY_TYPES = tuple(TRAJECTORY_TYPE_TO_CODE.keys())
REQUIRED_WINDOW_FIELDS = {
    "history_position",
    "history_velocity",
    "history_mask",
    "history_occluded",
    "history_random_dropout",
    "history_true_position",
    "future_position",
    "coordinate_origin",
    "sensor_position",
    "tree_centers",
    "tree_radii",
    "scene_id",
    "episode_id",
    "trajectory_type",
    "sample_start_index",
    "history_start_time",
    "future_start_time",
    "time_step_seconds",
    "history_visible_count",
    "last_valid_observation_age_steps",
    "valid_velocity_count",
    "history_max_consecutive_occlusion_steps",
    "occlusion_length_bin",
}


@pytest.fixture(scope="module")
def trajectory_times() -> np.ndarray:
    """Return deterministic 10 Hz sample times long enough for all events."""

    return np.arange(61, dtype=float) * 0.1


@pytest.fixture(scope="module")
def trajectory_parameters() -> TrajectoryParameters:
    """Return conservative parameters valid for all five motion models."""

    return TrajectoryParameters(
        initial_position=np.array([5.0, 5.0]),
        initial_velocity=np.array([0.6, 0.2]),
        acceleration=np.array([0.05, -0.02]),
        turn_rate=0.12,
        stop_start_time=2.0,
        stop_duration=1.0,
        piecewise_turn_time=2.0,
        piecewise_turn_angle=0.45,
        transition_duration=1.0,
    )


@pytest.fixture(scope="module")
def small_dataset_config() -> DatasetV2Config:
    """Load the real smoke schema and reduce only its scene/tree counts."""

    base = load_dataset_v2_config(PROJECT_ROOT / "configs" / "dataset_v2_smoke.yaml")
    return replace(
        base,
        scene_counts={name: 1 for name in SPLIT_NAMES},
        episodes_per_scene=5,
        trees=replace(base.trees, count=4),
    )


@pytest.fixture(scope="module")
def generated_dataset(small_dataset_config: DatasetV2Config) -> GeneratedDatasetV2:
    """Generate a fast in-memory dataset containing every motion type."""

    return generate_dataset_v2(small_dataset_config)


@pytest.fixture(scope="module")
def saved_dataset_directory(
    tmp_path_factory: pytest.TempPathFactory,
    generated_dataset: GeneratedDatasetV2,
) -> Path:
    """Save the in-memory fixture below pytest's temporary directory."""

    directory = tmp_path_factory.mktemp("dataset_v2")
    save_generated_dataset_v2(generated_dataset, directory)
    return directory


@pytest.mark.parametrize("trajectory_type", TRAJECTORY_TYPES)
def test_each_trajectory_type_has_the_expected_shape(
    trajectory_type: str,
    trajectory_times: np.ndarray,
    trajectory_parameters: TrajectoryParameters,
) -> None:
    """Every motion model must return position, velocity, and acceleration."""

    result = generate_trajectory(
        trajectory_type, trajectory_times, trajectory_parameters
    )

    assert result.positions.shape == (trajectory_times.size, 2)
    assert result.velocities.shape == result.positions.shape
    assert result.accelerations.shape == result.positions.shape


@pytest.mark.parametrize("trajectory_type", TRAJECTORY_TYPES)
def test_each_trajectory_type_contains_only_finite_values(
    trajectory_type: str,
    trajectory_times: np.ndarray,
    trajectory_parameters: TrajectoryParameters,
) -> None:
    """No motion model may emit NaN or infinite trajectory state values."""

    result = generate_trajectory(
        trajectory_type, trajectory_times, trajectory_parameters
    )

    assert np.all(np.isfinite(result.positions))
    assert np.all(np.isfinite(result.velocities))
    assert np.all(np.isfinite(result.accelerations))


@pytest.mark.parametrize("trajectory_type", TRAJECTORY_TYPES)
def test_each_trajectory_stays_inside_the_scene(
    trajectory_type: str,
    trajectory_times: np.ndarray,
    trajectory_parameters: TrajectoryParameters,
) -> None:
    """The deterministic test trajectories must remain inside their scene."""

    result = generate_trajectory(
        trajectory_type, trajectory_times, trajectory_parameters
    )
    scene_size = np.array([20.0, 20.0])
    boundary_margin = 0.15

    assert np.all(result.positions >= boundary_margin)
    assert np.all(result.positions <= scene_size - boundary_margin)


@pytest.mark.parametrize("trajectory_type", TRAJECTORY_TYPES)
def test_each_trajectory_keeps_clear_of_tree_trunks(
    trajectory_type: str,
    trajectory_times: np.ndarray,
    trajectory_parameters: TrajectoryParameters,
) -> None:
    """No sampled point or connecting segment may enter an expanded trunk."""

    result = generate_trajectory(
        trajectory_type, trajectory_times, trajectory_parameters
    )
    tree_center = np.array([15.0, 15.0])
    tree_radius = 0.6
    minimum_clearance = 0.2
    expanded_radius = tree_radius + minimum_clearance
    surface_clearances = (
        np.linalg.norm(result.positions - tree_center, axis=1) - tree_radius
    )

    assert np.all(surface_clearances >= minimum_clearance)
    for start, end in zip(result.positions[:-1], result.positions[1:]):
        assert not bool(
            segment_intersects_circle(start, end, tree_center, expanded_radius)
        )


@pytest.mark.parametrize("trajectory_type", TRAJECTORY_TYPES)
def test_each_trajectory_respects_speed_and_acceleration_limits(
    trajectory_type: str,
    trajectory_times: np.ndarray,
    trajectory_parameters: TrajectoryParameters,
) -> None:
    """All sampled states must respect the declared physical limits."""

    result = generate_trajectory(
        trajectory_type, trajectory_times, trajectory_parameters
    )
    maximum_speed = 1.2
    maximum_acceleration = 1.5

    assert float(np.max(np.linalg.norm(result.velocities, axis=1))) <= (
        maximum_speed + 1e-12
    )
    assert float(np.max(np.linalg.norm(result.accelerations, axis=1))) <= (
        maximum_acceleration + 1e-12
    )


@pytest.mark.parametrize("trajectory_type", TRAJECTORY_TYPES)
def test_trajectory_audit_derivatives_use_only_adjacent_time_steps(
    trajectory_type: str,
    trajectory_times: np.ndarray,
    trajectory_parameters: TrajectoryParameters,
) -> None:
    """Positions and accelerations must satisfy their documented causal rules."""

    result = generate_trajectory(
        trajectory_type, trajectory_times, trajectory_parameters
    )
    intervals = np.diff(trajectory_times)[:, np.newaxis]
    expected_position_steps = (
        0.5 * (result.velocities[:-1] + result.velocities[1:]) * intervals
    )
    expected_accelerations = np.diff(result.velocities, axis=0) / intervals

    np.testing.assert_allclose(
        np.diff(result.positions, axis=0), expected_position_steps
    )
    np.testing.assert_allclose(result.accelerations[0], np.zeros(2))
    np.testing.assert_allclose(result.accelerations[1:], expected_accelerations)


def test_trajectory_validation_rejects_a_segment_crossing_a_tree() -> None:
    """Collision validation must check segments, not sampled endpoints alone."""

    positions = np.array([[0.0, 1.0], [2.0, 1.0]])
    velocities = np.array([[1.0, 0.0], [1.0, 0.0]])
    accelerations = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError, match="tree"):
        validate_trajectory(
            positions,
            velocities,
            accelerations,
            scene_size=(3.0, 2.0),
            tree_centers=np.array([[1.0, 1.0]]),
            tree_radii=np.array([0.25]),
            min_tree_clearance=0.1,
            max_speed=2.0,
            max_acceleration=2.0,
        )


def test_generated_splits_contain_required_fields_and_shapes(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """Every split must expose the requested fixed-size model/audit arrays."""

    history_steps = small_dataset_config.window.history_steps
    future_steps = small_dataset_config.window.future_steps
    tree_count = small_dataset_config.trees.count
    assert history_steps == 20
    assert future_steps == 20
    assert small_dataset_config.window.window_stride == 5
    assert small_dataset_config.trajectory.sample_rate_hz == 10.0
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        assert REQUIRED_WINDOW_FIELDS <= set(arrays)
        window_count = arrays["scene_id"].shape[0]
        expected_shapes = {
            "history_position": (window_count, history_steps, 2),
            "history_velocity": (window_count, history_steps, 2),
            "history_mask": (window_count, history_steps),
            "history_occluded": (window_count, history_steps),
            "history_random_dropout": (window_count, history_steps),
            "history_true_position": (window_count, history_steps, 2),
            "future_position": (window_count, future_steps, 2),
            "coordinate_origin": (window_count, 2),
            "sensor_position": (window_count, 2),
            "tree_centers": (window_count, tree_count, 2),
            "tree_radii": (window_count, tree_count),
        }
        for field, expected_shape in expected_shapes.items():
            assert arrays[field].shape == expected_shape
        for field in REQUIRED_WINDOW_FIELDS - set(expected_shapes):
            assert arrays[field].shape == (window_count,)


def test_generated_episode_trajectories_are_finite_and_physically_valid(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """All saved full episodes must satisfy bounds, clearance, and limits."""

    scene_size = np.array(
        [small_dataset_config.scene.width, small_dataset_config.scene.height]
    )
    trajectory_config = small_dataset_config.trajectory
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        scene_index = {
            int(scene_id): index for index, scene_id in enumerate(arrays["scene_ids"])
        }
        for episode_index, scene_id_value in enumerate(arrays["episode_scene_ids"]):
            positions = arrays["episode_true_position_world"][episode_index]
            velocities = arrays["episode_velocity_world"][episode_index]
            accelerations = arrays["episode_acceleration_world"][episode_index]
            tree_index = scene_index[int(scene_id_value)]
            tree_centers = arrays["scene_tree_centers_world"][tree_index]
            tree_radii = arrays["scene_tree_radii"][tree_index]

            assert np.all(np.isfinite(positions))
            assert np.all(np.isfinite(velocities))
            assert np.all(np.isfinite(accelerations))
            assert np.all(positions >= trajectory_config.boundary_margin - 1e-10)
            assert np.all(
                positions
                <= scene_size - trajectory_config.boundary_margin + 1e-10
            )
            assert float(np.max(np.linalg.norm(velocities, axis=1))) <= (
                trajectory_config.max_speed + 1e-10
            )
            assert float(np.max(np.linalg.norm(accelerations, axis=1))) <= (
                trajectory_config.max_acceleration + 1e-10
            )

            distances = np.linalg.norm(
                positions[:, np.newaxis, :] - tree_centers[np.newaxis, :, :],
                axis=2,
            )
            surface_clearances = distances - tree_radii[np.newaxis, :]
            assert np.all(
                surface_clearances >= trajectory_config.min_tree_clearance - 1e-10
            )
            for start, end in zip(positions[:-1], positions[1:]):
                for center, radius in zip(tree_centers, tree_radii):
                    assert not bool(
                        segment_intersects_circle(
                            start,
                            end,
                            center,
                            float(radius + trajectory_config.min_tree_clearance),
                        )
                    )


def _sets_are_pairwise_disjoint(values: Mapping[str, Iterable[int]]) -> bool:
    """Return whether all named integer collections are pairwise disjoint."""

    sets = {name: set(items) for name, items in values.items()}
    for first_index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[first_index + 1 :]:
            if sets[first] & sets[second]:
                return False
    return True


def test_scene_level_split_sets_are_mutually_exclusive(
    generated_dataset: GeneratedDatasetV2,
) -> None:
    """No scene, and therefore no scene-owned window, may cross a split."""

    scene_ids = {
        name: generated_dataset.splits[name]["scene_ids"].tolist()
        for name in SPLIT_NAMES
    }
    window_scene_ids = {
        name: np.unique(generated_dataset.splits[name]["scene_id"]).tolist()
        for name in SPLIT_NAMES
    }

    assert _sets_are_pairwise_disjoint(scene_ids)
    assert _sets_are_pairwise_disjoint(window_scene_ids)
    for name in SPLIT_NAMES:
        assert set(window_scene_ids[name]) <= set(scene_ids[name])


def test_episode_level_split_sets_are_mutually_exclusive(
    generated_dataset: GeneratedDatasetV2,
) -> None:
    """Episode identifiers and their derived windows must never cross splits."""

    episode_ids = {
        name: generated_dataset.splits[name]["episode_ids"].tolist()
        for name in SPLIT_NAMES
    }
    window_episode_ids = {
        name: np.unique(generated_dataset.splits[name]["episode_id"]).tolist()
        for name in SPLIT_NAMES
    }

    assert _sets_are_pairwise_disjoint(episode_ids)
    assert _sets_are_pairwise_disjoint(window_episode_ids)
    for name in SPLIT_NAMES:
        assert set(window_episode_ids[name]) <= set(episode_ids[name])


def test_each_episode_belongs_to_exactly_one_scene_and_split(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """All rows for an episode must retain one scene/type owner in one split."""

    owners: Dict[int, Tuple[str, int, int]] = {}
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        for scene_id in arrays["scene_ids"]:
            assert int(np.sum(arrays["episode_scene_ids"] == scene_id)) == (
                small_dataset_config.episodes_per_scene
            )
        for episode_id in np.unique(arrays["episode_id"]):
            selected = arrays["episode_id"] == episode_id
            scenes = np.unique(arrays["scene_id"][selected])
            types = np.unique(arrays["trajectory_type"][selected])
            assert scenes.size == 1
            assert types.size == 1
            owner = (split_name, int(scenes[0]), int(types[0]))
            assert int(episode_id) not in owners
            owners[int(episode_id)] = owner


def test_windows_are_sliced_from_exactly_one_episode(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """Every local window must reconstruct the matching full episode slice."""

    history_steps = small_dataset_config.window.history_steps
    future_steps = small_dataset_config.window.future_steps
    total_steps = history_steps + future_steps
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        episode_index = {
            int(episode_id): index
            for index, episode_id in enumerate(arrays["episode_ids"])
        }
        scene_index = {
            int(scene_id): index for index, scene_id in enumerate(arrays["scene_ids"])
        }
        for row_index in range(arrays["episode_id"].size):
            episode_id = int(arrays["episode_id"][row_index])
            owner_index = episode_index[episode_id]
            scene_id = int(arrays["scene_id"][row_index])
            owner_scene_id = int(arrays["episode_scene_ids"][owner_index])
            start = int(arrays["sample_start_index"][row_index])
            end = start + total_steps
            origin = arrays["coordinate_origin"][row_index]
            reconstructed = np.concatenate(
                (
                    arrays["history_true_position"][row_index],
                    arrays["future_position"][row_index],
                ),
                axis=0,
            ) + origin
            expected = arrays["episode_true_position_world"][owner_index, start:end]

            assert scene_id == owner_scene_id
            assert end <= arrays["episode_true_position_world"].shape[1]
            assert int(arrays["trajectory_type"][row_index]) == int(
                arrays["episode_trajectory_types"][owner_index]
            )
            np.testing.assert_allclose(reconstructed, expected)

            owner_scene_index = scene_index[scene_id]
            np.testing.assert_allclose(
                arrays["sensor_position"][row_index] + origin,
                arrays["scene_sensor_position_world"][owner_scene_index],
            )
            np.testing.assert_allclose(
                arrays["tree_centers"][row_index] + origin,
                arrays["scene_tree_centers_world"][owner_scene_index],
            )
            np.testing.assert_allclose(
                arrays["tree_radii"][row_index],
                arrays["scene_tree_radii"][owner_scene_index],
            )


def test_history_and_future_indices_are_contiguous_and_nonoverlapping(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """Future index zero must immediately follow the final history index."""

    history_steps = small_dataset_config.window.history_steps
    future_steps = small_dataset_config.window.future_steps
    stride = small_dataset_config.window.window_stride
    expected_dt = 1.0 / small_dataset_config.trajectory.sample_rate_hz
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        starts = arrays["sample_start_index"]
        history_last_indices = starts + history_steps - 1
        future_first_indices = starts + history_steps
        future_last_indices = future_first_indices + future_steps - 1

        assert np.all(starts % stride == 0)
        assert np.all(future_first_indices == history_last_indices + 1)
        assert np.all(future_first_indices > history_last_indices)
        assert np.all(
            future_last_indices
            < arrays["episode_true_position_world"].shape[1]
        )
        np.testing.assert_allclose(arrays["time_step_seconds"], expected_dt)
        np.testing.assert_allclose(
            arrays["history_start_time"], starts.astype(float) * expected_dt
        )
        np.testing.assert_allclose(
            arrays["future_start_time"],
            future_first_indices.astype(float) * expected_dt,
        )


def test_history_velocity_is_a_causal_backward_difference(
    generated_dataset: GeneratedDatasetV2,
) -> None:
    """History velocity must use only current and preceding observations."""

    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        for history, mask, velocity, dt in zip(
            arrays["history_position"],
            arrays["history_mask"].astype(bool),
            arrays["history_velocity"],
            arrays["time_step_seconds"],
        ):
            expected = np.full(history.shape, np.nan, dtype=float)
            adjacent_visible = mask[1:] & mask[:-1]
            expected[1:][adjacent_visible] = (
                np.diff(history, axis=0)[adjacent_visible] / float(dt)
            )

            np.testing.assert_allclose(velocity, expected, equal_nan=True)


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare arrays exactly while treating matching NaNs as equal."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def test_fixed_split_seeds_reproduce_every_saved_field(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """A repeated in-memory build must reproduce every split and manifest."""

    repeated = generate_dataset_v2(small_dataset_config)

    assert repeated.manifest == generated_dataset.manifest
    for split_name in SPLIT_NAMES:
        first_arrays = generated_dataset.splits[split_name]
        second_arrays = repeated.splits[split_name]
        assert set(first_arrays) == set(second_arrays)
        for field in first_arrays:
            assert _arrays_equal(first_arrays[field], second_arrays[field]), (
                "{}:{} was not reproducible".format(split_name, field)
            )


def test_changing_split_seeds_changes_generated_data(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """Changing all independent split seeds must change every split's content."""

    changed_config = replace(
        small_dataset_config,
        split_seeds={
            name: small_dataset_config.split_seeds[name] + 10000
            for name in SPLIT_NAMES
        },
    )
    changed = generate_dataset_v2(changed_config)

    for split_name in SPLIT_NAMES:
        original = generated_dataset.splits[split_name]
        modified = changed.splits[split_name]
        assert not _arrays_equal(
            original["scene_tree_centers_world"],
            modified["scene_tree_centers_world"],
        )
        assert not _arrays_equal(
            original["episode_true_position_world"],
            modified["episode_true_position_world"],
        )


def _consecutive_true_run_lengths(mask_rows: np.ndarray) -> Tuple[int, ...]:
    """Independently compute row-bounded runs of true mask values."""

    lengths = []
    for row in np.asarray(mask_rows, dtype=bool):
        current = 0
        for value in row:
            if value:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        if current:
            lengths.append(current)
    return tuple(lengths)


def test_manifest_statistics_match_independent_npz_array_calculations(
    generated_dataset: GeneratedDatasetV2,
    small_dataset_config: DatasetV2Config,
) -> None:
    """Manifest counts, ratios, ranges, runs, and metadata must be auditable."""

    assert generated_dataset.manifest["split_seeds"] == (
        small_dataset_config.split_seeds
    )
    assert generated_dataset.manifest["trajectory_type_encoding"] == (
        TRAJECTORY_TYPE_TO_CODE
    )
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        statistics = generated_dataset.manifest["splits"][split_name]
        visible = arrays["history_mask"].astype(bool)
        occluded = arrays["history_occluded"].astype(bool)
        dropout = arrays["history_random_dropout"].astype(bool)
        speed = np.linalg.norm(arrays["episode_velocity_world"], axis=2)
        acceleration = np.linalg.norm(
            arrays["episode_acceleration_world"], axis=2
        )

        assert statistics["seed"] == small_dataset_config.split_seeds[split_name]
        assert statistics["scene_count"] == np.unique(arrays["scene_ids"]).size
        assert statistics["episode_count"] == np.unique(arrays["episode_ids"]).size
        assert statistics["window_count"] == arrays["scene_id"].size
        assert statistics["visible_ratio"] == pytest.approx(float(np.mean(visible)))
        assert statistics["geometric_occlusion_ratio"] == pytest.approx(
            float(np.mean(occluded))
        )
        assert statistics["random_dropout_ratio"] == pytest.approx(
            float(np.mean(dropout))
        )
        assert statistics["occlusion_dropout_overlap_count"] == int(
            np.sum(occluded & dropout)
        )
        np.testing.assert_allclose(
            statistics["speed_range"], [np.min(speed), np.max(speed)]
        )
        np.testing.assert_allclose(
            statistics["acceleration_range"],
            [np.min(acceleration), np.max(acceleration)],
        )
        assert tuple(statistics["consecutive_occlusion_lengths"]) == (
            _consecutive_true_run_lengths(arrays["episode_occluded_mask"])
        )

        window_span = (
            small_dataset_config.window.history_steps
            + small_dataset_config.window.future_steps
        )
        overlap_count = 0
        adjacent_count = 0
        for episode_id in arrays["episode_ids"]:
            starts = np.sort(
                arrays["sample_start_index"][arrays["episode_id"] == episode_id]
            )
            if starts.size > 1:
                adjacent_count += int(starts.size - 1)
                overlap_count += int(np.sum(np.diff(starts) < window_span))
        assert statistics["overlapping_adjacent_window_pairs"] == overlap_count
        assert statistics["adjacent_window_pair_count"] == adjacent_count
        assert statistics["overlapping_adjacent_window_ratio"] == pytest.approx(
            overlap_count / adjacent_count if adjacent_count else 0.0
        )

        episode_codes = arrays["episode_trajectory_types"]
        window_codes = arrays["trajectory_type"]
        for trajectory_type, code in TRAJECTORY_TYPE_TO_CODE.items():
            episode_count = int(np.sum(episode_codes == code))
            window_count = int(np.sum(window_codes == code))
            assert statistics["episode_trajectory_type_counts"][trajectory_type] == (
                episode_count
            )
            assert statistics["window_trajectory_type_counts"][trajectory_type] == (
                window_count
            )
            assert statistics["episode_trajectory_type_ratios"][trajectory_type] == (
                pytest.approx(episode_count / float(episode_codes.size))
            )
            assert statistics["window_trajectory_type_ratios"][trajectory_type] == (
                pytest.approx(window_count / float(window_codes.size))
            )

        for field, values in arrays.items():
            assert statistics["fields"][field] == {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }


def test_small_dataset_includes_every_motion_type_in_every_split(
    generated_dataset: GeneratedDatasetV2,
) -> None:
    """Five episodes per scene must exercise all five encoded motion modes."""

    expected_codes = set(TRAJECTORY_TYPE_TO_CODE.values())
    for split_name in SPLIT_NAMES:
        arrays = generated_dataset.splits[split_name]
        assert set(arrays["episode_trajectory_types"].tolist()) == expected_codes
        assert set(arrays["trajectory_type"].tolist()) == expected_codes


def test_trajectory_sampling_reports_max_attempts_exhaustion(
    small_dataset_config: DatasetV2Config,
) -> None:
    """Impossible physical limits must fail explicitly after the attempt cap."""

    impossible = replace(
        small_dataset_config,
        trajectory=replace(
            small_dataset_config.trajectory,
            max_speed=0.01,
            max_attempts=1,
        ),
    )

    with pytest.raises(RuntimeError, match=r"after 1 attempts"):
        generate_dataset_v2(impossible)


def test_saved_dataset_passes_the_independent_directory_validator(
    saved_dataset_directory: Path,
) -> None:
    """The public validator must audit saved NPZ files without regeneration."""

    summary = validate_dataset_directory(saved_dataset_directory)

    assert summary["status"] == "passed"
    assert all(not values for values in summary["scene_intersections"].values())
    assert all(not values for values in summary["episode_intersections"].values())
    for split_name in SPLIT_NAMES:
        assert summary["splits"][split_name]["scene_count"] == 1
        assert summary["splits"][split_name]["episode_count"] == 5
        assert summary["splits"][split_name]["window_count"] > 0


def test_visualizations_are_created_only_from_the_saved_dataset(
    saved_dataset_directory: Path,
) -> None:
    """The saved-data plotting entry point must produce all nine nonempty PNGs."""

    import matplotlib.pyplot as plt

    from thesis_experiment.visualization.dataset_v2 import (
        generate_dataset_v2_figures,
    )

    paths = generate_dataset_v2_figures(saved_dataset_directory)

    assert len(paths) == 9
    assert len(set(paths)) == 9
    assert all(path.parent == saved_dataset_directory / "figures" for path in paths)
    assert all(path.suffix == ".png" for path in paths)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert plt.get_fignums() == []
