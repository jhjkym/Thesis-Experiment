"""Unit tests for experiment 01 data generation primitives."""

import numpy as np
import pytest

from thesis_experiment.data.dataset import (
    create_dataset_windows,
    create_observations,
    local_to_world,
    world_to_local,
)
from thesis_experiment.geometry.forest import generate_tree_trunks
from thesis_experiment.geometry.occlusion import segment_intersects_circle
from scripts.run_experiment_01 import _visualization_sample


def test_segment_clear_when_circle_is_away_from_segment():
    """A circle away from the finite sensor-target segment must not occlude it."""
    intersects = segment_intersects_circle(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 2.0]),
        0.5,
    )

    assert not bool(intersects)


def test_segment_occluded_when_it_passes_through_circle_center():
    """A segment passing through a circle center must be marked occluded."""
    intersects = segment_intersects_circle(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
        0.25,
    )

    assert bool(intersects)


def test_tangent_segment_counts_as_occlusion():
    """Contact at exactly one point (tangency) counts as an intersection."""
    intersects = segment_intersects_circle(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 1.0]),
        1.0,
    )

    assert bool(intersects)


@pytest.fixture
def generated_forest():
    """Return a deterministic, moderately sparse forest for geometry checks."""
    return generate_tree_trunks(
        scene_size=(20.0, 15.0),
        tree_count=24,
        radius_range=(0.2, 0.45),
        min_spacing=0.15,
        sensor_position=np.array([1.0, 1.0]),
        seed=31415,
    )


def test_generated_tree_trunks_stay_inside_scene(generated_forest):
    """Every trunk circle, not only its center, must remain inside the scene."""
    centers, radii = generated_forest
    width, height = 20.0, 15.0

    assert centers.shape == (24, 2)
    assert radii.shape == (24,)
    assert np.all(radii >= 0.2)
    assert np.all(radii <= 0.45)
    assert np.all(centers[:, 0] - radii >= 0.0)
    assert np.all(centers[:, 0] + radii <= width)
    assert np.all(centers[:, 1] - radii >= 0.0)
    assert np.all(centers[:, 1] + radii <= height)


def test_generated_tree_trunks_have_minimum_edge_spacing(generated_forest):
    """All circle edges must be separated by at least ``min_spacing``."""
    centers, radii = generated_forest
    minimum_spacing = 0.15

    for first in range(len(radii)):
        for second in range(first + 1, len(radii)):
            center_distance = np.linalg.norm(centers[first] - centers[second])
            edge_distance = center_distance - radii[first] - radii[second]
            assert edge_distance >= minimum_spacing - 1e-12


def test_generated_tree_trunks_do_not_cover_sensor(generated_forest):
    """No generated trunk circle may contain the fixed sensor position."""
    centers, radii = generated_forest
    sensor_position = np.array([1.0, 1.0])
    center_distances = np.linalg.norm(centers - sensor_position, axis=1)

    assert np.all(center_distances > radii)


def test_tree_generation_is_reproducible_for_fixed_seed():
    """Repeated calls with the same seed must produce identical forests."""
    kwargs = {
        "scene_size": (12.0, 9.0),
        "tree_count": 12,
        "radius_range": (0.15, 0.35),
        "min_spacing": 0.1,
        "sensor_position": np.array([0.8, 0.8]),
        "seed": 2026,
    }

    first_centers, first_radii = generate_tree_trunks(**kwargs)
    second_centers, second_radii = generate_tree_trunks(**kwargs)

    np.testing.assert_array_equal(first_centers, second_centers)
    np.testing.assert_array_equal(first_radii, second_radii)


def test_local_coordinate_transform_round_trip():
    """World-to-local conversion must be exactly reversible up to float error."""
    world_points = np.array(
        [[-1.25, 4.0], [0.0, 0.0], [3.5, -2.75], [9.25, 8.125]],
        dtype=float,
    )
    origin = np.array([2.25, -3.0])

    local_points = world_to_local(world_points, origin)
    recovered_world_points = local_to_world(local_points, origin)

    np.testing.assert_allclose(recovered_world_points, world_points)
    np.testing.assert_allclose(local_points, world_points - origin)


@pytest.fixture
def alternating_observation_sequence():
    """Build a sequence whose samples alternate between visible and occluded."""
    sample_count = 60
    true_positions = np.empty((sample_count, 2), dtype=float)
    true_positions[:, 0] = 8.0 + 0.02 * np.arange(sample_count)
    true_positions[:, 1] = np.where(np.arange(sample_count) % 2 == 0, 2.0, 0.0)
    sensor_position = np.array([0.0, 0.0])
    tree_centers = np.array([[4.0, 0.0]])
    tree_radii = np.array([0.5])
    observations = create_observations(
        true_positions,
        sensor_position,
        tree_centers,
        tree_radii,
        noise_std=0.0,
        dropout_probability=0.0,
        rng=np.random.default_rng(1234),
    )
    return {
        "true_positions": true_positions,
        "sensor_position": sensor_position,
        "tree_centers": tree_centers,
        "tree_radii": tree_radii,
        "observations": observations,
    }


def test_occluded_observation_is_nan_while_true_position_remains_finite(
    alternating_observation_sequence,
):
    """Occlusion removes an observation but never removes its ground truth."""
    sequence = alternating_observation_sequence
    true_positions = sequence["true_positions"]
    observations = sequence["observations"]
    occluded = observations.occluded_mask.astype(bool)

    assert np.any(occluded)
    assert np.all(observations.visible_mask[occluded] == 0)
    assert np.all(np.isnan(observations.observed_positions[occluded]))
    assert np.all(np.isfinite(true_positions[occluded]))


def test_random_dropout_masks_otherwise_visible_observations():
    """A forced random dropout must set the mask to zero and store NaN."""
    true_positions = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    observations = create_observations(
        true_positions,
        sensor_position=np.array([0.0, 0.0]),
        tree_centers=np.empty((0, 2)),
        tree_radii=np.empty((0,)),
        noise_std=0.0,
        dropout_probability=1.0,
        rng=np.random.default_rng(4321),
    )

    assert np.all(observations.occluded_mask == 0)
    assert np.all(observations.dropout_mask == 1)
    assert np.all(observations.visible_mask == 0)
    assert np.all(np.isnan(observations.observed_positions))


def test_observation_noise_and_dropout_are_reproducible_for_fixed_seed():
    """Using equivalent RNG seeds must reproduce noise, masks, and dropouts."""
    true_positions = np.column_stack(
        (np.linspace(1.0, 4.0, 30), np.linspace(0.5, 2.0, 30))
    )
    arguments = {
        "true_positions": true_positions,
        "sensor_position": np.array([0.0, 0.0]),
        "tree_centers": np.empty((0, 2)),
        "tree_radii": np.empty((0,)),
        "noise_std": 0.15,
        "dropout_probability": 0.25,
    }

    first = create_observations(rng=np.random.default_rng(99), **arguments)
    second = create_observations(rng=np.random.default_rng(99), **arguments)

    np.testing.assert_allclose(
        first.observed_positions, second.observed_positions, equal_nan=True
    )
    np.testing.assert_array_equal(first.visible_mask, second.visible_mask)
    np.testing.assert_array_equal(first.occluded_mask, second.occluded_mask)
    np.testing.assert_array_equal(first.dropout_mask, second.dropout_mask)


@pytest.fixture
def dataset_windows(alternating_observation_sequence):
    """Create several local-coordinate windows from the synthetic sequence."""
    sequence = alternating_observation_sequence
    return create_dataset_windows(
        sequence["true_positions"],
        sequence["observations"],
        sequence["sensor_position"],
        sequence["tree_centers"],
        sequence["tree_radii"],
        history_steps=20,
        future_steps=20,
        num_samples=5,
        dt=0.1,
        scene_id=17,
    )


def test_dataset_window_dimensions_are_correct(dataset_windows):
    """Every required array must carry the expected sample and time axes."""
    expected_shapes = {
        "history_position": (5, 20, 2),
        "history_true_position": (5, 20, 2),
        "history_velocity": (5, 20, 2),
        "history_mask": (5, 20),
        "future_position": (5, 20, 2),
        "sensor_position": (5, 2),
        "tree_centers": (5, 1, 2),
        "tree_radii": (5, 1),
        "scene_id": (5,),
        "coordinate_origin": (5, 2),
        "history_occluded": (5, 20),
        "history_random_dropout": (5, 20),
    }

    for name, expected_shape in expected_shapes.items():
        assert name in dataset_windows
        assert dataset_windows[name].shape == expected_shape


def test_window_keeps_finite_true_history_at_masked_observations(dataset_windows):
    """Masked local observations are NaN while local ground truth stays finite."""
    masked = dataset_windows["history_mask"] == 0

    assert np.any(masked)
    assert np.all(np.isnan(dataset_windows["history_position"][masked]))
    assert np.all(np.isfinite(dataset_windows["history_true_position"][masked]))


def test_each_window_origin_is_its_last_valid_observation(dataset_windows):
    """The final valid local observation in every history must be the origin."""
    for history_position, history_mask in zip(
        dataset_windows["history_position"], dataset_windows["history_mask"]
    ):
        valid_indices = np.flatnonzero(history_mask)
        assert valid_indices.size > 0
        np.testing.assert_allclose(history_position[valid_indices[-1]], [0.0, 0.0])


def test_visualization_example_is_restored_from_one_dataset_row(dataset_windows):
    """Example-plot inputs must be traceable to one exact saved dataset row."""

    sample_index = 2
    example = _visualization_sample(dataset_windows, sample_index)
    origin = dataset_windows["coordinate_origin"][sample_index]

    assert example["sample_index"] == sample_index
    assert example["scene_id"] == 17
    assert example["start_index"] == dataset_windows["sample_start_index"][sample_index]
    np.testing.assert_allclose(
        example["history_position"],
        dataset_windows["history_position"][sample_index] + origin,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        example["history_true_position"],
        dataset_windows["history_true_position"][sample_index] + origin,
    )
    np.testing.assert_allclose(
        example["future_position"],
        dataset_windows["future_position"][sample_index] + origin,
    )
    np.testing.assert_array_equal(
        example["history_mask"],
        dataset_windows["history_mask"][sample_index].astype(bool),
    )
    np.testing.assert_allclose(
        example["tree_centers"],
        dataset_windows["tree_centers"][sample_index] + origin,
    )
