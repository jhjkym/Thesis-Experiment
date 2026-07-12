"""Regression tests for sensor-to-tree surface clearance generation."""

import numpy as np
import pytest

from thesis_experiment.geometry.forest import generate_tree_trunks


def _forest_arguments():
    """Return reusable arguments for a sparse deterministic forest."""

    return {
        "scene_size": (16.0, 12.0),
        "tree_count": 18,
        "radius_range": (0.2, 0.45),
        "min_spacing": 0.15,
        "sensor_position": np.array([3.0, 4.0]),
        "max_attempts": 10000,
    }


def test_generated_trees_respect_sensor_surface_clearance():
    """Every sensor-to-trunk surface distance must reach the requested value."""

    arguments = _forest_arguments()
    sensor = arguments["sensor_position"]
    centers, radii = generate_tree_trunks(
        **arguments, sensor_clearance=0.5, seed=321
    )
    surface_distances = np.linalg.norm(centers - sensor, axis=1) - radii

    assert np.all(surface_distances >= 0.5 - 1e-12)


def test_impossible_sensor_clearance_stops_after_max_attempts():
    """An impossible clearance must fail explicitly instead of looping forever."""

    with pytest.raises(RuntimeError, match="after 7 attempts"):
        generate_tree_trunks(
            scene_size=(2.0, 2.0),
            tree_count=1,
            radius_range=(0.4, 0.4),
            min_spacing=0.0,
            sensor_position=(1.0, 1.0),
            sensor_clearance=1.0,
            seed=7,
            max_attempts=7,
        )


def test_sensor_clearance_generation_is_reproducible_for_fixed_seed():
    """Equivalent seeds and clearances must generate identical forests."""

    arguments = _forest_arguments()
    first = generate_tree_trunks(
        **arguments, sensor_clearance=0.5, seed=987654
    )
    second = generate_tree_trunks(
        **arguments, sensor_clearance=0.5, seed=987654
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_default_sensor_clearance_matches_explicit_zero():
    """Omitting the new option must retain the experiment-1 random sequence."""

    arguments = _forest_arguments()
    default_result = generate_tree_trunks(**arguments, seed=2468)
    explicit_zero_result = generate_tree_trunks(
        **arguments, sensor_clearance=0.0, seed=2468
    )

    np.testing.assert_array_equal(default_result[0], explicit_zero_result[0])
    np.testing.assert_array_equal(default_result[1], explicit_zero_result[1])


@pytest.mark.parametrize("clearance", [-0.1, np.inf, -np.inf, np.nan])
def test_sensor_clearance_must_be_finite_and_non_negative(clearance):
    """Invalid sensor clearances must be rejected before sampling."""

    with pytest.raises(ValueError, match="sensor_clearance"):
        generate_tree_trunks(
            **_forest_arguments(), sensor_clearance=clearance, seed=1
        )
