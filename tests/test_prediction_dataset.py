"""Tests for the strict, NumPy-only prediction dataset boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from thesis_experiment.data.prediction_dataset import (
    FORBIDDEN_INPUT_FIELDS,
    INDEX_METADATA_FIELDS,
    MODEL_INPUT_FIELDS,
    PredictionDataset,
)


@pytest.fixture()
def prediction_archive(tmp_path: Path) -> Path:
    """Create a split archive containing both permitted and forbidden data."""

    history_position = np.asarray(
        [
            [[np.nan, np.nan], [1.0, 2.0], [2.0, 3.0], [np.nan, np.nan]],
            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]],
            [[9.0, 8.0], [8.0, 7.0], [np.nan, np.nan], [6.0, 5.0]],
        ],
        dtype=np.float64,
    )
    history_velocity = np.asarray(
        [
            [[np.nan, np.nan], [np.nan, np.nan], [10.0, 10.0], [np.nan, np.nan]],
            [[np.nan, np.nan], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
            [[np.nan, np.nan], [-10.0, -10.0], [np.nan, np.nan], [np.nan, -10.0]],
        ],
        dtype=np.float64,
    )
    history_mask = np.asarray(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1]], dtype=np.uint8
    )
    path = tmp_path / "train.npz"
    np.savez_compressed(
        str(path),
        history_position=history_position,
        history_velocity=history_velocity,
        history_mask=history_mask,
        time_step_seconds=np.full(3, 0.1, dtype=np.float64),
        future_position=np.arange(24, dtype=np.float64).reshape(3, 4, 2),
        scene_id=np.asarray([10, 10, 11], dtype=np.int64),
        episode_id=np.asarray([100, 101, 102], dtype=np.int64),
        sample_start_index=np.asarray([0, 5, 10], dtype=np.int64),
        history_true_position=np.zeros((3, 4, 2), dtype=np.float64),
        trajectory_type=np.asarray([0, 1, 2], dtype=np.int8),
        episode_true_position_world=np.zeros((3, 8, 2), dtype=np.float64),
        episode_turn_rate=np.ones(3, dtype=np.float64),
        episode_stop_start_time=np.ones(3, dtype=np.float64),
        episode_stop_duration=np.ones(3, dtype=np.float64),
        episode_piecewise_turn_time=np.ones(3, dtype=np.float64),
        episode_piecewise_turn_angle=np.ones(3, dtype=np.float64),
        episode_acceleration_parameter=np.ones((3, 2), dtype=np.float64),
    )
    return path


def test_default_loader_exposes_only_strict_whitelists(
    prediction_archive: Path,
) -> None:
    """Audit fields and future parameters must never appear as model inputs."""

    dataset = PredictionDataset(prediction_archive)
    sample = dataset[0]

    assert len(dataset) == 3
    assert dataset.input_fields == MODEL_INPUT_FIELDS
    assert dataset.metadata_fields == INDEX_METADATA_FIELDS
    assert dataset.target_field == "future_position"
    assert set(sample) == {"inputs", "target", "metadata"}
    assert tuple(sample["inputs"]) == MODEL_INPUT_FIELDS
    assert tuple(sample["metadata"]) == INDEX_METADATA_FIELDS
    assert FORBIDDEN_INPUT_FIELDS.isdisjoint(sample["inputs"])
    assert "future_position" not in sample["inputs"]
    assert sample["target"].shape == (4, 2)


def test_missing_values_are_filled_and_masks_are_preserved(
    prediction_archive: Path,
) -> None:
    """No non-finite state reaches a model, while both masks remain available."""

    dataset = PredictionDataset(prediction_archive, fill_value=-7.5)
    sample = dataset[0]
    inputs: Dict[str, np.ndarray] = sample["inputs"]

    assert np.all(np.isfinite(inputs["history_position"]))
    assert np.all(np.isfinite(inputs["history_velocity"]))
    assert np.array_equal(inputs["history_mask"], np.asarray([0, 1, 1, 0]))
    assert np.array_equal(
        inputs["history_velocity_mask"], np.asarray([0, 0, 1, 0])
    )
    assert np.all(inputs["history_position"][[0, 3]] == -7.5)
    assert np.all(inputs["history_velocity"][[0, 1, 3]] == -7.5)


def test_velocity_mask_requires_both_components_to_be_finite(
    prediction_archive: Path,
) -> None:
    """A partially missing 2-D velocity is invalid as a whole time step."""

    sample = PredictionDataset(prediction_archive)[2]

    assert np.array_equal(
        sample["inputs"]["history_velocity_mask"], np.asarray([0, 1, 0, 0])
    )
    assert np.all(np.isfinite(sample["inputs"]["history_velocity"]))


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "history_true_position",
        "future_position",
        "trajectory_type",
        "episode_true_position_world",
        "episode_turn_rate",
        "episode_stop_start_time",
        "episode_stop_duration",
        "episode_piecewise_turn_time",
        "episode_piecewise_turn_angle",
        "episode_acceleration_parameter",
    ],
)
def test_forbidden_fields_cannot_be_requested_as_inputs(
    prediction_archive: Path, forbidden_field: str
) -> None:
    """Even fields physically present in the archive fail closed as inputs."""

    with pytest.raises(ValueError, match="strict whitelist"):
        PredictionDataset(prediction_archive, input_fields=[forbidden_field])


def test_target_and_metadata_requests_are_also_strict(
    prediction_archive: Path,
) -> None:
    """Truth may only be a future label and grouping metadata stays minimal."""

    with pytest.raises(ValueError, match="target_field"):
        PredictionDataset(prediction_archive, target_field="history_true_position")
    with pytest.raises(ValueError, match="strict whitelist"):
        PredictionDataset(prediction_archive, metadata_fields=["trajectory_type"])


def test_get_batch_preserves_schema_and_does_not_expose_internal_views(
    prediction_archive: Path,
) -> None:
    """Batches use the same boundary and returned arrays cannot mutate storage."""

    dataset = PredictionDataset(prediction_archive)
    batch = dataset.get_batch([0, 2])

    assert batch["inputs"]["history_position"].shape == (2, 4, 2)
    assert batch["inputs"]["history_velocity_mask"].shape == (2, 4)
    assert batch["target"].shape == (2, 4, 2)
    assert batch["metadata"]["scene_id"].tolist() == [10, 11]
    batch["inputs"]["history_position"][0, 1, 0] = 999.0
    assert dataset[0]["inputs"]["history_position"][1, 0] == 1.0


def test_requested_subsets_do_not_load_or_return_extra_fields(
    prediction_archive: Path,
) -> None:
    """A derived mask can be requested without exposing its velocity source."""

    dataset = PredictionDataset(
        prediction_archive,
        input_fields=["history_velocity_mask"],
        metadata_fields=["episode_id"],
    )
    sample = dataset[1]

    assert set(sample["inputs"]) == {"history_velocity_mask"}
    assert set(sample["metadata"]) == {"episode_id"}
    assert sample["metadata"]["episode_id"] == 101
    assert not hasattr(dataset, "_archive")


def test_nonfinite_fill_value_is_rejected(prediction_archive: Path) -> None:
    """The configured replacement itself must be safe for numerical models."""

    with pytest.raises(ValueError, match="fill_value must be finite"):
        PredictionDataset(prediction_archive, fill_value=np.nan)


def _tampered_archive(
    source: Path, destination: Path, field: str, value: np.ndarray
) -> Path:
    """Copy a fixture archive while replacing one array."""

    with np.load(str(source), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays[field] = value
    np.savez_compressed(str(destination), **arrays)
    return destination


def test_loader_rejects_nonbinary_history_mask(
    prediction_archive: Path, tmp_path: Path
) -> None:
    """A malformed mask must not be passed through to a neural network."""

    with np.load(str(prediction_archive), allow_pickle=False) as archive:
        mask = archive["history_mask"].copy()
    mask[0, 0] = 2
    path = _tampered_archive(
        prediction_archive, tmp_path / "bad_mask.npz", "history_mask", mask
    )
    with pytest.raises(ValueError, match="only 0 and 1"):
        PredictionDataset(path)


def test_loader_rejects_position_mask_disagreement(
    prediction_archive: Path, tmp_path: Path
) -> None:
    """The retained position mask must describe every filled time step."""

    with np.load(str(prediction_archive), allow_pickle=False) as archive:
        positions = archive["history_position"].copy()
    positions[0, 1] = np.nan
    path = _tampered_archive(
        prediction_archive,
        tmp_path / "bad_position_mask.npz",
        "history_position",
        positions,
    )
    with pytest.raises(ValueError, match="do not match history_mask"):
        PredictionDataset(path)


def test_loader_rejects_nonpositive_time_step(
    prediction_archive: Path, tmp_path: Path
) -> None:
    """Time deltas supplied to a future model must be finite and positive."""

    path = _tampered_archive(
        prediction_archive,
        tmp_path / "bad_dt.npz",
        "time_step_seconds",
        np.asarray([0.1, 0.0, 0.1], dtype=float),
    )
    with pytest.raises(ValueError, match="must be positive"):
        PredictionDataset(path)
