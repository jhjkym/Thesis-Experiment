"""Unit tests for deterministic GRU inputs and train-only normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_experiment.data.prediction_dataset import PredictionDataset
from thesis_experiment.prediction.gru import (
    DeterministicGRU,
    build_gru_features,
)
from thesis_experiment.prediction.normalization import PredictionNormalizer


@pytest.fixture()
def train_archive(tmp_path: Path) -> Path:
    """Create observations whose invalid fill values would bias raw moments."""

    history_position = np.asarray(
        [
            [[1.0, 2.0], [np.nan, np.nan], [3.0, 6.0]],
            [[5.0, 10.0], [7.0, 14.0], [np.nan, np.nan]],
        ],
        dtype=np.float64,
    )
    history_velocity = np.asarray(
        [
            [[np.nan, np.nan], [np.nan, np.nan], [2.0, 4.0]],
            [[np.nan, np.nan], [2.0, 4.0], [np.nan, np.nan]],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "train.npz"
    np.savez_compressed(
        str(path),
        history_position=history_position,
        history_velocity=history_velocity,
        history_mask=np.asarray([[1, 0, 1], [1, 1, 0]], dtype=np.uint8),
        future_position=np.asarray(
            [
                [[8.0, 16.0], [9.0, 18.0]],
                [[10.0, 20.0], [11.0, 22.0]],
            ],
            dtype=np.float64,
        ),
        time_step_seconds=np.asarray([0.1, 0.1], dtype=np.float64),
        scene_id=np.asarray([1, 2], dtype=np.int64),
        episode_id=np.asarray([10, 20], dtype=np.int64),
        sample_start_index=np.asarray([0, 5], dtype=np.int64),
        history_true_position=np.full((2, 3, 2), 99999.0),
        trajectory_type=np.asarray([0, 4], dtype=np.int8),
        episode_turn_rate=np.asarray([99999.0, 99999.0]),
    )
    return path


def _model_inputs(batch_size: int, history_steps: int = 20) -> Dict[str, np.ndarray]:
    """Return a complete finite PredictionDataset-shaped input batch."""

    return {
        "history_position": np.zeros((batch_size, history_steps, 2), dtype=np.float32),
        "history_velocity": np.ones((batch_size, history_steps, 2), dtype=np.float32),
        "history_mask": np.ones((batch_size, history_steps), dtype=np.uint8),
        "history_velocity_mask": np.ones(
            (batch_size, history_steps), dtype=np.uint8
        ),
        "time_step_seconds": np.full(batch_size, 0.1, dtype=np.float32),
    }


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_gru_forward_supports_different_batch_sizes(batch_size: int) -> None:
    """The deterministic head always returns every configured future position."""

    features = build_gru_features(_model_inputs(batch_size))
    model = DeterministicGRU(
        input_size=7, hidden_size=12, num_layers=2, dropout=0.1, future_steps=20
    )
    prediction = model(features)

    assert features.shape == (batch_size, 20, 7)
    assert prediction.shape == (batch_size, 20, 2)
    assert bool(torch.isfinite(prediction).all())


def test_feature_builder_broadcasts_delta_t_and_preserves_order() -> None:
    """Seven channels follow the documented position/velocity/mask/dt order."""

    inputs = _model_inputs(2, history_steps=3)
    inputs["history_position"][0, 1] = [3.0, 4.0]
    inputs["history_velocity"][0, 1] = [5.0, 6.0]
    inputs["history_mask"][0, 1] = 0
    inputs["history_velocity_mask"][0, 1] = 0
    inputs["time_step_seconds"] = np.asarray([0.1, 0.2], dtype=np.float32)

    features = build_gru_features(inputs)

    assert torch.allclose(
        features[0, 1], torch.tensor([3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.1])
    )
    assert torch.allclose(features[1, :, -1], torch.full((3,), 0.2))


def test_feature_builder_accepts_one_prediction_sample(train_archive: Path) -> None:
    """A single strict PredictionDataset sample produces an unbatched sequence."""

    sample = PredictionDataset(train_archive, fill_value=0.0)[0]
    features = build_gru_features(sample["inputs"])

    assert features.shape == (3, 7)
    assert bool(torch.isfinite(features).all())


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_position", np.nan),
        ("history_velocity", np.inf),
        ("history_mask", 2.0),
        ("history_velocity_mask", -1.0),
        ("time_step_seconds", 0.0),
    ],
)
def test_feature_builder_rejects_invalid_network_inputs(
    field: str, value: float
) -> None:
    """NaN, Inf, non-binary masks and non-positive time never reach the GRU."""

    inputs = _model_inputs(1)
    if field in ("history_mask", "history_velocity_mask"):
        inputs[field] = np.full(
            inputs[field].shape, value, dtype=np.float32
        )
    else:
        inputs[field].flat[0] = value
    with pytest.raises(ValueError):
        build_gru_features(inputs)


def test_feature_builder_rejects_forbidden_or_missing_fields() -> None:
    """The feature boundary fails closed if callers append audit truth."""

    inputs = _model_inputs(1)
    inputs["history_true_position"] = np.zeros((1, 20, 2))
    with pytest.raises(ValueError, match="PredictionDataset whitelist"):
        build_gru_features(inputs)

    del inputs["history_true_position"]
    del inputs["history_mask"]
    with pytest.raises(ValueError, match="PredictionDataset whitelist"):
        build_gru_features(inputs)


def test_normalization_uses_only_mask_valid_training_observations(
    train_archive: Path,
) -> None:
    """Loader fill values and forbidden audit truth cannot affect moments."""

    dataset = PredictionDataset(train_archive, fill_value=1000000.0)
    normalizer = PredictionNormalizer.fit(dataset)
    statistics = normalizer.statistics

    expected_position = np.asarray([[1, 2], [3, 6], [5, 10], [7, 14]], dtype=float)
    expected_velocity = np.asarray([[2, 4], [2, 4]], dtype=float)
    assert np.allclose(statistics.position_mean, expected_position.mean(axis=0))
    assert np.allclose(statistics.position_scale, expected_position.std(axis=0))
    assert np.allclose(statistics.velocity_mean, expected_velocity.mean(axis=0))
    assert np.allclose(statistics.velocity_scale, [1.0e-8, 1.0e-8])
    assert statistics.valid_position_count == 4
    assert statistics.valid_velocity_count == 2


def test_normalized_inputs_are_finite_and_invalid_steps_are_zero(
    train_archive: Path,
) -> None:
    """Masks survive normalization while arbitrary fills become neutral zeros."""

    dataset = PredictionDataset(train_archive, fill_value=-12345.0)
    normalizer = PredictionNormalizer.fit(dataset)
    sample = dataset[0]
    normalized = normalizer.normalize_inputs(sample["inputs"])

    assert set(normalized) == set(sample["inputs"])
    assert np.array_equal(normalized["history_mask"], [1, 0, 1])
    assert np.array_equal(normalized["history_velocity_mask"], [0, 0, 1])
    assert np.all(normalized["history_position"][1] == 0.0)
    assert np.all(normalized["history_velocity"][:2] == 0.0)
    assert all(np.all(np.isfinite(value)) for value in normalized.values())


def test_target_normalization_and_inverse_are_consistent(
    train_archive: Path,
) -> None:
    """Predictions are restored to original local-coordinate position scale."""

    dataset = PredictionDataset(train_archive)
    normalizer = PredictionNormalizer.fit(dataset)
    target = dataset.get_batch(slice(None))["target"]

    normalized = normalizer.normalize_target(target)
    reconstructed = normalizer.inverse_target(normalized)

    assert np.allclose(reconstructed, target, rtol=1.0e-12, atol=1.0e-12)


def test_torch_normalization_preserves_device_dtype_and_inverse(
    train_archive: Path,
) -> None:
    """Mini-batch training can normalize and restore tensors without NumPy hops."""

    dataset = PredictionDataset(train_archive)
    normalizer = PredictionNormalizer.fit(dataset)
    target = torch.tensor(dataset.get_batch(slice(None))["target"], dtype=torch.float32)

    normalized = normalizer.normalize_target(target)
    reconstructed = normalizer.inverse_position(normalized)

    assert isinstance(normalized, torch.Tensor)
    assert normalized.dtype == torch.float32
    assert normalized.device == target.device
    assert torch.allclose(reconstructed, target, rtol=1.0e-5, atol=1.0e-6)


def test_normalization_json_round_trip(train_archive: Path, tmp_path: Path) -> None:
    """Saved train moments load identically for validation and test use."""

    normalizer = PredictionNormalizer.fit(PredictionDataset(train_archive))
    output = tmp_path / "normalization.json"
    normalizer.save(output)
    loaded = PredictionNormalizer.load(output)

    assert loaded.statistics == normalizer.statistics
    assert "train" in output.read_text(encoding="utf-8")


def test_normalizer_refuses_non_train_fit(train_archive: Path) -> None:
    """Validation and test splits cannot become normalization sources."""

    dataset = PredictionDataset(train_archive)
    with pytest.raises(ValueError, match="only be fitted on the train"):
        PredictionNormalizer.fit(dataset, split_name="validation")
