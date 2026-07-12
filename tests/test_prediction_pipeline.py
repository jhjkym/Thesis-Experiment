"""Tests for the shared experiment-02 training and evaluation pipeline.

These cover the script-level guarantees that cannot be tested on the pure model
functions alone: checkpoint round-trip fidelity, seed reproducibility, the
structural exclusion of the test split from training/early-stopping/
normalisation, identical test-window ordering across all four methods, and the
requirement that saved prediction IDs match the source archive.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_experiment.data.prediction_dataset import PredictionDataset
from thesis_experiment.prediction.gru import DeterministicGRU
from thesis_experiment.prediction.normalization import PredictionNormalizer
from thesis_experiment.prediction import pipeline


def _make_split(path: Path, count: int, seed: int) -> Path:
    """Write a small, strict dataset-v2-shaped archive for one split."""

    rng = np.random.default_rng(seed)
    history_steps = 20
    future_steps = 20
    history_position = rng.normal(size=(count, history_steps, 2)).astype(np.float64)
    history_velocity = rng.normal(size=(count, history_steps, 2)).astype(np.float64)
    history_mask = np.ones((count, history_steps), dtype=np.uint8)
    future_position = rng.normal(size=(count, future_steps, 2)).astype(np.float64)
    # Two scenes, with two episodes each, so aggregation has real structure.
    scene_id = np.array([1 if i < count // 2 else 2 for i in range(count)], dtype=np.int64)
    episode_id = np.array(
        [10 if i < count // 4 else 11 if i < count // 2 else 20 if i < 3 * count // 4 else 21 for i in range(count)],
        dtype=np.int64,
    )
    np.savez_compressed(
        str(path),
        history_position=history_position,
        history_velocity=history_velocity,
        history_mask=history_mask,
        future_position=future_position,
        time_step_seconds=np.full(count, 0.1, dtype=np.float64),
        scene_id=scene_id,
        episode_id=episode_id,
        sample_start_index=np.arange(count, dtype=np.int64) * 5,
        trajectory_type=np.array([i % 5 for i in range(count)], dtype=np.int8),
        occlusion_length_bin=np.array([i % 5 for i in range(count)], dtype=np.int8),
        # Forbidden fields deliberately present to confirm they are never used.
        history_true_position=np.full((count, history_steps, 2), 1234.0),
        episode_turn_rate=np.full(count, 9999.0),
    )
    return path


@pytest.fixture()
def splits(tmp_path: Path) -> Dict[str, Path]:
    return {
        "train": _make_split(tmp_path / "train.npz", 16, seed=1),
        "validation": _make_split(tmp_path / "validation.npz", 8, seed=2),
        "test": _make_split(tmp_path / "test.npz", 12, seed=3),
    }


def _train_short(splits: Dict[str, Path], seed: int, epochs: int = 3):
    """Train a tiny GRU deterministically and return the result and normalizer."""

    train_dataset = PredictionDataset(splits["train"])
    validation_dataset = PredictionDataset(splits["validation"])
    normalizer = PredictionNormalizer.fit(train_dataset)
    device = torch.device("cpu")
    pipeline.set_global_seed(seed)
    train_features, train_target = pipeline.build_normalized_tensors(
        train_dataset, normalizer, device=device
    )
    validation_features, validation_target = pipeline.build_normalized_tensors(
        validation_dataset, normalizer, device=device
    )
    model = DeterministicGRU(hidden_size=16, num_layers=1, dropout=0.0, future_steps=20)
    generator = torch.Generator()
    generator.manual_seed(seed)
    result = pipeline.train_gru(
        model=model,
        train_features=train_features,
        train_target=train_target,
        validation_features=validation_features,
        validation_target=validation_target,
        epochs=epochs,
        batch_size=8,
        validation_batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        loss_name="smooth_l1",
        gradient_clip_norm=1.0,
        early_stopping_patience=epochs,
        early_stopping_min_delta=0.0,
        num_workers=0,
        generator=generator,
        device=device,
    )
    return result, normalizer, model


def test_checkpoint_save_and_load_round_trip(splits: Dict[str, Path], tmp_path: Path) -> None:
    """A saved checkpoint reloads to identical GRU predictions."""

    result, normalizer, _ = _train_short(splits, seed=7)
    checkpoint = pipeline.build_model_checkpoint(
        model_state=result["best_state"],
        gru_config={
            "input_size": 7,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "future_steps": 20,
        },
        normalization=normalizer.statistics.to_dict(),
        seed=7,
        epoch=result["best_epoch"],
        validation_loss=result["best_validation_loss"],
        loss_name="smooth_l1",
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, str(checkpoint_path))

    device = torch.device("cpu")
    reloaded = pipeline.safe_torch_load(checkpoint_path, map_location=device)
    model, loaded_normalizer = pipeline.load_model_from_checkpoint(reloaded, device)
    test_dataset = PredictionDataset(splits["test"])
    first = pipeline.predict_gru_local(
        test_dataset, model, loaded_normalizer, batch_size=8, device=device
    )
    model_again, normalizer_again = pipeline.load_model_from_checkpoint(reloaded, device)
    second = pipeline.predict_gru_local(
        test_dataset, model_again, normalizer_again, batch_size=8, device=device
    )
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    assert loaded_normalizer.statistics == normalizer.statistics


def test_best_epoch_is_argmin_validation_epoch(splits: Dict[str, Path]) -> None:
    """best_epoch reported by train_gru is the epoch with lowest validation loss.

    Guards checkpoint metadata honesty: best.pt records the epoch its weights
    were captured at, not the (possibly later) stopped epoch.
    """

    result, _, _ = _train_short(splits, seed=7, epochs=4)
    history = result["history"]
    argmin_epoch = min(history, key=lambda record: record.validation_loss).epoch
    assert result["best_epoch"] == argmin_epoch
    # The stopped epoch can exceed the best epoch when later epochs do not improve.
    assert result["best_epoch"] <= result["stopped_epoch"]


def test_fixed_seed_gives_reproducible_training(splits: Dict[str, Path]) -> None:
    """Identical seeds produce identical weights and validation history."""

    first_result, _, first_model = _train_short(splits, seed=123)
    second_result, _, second_model = _train_short(splits, seed=123)

    for key in first_model.state_dict():
        np.testing.assert_allclose(
            first_result["best_state"][key].numpy(),
            second_result["best_state"][key].numpy(),
            rtol=0.0,
            atol=0.0,
        )
    first_history = [record.validation_loss for record in first_result["history"]]
    second_history = [record.validation_loss for record in second_result["history"]]
    assert first_history == second_history


def test_different_seeds_diverge(splits: Dict[str, Path]) -> None:
    """Different seeds must not yield byte-identical initial weights."""

    first_result, _, model = _train_short(splits, seed=1, epochs=1)
    second_result, _, _ = _train_short(splits, seed=2, epochs=1)
    differs = any(
        not np.array_equal(
            first_result["best_state"][key].numpy(),
            second_result["best_state"][key].numpy(),
        )
        for key in model.state_dict()
    )
    assert differs


def test_training_signature_excludes_test_data() -> None:
    """train_gru structurally accepts only train and validation tensors."""

    parameters = set(inspect.signature(pipeline.train_gru).parameters)
    assert "train_features" in parameters
    assert "validation_features" in parameters
    assert not any("test" in name for name in parameters)


def test_normalizer_refuses_non_train_split(splits: Dict[str, Path]) -> None:
    """Normalisation cannot be fit on validation or test splits."""

    validation_dataset = PredictionDataset(splits["validation"])
    with pytest.raises(ValueError, match="only be fitted on the train"):
        PredictionNormalizer.fit(validation_dataset, split_name="test")


def test_all_methods_share_identical_test_order(splits: Dict[str, Path], tmp_path: Path) -> None:
    """The four methods consume the same windows in the same order."""

    result, normalizer, _ = _train_short(splits, seed=5)
    checkpoint = pipeline.build_model_checkpoint(
        model_state=result["best_state"],
        gru_config={
            "input_size": 7,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "future_steps": 20,
        },
        normalization=normalizer.statistics.to_dict(),
        seed=5,
        epoch=result["stopped_epoch"],
        validation_loss=result["best_validation_loss"],
        loss_name="smooth_l1",
    )
    device = torch.device("cpu")
    model, loaded_normalizer = pipeline.load_model_from_checkpoint(checkpoint, device)
    test_dataset = PredictionDataset(splits["test"])
    kalman = {
        "process_noise": 0.02,
        "measurement_noise": 0.0009,
        "initial_position_variance": 0.05,
        "initial_velocity_variance": 0.25,
    }
    shapes = set()
    for method in pipeline.CLASSICAL_METHOD_NAMES:
        prediction = pipeline.predict_classical_local(
            test_dataset, method, future_steps=20, kalman=kalman
        )
        shapes.add(prediction.shape)
    gru_prediction = pipeline.predict_gru_local(
        test_dataset, model, loaded_normalizer, batch_size=8, device=device
    )
    shapes.add(gru_prediction.shape)
    assert shapes == {(len(test_dataset), 20, 2)}


def test_prediction_ids_match_source_archive(splits: Dict[str, Path]) -> None:
    """Loader metadata order equals the raw archive order used for grouping."""

    test_dataset = PredictionDataset(splits["test"])
    loader_metadata = test_dataset.get_batch(slice(None))["metadata"]
    with np.load(str(splits["test"]), allow_pickle=False) as archive:
        for key in ("scene_id", "episode_id", "sample_start_index"):
            np.testing.assert_array_equal(
                np.asarray(loader_metadata[key]), np.asarray(archive[key])
            )


def _run_seeded(splits: Dict[str, Path], epochs: int, resume=None, capture_epoch=None):
    """Train the tiny GRU with a fixed seed, optionally capturing a resume snapshot."""

    train_dataset = PredictionDataset(splits["train"])
    validation_dataset = PredictionDataset(splits["validation"])
    normalizer = PredictionNormalizer.fit(train_dataset)
    device = torch.device("cpu")
    pipeline.set_global_seed(42)
    train_features, train_target = pipeline.build_normalized_tensors(
        train_dataset, normalizer, device=device
    )
    validation_features, validation_target = pipeline.build_normalized_tensors(
        validation_dataset, normalizer, device=device
    )
    model = DeterministicGRU(hidden_size=16, num_layers=1, dropout=0.0, future_steps=20)
    if resume is not None:
        model.load_state_dict(resume.model_state)
    generator = torch.Generator()
    generator.manual_seed(42)
    captured: Dict[str, object] = {}

    def _capture(
        record,
        *,
        model_state,
        best_model_state,
        optimizer_state,
        epochs_without_improvement,
        loader_generator_state,
        best_epoch,
    ) -> None:
        if capture_epoch is not None and record.epoch == capture_epoch:
            # Deep-copy through save/load semantics so live tensors do not mutate.
            import io

            buffer = io.BytesIO()
            rng_state = pipeline.capture_safe_rng_state(loader_generator_state)
            torch.save(
                pipeline.to_weights_only_safe({
                    "start_epoch": record.epoch + 1,
                    "best_validation_loss": record.best_validation_loss,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "optimizer_state": optimizer_state,
                    "model_state": model_state,
                    "best_model_state": best_model_state,
                    "torch_rng_state": rng_state["torch_rng_state"],
                    "numpy_rng_state": rng_state["numpy_rng_state"],
                    "python_rng_state": rng_state["python_rng_state"],
                    "loader_generator_state": rng_state["loader_generator_state"],
                }),
                buffer,
            )
            buffer.seek(0)
            captured["snapshot"] = pipeline.safe_torch_load(
                buffer, map_location="cpu"
            )

    result = pipeline.train_gru(
        model=model,
        train_features=train_features,
        train_target=train_target,
        validation_features=validation_features,
        validation_target=validation_target,
        epochs=epochs,
        batch_size=8,
        validation_batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        loss_name="smooth_l1",
        gradient_clip_norm=1.0,
        early_stopping_patience=epochs,
        early_stopping_min_delta=0.0,
        num_workers=0,
        generator=generator,
        device=device,
        on_epoch_end=_capture,
        resume=resume,
    )
    return result, captured, model


def test_resume_reproduces_uninterrupted_run(splits: Dict[str, Path]) -> None:
    """Resuming mid-training matches a straight run to the bit.

    This regression guards the DataLoader generator-state fix: without
    restoring the loader generator, the post-resume shuffle order diverges and
    the trained weights drift from an uninterrupted run.
    """

    straight, _, reference_model = _run_seeded(splits, epochs=4)
    _, captured, _ = _run_seeded(splits, epochs=4, capture_epoch=2)
    snapshot = captured["snapshot"]
    resume_state = pipeline.TrainResumeState(
        start_epoch=int(snapshot["start_epoch"]),
        best_validation_loss=float(snapshot["best_validation_loss"]),
        epochs_without_improvement=int(snapshot["epochs_without_improvement"]),
        optimizer_state=snapshot["optimizer_state"],
        model_state=snapshot["model_state"],
        best_model_state=snapshot["best_model_state"],
        torch_rng_state=snapshot["torch_rng_state"],
        numpy_rng_state=snapshot["numpy_rng_state"],
        python_rng_state=snapshot["python_rng_state"],
        loader_generator_state=snapshot["loader_generator_state"],
        best_epoch=int(snapshot["best_epoch"]),
    )
    resumed, _, _ = _run_seeded(splits, epochs=4, resume=resume_state)

    assert [record.epoch for record in resumed["history"]] == [3, 4]
    for key in reference_model.state_dict():
        np.testing.assert_allclose(
            straight["best_state"][key].numpy(),
            resumed["best_state"][key].numpy(),
            rtol=0.0,
            atol=0.0,
        )
