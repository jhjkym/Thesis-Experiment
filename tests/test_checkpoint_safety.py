"""Regression tests for safe Experiment-02 checkpoints and exact resume."""

from __future__ import annotations

import io
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from thesis_experiment.prediction.gru import DeterministicGRU
from thesis_experiment.prediction import pipeline


def _normalization() -> Dict[str, Any]:
    """Return valid train-only statistics for a tiny model checkpoint."""

    return {
        "schema_version": 1,
        "source_split": "train",
        "position_mean": [0.25, -0.5],
        "position_scale": [1.5, 2.0],
        "velocity_mean": [0.1, -0.2],
        "velocity_scale": [0.7, 0.8],
        "valid_position_count": 12,
        "valid_velocity_count": 10,
    }


def _model_checkpoint(model: DeterministicGRU, epoch: int) -> Dict[str, Any]:
    return pipeline.build_model_checkpoint(
        model_state=model.state_dict(),
        gru_config={
            "input_size": 7,
            "hidden_size": 5,
            "num_layers": 1,
            "dropout": 0.0,
            "future_steps": 3,
        },
        normalization=_normalization(),
        seed=17,
        epoch=epoch,
        validation_loss=0.125,
        loss_name="smooth_l1",
    )


@pytest.mark.parametrize("filename,epoch", [("best.pt", 2), ("last.pt", 3)])
def test_model_checkpoint_loads_with_weights_only(
    tmp_path: Path, filename: str, epoch: int
) -> None:
    """Deployment best/last checkpoints contain no unsafe pickle globals."""

    pipeline.set_global_seed(17)
    model = DeterministicGRU(hidden_size=5, future_steps=3)
    path = tmp_path / filename
    torch.save(_model_checkpoint(model, epoch), str(path))

    # This is deliberately a direct strict load, not merely the project
    # wrapper, so a future unsafe checkpoint field causes this test to fail.
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        pytest.skip("installed PyTorch predates the weights_only argument")
    loaded = pipeline.safe_torch_load(path, map_location="cpu")
    restored, _ = pipeline.load_model_from_checkpoint(raw, torch.device("cpu"))

    assert int(loaded["epoch"]) == epoch
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0.0, atol=0.0)


def _contains_numpy_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_numpy_array(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numpy_array(item) for item in value)
    return False


def test_active_resume_checkpoint_is_weights_only_safe(tmp_path: Path) -> None:
    """A resume blob with optimizer and RNG state is strict-load compatible."""

    pipeline.set_global_seed(23)
    model = DeterministicGRU(hidden_size=5, future_steps=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(torch.randn(2, 4, 7)).square().mean().backward()
    optimizer.step()
    generator = torch.Generator().manual_seed(23)

    active_state = {
        "start_epoch": 2,
        "best_validation_loss": 0.25,
        "best_epoch": 1,
        "epochs_without_improvement": 0,
        "optimizer_state": optimizer.state_dict(),
        "model_state": model.state_dict(),
        "best_model_state": model.state_dict(),
    }
    active_state.update(pipeline.capture_safe_rng_state(generator.get_state()))
    raw_snapshot = {
        "completed_seed_indices": [],
        "history_rows": [{"epoch": 1, "validation_loss": 0.25}],
        "active_seed_index": 0,
        "active_state": active_state,
    }
    assert not _contains_numpy_array(active_state)

    path = tmp_path / "training_state.pt"
    torch.save(pipeline.to_weights_only_safe(raw_snapshot), str(path))
    try:
        strict = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        pytest.skip("installed PyTorch predates the weights_only argument")
    assert "active_state" in strict

    loaded = pipeline.safe_torch_load(path, map_location="cpu")
    saved = loaded["active_state"]
    resume = pipeline.TrainResumeState(
        start_epoch=int(saved["start_epoch"]),
        best_validation_loss=float(saved["best_validation_loss"]),
        epochs_without_improvement=int(saved["epochs_without_improvement"]),
        optimizer_state=saved["optimizer_state"],
        model_state=saved["model_state"],
        best_model_state=saved["best_model_state"],
        torch_rng_state=saved["torch_rng_state"],
        numpy_rng_state=saved["numpy_rng_state"],
        python_rng_state=saved["python_rng_state"],
        loader_generator_state=saved["loader_generator_state"],
        best_epoch=int(saved["best_epoch"]),
    )
    assert resume.start_epoch == 2
    assert isinstance(resume.numpy_rng_state["keys"], torch.Tensor)


def _training_tensors() -> Tuple[Any, Any, Any, Any]:
    generator = torch.Generator().manual_seed(101)
    train_x = torch.randn(12, 6, 7, generator=generator)
    train_y = torch.randn(12, 3, 2, generator=generator)
    validation_x = torch.randn(6, 6, 7, generator=generator)
    validation_y = torch.randn(6, 3, 2, generator=generator)
    return train_x, train_y, validation_x, validation_y


def _train(
    epochs: int,
    *,
    resume: Optional[pipeline.TrainResumeState] = None,
    capture_epoch: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Run a tiny deterministic job and optionally strict-round-trip a snapshot."""

    pipeline.set_global_seed(37)
    train_x, train_y, validation_x, validation_y = _training_tensors()
    model = DeterministicGRU(hidden_size=5, future_steps=3)
    loader_generator = torch.Generator().manual_seed(37)
    captured: Dict[str, Any] = {}

    def on_epoch_end(
        record,
        *,
        model_state,
        best_model_state,
        optimizer_state,
        epochs_without_improvement,
        loader_generator_state,
        best_epoch,
    ) -> None:
        if record.epoch != capture_epoch:
            return
        active = {
            "start_epoch": record.epoch + 1,
            "best_validation_loss": record.best_validation_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "optimizer_state": optimizer_state,
            "model_state": model_state,
            "best_model_state": best_model_state,
        }
        active.update(pipeline.capture_safe_rng_state(loader_generator_state))
        buffer = io.BytesIO()
        torch.save(pipeline.to_weights_only_safe({"active_state": active}), buffer)
        buffer.seek(0)
        captured.update(pipeline.safe_torch_load(buffer)["active_state"])

    result = pipeline.train_gru(
        model=model,
        train_features=train_x,
        train_target=train_y,
        validation_features=validation_x,
        validation_target=validation_y,
        epochs=epochs,
        batch_size=4,
        validation_batch_size=3,
        learning_rate=0.005,
        weight_decay=0.0,
        loss_name="mse",
        gradient_clip_norm=1.0,
        early_stopping_patience=(
            epochs if early_stopping_patience is None else early_stopping_patience
        ),
        early_stopping_min_delta=early_stopping_min_delta,
        num_workers=0,
        generator=loader_generator,
        device=torch.device("cpu"),
        on_epoch_end=on_epoch_end,
        resume=resume,
    )
    return result, captured or None


def test_strict_checkpoint_resume_matches_continuous_training_bitwise() -> None:
    """Strict-loaded interrupted training is bit-identical to a straight run."""

    continuous, _ = _train(epochs=3)
    _, saved = _train(epochs=3, capture_epoch=1)
    assert saved is not None
    resume = pipeline.TrainResumeState(
        start_epoch=int(saved["start_epoch"]),
        best_validation_loss=float(saved["best_validation_loss"]),
        epochs_without_improvement=int(saved["epochs_without_improvement"]),
        optimizer_state=saved["optimizer_state"],
        model_state=saved["model_state"],
        best_model_state=saved["best_model_state"],
        torch_rng_state=saved["torch_rng_state"],
        numpy_rng_state=saved["numpy_rng_state"],
        python_rng_state=saved["python_rng_state"],
        loader_generator_state=saved["loader_generator_state"],
        best_epoch=int(saved["best_epoch"]),
    )
    resumed, _ = _train(epochs=3, resume=resume)

    assert [record.epoch for record in resumed["history"]] == [2, 3]
    for key in continuous["last_state"]:
        torch.testing.assert_close(
            continuous["last_state"][key],
            resumed["last_state"][key],
            rtol=0.0,
            atol=0.0,
        )
    for key in continuous["best_state"]:
        torch.testing.assert_close(
            continuous["best_state"][key],
            resumed["best_state"][key],
            rtol=0.0,
            atol=0.0,
        )


def test_resume_after_terminal_early_stop_does_not_train_extra_epoch() -> None:
    """A snapshot saved just before the stop check remains terminal on resume."""

    continuous, _ = _train(
        epochs=4,
        capture_epoch=2,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0e9,
    )
    assert continuous["stopped_epoch"] == 2

    _, saved = _train(
        epochs=4,
        capture_epoch=2,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0e9,
    )
    assert saved is not None
    resume = pipeline.TrainResumeState(
        start_epoch=int(saved["start_epoch"]),
        best_validation_loss=float(saved["best_validation_loss"]),
        epochs_without_improvement=int(saved["epochs_without_improvement"]),
        optimizer_state=saved["optimizer_state"],
        model_state=saved["model_state"],
        best_model_state=saved["best_model_state"],
        torch_rng_state=saved["torch_rng_state"],
        numpy_rng_state=saved["numpy_rng_state"],
        python_rng_state=saved["python_rng_state"],
        loader_generator_state=saved["loader_generator_state"],
        cuda_rng_states=saved["cuda_rng_states"],
        best_epoch=int(saved["best_epoch"]),
    )
    resumed, _ = _train(
        epochs=4,
        resume=resume,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0e9,
    )

    assert resumed["history"] == []
    assert resumed["stopped_epoch"] == continuous["stopped_epoch"]
    assert resumed["best_validation_loss"] == continuous["best_validation_loss"]
    for key in continuous["last_state"]:
        torch.testing.assert_close(
            continuous["last_state"][key],
            resumed["last_state"][key],
            rtol=0.0,
            atol=0.0,
        )
    for key in continuous["best_state"]:
        torch.testing.assert_close(
            continuous["best_state"][key],
            resumed["best_state"][key],
            rtol=0.0,
            atol=0.0,
        )


def test_numpy_and_python_rng_serialization_restores_exact_sequence() -> None:
    pipeline.set_global_seed(91)
    numpy_state = pipeline.serialize_numpy_rng_state()
    python_state = pipeline.serialize_python_rng_state()
    expected_numpy = np.random.random_sample(8)
    expected_python = [random.random() for _ in range(8)]

    np.random.set_state(pipeline.deserialize_numpy_rng_state(numpy_state))
    random.setstate(pipeline.deserialize_python_rng_state(python_state))
    np.testing.assert_array_equal(np.random.random_sample(8), expected_numpy)
    assert [random.random() for _ in range(8)] == expected_python


def test_torch_thread_configuration_is_applied_in_fresh_process() -> None:
    """A non-default configured value is reflected by Torch getters."""

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    code = (
        "from thesis_experiment.prediction.pipeline import apply_torch_thread_config; "
        "import torch; "
        "result=apply_torch_thread_config(2, 2); "
        "assert result == {'torch_num_threads': 2, "
        "'torch_num_interop_threads': 2}; "
        "assert torch.get_num_threads() == 2; "
        "assert torch.get_num_interop_threads() == 2"
    )
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(source_root) + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(project_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert completed.returncode == 0, completed.stderr
