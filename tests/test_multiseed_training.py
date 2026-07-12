"""Regression tests for experiment-02 multi-seed training artefacts."""

from __future__ import annotations

import csv
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from thesis_experiment.prediction.gru import DeterministicGRU


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_prediction_baseline.py"
SPEC = importlib.util.spec_from_file_location("experiment_02_train_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)


def _training_archive(path: Path) -> Path:
    """Create a tiny archive containing masked observations and audit fields."""

    np.savez_compressed(
        str(path),
        history_position=np.asarray(
            [
                [[1.0, 2.0], [np.nan, np.nan], [1.0, 2.0]],
                [[1.0, 2.0], [1.0, 2.0], [np.nan, np.nan]],
            ],
            dtype=np.float64,
        ),
        history_velocity=np.asarray(
            [
                [[np.nan, np.nan], [np.nan, np.nan], [3.0, 4.0]],
                [[np.nan, np.nan], [3.0, 4.0], [np.nan, np.nan]],
            ],
            dtype=np.float64,
        ),
        history_mask=np.asarray([[1, 0, 1], [1, 1, 0]], dtype=np.uint8),
        future_position=np.zeros((2, 2, 2), dtype=np.float64),
        time_step_seconds=np.full(2, 0.1, dtype=np.float64),
        scene_id=np.asarray([1, 2], dtype=np.int64),
        episode_id=np.asarray([10, 20], dtype=np.int64),
        sample_start_index=np.asarray([0, 5], dtype=np.int64),
        history_true_position=np.full((2, 3, 2), 9999.0),
        trajectory_type=np.asarray([0, 4], dtype=np.int8),
    )
    return path


def _script_config(seeds=(11, 22, 33)):
    """Return only the validated config attributes used by script helpers."""

    return SimpleNamespace(
        seeds=tuple(seeds),
        fill_value=-17.25,
        normalization_epsilon=0.75,
        dataset=SimpleNamespace(
            train_split="train.npz",
            validation_split="validation.npz",
        ),
        gru=SimpleNamespace(
            input_size=7,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            future_steps=20,
        ),
        training=SimpleNamespace(loss="smooth_l1"),
    )


def test_training_script_applies_fill_value_and_normalization_epsilon(
    tmp_path: Path,
) -> None:
    """Non-default config values change loader fills and fitted scales."""

    archive = _training_archive(tmp_path / "train.npz")
    # The helper loads both splits through the exact same safe boundary.
    validation = tmp_path / "validation.npz"
    validation.write_bytes(archive.read_bytes())
    config = _script_config()

    train_dataset, validation_dataset = TRAIN_SCRIPT._load_training_datasets(
        config, tmp_path
    )
    assert train_dataset.fill_value == config.fill_value
    assert validation_dataset.fill_value == config.fill_value
    batch = train_dataset.get_batch(slice(None))["inputs"]
    assert np.all(batch["history_position"][0, 1] == config.fill_value)
    assert np.all(batch["history_velocity"][0, 0] == config.fill_value)

    normalizer = TRAIN_SCRIPT._fit_training_normalizer(config, train_dataset)
    # Every valid value is constant, so the configured floor is observable.
    assert normalizer.statistics.position_scale == (0.75, 0.75)
    assert normalizer.statistics.velocity_scale == (0.75, 0.75)


def test_all_seed_checkpoints_and_training_summary_are_preserved(
    tmp_path: Path,
) -> None:
    """Every configured seed gets best/last weights and an independent row."""

    archive = _training_archive(tmp_path / "train.npz")
    validation = tmp_path / "validation.npz"
    validation.write_bytes(archive.read_bytes())
    config = _script_config()
    train_dataset, _ = TRAIN_SCRIPT._load_training_datasets(config, tmp_path)
    normalizer = TRAIN_SCRIPT._fit_training_normalizer(config, train_dataset)
    checkpoint_directory = tmp_path / "output" / "checkpoints"
    checkpoint_directory.mkdir(parents=True)

    records = []
    losses = (0.3, 0.1, 0.2)
    for index, (seed, validation_loss) in enumerate(zip(config.seeds, losses)):
        torch.manual_seed(seed)
        model = DeterministicGRU(
            input_size=7,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            future_steps=20,
        )
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        result = {
            "best_state": state,
            "last_state": state,
            "best_epoch": 1,
            "stopped_epoch": 2,
            "best_validation_loss": validation_loss,
        }
        record = TRAIN_SCRIPT._save_seed_checkpoints(
            config=config,
            normalizer=normalizer,
            result=result,
            seed=seed,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            training_time_seconds=0.01 * (index + 1),
            checkpoint_directory=checkpoint_directory,
        )
        record["seed_index"] = index
        records.append(record)

    logger = logging.getLogger("test_multiseed_training")
    TRAIN_SCRIPT._finalize_checkpoints(
        config=config,
        normalizer=normalizer,
        per_seed_results=records,
        checkpoint_directory=checkpoint_directory,
        output_directory=checkpoint_directory.parent,
        logger=logger,
    )

    for seed in config.seeds:
        for filename in ("best.pt", "last.pt"):
            path = checkpoint_directory / "seed_{}".format(seed) / filename
            assert path.is_file()
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            assert checkpoint["seed"] == seed

    summary_path = checkpoint_directory.parent / "seed_training_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in summary] == list(config.seeds)
    assert [float(row["validation_loss"]) for row in summary] == list(losses)
    assert all(float(row["training_time"]) > 0.0 for row in summary)
    assert all(int(row["parameter_count"]) > 0 for row in summary)
    assert all(row["best_checkpoint"].endswith("/best.pt") for row in summary)
    assert all(row["last_checkpoint"].endswith("/last.pt") for row in summary)

    # Top-level best is a deployment convenience copy of the validation-best
    # seed, while all three independent records remain intact.
    top_best = torch.load(
        str(checkpoint_directory / "best.pt"),
        map_location="cpu",
        weights_only=True,
    )
    top_last = torch.load(
        str(checkpoint_directory / "last.pt"),
        map_location="cpu",
        weights_only=True,
    )
    assert top_best["seed"] == 22
    assert top_last["seed"] == 33
