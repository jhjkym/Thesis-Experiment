#!/usr/bin/env python
"""Train the deterministic GRU trajectory-prediction baseline (experiment 02).

The classical baselines (constant position, constant velocity, CV Kalman
filter) require no training; this script fits the single learnable baseline and
writes every artefact the evaluation stage consumes: the train-only
normalisation statistics, per-epoch CSV history, a copy of the resolved config,
and the best/last GRU checkpoints.  Model selection and early stopping read the
validation split only -- the test split is never opened here.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thesis_experiment.config_experiment_02 import (
    Experiment02Config,
    load_experiment_02_config,
)
from thesis_experiment.data.prediction_dataset import PredictionDataset
from thesis_experiment.prediction.gru import DeterministicGRU
from thesis_experiment.prediction.normalization import PredictionNormalizer
from thesis_experiment.prediction.pipeline import (
    EpochRecord,
    TrainResumeState,
    apply_torch_thread_config,
    build_model_checkpoint,
    build_normalized_tensors,
    capture_safe_rng_state,
    safe_torch_load,
    select_device,
    set_global_seed,
    to_weights_only_safe,
    train_gru,
)


CHECKPOINT_DIR_NAME = "checkpoints"
HISTORY_FILENAME = "training_history.csv"
NORMALIZATION_FILENAME = "normalization.json"
RESUME_FILENAME = "resume_state.pt"
SEED_SUMMARY_FILENAME = "seed_training_summary.csv"


def _arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experiment 02 GRU baseline.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from checkpoints/resume_state.pt.",
    )
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("experiment_02_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Rewrite the full CSV history so a resumed run stays consistent."""

    fields = [
        "seed",
        "epoch",
        "train_loss",
        "validation_loss",
        "best_validation_loss",
        "is_best",
        "learning_rate",
        "seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_seed_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write one explicit training record per configured random seed."""

    fields = [
        "seed",
        "best_epoch",
        "validation_loss",
        "training_time",
        "parameter_count",
        "best_checkpoint",
        "last_checkpoint",
        "stopped_epoch",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _load_training_datasets(
    config: Experiment02Config, dataset_directory: Path
) -> Tuple[PredictionDataset, PredictionDataset]:
    """Load train/validation through the strict whitelist using configured fill."""

    train_dataset = PredictionDataset(
        dataset_directory / config.dataset.train_split,
        fill_value=config.fill_value,
    )
    validation_dataset = PredictionDataset(
        dataset_directory / config.dataset.validation_split,
        fill_value=config.fill_value,
    )
    return train_dataset, validation_dataset


def _fit_training_normalizer(
    config: Experiment02Config, train_dataset: PredictionDataset
) -> PredictionNormalizer:
    """Fit train-only statistics using the configured minimum scale."""

    return PredictionNormalizer.fit(
        train_dataset,
        minimum_scale=config.normalization_epsilon,
    )


def _gru_config_dict(config: Experiment02Config) -> Dict[str, Any]:
    """Return the serializable GRU architecture stored in every checkpoint."""

    return {
        "input_size": config.gru.input_size,
        "hidden_size": config.gru.hidden_size,
        "num_layers": config.gru.num_layers,
        "dropout": config.gru.dropout,
        "future_steps": config.gru.future_steps,
    }


def _save_seed_checkpoints(
    *,
    config: Experiment02Config,
    normalizer: PredictionNormalizer,
    result: Dict[str, Any],
    seed: int,
    parameter_count: int,
    training_time_seconds: float,
    checkpoint_directory: Path,
) -> Dict[str, Any]:
    """Persist best/last model weights for one seed and return safe metadata.

    The returned mapping deliberately contains no model or optimizer state.  It
    can therefore be stored in the multi-seed resume manifest without retaining
    duplicate tensors for every completed seed.
    """

    seed_directory = checkpoint_directory / "seed_{}".format(seed)
    seed_directory.mkdir(parents=True, exist_ok=True)
    normalization = normalizer.statistics.to_dict()
    gru_config = _gru_config_dict(config)

    best_path = seed_directory / "best.pt"
    last_path = seed_directory / "last.pt"
    best_checkpoint = build_model_checkpoint(
        model_state=result["best_state"],
        gru_config=gru_config,
        normalization=normalization,
        seed=seed,
        epoch=int(result["best_epoch"]),
        validation_loss=float(result["best_validation_loss"]),
        loss_name=config.training.loss,
    )
    last_checkpoint = build_model_checkpoint(
        model_state=result["last_state"],
        gru_config=gru_config,
        normalization=normalization,
        seed=seed,
        epoch=int(result["stopped_epoch"]),
        validation_loss=float(result["best_validation_loss"]),
        loss_name=config.training.loss,
    )
    torch.save(best_checkpoint, str(best_path))
    torch.save(last_checkpoint, str(last_path))

    output_directory = checkpoint_directory.parent
    return {
        "seed": int(seed),
        "best_epoch": int(result["best_epoch"]),
        "validation_loss": float(result["best_validation_loss"]),
        "training_time": float(training_time_seconds),
        "parameter_count": int(parameter_count),
        "best_checkpoint": str(best_path.relative_to(output_directory)),
        "last_checkpoint": str(last_path.relative_to(output_directory)),
        "stopped_epoch": int(result["stopped_epoch"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arguments(argv)
    config_path = _project_path(args.config)
    config = load_experiment_02_config(config_path)
    applied_threads = apply_torch_thread_config(
        config.runtime.torch_num_threads,
        config.runtime.torch_num_interop_threads,
    )

    output_directory = _project_path(config.output_directory)
    checkpoint_directory = output_directory / CHECKPOINT_DIR_NAME
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    logger = _logger(output_directory / "run.log")
    device = select_device()
    logger.info("Starting experiment 02 training with config %s", config_path)
    logger.info("Device: %s | torch %s", device, torch.__version__)
    logger.info(
        "PyTorch threads: intra_op=%d | inter_op=%d",
        applied_threads["torch_num_threads"],
        applied_threads["torch_num_interop_threads"],
    )
    logger.info("Configured seeds: %s", list(config.seeds))

    shutil.copyfile(str(config_path), str(output_directory / "config.yaml"))

    dataset_directory = _project_path(config.dataset.directory)
    train_dataset, validation_dataset = _load_training_datasets(
        config, dataset_directory
    )
    logger.info(
        "Loaded train=%d and validation=%d windows with fill_value=%g "
        "(test split not opened here)",
        len(train_dataset),
        len(validation_dataset),
        config.fill_value,
    )

    # Normalisation is fit on the train split only and reused for validation.
    normalizer = _fit_training_normalizer(config, train_dataset)
    normalizer.save(output_directory / NORMALIZATION_FILENAME)
    logger.info(
        "Fitted train-only normalization with minimum_scale=%g "
        "(valid position steps=%d, velocity steps=%d)",
        config.normalization_epsilon,
        normalizer.statistics.valid_position_count,
        normalizer.statistics.valid_velocity_count,
    )

    resume_path = checkpoint_directory / RESUME_FILENAME
    resume_blob: Dict[str, Any] = {}
    if args.resume and resume_path.is_file():
        resume_blob = safe_torch_load(resume_path, map_location="cpu")
        logger.info(
            "Resuming from %s (completed seeds=%s, active seed index=%s, epoch=%s)",
            resume_path,
            resume_blob.get("completed_seed_indices"),
            resume_blob.get("active_seed_index"),
            resume_blob.get("active_epoch"),
        )
    elif args.resume:
        logger.info("No resume state found at %s; starting fresh", resume_path)

    history_rows: List[Dict[str, Any]] = list(resume_blob.get("history_rows", []))
    completed_seed_indices = set(resume_blob.get("completed_seed_indices", []))
    per_seed_results: List[Dict[str, Any]] = list(resume_blob.get("per_seed_results", []))

    validation_features_cache = None

    for seed_index, seed in enumerate(config.seeds):
        if seed_index in completed_seed_indices:
            logger.info("Seed index %d (seed=%d) already complete; skipping", seed_index, seed)
            continue

        set_global_seed(seed)
        train_features, train_target = build_normalized_tensors(
            train_dataset, normalizer, dtype=torch.float32, device=device
        )
        if validation_features_cache is None:
            validation_features_cache = build_normalized_tensors(
                validation_dataset, normalizer, dtype=torch.float32, device=device
            )
        validation_features, validation_target = validation_features_cache

        model = DeterministicGRU(
            input_size=config.gru.input_size,
            hidden_size=config.gru.hidden_size,
            num_layers=config.gru.num_layers,
            dropout=config.gru.dropout,
            future_steps=config.gru.future_steps,
        ).to(device)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        logger.info(
            "Seed %d: GRU hidden=%d layers=%d params=%d",
            seed,
            config.gru.hidden_size,
            config.gru.num_layers,
            parameter_count,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        resume_state: Optional[TrainResumeState] = None
        active = resume_blob.get("active_seed_index")
        if active == seed_index and "active_state" in resume_blob:
            saved = resume_blob["active_state"]
            model.load_state_dict(saved["model_state"])
            resume_state = TrainResumeState(
                start_epoch=int(saved["start_epoch"]),
                best_validation_loss=float(saved["best_validation_loss"]),
                epochs_without_improvement=int(saved["epochs_without_improvement"]),
                optimizer_state=saved["optimizer_state"],
                model_state=saved["model_state"],
                best_model_state=saved["best_model_state"],
                torch_rng_state=saved["torch_rng_state"],
                numpy_rng_state=saved["numpy_rng_state"],
                python_rng_state=saved["python_rng_state"],
                loader_generator_state=saved.get("loader_generator_state"),
                cuda_rng_states=saved.get("cuda_rng_states"),
                best_epoch=int(saved.get("best_epoch", 0)),
            )
        def _on_epoch_end(
            record: EpochRecord,
            *,
            model_state,
            best_model_state,
            optimizer_state,
            epochs_without_improvement,
            loader_generator_state,
            best_epoch,
            _seed=seed,
            _seed_index=seed_index,
        ) -> None:
            history_rows.append(
                {
                    "seed": _seed,
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "validation_loss": record.validation_loss,
                    "best_validation_loss": record.best_validation_loss,
                    "is_best": int(record.is_best),
                    "learning_rate": record.learning_rate,
                    "seconds": record.seconds,
                }
            )
            _write_history(output_directory / HISTORY_FILENAME, history_rows)
            rng_state = capture_safe_rng_state(loader_generator_state)
            resume_snapshot = {
                "completed_seed_indices": sorted(completed_seed_indices),
                "per_seed_results": per_seed_results,
                "history_rows": history_rows,
                "active_seed_index": _seed_index,
                "active_epoch": record.epoch,
                "active_state": {
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
                    "loader_generator_state": rng_state[
                        "loader_generator_state"
                    ],
                },
            }
            torch.save(to_weights_only_safe(resume_snapshot), str(resume_path))

        result = train_gru(
            model=model,
            train_features=train_features,
            train_target=train_target,
            validation_features=validation_features,
            validation_target=validation_target,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            validation_batch_size=config.training.validation_batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            loss_name=config.training.loss,
            gradient_clip_norm=config.training.gradient_clip_norm,
            early_stopping_patience=config.training.early_stopping_patience,
            early_stopping_min_delta=config.training.early_stopping_min_delta,
            num_workers=config.training.num_workers,
            generator=generator,
            device=device,
            on_epoch_end=_on_epoch_end,
            resume=resume_state,
        )
        if not result["history"] and resume_state is not None:
            # The process may have stopped after the final epoch snapshot but
            # before marking this seed complete.  Preserve that completed
            # epoch and its last weights instead of emitting epoch zero.
            result["stopped_epoch"] = int(
                resume_blob.get("active_epoch", resume_state.start_epoch - 1)
            )
            result["last_state"] = resume_state.model_state
        logger.info(
            "Seed %d finished at epoch %d with best validation loss %.6f",
            seed,
            result["stopped_epoch"],
            result["best_validation_loss"],
        )
        training_time_seconds = sum(
            float(row["seconds"])
            for row in history_rows
            if int(row["seed"]) == int(seed)
        )
        seed_result = _save_seed_checkpoints(
            config=config,
            normalizer=normalizer,
            result=result,
            seed=int(seed),
            parameter_count=parameter_count,
            training_time_seconds=training_time_seconds,
            checkpoint_directory=checkpoint_directory,
        )
        seed_result["seed_index"] = int(seed_index)
        per_seed_results.append(seed_result)
        logger.info(
            "Seed %d checkpoints: %s and %s (training_time=%.3fs)",
            seed,
            seed_result["best_checkpoint"],
            seed_result["last_checkpoint"],
            training_time_seconds,
        )
        completed_seed_indices.add(seed_index)
        # Clear the active-seed resume slot now that this seed is complete.
        resume_blob.pop("active_state", None)
        resume_blob["active_seed_index"] = None
        torch.save(
            to_weights_only_safe({
                "completed_seed_indices": sorted(completed_seed_indices),
                "per_seed_results": per_seed_results,
                "history_rows": history_rows,
                "active_seed_index": None,
            }),
            str(resume_path),
        )
        _write_seed_summary(
            output_directory / SEED_SUMMARY_FILENAME,
            _ordered_seed_results(config, per_seed_results, require_all=False),
        )

    _finalize_checkpoints(
        config=config,
        normalizer=normalizer,
        per_seed_results=per_seed_results,
        checkpoint_directory=checkpoint_directory,
        output_directory=output_directory,
        logger=logger,
    )
    logger.info("Experiment 02 training completed successfully")
    return 0


def _ordered_seed_results(
    config: Experiment02Config,
    per_seed_results: List[Dict[str, Any]],
    *,
    require_all: bool,
) -> List[Dict[str, Any]]:
    """Return unique completed-seed records in configured seed order."""

    by_seed: Dict[int, Dict[str, Any]] = {}
    configured = set(int(seed) for seed in config.seeds)
    for result in per_seed_results:
        seed = int(result["seed"])
        if seed not in configured:
            raise ValueError("resume state contains unconfigured seed {}".format(seed))
        if seed in by_seed:
            raise ValueError("duplicate completed result for seed {}".format(seed))
        by_seed[seed] = result
    missing = [int(seed) for seed in config.seeds if int(seed) not in by_seed]
    if require_all and missing:
        raise RuntimeError(
            "training did not produce results for configured seeds: {}".format(missing)
        )
    return [by_seed[int(seed)] for seed in config.seeds if int(seed) in by_seed]


def _finalize_checkpoints(
    *,
    config: Experiment02Config,
    normalizer: PredictionNormalizer,
    per_seed_results: List[Dict[str, Any]],
    checkpoint_directory: Path,
    output_directory: Path,
    logger: logging.Logger,
) -> None:
    """Write summary plus optional top-level deployment convenience copies.

    Every configured seed remains represented by its own checkpoint directory
    and summary row.  The validation-best top-level copy is never used as a
    substitute for multi-seed reporting.
    """

    if not per_seed_results:
        raise RuntimeError("no seed produced a trained model")
    del normalizer  # Per-seed checkpoints already contain these statistics.
    ordered = _ordered_seed_results(config, per_seed_results, require_all=True)
    _write_seed_summary(
        output_directory / SEED_SUMMARY_FILENAME,
        ordered,
    )

    best_seed = min(ordered, key=lambda item: float(item["validation_loss"]))
    last_seed = ordered[-1]
    best_source = output_directory / str(best_seed["best_checkpoint"])
    last_source = output_directory / str(last_seed["last_checkpoint"])
    if not best_source.is_file() or not last_source.is_file():
        raise FileNotFoundError("one or more per-seed checkpoints are missing")
    shutil.copyfile(str(best_source), str(checkpoint_directory / "best.pt"))
    shutil.copyfile(str(last_source), str(checkpoint_directory / "last.pt"))
    logger.info(
        "Saved deployment convenience best.pt (seed=%d, best_epoch=%d, "
        "val_loss=%.6f) and last.pt (seed=%d, epoch=%d); all seed results "
        "remain in seed-specific directories and %s",
        best_seed["seed"],
        best_seed["best_epoch"],
        best_seed["validation_loss"],
        last_seed["seed"],
        last_seed["stopped_epoch"],
        SEED_SUMMARY_FILENAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
