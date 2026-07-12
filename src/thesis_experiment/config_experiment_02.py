"""Configuration loading for deterministic trajectory prediction baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import math
import yaml


@dataclass(frozen=True)
class DatasetPathsConfig:
    """Dataset-v2 directory and split filenames."""

    directory: Path
    train_split: str
    validation_split: str
    test_split: str


@dataclass(frozen=True)
class KalmanConfig:
    """Shared CV Kalman filter noise parameters."""

    process_noise: float
    measurement_noise: float
    initial_position_variance: float
    initial_velocity_variance: float


@dataclass(frozen=True)
class GRUConfig:
    """Deterministic GRU architecture."""

    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    future_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization, batching, and early-stopping settings."""

    epochs: int
    batch_size: int
    validation_batch_size: int
    learning_rate: float
    weight_decay: float
    loss: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_norm: float
    num_workers: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Inference batching, timing, plots, and reporting settings."""

    batch_size: int
    runtime_warmup: int
    runtime_repeats: int
    trajectory_example_count: int
    default_statistical_unit: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Explicit PyTorch CPU thread-pool settings."""

    torch_num_threads: int
    torch_num_interop_threads: int


@dataclass(frozen=True)
class Experiment02Config:
    """Validated experiment-02 configuration."""

    seeds: Tuple[int, ...]
    dataset: DatasetPathsConfig
    output_directory: Path
    normalization_epsilon: float
    fill_value: float
    kalman: KalmanConfig
    gru: GRUConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    source_path: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Configuration section '{}' must be a mapping".format(name))
    return value


def load_experiment_02_config(path: Path) -> Experiment02Config:
    """Load and validate an experiment-02 YAML file without accessing data."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    dataset = _mapping(root.get("dataset"), "dataset")
    normalization = _mapping(root.get("normalization"), "normalization")
    kalman = _mapping(root.get("kalman_filter"), "kalman_filter")
    gru = _mapping(root.get("gru"), "gru")
    training = _mapping(root.get("training"), "training")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    runtime = _mapping(root.get("runtime"), "runtime")

    seeds_value = experiment.get("seeds")
    if not isinstance(seeds_value, (list, tuple)):
        raise ValueError("experiment.seeds must be a sequence")
    config = Experiment02Config(
        seeds=tuple(int(value) for value in seeds_value),
        dataset=DatasetPathsConfig(
            directory=Path(str(dataset["directory"])),
            train_split=str(dataset.get("train_split", "train.npz")),
            validation_split=str(
                dataset.get("validation_split", "validation.npz")
            ),
            test_split=str(dataset.get("test_split", "test.npz")),
        ),
        output_directory=Path(str(root.get("output_directory", "outputs/experiment_02"))),
        normalization_epsilon=float(normalization.get("epsilon", 1.0e-6)),
        fill_value=float(normalization.get("fill_value", 0.0)),
        kalman=KalmanConfig(
            process_noise=float(kalman["process_noise"]),
            measurement_noise=float(kalman["measurement_noise"]),
            initial_position_variance=float(kalman["initial_position_variance"]),
            initial_velocity_variance=float(kalman["initial_velocity_variance"]),
        ),
        gru=GRUConfig(
            input_size=int(gru.get("input_size", 7)),
            hidden_size=int(gru["hidden_size"]),
            num_layers=int(gru["num_layers"]),
            dropout=float(gru["dropout"]),
            future_steps=int(gru["future_steps"]),
        ),
        training=TrainingConfig(
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            validation_batch_size=int(
                training.get("validation_batch_size", training["batch_size"])
            ),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
            loss=str(training.get("loss", "smooth_l1")),
            early_stopping_patience=int(training["early_stopping_patience"]),
            early_stopping_min_delta=float(
                training.get("early_stopping_min_delta", 0.0)
            ),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            num_workers=int(training.get("num_workers", 0)),
        ),
        evaluation=EvaluationConfig(
            batch_size=int(evaluation["batch_size"]),
            runtime_warmup=int(evaluation.get("runtime_warmup", 1)),
            runtime_repeats=int(evaluation.get("runtime_repeats", 5)),
            trajectory_example_count=int(
                evaluation.get("trajectory_example_count", 4)
            ),
            default_statistical_unit=str(
                evaluation.get("default_statistical_unit", "episode")
            ),
        ),
        runtime=RuntimeConfig(
            torch_num_threads=int(runtime.get("torch_num_threads", 1)),
            torch_num_interop_threads=int(
                runtime.get("torch_num_interop_threads", 1)
            ),
        ),
        source_path=config_path,
    )
    validate_experiment_02_config(config)
    return config


def validate_experiment_02_config(config: Experiment02Config) -> None:
    """Raise ``ValueError`` for unsafe or internally inconsistent settings."""

    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("experiment.seeds must be non-empty and unique")
    if any(not name.endswith(".npz") for name in (
        config.dataset.train_split,
        config.dataset.validation_split,
        config.dataset.test_split,
    )):
        raise ValueError("dataset split filenames must end with .npz")
    if len({
        config.dataset.train_split,
        config.dataset.validation_split,
        config.dataset.test_split,
    }) != 3:
        raise ValueError("train, validation, and test split files must be distinct")
    finite_values = (
        config.normalization_epsilon,
        config.fill_value,
        config.kalman.process_noise,
        config.kalman.measurement_noise,
        config.kalman.initial_position_variance,
        config.kalman.initial_velocity_variance,
        config.gru.dropout,
        config.training.learning_rate,
        config.training.weight_decay,
        config.training.early_stopping_min_delta,
        config.training.gradient_clip_norm,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("all floating-point configuration values must be finite")
    if config.normalization_epsilon <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    if min(
        config.kalman.process_noise,
        config.kalman.measurement_noise,
        config.kalman.initial_position_variance,
        config.kalman.initial_velocity_variance,
    ) <= 0.0:
        raise ValueError("Kalman covariance/noise values must be positive")
    if config.gru.input_size != 7:
        raise ValueError("deterministic GRU input_size must be 7")
    if min(config.gru.hidden_size, config.gru.num_layers, config.gru.future_steps) <= 0:
        raise ValueError("GRU dimensions must be positive")
    if not 0.0 <= config.gru.dropout < 1.0:
        raise ValueError("GRU dropout must be in [0, 1)")
    training_integers = (
        config.training.epochs,
        config.training.batch_size,
        config.training.validation_batch_size,
        config.training.early_stopping_patience,
    )
    if min(training_integers) <= 0 or config.training.num_workers < 0:
        raise ValueError("training counts/batch sizes must be positive")
    if (
        config.training.learning_rate <= 0.0
        or config.training.weight_decay < 0.0
        or config.training.early_stopping_min_delta < 0.0
        or config.training.gradient_clip_norm <= 0.0
    ):
        raise ValueError("invalid optimizer or early-stopping values")
    if config.training.loss not in ("mse", "smooth_l1"):
        raise ValueError("training.loss must be mse or smooth_l1")
    if min(
        config.evaluation.batch_size,
        config.evaluation.runtime_repeats,
        config.evaluation.trajectory_example_count,
    ) <= 0 or config.evaluation.runtime_warmup < 0:
        raise ValueError("evaluation counts must be positive")
    if config.evaluation.default_statistical_unit not in ("episode", "scene"):
        raise ValueError("default_statistical_unit must be episode or scene")
    if min(
        config.runtime.torch_num_threads,
        config.runtime.torch_num_interop_threads,
    ) <= 0:
        raise ValueError("PyTorch thread counts must be positive")


__all__ = [
    "DatasetPathsConfig",
    "EvaluationConfig",
    "Experiment02Config",
    "GRUConfig",
    "KalmanConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "load_experiment_02_config",
    "validate_experiment_02_config",
]
