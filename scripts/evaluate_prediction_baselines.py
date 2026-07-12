#!/usr/bin/env python
"""Evaluate all four experiment-02 trajectory-prediction baselines.

Every method (constant position, constant velocity, CV Kalman filter, and the
trained deterministic GRU) is scored on exactly the same ``test.npz`` in exactly
the same window order.  Model inputs pass through :class:`PredictionDataset`, so
supervision, audit truth, trajectory type, and future motion parameters can
never enter a predictor.  ``trajectory_type`` and ``occlusion_length_bin`` are
read straight from the archive *after* inference and are used only to group and
visualise results.

All figures are produced from the persisted CSV/NPZ artefacts, never from a
fresh, unsaved computation, so the plots always match the reported numbers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from thesis_experiment.config_experiment_02 import load_experiment_02_config
from thesis_experiment.data.prediction_dataset import PredictionDataset
from thesis_experiment.evaluation.prediction_metrics import (
    compute_prediction_metrics,
    prediction_metric_records,
    seed_mean_and_sample_std,
)
from thesis_experiment.prediction.pipeline import (
    CLASSICAL_METHOD_NAMES,
    GRU_METHOD_NAME,
    apply_torch_thread_config,
    load_model_from_checkpoint,
    predict_classical_local,
    predict_gru_local,
    safe_torch_load,
    select_device,
    set_global_seed,
)

TRAJECTORY_TYPE_NAMES = {
    0: "constant_velocity",
    1: "constant_acceleration",
    2: "constant_turn",
    3: "stop_and_go",
    4: "piecewise_direction",
}
OCCLUSION_BIN_NAMES = {0: "0", 1: "1-5", 2: "6-10", 3: "11-15", 4: "16-20"}
PredictionKey = Tuple[str, Optional[int]]


def _arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate experiment 02 baselines.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_evaluation_dataset(config, dataset_directory: Path):
    """Load the configured test split through the strict prediction boundary."""

    test_archive = dataset_directory / config.dataset.test_split
    return (
        PredictionDataset(test_archive, fill_value=config.fill_value),
        test_archive,
    )


def _logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("experiment_02_eval")
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


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_grouping_metadata(archive_path: Path, dataset: PredictionDataset) -> Dict[str, np.ndarray]:
    """Load post-inference grouping labels straight from the test archive.

    These arrays are used only for grouping, identification, and plotting.  The
    row order of the raw archive matches ``PredictionDataset`` (which does not
    reorder samples), and this is verified against the loader metadata so that
    predictions, labels, and IDs always describe the same window.
    """

    with np.load(str(archive_path), allow_pickle=False) as archive:
        metadata = {
            "scene_id": np.asarray(archive["scene_id"]),
            "episode_id": np.asarray(archive["episode_id"]),
            "sample_start_index": np.asarray(archive["sample_start_index"]),
            "trajectory_type": np.asarray(archive["trajectory_type"]),
            "occlusion_length_bin": np.asarray(archive["occlusion_length_bin"]),
            "future_position": np.asarray(archive["future_position"], dtype=np.float64),
        }
    # Cross-check IDs against the strict loader's own metadata view.
    loader_batch = dataset.get_batch(slice(None))["metadata"]
    for key in ("scene_id", "episode_id", "sample_start_index"):
        if not np.array_equal(np.asarray(loader_batch[key]), metadata[key]):
            raise ValueError(
                "archive {} order does not match PredictionDataset order".format(key)
            )
    return metadata


def _generate_predictions(
    dataset: PredictionDataset,
    config,
    device: torch.device,
    checkpoint_paths: Mapping[int, Path],
    logger: logging.Logger,
) -> Dict[PredictionKey, np.ndarray]:
    """Run deterministic baselines once and every configured GRU seed.

    ``None`` is the seed marker for classical methods.  They are deterministic
    and are intentionally not copied once per GRU seed, which would falsely
    inflate their replicate count in formal summaries.
    """

    future_steps = config.gru.future_steps
    kalman = {
        "process_noise": config.kalman.process_noise,
        "measurement_noise": config.kalman.measurement_noise,
        "initial_position_variance": config.kalman.initial_position_variance,
        "initial_velocity_variance": config.kalman.initial_velocity_variance,
    }
    predictions: Dict[PredictionKey, np.ndarray] = {}
    for method in CLASSICAL_METHOD_NAMES:
        key = (method, None)
        predictions[key] = predict_classical_local(
            dataset, method, future_steps=future_steps, kalman=kalman
        )
        logger.info("Generated %s predictions %s", method, predictions[key].shape)

    for seed in config.seeds:
        checkpoint = safe_torch_load(checkpoint_paths[int(seed)], map_location=device)
        if int(checkpoint["seed"]) != int(seed):
            raise ValueError(
                "checkpoint seed {} does not match configured seed {}".format(
                    checkpoint["seed"], seed
                )
            )
        model, normalizer = load_model_from_checkpoint(checkpoint, device)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        logger.info(
            "Loaded GRU checkpoint (seed=%d, epoch=%d, params=%d)",
            int(checkpoint["seed"]),
            int(checkpoint["epoch"]),
            parameter_count,
        )
        key = (GRU_METHOD_NAME, int(seed))
        predictions[key] = predict_gru_local(
            dataset,
            model,
            normalizer,
            batch_size=config.evaluation.batch_size,
            device=device,
        )
        logger.info(
            "Generated %s seed=%d predictions %s",
            GRU_METHOD_NAME,
            seed,
            predictions[key].shape,
        )
    return predictions


def _mean_std_summary(
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate independently evaluated seeds into mean and sample std.

    Classical baselines have one row with an empty seed and therefore report
    ``seed_count=1`` and ``std=0``.  GRU rows are grouped over their actual
    checkpoints; no validation-based best-seed selection is performed.
    """

    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in summary_rows:
        key = (
            str(row["model_name"]),
            str(row["metric"]),
            str(row["aggregation_level"]),
        )
        grouped.setdefault(key, []).append(float(row["mean"]))
    result: List[Dict[str, Any]] = []
    for (model_name, metric, aggregation_level), values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        mean, standard_deviation = seed_mean_and_sample_std(array)
        result.append(
            {
                "model_name": model_name,
                "metric": metric,
                "aggregation_level": aggregation_level,
                "seed_count": int(array.size),
                "mean": mean,
                "std": standard_deviation,
            }
        )
    return result


def _compute_and_write_metrics(
    predictions: Mapping[PredictionKey, np.ndarray],
    metadata: Dict[str, np.ndarray],
    output_directory: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Compute per-window metrics then aggregate to every reporting level."""

    future_position = metadata["future_position"]
    scene_id = metadata["scene_id"]
    episode_id = metadata["episode_id"]
    sample_start_index = metadata["sample_start_index"]
    trajectory_names = np.asarray(
        [TRAJECTORY_TYPE_NAMES[int(v)] for v in metadata["trajectory_type"]]
    )
    occlusion_names = np.asarray(
        [OCCLUSION_BIN_NAMES[int(v)] for v in metadata["occlusion_length_bin"]]
    )

    window_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    horizon_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    motion_rows: List[Dict[str, Any]] = []
    occlusion_rows: List[Dict[str, Any]] = []

    for (method, seed), prediction in predictions.items():
        seed_value: Any = "" if seed is None else int(seed)
        metrics = compute_prediction_metrics(prediction, future_position)
        ade = metrics["ade"]
        fde = metrics["fde"]

        for index in range(ade.shape[0]):
            window_rows.append(
                {
                    "model_name": method,
                    "seed": seed_value,
                    "window_index": index,
                    "scene_id": int(scene_id[index]),
                    "episode_id": int(episode_id[index]),
                    "trajectory_type": str(trajectory_names[index]),
                    "occlusion_length_bin": str(occlusion_names[index]),
                    "sample_start_index": int(sample_start_index[index]),
                    "ade": float(ade[index]),
                    "fde": float(fde[index]),
                }
            )

        for step, mean_distance, rmse in zip(
            metrics["per_horizon_step"],
            metrics["per_horizon_mean_euclidean_distance"],
            metrics["per_horizon_rmse"],
        ):
            horizon_rows.append(
                {
                    "model_name": method,
                    "seed": seed_value,
                    "horizon_step": int(step),
                    "mean_euclidean_distance": float(mean_distance),
                    "rmse": float(rmse),
                }
            )

        records = prediction_metric_records(
            {"ade": ade, "fde": fde},
            scene_id=scene_id,
            episode_id=episode_id,
            trajectory_type=trajectory_names,
            occlusion_length_bin=occlusion_names,
            sample_start_index=sample_start_index,
            model_name=method,
        )
        for table_name in ("summary", "episode", "scene"):
            for row in records[table_name]:
                row["seed"] = seed_value
        summary_rows.extend(records["summary"])
        episode_rows.extend(records["episode"])
        scene_rows.extend(records["scene"])

        # Equal-weight episode mean within each trajectory type (episode -> one type).
        for metric_name in ("ade", "fde"):
            type_to_episode_means: Dict[str, List[float]] = {}
            for row in records["episode"]:
                if row["metric"] != metric_name:
                    continue
                type_to_episode_means.setdefault(row["trajectory_type"], []).append(
                    row["mean"]
                )
            for type_name, means in type_to_episode_means.items():
                motion_rows.append(
                    {
                        "model_name": method,
                        "seed": seed_value,
                        "metric": metric_name,
                        "trajectory_type": type_name,
                        "episode_count": len(means),
                        "episode_mean": float(np.mean(means)),
                    }
                )
            for group in records["trajectory_type"]:
                if group["metric"] != metric_name:
                    continue
                for motion_row in motion_rows:
                    if (
                        motion_row["model_name"] == method
                        and motion_row["seed"] == seed_value
                        and motion_row["metric"] == metric_name
                        and motion_row["trajectory_type"] == group["trajectory_type"]
                    ):
                        motion_row["window_mean"] = float(group["window_mean"])
                        motion_row["window_count"] = int(group["window_count"])

        for group in records["occlusion_length_bin"]:
            occlusion_rows.append(
                {
                    "model_name": method,
                    "seed": seed_value,
                    "metric": group["metric"],
                    "occlusion_length_bin": group["occlusion_length_bin"],
                    "window_count": int(group["window_count"]),
                    "window_mean": float(group["window_mean"]),
                }
            )

    window_fields = [
            "model_name",
            "seed",
            "window_index",
            "scene_id",
            "episode_id",
            "trajectory_type",
            "occlusion_length_bin",
            "sample_start_index",
            "ade",
            "fde",
        ]
    episode_fields = [
        "model_name", "seed", "metric", "episode_id", "scene_id",
        "trajectory_type", "window_count", "mean",
    ]
    scene_fields = [
        "model_name", "seed", "metric", "scene_id", "episode_count",
        "window_count", "mean",
    ]
    horizon_fields = [
        "model_name", "seed", "horizon_step", "mean_euclidean_distance", "rmse",
    ]
    summary_fields = [
        "model_name", "seed", "metric", "aggregation_level", "unit_count", "mean",
    ]
    # The by-seed tables are the authoritative multi-run artefacts.  Legacy
    # filenames are retained with the same rows and seed column for downstream
    # consumers written for the original single-seed smoke experiment.
    for filename in ("per_window_metrics_by_seed.csv", "per_window_metrics.csv"):
        _write_csv(output_directory / filename, window_rows, window_fields)
    for filename in ("per_episode_metrics_by_seed.csv", "per_episode_metrics.csv"):
        _write_csv(output_directory / filename, episode_rows, episode_fields)
    for filename in ("per_scene_metrics_by_seed.csv", "per_scene_metrics.csv"):
        _write_csv(output_directory / filename, scene_rows, scene_fields)
    for filename in ("per_horizon_metrics_by_seed.csv", "per_horizon_metrics.csv"):
        _write_csv(output_directory / filename, horizon_rows, horizon_fields)
    for filename in ("summary_metrics_by_seed.csv", "summary_metrics.csv"):
        _write_csv(output_directory / filename, summary_rows, summary_fields)

    mean_std_rows = _mean_std_summary(summary_rows)
    _write_csv(
        output_directory / "summary_metrics_mean_std.csv",
        mean_std_rows,
        ["model_name", "metric", "aggregation_level", "seed_count", "mean", "std"],
    )
    _write_csv(
        output_directory / "per_motion_type_metrics.csv",
        motion_rows,
        [
            "model_name",
            "seed",
            "metric",
            "trajectory_type",
            "episode_count",
            "episode_mean",
            "window_count",
            "window_mean",
        ],
    )
    _write_csv(
        output_directory / "per_occlusion_group_metrics.csv",
        occlusion_rows,
        ["model_name", "seed", "metric", "occlusion_length_bin", "window_count", "window_mean"],
    )
    logger.info(
        "Wrote metric tables: %d window rows, %d episode rows, %d scene rows",
        len(window_rows),
        len(episode_rows),
        len(scene_rows),
    )
    return {"summary_rows": summary_rows, "mean_std_rows": mean_std_rows}


def _save_predictions(
    predictions: Mapping[PredictionKey, np.ndarray],
    metadata: Dict[str, np.ndarray],
    output_directory: Path,
) -> None:
    """Persist forecasts and lightweight identifiers, not full episode arrays."""

    prediction_keys = list(predictions)
    model_names = [key[0] for key in prediction_keys]
    # NPZ has no nullable integer dtype without pickle.  -1 is the documented
    # sentinel for deterministic classical baselines; configured seeds must be
    # non-negative for experiment 02.
    seeds = [-1 if key[1] is None else int(key[1]) for key in prediction_keys]
    stacked = np.stack([predictions[key] for key in prediction_keys], axis=0)
    np.savez_compressed(
        str(output_directory / "predictions.npz"),
        model_name=np.asarray(model_names),
        seed=np.asarray(seeds, dtype=np.int64),
        prediction=stacked.astype(np.float64),
        future_position=metadata["future_position"].astype(np.float64),
        scene_id=metadata["scene_id"],
        episode_id=metadata["episode_id"],
        sample_start_index=metadata["sample_start_index"],
        trajectory_type=metadata["trajectory_type"],
        occlusion_length_bin=metadata["occlusion_length_bin"],
    )


def _time_call(function, warmup: int, repeats: int) -> float:
    """Return mean seconds per call after warmup, using a monotonic clock."""

    for _ in range(max(warmup, 0)):
        function()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - started) / repeats


def _measure_runtime(
    dataset: PredictionDataset,
    config,
    device: torch.device,
    checkpoint_paths: Mapping[int, Path],
    output_directory: Path,
    logger: logging.Logger,
) -> None:
    """Measure single-sample and batch inference time on CPU (and GPU if any)."""

    warmup = config.evaluation.runtime_warmup
    repeats = config.evaluation.runtime_repeats
    future_steps = config.gru.future_steps
    kalman = {
        "process_noise": config.kalman.process_noise,
        "measurement_noise": config.kalman.measurement_noise,
        "initial_position_variance": config.kalman.initial_position_variance,
        "initial_velocity_variance": config.kalman.initial_velocity_variance,
    }
    rows: List[Dict[str, Any]] = []

    def _single_dataset() -> PredictionDataset:
        return dataset

    batch_size = len(dataset)

    # Classical methods run on CPU (NumPy) only.
    for method in CLASSICAL_METHOD_NAMES:
        single = _time_call(
            lambda m=method: predict_classical_local(
                _SingleWindowDataset(dataset, 0), m, future_steps=future_steps, kalman=kalman
            ),
            warmup,
            repeats,
        )
        full = _time_call(
            lambda m=method: predict_classical_local(
                dataset, m, future_steps=future_steps, kalman=kalman
            ),
            warmup,
            repeats,
        )
        rows.append(
            {
                "model_name": method,
                "seed": "",
                "device": "cpu",
                "mode": "single_sample",
                "batch_size": 1,
                "mean_seconds": single,
                "per_window_ms": single * 1000.0,
            }
        )
        rows.append(
            {
                "model_name": method,
                "seed": "",
                "device": "cpu",
                "mode": "batch",
                "batch_size": batch_size,
                "mean_seconds": full,
                "per_window_ms": full / batch_size * 1000.0,
            }
        )

    # GRU timed on every available device.
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for seed in config.seeds:
        for run_device in devices:
            checkpoint = safe_torch_load(
                checkpoint_paths[int(seed)], map_location=run_device
            )
            model, normalizer = load_model_from_checkpoint(checkpoint, run_device)
            single = _time_call(
                lambda: predict_gru_local(
                    _SingleWindowDataset(dataset, 0),
                    model,
                    normalizer,
                    batch_size=1,
                    device=run_device,
                ),
                warmup,
                repeats,
            )
            full = _time_call(
                lambda: predict_gru_local(
                    dataset,
                    model,
                    normalizer,
                    batch_size=config.evaluation.batch_size,
                    device=run_device,
                ),
                warmup,
                repeats,
            )
            label = "cuda" if run_device.type == "cuda" else "cpu"
            rows.append(
                {
                    "model_name": GRU_METHOD_NAME,
                    "seed": int(seed),
                    "device": label,
                    "mode": "single_sample",
                    "batch_size": 1,
                    "mean_seconds": single,
                    "per_window_ms": single * 1000.0,
                }
            )
            rows.append(
                {
                    "model_name": GRU_METHOD_NAME,
                    "seed": int(seed),
                    "device": label,
                    "mode": "batch",
                    "batch_size": batch_size,
                    "mean_seconds": full,
                    "per_window_ms": full / batch_size * 1000.0,
                }
            )

    _write_csv(
        output_directory / "runtime.csv",
        rows,
        ["model_name", "seed", "device", "mode", "batch_size", "mean_seconds", "per_window_ms"],
    )
    logger.info("Wrote runtime.csv with %d timing rows", len(rows))


class _SingleWindowDataset:
    """Adapter exposing exactly one window with the PredictionDataset API.

    Used only for single-sample runtime timing.  It forwards the strict input
    whitelist for one row and keeps the same ``get_batch``/``input_fields``
    contract the pipeline relies on.
    """

    def __init__(self, dataset: PredictionDataset, index: int) -> None:
        self._batch = dataset.get_batch(np.asarray([index], dtype=np.int64))
        self._input_fields = dataset.input_fields

    @property
    def input_fields(self):
        return self._input_fields

    def __len__(self) -> int:
        return 1

    def get_batch(self, indices):
        return self._batch


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arguments(argv)
    config_path = _project_path(args.config)
    config = load_experiment_02_config(config_path)

    output_directory = _project_path(config.output_directory)
    checkpoint_paths = {
        int(seed): output_directory / "checkpoints" / "seed_{}".format(seed) / "best.pt"
        for seed in config.seeds
    }
    missing_checkpoints = [
        path for path in checkpoint_paths.values() if not path.is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "missing per-seed GRU checkpoint(s) {}; run "
            "train_prediction_baseline.py first".format(
                ", ".join(str(path) for path in missing_checkpoints)
            )
        )
    figures_directory = output_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    logger = _logger(output_directory / "run.log")
    applied_threads = apply_torch_thread_config(
        config.runtime.torch_num_threads,
        config.runtime.torch_num_interop_threads,
    )
    device = select_device()
    # A fixed seed keeps deterministic-GRU inference and timing reproducible.
    set_global_seed(config.seeds[0])
    logger.info("Starting experiment 02 evaluation with config %s", config_path)
    logger.info("Device: %s | torch %s", device, torch.__version__)
    logger.info(
        "PyTorch threads: intra-op=%d inter-op=%d",
        applied_threads["torch_num_threads"],
        applied_threads["torch_num_interop_threads"],
    )
    logger.info(
        "Evaluating GRU seeds %s; primary plots use %s-level aggregation",
        list(config.seeds),
        config.evaluation.default_statistical_unit,
    )

    dataset_directory = _project_path(config.dataset.directory)
    test_dataset, test_archive = _load_evaluation_dataset(config, dataset_directory)
    logger.info("Loaded %d test windows from %s", len(test_dataset), test_archive)

    predictions = _generate_predictions(
        test_dataset, config, device, checkpoint_paths, logger
    )
    # Grouping-only labels and future supervision are loaded strictly after all
    # predictors have finished, so neither trajectory type nor occlusion group
    # can influence model inference.
    metadata = _load_grouping_metadata(test_archive, test_dataset)
    _compute_and_write_metrics(predictions, metadata, output_directory, logger)
    _save_predictions(predictions, metadata, output_directory)
    _measure_runtime(
        test_dataset, config, device, checkpoint_paths, output_directory, logger
    )

    os.environ.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib"))
    from experiment_02_figures import generate_all_figures

    generate_all_figures(
        output_directory,
        figures_directory,
        logger,
        default_statistical_unit=config.evaluation.default_statistical_unit,
    )
    logger.info("Experiment 02 evaluation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
