"""Figure generation for experiment 02, driven only by persisted artefacts.

Every figure reads the CSV tables and ``predictions.npz`` written by the
evaluation script.  Nothing here re-runs a model or recomputes a random
quantity, so the plots are guaranteed to match the reported numbers.  Window
level (descriptive) and episode/scene level (formal) results are labelled
explicitly wherever both appear.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_experiment.evaluation.prediction_metrics import seed_mean_and_sample_std


METHOD_ORDER = [
    "constant_position",
    "constant_velocity",
    "cv_kalman_filter",
    "deterministic_gru",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(figure, path: Path) -> None:
    figure.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_train_validation_loss(output_directory: Path, figures_directory: Path) -> None:
    rows = _read_csv(output_directory / "training_history.csv")
    figure, axes = plt.subplots(figsize=(7, 4))
    seeds = sorted({row["seed"] for row in rows})
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        epochs = [int(row["epoch"]) for row in seed_rows]
        train = [float(row["train_loss"]) for row in seed_rows]
        validation = [float(row["validation_loss"]) for row in seed_rows]
        axes.plot(epochs, train, marker="o", label="train (seed {})".format(seed))
        axes.plot(
            epochs, validation, marker="s", linestyle="--",
            label="validation (seed {})".format(seed),
        )
    axes.set_xlabel("epoch")
    axes.set_ylabel("loss")
    axes.set_title("Deterministic GRU training and validation loss")
    axes.legend(fontsize=8)
    axes.grid(True, alpha=0.3)
    _save(figure, figures_directory / "train_validation_loss.png")


def metric_comparison_values(
    rows: Sequence[Dict[str, str]], metric: str, statistical_unit: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return method-ordered mean/std values for one configured formal unit.

    This small pure helper makes it testable that changing
    ``evaluation.default_statistical_unit`` changes the data selected for the
    primary ADE/FDE plots rather than merely changing a label.
    """

    if statistical_unit not in ("episode", "scene"):
        raise ValueError("statistical_unit must be episode or scene")
    matching = {
        (row["model_name"], row["metric"], row["aggregation_level"]): row
        for row in rows
    }
    means = []
    standard_deviations = []
    for method in METHOD_ORDER:
        row = matching.get((method, metric, statistical_unit))
        means.append(float(row["mean"]) if row is not None else np.nan)
        standard_deviations.append(float(row["std"]) if row is not None else np.nan)
    return np.asarray(means), np.asarray(standard_deviations)


def _plot_metric_comparison(
    output_directory: Path,
    figures_directory: Path,
    metric: str,
    filename: str,
    default_statistical_unit: str,
) -> None:
    rows = _read_csv(output_directory / "summary_metrics_mean_std.csv")
    values, standard_deviations = metric_comparison_values(
        rows, metric, default_statistical_unit
    )
    positions = np.arange(len(METHOD_ORDER))
    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.bar(
        positions,
        values,
        yerr=standard_deviations,
        capsize=4,
        label="{}-equal mean +/- seed std".format(default_statistical_unit),
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(METHOD_ORDER, rotation=20, ha="right", fontsize=8)
    axes.set_ylabel("{} (local position units)".format(metric.upper()))
    axes.set_title(
        "{} ({}-level formal comparison)".format(
            metric.upper(), default_statistical_unit
        )
    )
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", alpha=0.3)
    _save(figure, figures_directory / filename)


def _plot_error_by_horizon(output_directory: Path, figures_directory: Path) -> None:
    rows = _read_csv(output_directory / "per_horizon_metrics_by_seed.csv")
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for method in METHOD_ORDER:
        steps = sorted(
            {int(row["horizon_step"]) for row in rows if row["model_name"] == method}
        )
        per_step = [
            [
                float(row["mean_euclidean_distance"])
                for row in rows
                if row["model_name"] == method
                and int(row["horizon_step"]) == step
            ]
            for step in steps
        ]
        statistics = [seed_mean_and_sample_std(values) for values in per_step]
        distance = np.asarray([item[0] for item in statistics])
        deviation = np.asarray([item[1] for item in statistics])
        axes.plot(steps, distance, marker="o", markersize=3, label=method)
        if method == "deterministic_gru" and any(len(values) > 1 for values in per_step):
            axes.fill_between(
                steps, distance - deviation, distance + deviation, alpha=0.2
            )
    axes.set_xlabel("future horizon step")
    axes.set_ylabel("mean Euclidean error (window-level)")
    axes.set_title("Error growth over the forecast horizon (descriptive window mean)")
    axes.legend(fontsize=8)
    axes.grid(True, alpha=0.3)
    _save(figure, figures_directory / "error_by_horizon.png")


def _plot_error_by_motion_type(output_directory: Path, figures_directory: Path) -> None:
    rows = [
        row
        for row in _read_csv(output_directory / "per_motion_type_metrics.csv")
        if row["metric"] == "ade"
    ]
    motion_types = sorted({row["trajectory_type"] for row in rows})
    positions = np.arange(len(motion_types))
    width = 0.2
    figure, axes = plt.subplots(figsize=(9, 4.5))
    for offset, method in enumerate(METHOD_ORDER):
        values = []
        errors = []
        for motion in motion_types:
            match = [
                row
                for row in rows
                if row["model_name"] == method and row["trajectory_type"] == motion
            ]
            seed_values = [float(row["episode_mean"]) for row in match]
            if seed_values:
                mean, deviation = seed_mean_and_sample_std(seed_values)
                values.append(mean)
                errors.append(deviation)
            else:
                values.append(np.nan)
                errors.append(np.nan)
        axes.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            yerr=errors if method == "deterministic_gru" else None,
            capsize=3 if method == "deterministic_gru" else 0,
            label=method,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(motion_types, rotation=20, ha="right", fontsize=8)
    axes.set_ylabel("ADE (episode-mean within type)")
    axes.set_title("ADE by trajectory type (episode-level formal grouping)")
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", alpha=0.3)
    _save(figure, figures_directory / "error_by_motion_type.png")


def _plot_error_by_occlusion_group(output_directory: Path, figures_directory: Path) -> None:
    rows = [
        row
        for row in _read_csv(output_directory / "per_occlusion_group_metrics.csv")
        if row["metric"] == "ade"
    ]
    bin_order = ["0", "1-5", "6-10", "11-15", "16-20"]
    present = [b for b in bin_order if any(row["occlusion_length_bin"] == b for row in rows)]
    positions = np.arange(len(present))
    width = 0.2
    figure, axes = plt.subplots(figsize=(9, 4.5))
    for offset, method in enumerate(METHOD_ORDER):
        values = []
        errors = []
        for group in present:
            match = [
                row
                for row in rows
                if row["model_name"] == method and row["occlusion_length_bin"] == group
            ]
            seed_values = [float(row["window_mean"]) for row in match]
            if seed_values:
                mean, deviation = seed_mean_and_sample_std(seed_values)
                values.append(mean)
                errors.append(deviation)
            else:
                values.append(np.nan)
                errors.append(np.nan)
        axes.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            yerr=errors if method == "deterministic_gru" else None,
            capsize=3 if method == "deterministic_gru" else 0,
            label=method,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(present, fontsize=8)
    axes.set_xlabel("occlusion length bin")
    axes.set_ylabel("ADE (window-level descriptive)")
    axes.set_title("ADE by occlusion length (descriptive window mean)")
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", alpha=0.3)
    _save(figure, figures_directory / "error_by_occlusion_group.png")


def _plot_trajectory_examples(output_directory: Path, figures_directory: Path) -> None:
    with np.load(str(output_directory / "predictions.npz"), allow_pickle=False) as blob:
        model_names = [str(name) for name in blob["model_name"]]
        seeds = np.asarray(blob["seed"], dtype=np.int64)
        prediction = blob["prediction"]
        future_position = blob["future_position"]
        episode_id = blob["episode_id"]
    example_count = min(4, future_position.shape[0])
    # Deterministically pick spread-out windows from the saved arrays.
    indices = np.linspace(0, future_position.shape[0] - 1, example_count, dtype=int)
    columns = 2
    rows_count = int(np.ceil(example_count / columns))
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(5.5 * columns, 4.0 * rows_count), squeeze=False
    )
    for plot_index, window_index in enumerate(indices):
        axis = axes[plot_index // columns][plot_index % columns]
        truth = future_position[window_index]
        axis.plot(
            truth[:, 0], truth[:, 1], marker="o", markersize=3, color="black",
            label="future truth",
        )
        for method_index, method in enumerate(model_names):
            path = prediction[method_index, window_index]
            label = method
            if seeds[method_index] >= 0:
                label = "{} (seed {})".format(method, int(seeds[method_index]))
            axis.plot(path[:, 0], path[:, 1], marker=".", markersize=2, label=label)
        axis.set_title(
            "window {} (episode {})".format(int(window_index), int(episode_id[window_index])),
            fontsize=9,
        )
        axis.set_xlabel("local x")
        axis.set_ylabel("local y")
        axis.grid(True, alpha=0.3)
        if plot_index == 0:
            axis.legend(fontsize=7)
    for empty_index in range(example_count, rows_count * columns):
        axes[empty_index // columns][empty_index % columns].axis("off")
    figure.suptitle("Example forecasts vs. future truth (local coordinates)")
    _save(figure, figures_directory / "trajectory_examples.png")


def _plot_runtime_comparison(output_directory: Path, figures_directory: Path) -> None:
    rows = _read_csv(output_directory / "runtime.csv")
    labels = []
    values = []
    deviations = []
    groups: Dict[tuple, List[float]] = {}
    for row in rows:
        key = (row["model_name"], row["device"], row["mode"])
        groups.setdefault(key, []).append(float(row["per_window_ms"]))
    for (model_name, device, mode), timings in groups.items():
        labels.append(
            "{}\n{} / {}".format(model_name, device, mode)
        )
        values.append(float(np.mean(timings)))
        # This is dispersion over repeated timing measurements, not independent
        # random seeds; retain the existing population definition.
        deviations.append(float(np.std(timings, ddof=0)))
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.5))
    axes.bar(positions, values, yerr=deviations, capsize=3, color="tab:blue")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes.set_ylabel("per-window inference time (ms)")
    axes.set_yscale("log")
    axes.set_title("Inference time per window (log scale)")
    axes.grid(True, axis="y", alpha=0.3)
    _save(figure, figures_directory / "runtime_comparison.png")


def generate_all_figures(
    output_directory: Path,
    figures_directory: Path,
    logger: logging.Logger,
    *,
    default_statistical_unit: str = "episode",
) -> None:
    """Produce all eight required figures from persisted artefacts only."""

    _plot_train_validation_loss(output_directory, figures_directory)
    _plot_metric_comparison(
        output_directory,
        figures_directory,
        "ade",
        "ade_comparison.png",
        default_statistical_unit,
    )
    _plot_metric_comparison(
        output_directory,
        figures_directory,
        "fde",
        "fde_comparison.png",
        default_statistical_unit,
    )
    _plot_error_by_horizon(output_directory, figures_directory)
    _plot_error_by_motion_type(output_directory, figures_directory)
    _plot_error_by_occlusion_group(output_directory, figures_directory)
    _plot_trajectory_examples(output_directory, figures_directory)
    _plot_runtime_comparison(output_directory, figures_directory)
    logger.info("Generated 8 figures in %s", figures_directory)


__all__ = ["generate_all_figures", "metric_comparison_values"]
