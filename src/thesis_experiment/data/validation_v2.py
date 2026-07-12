"""Independent validation of saved dataset-v2 split archives and manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import json
import numpy as np

from thesis_experiment.config_v2 import SPLIT_NAMES
from thesis_experiment.data.dataset_v2 import (
    OCCLUSION_LENGTH_BIN_TO_CODE,
    validate_trajectory,
)
from thesis_experiment.data.prediction_dataset import (
    INDEX_METADATA_FIELDS,
    MODEL_INPUT_FIELDS,
    SUPERVISION_FIELDS,
)
from thesis_experiment.geometry.occlusion import segment_intersects_circles


REQUIRED_WINDOW_FIELDS = (
    "history_position",
    "history_velocity",
    "history_mask",
    "history_occluded",
    "history_random_dropout",
    "history_true_position",
    "future_position",
    "coordinate_origin",
    "sensor_position",
    "tree_centers",
    "tree_radii",
    "scene_id",
    "episode_id",
    "trajectory_type",
    "sample_start_index",
    "history_start_time",
    "future_start_time",
    "time_step_seconds",
    "history_visible_count",
    "last_valid_observation_age_steps",
    "valid_velocity_count",
    "history_max_consecutive_occlusion_steps",
    "occlusion_length_bin",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("missing manifest: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return value


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError("missing split archive: {}".format(path))
    with np.load(str(path), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _assert(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
        raise ValueError(
            "manifest mismatch for {}: {} != {}".format(name, actual, expected)
        )


def _assert_binary(array: np.ndarray, name: str) -> None:
    """Require an integer-like mask to contain only zero and one."""

    values = np.asarray(array)
    _assert(
        bool(np.all(np.isfinite(values))),
        "{} contains non-finite values".format(name),
    )
    _assert(
        bool(np.all((values == 0) | (values == 1))),
        "{} must contain only 0 and 1".format(name),
    )


def _scene_fingerprints(arrays: Dict[str, np.ndarray]) -> List[bytes]:
    """Return exact content fingerprints for saved scene definitions."""

    fingerprints: List[bytes] = []
    for index in range(arrays["scene_ids"].size):
        parts = (
            np.ascontiguousarray(arrays["scene_sensor_position_world"][index]),
            np.ascontiguousarray(arrays["scene_tree_centers_world"][index]),
            np.ascontiguousarray(arrays["scene_tree_radii"][index]),
        )
        fingerprints.append(b"".join(part.tobytes() for part in parts))
    return fingerprints


def _runs(mask_rows: np.ndarray) -> List[int]:
    lengths: List[int] = []
    for row in np.asarray(mask_rows, dtype=bool):
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        differences = np.diff(padded)
        starts = np.flatnonzero(differences == 1)
        ends = np.flatnonzero(differences == -1)
        lengths.extend((ends - starts).astype(int).tolist())
    return lengths


def _maximum_run(mask: np.ndarray) -> int:
    lengths = _runs(np.asarray(mask, dtype=bool)[np.newaxis, :])
    return int(max(lengths)) if lengths else 0


def _histogram(values: np.ndarray) -> Dict[str, int]:
    array = np.asarray(values, dtype=int)
    return {
        str(int(value)): int(np.sum(array == value)) for value in np.unique(array)
    }


def _bin_counts(values: np.ndarray) -> Dict[str, int]:
    array = np.asarray(values, dtype=int)
    return {
        name: int(np.sum(array == code))
        for name, code in OCCLUSION_LENGTH_BIN_TO_CODE.items()
    }


def _type_counts(codes: np.ndarray, encoding: Mapping[str, int]) -> Dict[str, int]:
    return {name: int(np.sum(codes == code)) for name, code in encoding.items()}


def _validate_field_roles(
    manifest: Mapping[str, Any], arrays_by_split: Mapping[str, Dict[str, np.ndarray]]
) -> None:
    """Validate the saved training boundary against the strict loader contract."""

    roles = manifest["field_roles"]
    _assert(
        tuple(roles["model_input_fields"]) == MODEL_INPUT_FIELDS,
        "manifest model input whitelist does not match PredictionDataset",
    )
    _assert(
        tuple(roles["supervision_label_fields"]) == SUPERVISION_FIELDS,
        "manifest supervision whitelist does not match PredictionDataset",
    )
    _assert(
        tuple(roles["sample_index_metadata_fields"]) == INDEX_METADATA_FIELDS,
        "manifest metadata whitelist does not match PredictionDataset",
    )
    inputs = set(roles["model_input_fields"])
    _assert(
        roles["trajectory_type_usage"] == "metadata_only",
        "trajectory_type must be metadata only",
    )
    _assert(
        "history_true_position" in roles["audit_only_fields"],
        "history_true_position must be audit only",
    )
    _assert(
        inputs.isdisjoint(roles["audit_only_fields"]),
        "audit-only fields appear in the model input whitelist",
    )
    _assert(
        inputs.isdisjoint(roles["future_motion_parameter_fields"]),
        "future motion parameters appear in the model input whitelist",
    )
    _assert(
        inputs.isdisjoint(roles["forbidden_model_input_fields"]),
        "forbidden fields appear in the model input whitelist",
    )
    _assert(
        "trajectory_type" not in inputs,
        "trajectory_type must not be a default model input",
    )
    for split_name, arrays in arrays_by_split.items():
        expected_forbidden = set(arrays).difference(inputs)
        _assert(
            set(roles["forbidden_model_input_fields"]) == expected_forbidden,
            "{} forbidden field list does not cover every saved non-input field".format(
                split_name
            ),
        )


def _validate_manifest_statistics(
    split_name: str,
    arrays: Dict[str, np.ndarray],
    manifest_split: Mapping[str, Any],
    encoding: Mapping[str, int],
) -> None:
    """Independently recompute every required saved split statistic."""

    _assert(
        int(manifest_split["scene_count"]) == int(arrays["scene_ids"].size),
        "{} scene_count mismatch".format(split_name),
    )
    _assert(
        int(manifest_split["episode_count"]) == int(arrays["episode_ids"].size),
        "{} episode_count mismatch".format(split_name),
    )
    _assert(
        int(manifest_split["window_count"]) == int(arrays["scene_id"].size),
        "{} window_count mismatch".format(split_name),
    )
    _assert(
        manifest_split["scene_ids"] == arrays["scene_ids"].astype(int).tolist(),
        "{} scene_ids mismatch".format(split_name),
    )
    _assert(
        manifest_split["episode_ids"] == arrays["episode_ids"].astype(int).tolist(),
        "{} episode_ids mismatch".format(split_name),
    )

    episode_counts = _type_counts(arrays["episode_trajectory_types"], encoding)
    window_counts = _type_counts(arrays["trajectory_type"], encoding)
    _assert(
        manifest_split["episode_trajectory_type_counts"] == episode_counts,
        "{} episode type count mismatch".format(split_name),
    )
    _assert(
        manifest_split["window_trajectory_type_counts"] == window_counts,
        "{} window type count mismatch".format(split_name),
    )
    episode_total = int(arrays["episode_ids"].size)
    window_total = int(arrays["episode_id"].size)
    for name in encoding:
        _assert_close(
            manifest_split["episode_trajectory_type_ratios"][name],
            episode_counts[name] / episode_total,
            "{}.episode_type_ratio.{}".format(split_name, name),
        )
        _assert_close(
            manifest_split["window_trajectory_type_ratios"][name],
            window_counts[name] / window_total,
            "{}.window_type_ratio.{}".format(split_name, name),
        )

    visible = arrays["history_mask"].astype(bool)
    occluded = arrays["history_occluded"].astype(bool)
    dropout = arrays["history_random_dropout"].astype(bool)
    _assert_close(
        manifest_split["visible_ratio"], np.mean(visible), split_name + ".visible_ratio"
    )
    _assert_close(
        manifest_split["geometric_occlusion_ratio"],
        np.mean(occluded),
        split_name + ".geometric_occlusion_ratio",
    )
    _assert_close(
        manifest_split["random_dropout_ratio"],
        np.mean(dropout),
        split_name + ".random_dropout_ratio",
    )
    _assert(
        int(manifest_split["occlusion_dropout_overlap_count"])
        == int(np.sum(occluded & dropout)),
        "{} overlap count mismatch".format(split_name),
    )

    speeds = np.linalg.norm(arrays["episode_velocity_world"], axis=2)
    accelerations = np.linalg.norm(arrays["episode_acceleration_world"], axis=2)
    for actual, expected, suffix in (
        (manifest_split["speed_range"][0], np.min(speeds), "speed_min"),
        (manifest_split["speed_range"][1], np.max(speeds), "speed_max"),
        (
            manifest_split["acceleration_range"][0],
            np.min(accelerations),
            "acceleration_min",
        ),
        (
            manifest_split["acceleration_range"][1],
            np.max(accelerations),
            "acceleration_max",
        ),
    ):
        _assert_close(actual, expected, split_name + "." + suffix)

    run_lengths = _runs(arrays["episode_occluded_mask"])
    _assert(
        manifest_split["consecutive_occlusion_lengths"] == run_lengths,
        "{} occlusion lengths mismatch".format(split_name),
    )
    histogram: Dict[str, int] = {}
    for length in run_lengths:
        key = str(length)
        histogram[key] = histogram.get(key, 0) + 1
    _assert(
        manifest_split["consecutive_occlusion_length_histogram"] == histogram,
        "{} occlusion histogram mismatch".format(split_name),
    )

    window_span = int(
        arrays["history_position"].shape[1] + arrays["future_position"].shape[1]
    )
    overlap_count = 0
    adjacent_count = 0
    for episode_id in arrays["episode_ids"]:
        starts = np.sort(
            arrays["sample_start_index"][arrays["episode_id"] == episode_id]
        )
        if starts.size > 1:
            adjacent_count += int(starts.size - 1)
            overlap_count += int(np.sum(np.diff(starts) < window_span))
    _assert(
        int(manifest_split["overlapping_adjacent_window_pairs"]) == overlap_count,
        "{} overlapping window count mismatch".format(split_name),
    )
    _assert(
        int(manifest_split["adjacent_window_pair_count"]) == adjacent_count,
        "{} adjacent window count mismatch".format(split_name),
    )
    _assert_close(
        manifest_split["overlapping_adjacent_window_ratio"],
        overlap_count / adjacent_count if adjacent_count else 0.0,
        split_name + ".overlapping_window_ratio",
    )
    _assert(
        manifest_split["history_visible_count_distribution"]
        == _histogram(arrays["history_visible_count"]),
        "{} visible history distribution mismatch".format(split_name),
    )
    _assert(
        manifest_split["last_valid_observation_age_steps_distribution"]
        == _histogram(arrays["last_valid_observation_age_steps"]),
        "{} last-observation distribution mismatch".format(split_name),
    )
    _assert(
        manifest_split["valid_velocity_count_distribution"]
        == _histogram(arrays["valid_velocity_count"]),
        "{} valid velocity distribution mismatch".format(split_name),
    )
    _assert(
        int(manifest_split["windows_without_valid_velocity_count"])
        == int(np.sum(arrays["valid_velocity_count"] == 0)),
        "{} zero-velocity-window count mismatch".format(split_name),
    )
    _assert(
        manifest_split["window_occlusion_length_bin_counts"]
        == _bin_counts(arrays["occlusion_length_bin"]),
        "{} occlusion bin count mismatch".format(split_name),
    )

    rejection_by_type = manifest_split["rejected_episode_counts_by_trajectory_type"]
    fully_total = 0
    insufficient_total = 0
    physical_total = 0
    for name, code in encoding.items():
        type_audit = manifest_split["trajectory_type_window_audit"][name]
        selected = arrays["trajectory_type"] == code
        _assert(
            int(type_audit["episode_count"])
            == int(np.sum(arrays["episode_trajectory_types"] == code)),
            "{}.{} episode audit mismatch".format(split_name, name),
        )
        _assert(
            int(type_audit["valid_window_count"]) == int(np.sum(selected)),
            "{}.{} window audit mismatch".format(split_name, name),
        )
        _assert(
            int(type_audit["valid_window_count"]) > 0,
            "{}.{} has no valid windows".format(split_name, name),
        )
        _assert(
            type_audit["occlusion_length_bin_counts"]
            == _bin_counts(arrays["occlusion_length_bin"][selected]),
            "{}.{} occlusion-bin audit mismatch".format(split_name, name),
        )
        rejected = rejection_by_type[name]
        _assert(
            int(rejected["total"])
            == int(rejected["fully_unobservable"])
            + int(rejected["insufficient_history"]),
            "{}.{} rejected total mismatch".format(split_name, name),
        )
        _assert(
            int(type_audit["rejected_episode_count"]) == int(rejected["total"]),
            "{}.{} rejected audit mismatch".format(split_name, name),
        )
        _assert(
            int(type_audit["rejected_physical_candidate_count"])
            == int(rejected["physical"]),
            "{}.{} physical rejection audit mismatch".format(split_name, name),
        )
        _assert(
            int(type_audit["rejected_total_candidate_count"])
            == int(rejected["total"]) + int(rejected["physical"]),
            "{}.{} total candidate rejection audit mismatch".format(
                split_name, name
            ),
        )
        fully_total += int(rejected["fully_unobservable"])
        insufficient_total += int(rejected["insufficient_history"])
        physical_total += int(rejected["physical"])
    _assert(
        fully_total == int(manifest_split["rejected_fully_unobservable_episode_count"]),
        split_name + " fully-unobservable rejection mismatch",
    )
    _assert(
        insufficient_total
        == int(manifest_split["rejected_insufficient_history_episode_count"]),
        split_name + " insufficient-history rejection mismatch",
    )
    _assert(
        physical_total
        == int(manifest_split["rejected_physical_episode_candidate_count"]),
        split_name + " physical rejection mismatch",
    )

    fields = manifest_split["fields"]
    _assert(set(fields) == set(arrays), "{} manifest field set mismatch".format(split_name))
    for name, array in arrays.items():
        _assert(
            fields[name]["shape"] == list(array.shape),
            "{}.{} shape mismatch".format(split_name, name),
        )
        _assert(
            fields[name]["dtype"] == str(array.dtype),
            "{}.{} dtype mismatch".format(split_name, name),
        )


def _validate_window_content(
    split_name: str,
    arrays: Dict[str, np.ndarray],
    history_steps: int,
    future_steps: int,
    window_stride: int,
    minimum_visible_history_steps: int,
    minimum_consecutive_visible_steps: int,
    minimum_windows_per_episode: int,
) -> Dict[str, float]:
    for field in REQUIRED_WINDOW_FIELDS:
        _assert(field in arrays, "{} missing field {}".format(split_name, field))
    window_count = int(arrays["episode_id"].size)
    for name in (
        "scene_id",
        "trajectory_type",
        "sample_start_index",
        "history_start_time",
        "future_start_time",
        "time_step_seconds",
    ):
        _assert(
            arrays[name].shape == (window_count,),
            "{}.{} must have one value per window".format(split_name, name),
        )
    _assert(
        arrays["history_position"].shape == (window_count, history_steps, 2),
        "{} history_position shape mismatch".format(split_name),
    )
    _assert(
        arrays["future_position"].shape == (window_count, future_steps, 2),
        "{} future_position shape mismatch".format(split_name),
    )
    for name in (
        "history_visible_count",
        "last_valid_observation_age_steps",
        "valid_velocity_count",
        "history_max_consecutive_occlusion_steps",
        "occlusion_length_bin",
    ):
        _assert(
            arrays[name].shape == (window_count,),
            "{}.{} must have one value per window".format(split_name, name),
        )
    mask = arrays["history_mask"].astype(bool)
    occluded = arrays["history_occluded"].astype(bool)
    dropout = arrays["history_random_dropout"].astype(bool)
    _assert_binary(arrays["history_mask"], split_name + ".history_mask")
    _assert_binary(arrays["history_occluded"], split_name + ".history_occluded")
    _assert_binary(
        arrays["history_random_dropout"],
        split_name + ".history_random_dropout",
    )
    _assert(np.array_equal(~mask, occluded | dropout), split_name + " mask causes mismatch")
    _assert(not np.any(occluded & dropout), split_name + " causes must be disjoint")
    _assert(
        np.all(np.isnan(arrays["history_position"][~mask])),
        split_name + " masked history must be NaN",
    )
    _assert(
        np.all(np.isfinite(arrays["history_position"][mask])),
        split_name + " visible history must be finite",
    )
    _assert(
        np.all(np.isfinite(arrays["history_true_position"])),
        split_name + " history truth must be finite",
    )
    _assert(
        np.all(np.isfinite(arrays["future_position"])),
        split_name + " future labels must be finite",
    )

    episode_lookup = {
        int(episode_id): index
        for index, episode_id in enumerate(arrays["episode_ids"].tolist())
    }
    scene_lookup = {
        int(scene_id): index
        for index, scene_id in enumerate(arrays["scene_ids"].tolist())
    }
    max_truth_error = 0.0
    max_velocity_error = 0.0
    max_observation_error = 0.0
    max_local_scene_error = 0.0
    for episode_id in arrays["episode_ids"]:
        _assert(
            int(np.sum(arrays["episode_id"] == episode_id))
            >= minimum_windows_per_episode,
            "{} episode {} has too few valid windows".format(
                split_name, int(episode_id)
            ),
        )
    for window_index in range(window_count):
        episode_id = int(arrays["episode_id"][window_index])
        _assert(
            episode_id in episode_lookup,
            "{} window references unknown episode {}".format(split_name, episode_id),
        )
        episode_index = episode_lookup[episode_id]
        scene_id = int(arrays["scene_id"][window_index])
        _assert(
            scene_id in scene_lookup,
            "{} window references unknown scene {}".format(split_name, scene_id),
        )
        scene_index = scene_lookup[scene_id]
        _assert(
            scene_id
            == int(arrays["episode_scene_ids"][episode_index]),
            "{} window scene/episode mismatch".format(split_name),
        )
        _assert(
            int(arrays["trajectory_type"][window_index])
            == int(arrays["episode_trajectory_types"][episode_index]),
            "{} window trajectory type mismatch".format(split_name),
        )
        start = int(arrays["sample_start_index"][window_index])
        _assert(start % window_stride == 0, split_name + " window violates stride")
        end = start + history_steps + future_steps
        truth = arrays["episode_true_position_world"][episode_index]
        _assert(end <= truth.shape[0], split_name + " window crosses episode end")
        origin = arrays["coordinate_origin"][window_index]
        recovered_history = arrays["history_true_position"][window_index] + origin
        recovered_future = arrays["future_position"][window_index] + origin
        recovered_sensor = arrays["sensor_position"][window_index] + origin
        recovered_trees = arrays["tree_centers"][window_index] + origin
        history_error = float(
            np.max(np.abs(recovered_history - truth[start : start + history_steps]))
        )
        future_error = float(
            np.max(np.abs(recovered_future - truth[start + history_steps : end]))
        )
        max_truth_error = max(max_truth_error, history_error, future_error)
        local_scene_error = max(
            float(
                np.max(
                    np.abs(
                        recovered_sensor
                        - arrays["scene_sensor_position_world"][scene_index]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        recovered_trees
                        - arrays["scene_tree_centers_world"][scene_index]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        arrays["tree_radii"][window_index]
                        - arrays["scene_tree_radii"][scene_index]
                    )
                )
            ),
        )
        max_local_scene_error = max(max_local_scene_error, local_scene_error)
        dt = float(arrays["time_step_seconds"][window_index])
        episode_times = arrays["episode_times"][episode_index]
        _assert_close(
            arrays["history_start_time"][window_index],
            episode_times[start],
            split_name + ".history_start_time",
        )
        _assert_close(
            arrays["future_start_time"][window_index],
            episode_times[start + history_steps],
            split_name + ".future_start_time",
        )
        local_observed = arrays["history_position"][window_index]
        local_mask = mask[window_index]
        expected_slice = slice(start, start + history_steps)
        _assert(
            np.array_equal(
                local_mask,
                arrays["episode_visible_mask"][episode_index][expected_slice].astype(
                    bool
                ),
            ),
            split_name + " window/full-episode visible mask mismatch",
        )
        _assert(
            np.array_equal(
                occluded[window_index],
                arrays["episode_occluded_mask"][episode_index][expected_slice].astype(
                    bool
                ),
            ),
            split_name + " window/full-episode occlusion mismatch",
        )
        _assert(
            np.array_equal(
                dropout[window_index],
                arrays["episode_random_dropout_mask"][episode_index][
                    expected_slice
                ].astype(bool),
            ),
            split_name + " window/full-episode dropout mismatch",
        )
        recovered_observed = local_observed + origin
        expected_observed = arrays["episode_observed_position_world"][episode_index][
            expected_slice
        ]
        _assert(
            bool(
                np.allclose(
                    recovered_observed,
                    expected_observed,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            ),
            split_name + " observed history was altered or interpolated",
        )
        if np.any(local_mask):
            max_observation_error = max(
                max_observation_error,
                float(
                    np.max(
                        np.abs(
                            recovered_observed[local_mask]
                            - expected_observed[local_mask]
                        )
                    )
                ),
            )
        visible_count = int(np.sum(local_mask))
        _assert(
            visible_count >= minimum_visible_history_steps,
            split_name + " window has insufficient visible history",
        )
        _assert(
            _maximum_run(local_mask) >= minimum_consecutive_visible_steps,
            split_name + " window has insufficient consecutive visibility",
        )
        _assert(
            int(arrays["history_visible_count"][window_index]) == visible_count,
            split_name + " visible history audit mismatch",
        )
        last_valid = int(np.flatnonzero(local_mask)[-1])
        _assert(
            int(arrays["last_valid_observation_age_steps"][window_index])
            == history_steps - 1 - last_valid,
            split_name + " last observation age mismatch",
        )
        expected_velocity = np.full((history_steps, 2), np.nan, dtype=float)
        adjacent = local_mask[1:] & local_mask[:-1]
        adjacent_indices = np.flatnonzero(adjacent) + 1
        expected_velocity[adjacent_indices] = (
            np.diff(local_observed, axis=0)[adjacent] / dt
        )
        saved_velocity = arrays["history_velocity"][window_index]
        finite = np.isfinite(expected_velocity)
        _assert(
            np.array_equal(np.isfinite(saved_velocity), finite),
            split_name + " history velocity mask mismatch",
        )
        if np.any(finite):
            max_velocity_error = max(
                max_velocity_error,
                float(np.max(np.abs(saved_velocity[finite] - expected_velocity[finite]))),
            )
        _assert(
            int(arrays["valid_velocity_count"][window_index])
            == int(np.sum(np.all(np.isfinite(saved_velocity), axis=1))),
            split_name + " valid velocity count mismatch",
        )
        maximum_occlusion = _maximum_run(occluded[window_index])
        _assert(
            int(arrays["history_max_consecutive_occlusion_steps"][window_index])
            == maximum_occlusion,
            split_name + " maximum occlusion length mismatch",
        )
        if maximum_occlusion == 0:
            expected_bin = OCCLUSION_LENGTH_BIN_TO_CODE["0"]
        elif maximum_occlusion <= 5:
            expected_bin = OCCLUSION_LENGTH_BIN_TO_CODE["1-5"]
        elif maximum_occlusion <= 10:
            expected_bin = OCCLUSION_LENGTH_BIN_TO_CODE["6-10"]
        elif maximum_occlusion <= 15:
            expected_bin = OCCLUSION_LENGTH_BIN_TO_CODE["11-15"]
        else:
            expected_bin = OCCLUSION_LENGTH_BIN_TO_CODE["16-20"]
        _assert(
            int(arrays["occlusion_length_bin"][window_index]) == expected_bin,
            split_name + " occlusion bin mismatch",
        )
    _assert(max_truth_error <= 1e-12, split_name + " local reconstruction mismatch")
    _assert(max_velocity_error <= 1e-12, split_name + " history velocity mismatch")
    _assert(
        max_local_scene_error <= 1e-12,
        split_name + " local scene reconstruction mismatch",
    )
    return {
        "maximum_world_reconstruction_error": max_truth_error,
        "maximum_history_velocity_error": max_velocity_error,
        "maximum_observation_reconstruction_error": max_observation_error,
        "maximum_local_scene_reconstruction_error": max_local_scene_error,
    }


def validate_dataset_directory(dataset_directory: Path) -> Dict[str, Any]:
    """Validate saved splits independently and return a JSON-ready summary."""

    directory = Path(dataset_directory)
    manifest = _load_json(directory / "dataset_manifest.json")
    _assert(manifest.get("dataset_version") == "2.0", "unexpected dataset version")
    encoding = {
        str(name): int(code)
        for name, code in manifest["trajectory_type_encoding"].items()
    }
    _assert(len(encoding) == 5, "trajectory encoding must define five modes")
    _assert(
        {
            str(name): int(code)
            for name, code in manifest["occlusion_length_bin_encoding"].items()
        }
        == OCCLUSION_LENGTH_BIN_TO_CODE,
        "unexpected occlusion length bin encoding",
    )
    split_seeds = manifest["split_seeds"]
    _assert(
        len({int(split_seeds[name]) for name in SPLIT_NAMES}) == 3,
        "split seeds must be distinct",
    )
    arrays_by_split = {
        name: _load_npz(directory / (name + ".npz")) for name in SPLIT_NAMES
    }
    _validate_field_roles(manifest, arrays_by_split)
    scene_sets: Dict[str, Set[int]] = {}
    episode_sets: Dict[str, Set[int]] = {}
    scene_fingerprint_sets: Dict[str, Set[bytes]] = {}
    split_summaries: Dict[str, Any] = {}
    history_steps = int(manifest["window"]["history_steps"])
    future_steps = int(manifest["window"]["future_steps"])
    stride = int(manifest["window"]["window_stride"])
    minimum_visible_history_steps = int(
        manifest["window"]["minimum_visible_history_steps"]
    )
    minimum_consecutive_visible_steps = int(
        manifest["window"]["minimum_consecutive_visible_steps"]
    )
    minimum_windows_per_episode = int(
        manifest["window"]["minimum_windows_per_episode"]
    )
    scene_size = (float(manifest["scene"]["width"]), float(manifest["scene"]["height"]))
    required_sensor_tree_clearance = float(
        manifest["scene"]["sensor_tree_clearance"]
    )
    trajectory_manifest = manifest["trajectory"]
    episodes_per_scene = int(manifest["episodes_per_scene"])

    for split_name in SPLIT_NAMES:
        arrays = arrays_by_split[split_name]
        scene_sets[split_name] = set(arrays["scene_ids"].astype(int).tolist())
        episode_sets[split_name] = set(arrays["episode_ids"].astype(int).tolist())
        fingerprints = _scene_fingerprints(arrays)
        _assert(
            len(set(fingerprints)) == len(fingerprints),
            split_name + " contains duplicated scene content",
        )
        scene_fingerprint_sets[split_name] = set(fingerprints)
        for field in (
            "episode_visible_mask",
            "episode_occluded_mask",
            "episode_random_dropout_mask",
        ):
            _assert_binary(arrays[field], "{}.{}".format(split_name, field))
        episode_visible = arrays["episode_visible_mask"].astype(bool)
        episode_occluded = arrays["episode_occluded_mask"].astype(bool)
        episode_dropout = arrays["episode_random_dropout_mask"].astype(bool)
        _assert(
            np.array_equal(~episode_visible, episode_occluded | episode_dropout),
            split_name + " full-episode observation causes mismatch",
        )
        _assert(
            not np.any(episode_occluded & episode_dropout),
            split_name + " full-episode missing causes must be disjoint",
        )
        _assert(
            np.all(
                np.isnan(
                    arrays["episode_observed_position_world"][~episode_visible]
                )
            ),
            split_name + " hidden full-episode observations must be NaN",
        )
        _assert(
            np.all(
                np.isfinite(
                    arrays["episode_observed_position_world"][episode_visible]
                )
            ),
            split_name + " visible full-episode observations must be finite",
        )
        _assert(
            set(arrays["scene_id"].astype(int).tolist()).issubset(scene_sets[split_name]),
            split_name + " window references foreign scene",
        )
        _assert(
            set(arrays["episode_scene_ids"].astype(int).tolist()).issubset(
                scene_sets[split_name]
            ),
            split_name + " episode references foreign scene",
        )
        _assert(
            int(manifest["scene_counts"][split_name]) == len(scene_sets[split_name]),
            split_name + " configured scene count mismatch",
        )
        for scene_id in scene_sets[split_name]:
            _assert(
                int(np.sum(arrays["episode_scene_ids"] == scene_id))
                == episodes_per_scene,
                "{} scene {} episode count mismatch".format(split_name, scene_id),
            )
        window_metrics = _validate_window_content(
            split_name,
            arrays,
            history_steps,
            future_steps,
            stride,
            minimum_visible_history_steps,
            minimum_consecutive_visible_steps,
            minimum_windows_per_episode,
        )

        scene_lookup = {
            int(scene_id): index
            for index, scene_id in enumerate(arrays["scene_ids"].tolist())
        }
        min_clearance = float("inf")
        min_sensor_tree_clearance = float("inf")
        max_speed = 0.0
        max_acceleration = 0.0
        geometry_checked_frames = 0
        geometry_mismatch_count = 0
        for scene_index in range(arrays["scene_ids"].size):
            sensor_surface_distances = np.linalg.norm(
                arrays["scene_tree_centers_world"][scene_index]
                - arrays["scene_sensor_position_world"][scene_index],
                axis=1,
            ) - arrays["scene_tree_radii"][scene_index]
            scene_minimum = float(np.min(sensor_surface_distances))
            _assert(
                scene_minimum >= required_sensor_tree_clearance - 1e-10,
                "{} scene {} violates sensor_tree_clearance".format(
                    split_name, int(arrays["scene_ids"][scene_index])
                ),
            )
            min_sensor_tree_clearance = min(
                min_sensor_tree_clearance, scene_minimum
            )
        for episode_index, scene_id_value in enumerate(arrays["episode_scene_ids"]):
            _assert(
                bool(np.any(arrays["episode_visible_mask"][episode_index])),
                "{} accepted a fully unobservable episode".format(split_name),
            )
            scene_index = scene_lookup[int(scene_id_value)]
            recomputed_occlusion = np.asarray(
                [
                    bool(
                        np.any(
                            segment_intersects_circles(
                                arrays["scene_sensor_position_world"][scene_index],
                                target,
                                arrays["scene_tree_centers_world"][scene_index],
                                arrays["scene_tree_radii"][scene_index],
                            )
                        )
                    )
                    for target in arrays["episode_true_position_world"][episode_index]
                ],
                dtype=bool,
            )
            saved_occlusion = arrays["episode_occluded_mask"][episode_index].astype(
                bool
            )
            geometry_checked_frames += int(recomputed_occlusion.size)
            geometry_mismatch_count += int(
                np.sum(recomputed_occlusion != saved_occlusion)
            )
            metrics = validate_trajectory(
                arrays["episode_true_position_world"][episode_index],
                arrays["episode_velocity_world"][episode_index],
                arrays["episode_acceleration_world"][episode_index],
                scene_size,
                arrays["scene_tree_centers_world"][scene_index],
                arrays["scene_tree_radii"][scene_index],
                float(trajectory_manifest["min_tree_clearance"]),
                float(trajectory_manifest["max_speed"]),
                float(trajectory_manifest["max_acceleration"]),
                float(trajectory_manifest["boundary_margin"]),
            )
            min_clearance = min(min_clearance, metrics["minimum_tree_surface_distance"])
            max_speed = max(max_speed, metrics["maximum_speed"])
            max_acceleration = max(
                max_acceleration, metrics["maximum_acceleration"]
            )
        _assert(
            geometry_mismatch_count == 0,
            "{} saved occlusion differs from geometry".format(split_name),
        )
        _validate_manifest_statistics(
            split_name, arrays, manifest["splits"][split_name], encoding
        )
        split_summaries[split_name] = {
            "scene_count": len(scene_sets[split_name]),
            "episode_count": len(episode_sets[split_name]),
            "window_count": int(arrays["episode_id"].size),
            "minimum_tree_surface_distance": min_clearance,
            "minimum_sensor_tree_surface_distance": min_sensor_tree_clearance,
            "maximum_speed": max_speed,
            "maximum_acceleration": max_acceleration,
            "occlusion_geometry_checked_frame_count": geometry_checked_frames,
            "occlusion_geometry_mismatch_count": geometry_mismatch_count,
            **window_metrics
        }

    scene_intersections: Dict[str, List[int]] = {}
    episode_intersections: Dict[str, List[int]] = {}
    scene_content_intersections: Dict[str, int] = {}
    for first_index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[first_index + 1 :]:
            key = first + "__" + second
            scene_intersections[key] = sorted(scene_sets[first] & scene_sets[second])
            episode_intersections[key] = sorted(
                episode_sets[first] & episode_sets[second]
            )
            scene_content_intersections[key] = len(
                scene_fingerprint_sets[first] & scene_fingerprint_sets[second]
            )
            _assert(not scene_intersections[key], "scene leakage for " + key)
            _assert(not episode_intersections[key], "episode leakage for " + key)
            _assert(
                scene_content_intersections[key] == 0,
                "duplicated scene content leakage for " + key,
            )

    return {
        "status": "passed",
        "dataset_directory": str(directory),
        "split_seeds": {name: int(split_seeds[name]) for name in SPLIT_NAMES},
        "scene_intersections": scene_intersections,
        "scene_content_intersection_counts": scene_content_intersections,
        "episode_intersections": episode_intersections,
        "splits": split_summaries,
    }
