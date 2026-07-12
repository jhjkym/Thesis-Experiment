"""Strict, NumPy-only loader for future trajectory prediction samples.

The dataset-v2 archives deliberately contain audit arrays and complete episode
state in addition to the fields needed by a prediction model.  This module is
the boundary between those archives and future model code: only the explicit
whitelists below can leave the loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


MODEL_INPUT_FIELDS: Tuple[str, ...] = (
    "history_position",
    "history_velocity",
    "history_mask",
    "history_velocity_mask",
    "time_step_seconds",
)
"""Fields that may be exposed to a future prediction model."""

SUPERVISION_FIELDS: Tuple[str, ...] = ("future_position",)
"""Fields that may be exposed as supervision labels."""

INDEX_METADATA_FIELDS: Tuple[str, ...] = (
    "scene_id",
    "episode_id",
    "sample_start_index",
)
"""Non-model metadata permitted for grouping and sample identification."""

FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "history_true_position",
        "future_position",
        "trajectory_type",
        "coordinate_origin",
        "sensor_position",
        "tree_centers",
        "tree_radii",
        "history_occluded",
        "history_random_dropout",
        "history_start_time",
        "future_start_time",
        "scene_ids",
        "scene_sensor_position_world",
        "scene_tree_centers_world",
        "scene_tree_radii",
        "episode_ids",
        "episode_scene_ids",
        "episode_trajectory_types",
        "episode_true_position_world",
        "episode_observed_position_world",
        "episode_visible_mask",
        "episode_occluded_mask",
        "episode_random_dropout_mask",
        "episode_velocity_world",
        "episode_acceleration_world",
        "episode_times",
        "episode_initial_position",
        "episode_initial_velocity",
        "episode_acceleration_parameter",
        "episode_turn_rate",
        "episode_stop_start_time",
        "episode_stop_duration",
        "episode_piecewise_turn_time",
        "episode_piecewise_turn_angle",
    }
)
"""Known audit, truth, and future-parameter fields forbidden as inputs.

The model-input whitelist remains authoritative: fields not listed here are
also rejected unless they are present in :data:`MODEL_INPUT_FIELDS`.
"""


def _normalise_field_selection(
    requested: Optional[Sequence[str]],
    default: Tuple[str, ...],
    allowed: Tuple[str, ...],
    selection_name: str,
) -> Tuple[str, ...]:
    """Return a validated, duplicate-free field selection."""

    if requested is None:
        selected = default
    else:
        if isinstance(requested, str):
            raise TypeError("{} must be a sequence of field names".format(selection_name))
        selected = tuple(requested)
    if len(set(selected)) != len(selected):
        raise ValueError("{} contains duplicate fields".format(selection_name))
    disallowed = sorted(set(selected).difference(allowed))
    if disallowed:
        raise ValueError(
            "{} contains fields outside the strict whitelist: {}".format(
                selection_name, ", ".join(disallowed)
            )
        )
    return selected


def _filled_finite(array: np.ndarray, fill_value: float) -> np.ndarray:
    """Copy an array and replace every non-finite value with ``fill_value``."""

    values = np.asarray(array)
    try:
        finite = np.isfinite(values)
    except TypeError as error:
        raise ValueError("prediction inputs must have numeric dtypes") from error
    return np.where(finite, values, fill_value)


def _selected_value(array: np.ndarray, index: Any) -> Any:
    """Index an internal array without exposing a mutable internal view."""

    value = array[index]
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


class PredictionDataset:
    """Load leakage-safe prediction samples from one dataset-v2 split.

    Args:
        archive_path: Path to a split ``.npz`` file.
        fill_value: Finite replacement for missing position and velocity
            components.  The default is zero.  The original observation mask
            and the derived velocity mask remain available to distinguish
            replacements from observations.
        input_fields: Optional ordered subset of :data:`MODEL_INPUT_FIELDS`.
            Any other field is rejected rather than silently exposed.
        metadata_fields: Optional ordered subset of
            :data:`INDEX_METADATA_FIELDS`.
        target_field: Supervision field.  It must be ``future_position``.

    ``__getitem__`` and :meth:`get_batch` return dictionaries with exactly
    three top-level entries: ``inputs``, ``target``, and ``metadata``.  The
    target value is the ``future_position`` array itself, not a model input.
    The underlying ``NpzFile`` is closed during construction and only arrays
    needed for the requested view are retained.
    """

    def __init__(
        self,
        archive_path: Union[str, Path],
        fill_value: float = 0.0,
        input_fields: Optional[Sequence[str]] = None,
        metadata_fields: Optional[Sequence[str]] = None,
        target_field: str = "future_position",
    ) -> None:
        self._path = Path(archive_path)
        if not self._path.is_file():
            raise FileNotFoundError("prediction archive not found: {}".format(self._path))

        self._fill_value = float(fill_value)
        if not np.isfinite(self._fill_value):
            raise ValueError("fill_value must be finite")

        self._input_fields = _normalise_field_selection(
            input_fields,
            MODEL_INPUT_FIELDS,
            MODEL_INPUT_FIELDS,
            "input_fields",
        )
        self._metadata_fields = _normalise_field_selection(
            metadata_fields,
            INDEX_METADATA_FIELDS,
            INDEX_METADATA_FIELDS,
            "metadata_fields",
        )
        if target_field not in SUPERVISION_FIELDS:
            raise ValueError(
                "target_field must be one of: {}".format(
                    ", ".join(SUPERVISION_FIELDS)
                )
            )
        self._target_field = target_field

        archive_input_fields = {
            field
            for field in self._input_fields
            if field != "history_velocity_mask"
        }
        if "history_velocity_mask" in self._input_fields:
            archive_input_fields.add("history_velocity")
        required_fields = (
            archive_input_fields
            | set(self._metadata_fields)
            | {self._target_field}
        )

        loaded: Dict[str, np.ndarray] = {}
        with np.load(str(self._path), allow_pickle=False) as archive:
            missing = sorted(required_fields.difference(archive.files))
            if missing:
                raise ValueError(
                    "prediction archive is missing required fields: {}".format(
                        ", ".join(missing)
                    )
                )
            for field in required_fields:
                loaded[field] = np.array(archive[field], copy=True)

        target = loaded[self._target_field]
        if target.ndim < 1:
            raise ValueError("future_position must have a sample dimension")
        if not np.all(np.isfinite(target)):
            raise ValueError("future_position contains NaN or infinite values")
        self._length = int(target.shape[0])

        for field, values in loaded.items():
            if values.ndim < 1 or values.shape[0] != self._length:
                raise ValueError(
                    "field {} does not share the target sample dimension".format(field)
                )

        history_length: Optional[int] = None
        if "history_mask" in loaded:
            history_mask = loaded["history_mask"]
            if history_mask.ndim != 2:
                raise ValueError("history_mask must have shape (N, H)")
            if not bool(
                np.all((history_mask == 0) | (history_mask == 1))
            ):
                raise ValueError("history_mask must contain only 0 and 1")
            history_length = int(history_mask.shape[1])
        for field in ("history_position", "history_velocity"):
            if field not in loaded:
                continue
            values = loaded[field]
            if values.ndim != 3 or values.shape[-1] != 2:
                raise ValueError("{} must have shape (N, H, 2)".format(field))
            if history_length is None:
                history_length = int(values.shape[1])
            elif values.shape[1] != history_length:
                raise ValueError("history fields must share the same H dimension")
        if "history_position" in loaded and "history_mask" in loaded:
            position_finite = np.all(
                np.isfinite(loaded["history_position"]), axis=-1
            )
            if not np.array_equal(position_finite, loaded["history_mask"].astype(bool)):
                raise ValueError(
                    "history_position finite values do not match history_mask"
                )

        inputs: Dict[str, np.ndarray] = {}
        if "history_position" in self._input_fields:
            history_position = loaded["history_position"]
            inputs["history_position"] = _filled_finite(
                history_position, self._fill_value
            )
        if "history_velocity" in self._input_fields or (
            "history_velocity_mask" in self._input_fields
        ):
            history_velocity = loaded["history_velocity"]
            velocity_mask = np.all(np.isfinite(history_velocity), axis=-1).astype(
                np.uint8
            )
            if "history_velocity" in self._input_fields:
                inputs["history_velocity"] = _filled_finite(
                    history_velocity, self._fill_value
                )
            if "history_velocity_mask" in self._input_fields:
                inputs["history_velocity_mask"] = velocity_mask
        if "history_mask" in self._input_fields:
            history_mask = loaded["history_mask"]
            inputs["history_mask"] = history_mask
        if "time_step_seconds" in self._input_fields:
            time_step = loaded["time_step_seconds"]
            if not np.all(np.isfinite(time_step)):
                raise ValueError("time_step_seconds contains non-finite values")
            if not np.all(time_step > 0.0):
                raise ValueError("time_step_seconds must be positive")
            inputs["time_step_seconds"] = time_step

        self._inputs = {field: inputs[field] for field in self._input_fields}
        self._target = target
        self._metadata = {
            field: loaded[field] for field in self._metadata_fields
        }

    @property
    def archive_path(self) -> Path:
        """Return the source archive path without exposing its contents."""

        return self._path

    @property
    def fill_value(self) -> float:
        """Return the configured finite missing-value replacement."""

        return self._fill_value

    @property
    def input_fields(self) -> Tuple[str, ...]:
        """Return the ordered model-input whitelist selected by this loader."""

        return self._input_fields

    @property
    def target_field(self) -> str:
        """Return the sole supervision field name."""

        return self._target_field

    @property
    def metadata_fields(self) -> Tuple[str, ...]:
        """Return the ordered sample-index metadata fields."""

        return self._metadata_fields

    def __len__(self) -> int:
        """Return the number of window samples in this split."""

        return self._length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one sample with strict input, target, and metadata sections."""

        if not isinstance(index, (int, np.integer)):
            raise TypeError("PredictionDataset indices must be integers")
        normalised_index = int(index)
        if normalised_index < 0:
            normalised_index += self._length
        if normalised_index < 0 or normalised_index >= self._length:
            raise IndexError("PredictionDataset index out of range")
        return self._make_result(normalised_index)

    def get_batch(self, indices: Union[slice, Sequence[int], np.ndarray]) -> Dict[str, Any]:
        """Return a selected batch while preserving the same strict schema."""

        if isinstance(indices, slice):
            selection: Any = indices
        else:
            selection_array = np.asarray(indices)
            if selection_array.ndim != 1 or not np.issubdtype(
                selection_array.dtype, np.integer
            ):
                raise TypeError("batch indices must be a one-dimensional integer sequence")
            selection = selection_array.astype(np.int64, copy=False)
        return self._make_result(selection)

    def _make_result(self, index: Any) -> Dict[str, Any]:
        """Build a detached sample or batch result for an already-valid index."""

        return {
            "inputs": {
                field: _selected_value(self._inputs[field], index)
                for field in self._input_fields
            },
            "target": _selected_value(self._target, index),
            "metadata": {
                field: _selected_value(self._metadata[field], index)
                for field in self._metadata_fields
            },
        }


__all__ = [
    "FORBIDDEN_INPUT_FIELDS",
    "INDEX_METADATA_FIELDS",
    "MODEL_INPUT_FIELDS",
    "PredictionDataset",
    "SUPERVISION_FIELDS",
]
