"""Mask-aware training-set normalization for deterministic prediction models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from thesis_experiment.data.prediction_dataset import (
    MODEL_INPUT_FIELDS,
    PredictionDataset,
)


NumericArray = Union[np.ndarray, Tensor]
NORMALIZATION_SCHEMA_VERSION = 1


def _pair(values: Any, name: str, *, positive: bool = False) -> Tuple[float, float]:
    """Parse a finite length-two JSON vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2,) or not bool(np.all(np.isfinite(array))):
        raise ValueError("{} must contain two finite values".format(name))
    if positive and not bool(np.all(array > 0.0)):
        raise ValueError("{} must contain positive values".format(name))
    return float(array[0]), float(array[1])


@dataclass(frozen=True)
class NormalizationStatistics:
    """Serializable position and velocity statistics fitted on train only."""

    position_mean: Tuple[float, float]
    position_scale: Tuple[float, float]
    velocity_mean: Tuple[float, float]
    velocity_scale: Tuple[float, float]
    valid_position_count: int
    valid_velocity_count: int
    source_split: str = "train"
    schema_version: int = NORMALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_mean", _pair(self.position_mean, "position_mean")
        )
        object.__setattr__(
            self,
            "position_scale",
            _pair(self.position_scale, "position_scale", positive=True),
        )
        object.__setattr__(
            self, "velocity_mean", _pair(self.velocity_mean, "velocity_mean")
        )
        object.__setattr__(
            self,
            "velocity_scale",
            _pair(self.velocity_scale, "velocity_scale", positive=True),
        )
        if self.valid_position_count <= 0:
            raise ValueError("valid_position_count must be positive")
        if self.valid_velocity_count <= 0:
            raise ValueError("valid_velocity_count must be positive")
        if self.source_split != "train":
            raise ValueError("normalization statistics must come from train")
        if self.schema_version != NORMALIZATION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported normalization schema version: {}".format(
                    self.schema_version
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "source_split": self.source_split,
            "position_mean": list(self.position_mean),
            "position_scale": list(self.position_scale),
            "velocity_mean": list(self.velocity_mean),
            "velocity_scale": list(self.velocity_scale),
            "valid_position_count": self.valid_position_count,
            "valid_velocity_count": self.valid_velocity_count,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "NormalizationStatistics":
        """Validate and construct statistics read from JSON."""

        required = {
            "schema_version",
            "source_split",
            "position_mean",
            "position_scale",
            "velocity_mean",
            "velocity_scale",
            "valid_position_count",
            "valid_velocity_count",
        }
        if set(values.keys()) != required:
            raise ValueError(
                "normalization JSON fields do not match schema; missing={}, "
                "unexpected={}".format(
                    sorted(required.difference(values.keys())),
                    sorted(set(values.keys()).difference(required)),
                )
            )
        return cls(
            position_mean=values["position_mean"],
            position_scale=values["position_scale"],
            velocity_mean=values["velocity_mean"],
            velocity_scale=values["velocity_scale"],
            valid_position_count=int(values["valid_position_count"]),
            valid_velocity_count=int(values["valid_velocity_count"]),
            source_split=str(values["source_split"]),
            schema_version=int(values["schema_version"]),
        )


def _masked_statistics(
    values: np.ndarray, mask: np.ndarray, name: str, minimum_scale: float
) -> Tuple[Tuple[float, float], Tuple[float, float], int]:
    """Compute component-wise population moments from valid time steps only."""

    numeric_values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(mask, dtype=bool)
    if numeric_values.ndim != 3 or numeric_values.shape[-1] != 2:
        raise ValueError("{} values must have shape (N, H, 2)".format(name))
    if valid_mask.shape != numeric_values.shape[:-1]:
        raise ValueError("{} mask shape does not match values".format(name))
    selected = numeric_values[valid_mask]
    if selected.shape[0] == 0:
        raise ValueError("{} has no valid observations".format(name))
    if not bool(np.all(np.isfinite(selected))):
        raise ValueError("{} valid observations must be finite".format(name))
    mean = np.mean(selected, axis=0)
    scale = np.std(selected, axis=0)
    scale = np.maximum(scale, minimum_scale)
    return (
        (float(mean[0]), float(mean[1])),
        (float(scale[0]), float(scale[1])),
        int(selected.shape[0]),
    )


class PredictionNormalizer:
    """Apply train-only, mask-aware position and velocity normalization."""

    def __init__(self, statistics: NormalizationStatistics) -> None:
        self.statistics = statistics

    @classmethod
    def fit(
        cls,
        dataset: PredictionDataset,
        *,
        split_name: str = "train",
        minimum_scale: float = 1.0e-8,
    ) -> "PredictionNormalizer":
        """Fit using only valid history observations from a train split.

        Fill values are excluded through ``history_mask`` and the derived
        ``history_velocity_mask``.  Future labels are not used to estimate any
        statistic.
        """

        if not isinstance(dataset, PredictionDataset):
            raise TypeError("dataset must be a PredictionDataset")
        if split_name != "train":
            raise ValueError("normalization may only be fitted on the train split")
        if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be finite and positive")
        if set(dataset.input_fields) != set(MODEL_INPUT_FIELDS):
            raise ValueError(
                "normalization fitting requires the complete PredictionDataset "
                "model-input whitelist"
            )

        batch = dataset.get_batch(slice(None))
        inputs = batch["inputs"]
        position_mean, position_scale, position_count = _masked_statistics(
            inputs["history_position"],
            inputs["history_mask"],
            "history_position",
            float(minimum_scale),
        )
        velocity_mean, velocity_scale, velocity_count = _masked_statistics(
            inputs["history_velocity"],
            inputs["history_velocity_mask"],
            "history_velocity",
            float(minimum_scale),
        )
        return cls(
            NormalizationStatistics(
                position_mean=position_mean,
                position_scale=position_scale,
                velocity_mean=velocity_mean,
                velocity_scale=velocity_scale,
                valid_position_count=position_count,
                valid_velocity_count=velocity_count,
                source_split="train",
            )
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save validated train statistics as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(self.statistics.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PredictionNormalizer":
        """Load and validate train normalization statistics from JSON."""

        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("normalization JSON root must be an object")
        return cls(NormalizationStatistics.from_dict(values))

    @staticmethod
    def _affine(
        values: NumericArray,
        center: Tuple[float, float],
        scale: Tuple[float, float],
        *,
        inverse: bool,
    ) -> NumericArray:
        """Apply a two-component affine operation preserving NumPy/Torch type."""

        if isinstance(values, Tensor):
            if not values.is_floating_point():
                values = values.float()
            center_value = values.new_tensor(center)
            scale_value = values.new_tensor(scale)
            result = (
                values * scale_value + center_value
                if inverse
                else (values - center_value) / scale_value
            )
            return result
        array = np.asarray(values)
        if array.dtype.kind not in "fc":
            array = array.astype(np.float64)
        center_value = np.asarray(center, dtype=array.dtype)
        scale_value = np.asarray(scale, dtype=array.dtype)
        return (
            array * scale_value + center_value
            if inverse
            else (array - center_value) / scale_value
        )

    @staticmethod
    def _zero_invalid(values: NumericArray, mask: NumericArray) -> NumericArray:
        """Force filled/missing time steps to zero after normalization."""

        if isinstance(values, Tensor):
            mask_tensor = torch.as_tensor(mask, device=values.device).bool()
            if tuple(mask_tensor.shape) != tuple(values.shape[:-1]):
                raise ValueError("normalization mask shape does not match values")
            return torch.where(mask_tensor.unsqueeze(-1), values, torch.zeros_like(values))
        array = np.asarray(values)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != array.shape[:-1]:
            raise ValueError("normalization mask shape does not match values")
        return np.where(mask_array[..., None], array, np.zeros_like(array))

    def normalize_inputs(
        self, inputs: Mapping[str, NumericArray]
    ) -> Dict[str, NumericArray]:
        """Normalize strict model inputs and retain both masks and time step.

        Missing position and velocity steps are set to zero *after*
        normalization, ensuring that an arbitrary loader fill value cannot be
        mistaken for an observed standardized value.
        """

        expected = set(MODEL_INPUT_FIELDS)
        if set(inputs.keys()) != expected:
            raise ValueError(
                "normalization inputs must exactly match PredictionDataset whitelist"
            )
        position = self._affine(
            inputs["history_position"],
            self.statistics.position_mean,
            self.statistics.position_scale,
            inverse=False,
        )
        velocity = self._affine(
            inputs["history_velocity"],
            self.statistics.velocity_mean,
            self.statistics.velocity_scale,
            inverse=False,
        )
        position = self._zero_invalid(position, inputs["history_mask"])
        velocity = self._zero_invalid(
            velocity, inputs["history_velocity_mask"]
        )
        return {
            "history_position": position,
            "history_velocity": velocity,
            "history_mask": inputs["history_mask"],
            "history_velocity_mask": inputs["history_velocity_mask"],
            "time_step_seconds": inputs["time_step_seconds"],
        }

    def normalize_target(self, future_position: NumericArray) -> NumericArray:
        """Normalize future supervision using train history-position moments."""

        return self._affine(
            future_position,
            self.statistics.position_mean,
            self.statistics.position_scale,
            inverse=False,
        )

    def inverse_position(self, normalized_position: NumericArray) -> NumericArray:
        """Restore normalized predictions or labels to local position units."""

        return self._affine(
            normalized_position,
            self.statistics.position_mean,
            self.statistics.position_scale,
            inverse=True,
        )

    def inverse_target(self, normalized_target: NumericArray) -> NumericArray:
        """Alias for restoring a normalized future label or prediction."""

        return self.inverse_position(normalized_target)


def fit_prediction_normalizer(
    dataset: PredictionDataset,
    *,
    split_name: str = "train",
    minimum_scale: float = 1.0e-8,
) -> PredictionNormalizer:
    """Functional wrapper around :meth:`PredictionNormalizer.fit`."""

    return PredictionNormalizer.fit(
        dataset, split_name=split_name, minimum_scale=minimum_scale
    )


__all__ = [
    "NORMALIZATION_SCHEMA_VERSION",
    "NormalizationStatistics",
    "PredictionNormalizer",
    "fit_prediction_normalizer",
]
