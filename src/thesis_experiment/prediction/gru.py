"""Deterministic GRU components for leakage-safe trajectory prediction.

The feature builder in this module deliberately accepts only the five fields
exposed by :class:`thesis_experiment.data.prediction_dataset.PredictionDataset`.
It therefore cannot silently append audit truth, trajectory labels, or future
motion parameters to a neural-network input.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from thesis_experiment.data.prediction_dataset import MODEL_INPUT_FIELDS


GRU_FEATURE_NAMES: Tuple[str, ...] = (
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "position_mask",
    "velocity_mask",
    "delta_t",
)
"""Ordered names of the seven features supplied at every history step."""


ArrayLike = Union[np.ndarray, Tensor, float]


def _as_tensor(
    value: ArrayLike,
    *,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> Tensor:
    """Convert a numeric PredictionDataset value without copying unnecessarily."""

    if isinstance(value, Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _require_finite(name: str, value: Tensor) -> None:
    """Raise a clear error before a non-finite value reaches the recurrent net."""

    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("{} contains NaN or infinite values".format(name))


def _validate_binary_mask(name: str, value: Tensor) -> None:
    """Validate a floating or integer tensor used as a binary feature mask."""

    _require_finite(name, value)
    if not bool(torch.logical_or(value == 0, value == 1).all().item()):
        raise ValueError("{} must contain only 0 and 1".format(name))


def build_gru_features(
    inputs: Mapping[str, ArrayLike],
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """Build the strict seven-dimensional GRU history sequence.

    Parameters
    ----------
    inputs:
        The ``inputs`` mapping returned by ``PredictionDataset`` (or a batch
        produced from it).  Its keys must be exactly ``MODEL_INPUT_FIELDS``.
        Requiring exact equality makes the leakage boundary fail closed if an
        audit or future field is accidentally added by calling code.
    dtype, device:
        Floating dtype and device of the returned tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(H, 7)`` for one sample or ``(N, H, 7)`` for a batch.  Feature
        order is position, velocity, their two masks, and the broadcast sample
        time step.

    Raises
    ------
    ValueError
        If fields, shapes, masks, time steps, or numerical values are invalid.
    """

    expected = set(MODEL_INPUT_FIELDS)
    actual = set(inputs.keys())
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            "GRU inputs must exactly match the PredictionDataset whitelist; "
            "missing={}, unexpected={}".format(missing, unexpected)
        )

    position = _as_tensor(
        inputs["history_position"], dtype=dtype, device=device
    )
    velocity = _as_tensor(
        inputs["history_velocity"], dtype=dtype, device=device
    )
    position_mask = _as_tensor(
        inputs["history_mask"], dtype=dtype, device=device
    )
    velocity_mask = _as_tensor(
        inputs["history_velocity_mask"], dtype=dtype, device=device
    )
    delta_t = _as_tensor(
        inputs["time_step_seconds"], dtype=dtype, device=device
    )

    if position.ndim not in (2, 3) or position.shape[-1] != 2:
        raise ValueError(
            "history_position must have shape (H, 2) or (N, H, 2)"
        )
    if velocity.shape != position.shape:
        raise ValueError("history_velocity must match history_position shape")
    history_shape = position.shape[:-1]
    if tuple(position_mask.shape) != tuple(history_shape):
        raise ValueError("history_mask must match the history time dimensions")
    if tuple(velocity_mask.shape) != tuple(history_shape):
        raise ValueError(
            "history_velocity_mask must match the history time dimensions"
        )

    expected_delta_shape = position.shape[:-2]
    if tuple(delta_t.shape) != tuple(expected_delta_shape):
        raise ValueError(
            "time_step_seconds must be scalar for one sample or shape (N,) "
            "for a batch"
        )

    _require_finite("history_position", position)
    _require_finite("history_velocity", velocity)
    _validate_binary_mask("history_mask", position_mask)
    _validate_binary_mask("history_velocity_mask", velocity_mask)
    _require_finite("time_step_seconds", delta_t)
    if not bool((delta_t > 0).all().item()):
        raise ValueError("time_step_seconds must be positive")

    history_steps = int(position.shape[-2])
    broadcast_shape = tuple(expected_delta_shape) + (history_steps, 1)
    delta_feature = delta_t.unsqueeze(-1).unsqueeze(-1).expand(broadcast_shape)
    features = torch.cat(
        (
            position,
            velocity,
            position_mask.unsqueeze(-1),
            velocity_mask.unsqueeze(-1),
            delta_feature,
        ),
        dim=-1,
    )
    _require_finite("GRU feature tensor", features)
    return features


class DeterministicGRU(nn.Module):
    """GRU encoder with a deterministic multi-horizon position head.

    The model predicts every future local position directly from the final GRU
    hidden state.  It contains no probabilistic head and accepts no metadata.
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        future_steps: int = 20,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.future_steps = int(future_steps)
        recurrent_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_head = nn.Linear(self.hidden_size, self.future_steps * 2)

    def forward(self, features: Tensor) -> Tensor:
        """Predict ``(batch, future_steps, 2)`` local future positions."""

        if features.ndim != 3:
            raise ValueError("GRU features must have shape (N, H, input_size)")
        if features.shape[-1] != self.input_size:
            raise ValueError(
                "GRU feature size {} does not match configured input_size {}".format(
                    features.shape[-1], self.input_size
                )
            )
        if features.shape[0] <= 0 or features.shape[1] <= 0:
            raise ValueError("GRU batch and history dimensions must be non-empty")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("GRU features contain NaN or infinite values")

        _, hidden = self.gru(features)
        prediction = self.output_head(hidden[-1])
        return prediction.reshape(features.shape[0], self.future_steps, 2)


__all__ = [
    "DeterministicGRU",
    "GRU_FEATURE_NAMES",
    "build_gru_features",
]
