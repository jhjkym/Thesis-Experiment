"""Shared, leakage-safe training and inference pipeline for experiment 02.

This module is the single source of truth for the deterministic prediction
pipeline used by both ``scripts/train_prediction_baseline.py`` and
``scripts/evaluate_prediction_baselines.py``.  Keeping the normalize -> feature
-> forward -> inverse path in one place guarantees that training and evaluation
cannot diverge, and that every predictor sees the test windows in exactly the
same order.

Only the strict :data:`PredictionDataset` whitelist reaches any model.  The
future label is used solely as supervision (already normalised with train-only
position statistics) and as the metric target; it is never a model input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from thesis_experiment.data.prediction_dataset import (
    MODEL_INPUT_FIELDS,
    PredictionDataset,
)
from thesis_experiment.prediction.classical import (
    CVKalmanFilter,
    constant_position,
    constant_velocity,
)
from thesis_experiment.prediction.gru import DeterministicGRU, build_gru_features
from thesis_experiment.prediction.normalization import PredictionNormalizer


CLASSICAL_METHOD_NAMES: Tuple[str, ...] = (
    "constant_position",
    "constant_velocity",
    "cv_kalman_filter",
)
GRU_METHOD_NAME = "deterministic_gru"
ALL_METHOD_NAMES: Tuple[str, ...] = CLASSICAL_METHOD_NAMES + (GRU_METHOD_NAME,)


_SAFE_VALUE_TAG = "__thesis_experiment_weights_only_type__"


def to_weights_only_safe(value: Any) -> Any:
    """Encode a checkpoint value using only weights-only-safe primitives.

    Modern PyTorch intentionally rejects arbitrary pickle globals when
    ``torch.load(..., weights_only=True)`` is used.  Checkpoints written by
    this project therefore contain only tensors and Python scalar/container
    primitives.  NumPy arrays/scalars and floating-point values are tagged so
    the same files also work with the stricter, experimental weights-only
    loader shipped by PyTorch 1.13.

    The conversion is recursive and rejects unknown objects instead of
    silently introducing an unsafe pickle payload.
    """

    if isinstance(value, Tensor):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError(
                "unsupported NumPy checkpoint dtype '{}'".format(array.dtype)
            )
        # Old Torch versions cannot construct tensors from every unsigned
        # NumPy dtype (notably uint32, used by MT19937), so encode unsigned
        # integer arrays losslessly as int64 tensors.
        tensor_array = (
            array.astype(np.int64)
            if array.dtype.kind == "u" and array.dtype.itemsize > 1
            else array
        )
        return {
            _SAFE_VALUE_TAG: "numpy_array",
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "tensor": torch.as_tensor(tensor_array).clone(),
        }
    if isinstance(value, np.generic):
        return to_weights_only_safe(value.item())
    if isinstance(value, float):
        # PyTorch 1.13's experimental weights-only unpickler does not support
        # the BINFLOAT opcode.  repr is round-trip exact for Python floats.
        return {_SAFE_VALUE_TAG: "float", "value": repr(value)}
    if value is None:
        return {_SAFE_VALUE_TAG: "none"}
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, list):
        return [to_weights_only_safe(item) for item in value]
    if isinstance(value, tuple):
        return {
            _SAFE_VALUE_TAG: "tuple",
            "items": [to_weights_only_safe(item) for item in value],
        }
    if isinstance(value, Mapping):
        result: Dict[Any, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (bool, int, str)):
                raise TypeError(
                    "checkpoint mapping keys must be bool, int, or str; got {}".format(
                        type(key).__name__
                    )
                )
            result[key] = to_weights_only_safe(item)
        return result
    raise TypeError(
        "unsupported checkpoint value type '{}'".format(type(value).__name__)
    )


def from_weights_only_safe(value: Any) -> Any:
    """Decode a value produced by :func:`to_weights_only_safe`."""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, list):
        return [from_weights_only_safe(item) for item in value]
    if isinstance(value, tuple):
        # Untagged tuples may occur in otherwise safe third-party state dicts.
        return tuple(from_weights_only_safe(item) for item in value)
    if isinstance(value, Mapping):
        tag = value.get(_SAFE_VALUE_TAG)
        if tag == "float":
            return float(value["value"])
        if tag == "none":
            return None
        if tag == "tuple":
            return tuple(from_weights_only_safe(item) for item in value["items"])
        if tag == "numpy_array":
            tensor = value["tensor"]
            if not isinstance(tensor, Tensor):
                raise ValueError("encoded NumPy array payload must be a tensor")
            shape = tuple(int(item) for item in value["shape"])
            array = tensor.detach().cpu().numpy().astype(
                np.dtype(str(value["dtype"])), copy=False
            )
            if array.shape != shape:
                raise ValueError("encoded NumPy array shape does not match payload")
            return array.copy()
        if tag is not None:
            raise ValueError("unknown weights-only checkpoint tag '{}'".format(tag))
        return {key: from_weights_only_safe(item) for key, item in value.items()}
    return value


def safe_torch_load(
    source: Union[str, Path, Any], *, map_location: Any = "cpu"
) -> Any:
    """Load a checkpoint without enabling arbitrary pickle execution.

    The loader always requests ``weights_only=True`` and never retries with
    unsafe pickle loading.  A PyTorch release too old to expose that argument
    is rejected explicitly; formal experiments must use the recommended
    modern environment.
    """

    try:
        loaded = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError as error:
        if "weights_only" in str(error):
            raise RuntimeError(
                "installed PyTorch does not support safe weights-only checkpoint "
                "loading; use the recommended modern environment"
            ) from error
        raise
    return from_weights_only_safe(loaded)


def serialize_numpy_rng_state(state: Optional[Tuple[Any, ...]] = None) -> Dict[str, Any]:
    """Return NumPy's legacy RNG state without an embedded ndarray."""

    current = np.random.get_state() if state is None else state
    if not isinstance(current, tuple) or len(current) != 5:
        raise ValueError("NumPy RNG state must be a five-item tuple")
    bit_generator, keys, position, has_gauss, cached_gaussian = current
    key_array = np.asarray(keys)
    if key_array.ndim != 1 or key_array.dtype.kind not in "ui":
        raise ValueError("NumPy RNG key state must be a one-dimensional integer array")
    return {
        "format": "numpy_random_state_v1",
        "bit_generator": str(bit_generator),
        "keys": torch.as_tensor(key_array.astype(np.int64)).clone(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": repr(float(cached_gaussian)),
    }


def deserialize_numpy_rng_state(state: Any) -> Tuple[Any, ...]:
    """Decode :func:`serialize_numpy_rng_state` output for NumPy."""

    if isinstance(state, tuple):
        # Backwards compatibility for an in-memory legacy TrainResumeState.
        return state
    if not isinstance(state, Mapping) or state.get("format") != "numpy_random_state_v1":
        raise ValueError("unsupported NumPy RNG checkpoint state")
    keys = state.get("keys")
    if not isinstance(keys, Tensor):
        raise ValueError("NumPy RNG checkpoint keys must be a tensor")
    return (
        str(state["bit_generator"]),
        keys.detach().cpu().numpy().astype(np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def serialize_python_rng_state(state: Optional[Tuple[Any, ...]] = None) -> Dict[str, Any]:
    """Return Python's RNG state using only safe scalar/container values."""

    current = random.getstate() if state is None else state
    if not isinstance(current, tuple) or len(current) != 3:
        raise ValueError("Python RNG state must be a three-item tuple")
    version, internal_state, gaussian = current
    if not isinstance(internal_state, tuple):
        raise ValueError("Python RNG internal state must be a tuple")
    return {
        "format": "python_random_state_v1",
        "version": int(version),
        "internal_state": [int(item) for item in internal_state],
        "has_gaussian": gaussian is not None,
        "gaussian": "0.0" if gaussian is None else repr(float(gaussian)),
    }


def deserialize_python_rng_state(state: Any) -> Tuple[Any, ...]:
    """Decode :func:`serialize_python_rng_state` output for :mod:`random`."""

    if isinstance(state, tuple):
        # Backwards compatibility for an in-memory legacy TrainResumeState.
        return state
    if not isinstance(state, Mapping) or state.get("format") != "python_random_state_v1":
        raise ValueError("unsupported Python RNG checkpoint state")
    gaussian = float(state["gaussian"]) if bool(state["has_gaussian"]) else None
    return (
        int(state["version"]),
        tuple(int(item) for item in state["internal_state"]),
        gaussian,
    )


def capture_safe_rng_state(loader_generator_state: Any = None) -> Dict[str, Any]:
    """Capture all RNGs needed for bit-exact interrupted training resume."""

    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "numpy_rng_state": serialize_numpy_rng_state(),
        "python_rng_state": serialize_python_rng_state(),
        "loader_generator_state": loader_generator_state,
    }


def apply_torch_thread_config(
    num_threads: int, num_interop_threads: int
) -> Dict[str, int]:
    """Apply and report explicit PyTorch intra/inter-op thread limits.

    PyTorch permits setting the inter-op pool only before parallel work starts.
    A repeated call requesting its already-active value is harmless; a failed
    attempt to change it is propagated so configuration cannot silently drift.
    """

    if isinstance(num_threads, bool) or int(num_threads) != num_threads or num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")
    if (
        isinstance(num_interop_threads, bool)
        or int(num_interop_threads) != num_interop_threads
        or num_interop_threads <= 0
    ):
        raise ValueError("num_interop_threads must be a positive integer")
    requested_threads = int(num_threads)
    requested_interop = int(num_interop_threads)
    torch.set_num_threads(requested_threads)
    if int(torch.get_num_interop_threads()) != requested_interop:
        try:
            torch.set_num_interop_threads(requested_interop)
        except RuntimeError:
            if int(torch.get_num_interop_threads()) != requested_interop:
                raise
    applied = {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }
    if applied["torch_num_threads"] != requested_threads:
        raise RuntimeError("PyTorch intra-op thread setting was not applied")
    if applied["torch_num_interop_threads"] != requested_interop:
        raise RuntimeError("PyTorch inter-op thread setting was not applied")
    return applied


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy, and Torch RNGs for reproducible smoke results."""

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA when available and requested, otherwise CPU."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loss(name: str) -> nn.Module:
    """Return the configured regression loss module."""

    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError("unsupported loss '{}'".format(name))


def _raw_batch(dataset: PredictionDataset) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return the full strict input mapping and future target for a split."""

    if set(dataset.input_fields) != set(MODEL_INPUT_FIELDS):
        raise ValueError(
            "pipeline requires the full PredictionDataset model-input whitelist"
        )
    batch = dataset.get_batch(slice(None))
    inputs = {field: np.asarray(batch["inputs"][field]) for field in MODEL_INPUT_FIELDS}
    target = np.asarray(batch["target"], dtype=np.float64)
    return inputs, target


def build_normalized_tensors(
    dataset: PredictionDataset,
    normalizer: PredictionNormalizer,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Return ``(features, normalized_target)`` tensors for one split.

    Inputs are normalised with the train-only statistics, converted to the
    strict seven-channel GRU feature tensor, and the future label is normalised
    with the same train position statistics.  The label never enters the
    feature tensor.
    """

    inputs, target = _raw_batch(dataset)
    normalized_inputs = normalizer.normalize_inputs(inputs)
    features = build_gru_features(normalized_inputs, dtype=dtype, device=device)
    normalized_target = normalizer.normalize_target(
        torch.as_tensor(target, dtype=dtype, device=device)
    )
    if features.ndim != 3:
        raise ValueError("expected batched features of shape (N, H, 7)")
    if normalized_target.ndim != 3 or normalized_target.shape[-1] != 2:
        raise ValueError("expected normalized target of shape (N, T, 2)")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("normalized features contain non-finite values")
    if not bool(torch.isfinite(normalized_target).all()):
        raise ValueError("normalized target contains non-finite values")
    return features, normalized_target


@dataclass(frozen=True)
class EpochRecord:
    """One training epoch summary written to the CSV history log."""

    epoch: int
    train_loss: float
    validation_loss: float
    best_validation_loss: float
    is_best: bool
    learning_rate: float
    seconds: float


def _evaluate_loss(
    model: DeterministicGRU,
    features: Tensor,
    target: Tensor,
    loss_fn: nn.Module,
    batch_size: int,
) -> float:
    """Return the mean validation loss without accumulating gradients."""

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = start + batch_size
            batch_features = features[start:stop]
            batch_target = target[start:stop]
            prediction = model(batch_features)
            loss = loss_fn(prediction, batch_target)
            total += float(loss.item()) * batch_features.shape[0]
            count += batch_features.shape[0]
    if count == 0:
        raise ValueError("validation split is empty")
    return total / count


@dataclass
class TrainResumeState:
    """State restored from a safe resume checkpoint for one GRU seed.

    ``numpy_rng_state`` and ``python_rng_state`` may be either the safe mapping
    produced by this module or an in-memory legacy tuple.  New on-disk files
    must always use the safe mapping representation.
    """

    start_epoch: int
    best_validation_loss: float
    epochs_without_improvement: int
    optimizer_state: Dict[str, Any]
    model_state: Dict[str, Any]
    best_model_state: Dict[str, Any]
    torch_rng_state: Any
    numpy_rng_state: Any
    python_rng_state: Any
    loader_generator_state: Any = None
    cuda_rng_states: Any = None
    best_epoch: int = 0


def train_gru(
    *,
    model: DeterministicGRU,
    train_features: Tensor,
    train_target: Tensor,
    validation_features: Tensor,
    validation_target: Tensor,
    epochs: int,
    batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    gradient_clip_norm: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    num_workers: int,
    generator: torch.Generator,
    device: torch.device,
    on_epoch_end=None,
    resume: Optional[TrainResumeState] = None,
) -> Dict[str, Any]:
    """Train one deterministic GRU with validation-only model selection.

    Model selection and early stopping read the validation split only.  The
    caller supplies test data nowhere in this function, structurally preventing
    test leakage into training decisions.  ``on_epoch_end`` receives each
    :class:`EpochRecord` and the current best/last states so the script layer
    can persist a resumable checkpoint after every epoch.
    """

    loss_fn = make_loss(loss_name)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    dataset = TensorDataset(train_features, train_target)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )

    best_validation_loss = float("inf")
    best_state: Dict[str, Any] = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    history: List[EpochRecord] = []

    if resume is not None:
        model.load_state_dict(resume.model_state)
        optimizer.load_state_dict(resume.optimizer_state)
        best_validation_loss = resume.best_validation_loss
        best_state = resume.best_model_state
        best_epoch = resume.best_epoch
        epochs_without_improvement = resume.epochs_without_improvement
        start_epoch = resume.start_epoch
        torch.set_rng_state(resume.torch_rng_state)
        if device.type == "cuda":
            if resume.cuda_rng_states is None:
                raise ValueError(
                    "CUDA resume requires saved CUDA RNG states for exact recovery"
                )
            torch.cuda.set_rng_state_all(list(resume.cuda_rng_states))
        np.random.set_state(deserialize_numpy_rng_state(resume.numpy_rng_state))
        random.setstate(deserialize_python_rng_state(resume.python_rng_state))
        if resume.loader_generator_state is not None:
            generator.set_state(resume.loader_generator_state)

    import time

    # The epoch callback persists its snapshot before the loop evaluates early
    # stopping.  If interruption occurs in that narrow interval, the restored
    # counter already represents a terminal state and must not run one extra
    # epoch.
    if epochs_without_improvement >= early_stopping_patience:
        return {
            "history": history,
            "best_validation_loss": best_validation_loss,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "last_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "stopped_epoch": max(start_epoch - 1, 0),
        }

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        running = 0.0
        seen = 0
        for batch_features, batch_target in loader:
            optimizer.zero_grad()
            prediction = model(batch_features)
            loss = loss_fn(prediction, batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            running += float(loss.item()) * batch_features.shape[0]
            seen += batch_features.shape[0]
        train_loss = running / max(seen, 1)
        validation_loss = _evaluate_loss(
            model, validation_features, validation_target, loss_fn, validation_batch_size
        )

        is_best = validation_loss < best_validation_loss - early_stopping_min_delta
        if is_best:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
            is_best=is_best,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - started,
        )
        history.append(record)
        if on_epoch_end is not None:
            on_epoch_end(
                record,
                model_state={
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                best_model_state=best_state,
                optimizer_state=optimizer.state_dict(),
                epochs_without_improvement=epochs_without_improvement,
                loader_generator_state=generator.get_state(),
                best_epoch=best_epoch,
            )
        if epochs_without_improvement >= early_stopping_patience:
            break

    return {
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "last_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "stopped_epoch": history[-1].epoch if history else 0,
    }


CHECKPOINT_SCHEMA_VERSION = 1


def build_model_checkpoint(
    *,
    model_state: Dict[str, Any],
    gru_config: Mapping[str, int],
    normalization: Mapping[str, Any],
    seed: int,
    epoch: int,
    validation_loss: float,
    loss_name: str,
) -> Dict[str, Any]:
    """Assemble a self-describing, weights-only-safe GRU checkpoint."""

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": "deterministic_gru",
        "input_whitelist": list(MODEL_INPUT_FIELDS),
        "gru_config": {
            "input_size": int(gru_config["input_size"]),
            "hidden_size": int(gru_config["hidden_size"]),
            "num_layers": int(gru_config["num_layers"]),
            "dropout": float(gru_config["dropout"]),
            "future_steps": int(gru_config["future_steps"]),
        },
        "normalization": dict(normalization),
        "seed": int(seed),
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "loss_name": str(loss_name),
        "model_state": model_state,
    }
    return to_weights_only_safe(checkpoint)


def load_model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> Tuple[DeterministicGRU, PredictionNormalizer]:
    """Rebuild the GRU and its train-only normalizer from a checkpoint dict."""

    checkpoint = from_weights_only_safe(checkpoint)
    if int(checkpoint.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if list(checkpoint.get("input_whitelist", [])) != list(MODEL_INPUT_FIELDS):
        raise ValueError("checkpoint input whitelist does not match dataset boundary")
    config = checkpoint["gru_config"]
    model = DeterministicGRU(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        future_steps=int(config["future_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    from thesis_experiment.prediction.normalization import NormalizationStatistics

    normalizer = PredictionNormalizer(
        NormalizationStatistics.from_dict(dict(checkpoint["normalization"]))
    )
    return model, normalizer


def predict_gru_local(
    dataset: PredictionDataset,
    model: DeterministicGRU,
    normalizer: PredictionNormalizer,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return GRU forecasts restored to the original local coordinate scale."""

    features, _ = build_normalized_tensors(
        dataset, normalizer, dtype=torch.float32, device=device
    )
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size]
            normalized = model(batch)
            local = normalizer.inverse_position(normalized)
            predictions.append(local.detach().cpu().numpy())
    result = np.concatenate(predictions, axis=0).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("GRU produced non-finite predictions")
    return result


def predict_classical_local(
    dataset: PredictionDataset,
    method_name: str,
    *,
    future_steps: int,
    kalman: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return a classical baseline forecast in local coordinates.

    Classical baselines operate directly on the raw local history (the loader
    already fills missing steps and exposes both masks), so no normalisation is
    applied.  The result is ``(N, future_steps, 2)`` in the same window order.
    """

    inputs, _ = _raw_batch(dataset)
    if method_name == "constant_position":
        return constant_position(inputs, future_steps=future_steps).astype(np.float64)
    if method_name == "constant_velocity":
        return constant_velocity(inputs, future_steps=future_steps).astype(np.float64)
    if method_name == "cv_kalman_filter":
        if kalman is None:
            raise ValueError("cv_kalman_filter requires Kalman parameters")
        filter_model = CVKalmanFilter(
            process_noise=float(kalman["process_noise"]),
            observation_noise=float(kalman["measurement_noise"]),
            initial_position_variance=float(kalman["initial_position_variance"]),
            initial_velocity_variance=float(kalman["initial_velocity_variance"]),
        )
        return filter_model.predict(inputs, future_steps=future_steps).astype(np.float64)
    raise ValueError("unknown classical method '{}'".format(method_name))


__all__ = [
    "ALL_METHOD_NAMES",
    "CHECKPOINT_SCHEMA_VERSION",
    "CLASSICAL_METHOD_NAMES",
    "EpochRecord",
    "GRU_METHOD_NAME",
    "TrainResumeState",
    "apply_torch_thread_config",
    "build_model_checkpoint",
    "build_normalized_tensors",
    "capture_safe_rng_state",
    "deserialize_numpy_rng_state",
    "deserialize_python_rng_state",
    "from_weights_only_safe",
    "load_model_from_checkpoint",
    "make_loss",
    "predict_classical_local",
    "predict_gru_local",
    "safe_torch_load",
    "select_device",
    "serialize_numpy_rng_state",
    "serialize_python_rng_state",
    "set_global_seed",
    "to_weights_only_safe",
    "train_gru",
]
