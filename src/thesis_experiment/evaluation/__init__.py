"""Model-independent evaluation utilities for trajectory prediction."""

from .aggregation import INDEPENDENCE_WARNING, aggregate_window_metrics

__all__ = ["INDEPENDENCE_WARNING", "aggregate_window_metrics"]
