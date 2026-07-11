"""Geometry utilities for forest generation and line-of-sight checks."""

from .forest import generate_tree_trunks
from .occlusion import (
    is_occluded,
    occlusion_mask,
    segment_intersects_circle,
    segment_intersects_circles,
)

__all__ = [
    "generate_tree_trunks",
    "is_occluded",
    "occlusion_mask",
    "segment_intersects_circle",
    "segment_intersects_circles",
]
