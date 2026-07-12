"""Pytest configuration for stable, bounded test execution."""

import os
import sys
from pathlib import Path


# Set native-library limits before importing Torch.  Assignment (rather than
# setdefault) intentionally overrides high-thread-count workstation defaults so
# a bare ``python -m pytest -q`` remains fast and reproducible.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import torch


torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
