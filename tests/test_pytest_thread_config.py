"""Verify that a bare pytest invocation applies bounded Torch threading."""

import os

import torch


def test_pytest_process_uses_single_torch_thread() -> None:
    """The test process is 1/1 without user-provided environment variables."""

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
