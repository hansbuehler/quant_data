from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from quantdata.torch import DeviceLock


def test_device_lock_returns_documented_cpu_fallback_without_cuda() -> None:
    with patch("torch.cuda.device_count", return_value=0):
        with pytest.warns(UserWarning, match="falling back to CPU"):
            with DeviceLock(fall_back=True) as lock:
                assert lock.device == torch.device("cpu")


def test_device_lock_rejects_missing_cuda_without_fallback() -> None:
    with patch("torch.cuda.device_count", return_value=0):
        with pytest.raises(AttributeError, match="No CUDA devices"):
            DeviceLock(fall_back=False)
