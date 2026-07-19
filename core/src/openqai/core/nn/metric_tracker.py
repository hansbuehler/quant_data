from __future__ import annotations

from collections.abc import Mapping

import numpy as nnp
import jax as jax
import flax as flax
import optax as optax


class MetricTracker(object):
    """
    Simple utility class to track metrics during training
    """

    def __init__(self):
        self.metrics = {}

    def record(self, epoch: int, metrics: Mapping[str, tuple[float, float | None]]):
        """Update the metric tracker with new metrics and their corresponding weights"""
        assert "loss" in metrics, "'loss' should be included in the metrics directory"
        for k, v in metrics.items():
            if isinstance(v, tuple):
                assert (
                    len(v) == 2
                ), f"Expected metric '{k}' to be a tuple of (mean, std) or (value, None), but got {v} with length {len(v)}."
                self.record_metric(epoch, k, mean=v[0], std=v[1])
            else:
                self.record_metric(epoch, k, mean=v, std=None)

    def record_metric(self, epoch: int, name: str, mean: float, std: float | None = None):
        """Add a single metric"""
        tm = self.metrics.get(name)
        if tm is None:
            tm = dict(epochs=[epoch], values=[mean], stds=[std] if std is not None else None)
            self.metrics[name] = tm
        else:
            assert (
                tm["epochs"][-1] < epoch
            ), f"Epoch {epoch} is not greater than last epoch {tm['epochs'][-1]} for metric {name}"
            tm["epochs"].append(epoch)
            tm["values"].append(mean)
            if std is None:
                assert (
                    tm["stds"] is None
                ), f"Metric {name} was previously recorded with stds, but now std is None"
            else:
                assert (
                    tm["stds"] is not None
                ), f"Metric {name} was previously recorded without stds, but now std is provided"
                tm["stds"].append(std)

    def items(self):
        """Get all metrics as an iterable of (name, epochs, values) tuples"""
        return self.metrics.items()

    def __getitem__(self, name):
        """Get metric by name"""
        return self.metrics[name]

    def get(self, name, default=None):
        """Get metric by name with default"""
        return self.metrics.get(name, default)

    def state_dict(self):
        """Get a serializable copy of the tracked metrics as numpy arrays"""
        return {
            k: {
                "epochs": nnp.asarray(v["epochs"], dtype=nnp.int32),
                "values": nnp.asarray(v["values"], dtype=nnp.float32),
                "stds": nnp.asarray(v["stds"], dtype=nnp.float32)
                if v["stds"] is not None
                else None,
            }
            for k, v in self.metrics.items()
        }

    @staticmethod
    def load_state_dict(state):
        """Load tracked metrics from a serializable state"""
        assert isinstance(
            state, dict
        ), f"Expected tracker state to be a dict, but got {type(state)}"
        tracker = MetricTracker()
        tracker.metrics = {
            k: {
                "epochs": x["epochs"].tolist(),
                "values": x["values"].tolist(),
                "stds": x["stds"].tolist() if x["stds"] is not None else None,
            }
            for k, x in state.items()
        }
        return tracker
