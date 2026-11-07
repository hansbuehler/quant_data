from datetime import date, timedelta

import torch

from quantdata.dysanos.v11.benchmark_rolling_affine import install_rolling_weights


def test_warm_start_aligns_dates_and_keeps_decoder() -> None:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_decoder = torch.nn.Linear(2, 3)
            self.daily_state_embedding = torch.nn.Embedding(4, 2)

    model = Model()
    start = date(2025, 1, 1)
    old = tuple(start + timedelta(days=i) for i in range(4))
    new = tuple(start + timedelta(days=i) for i in range(2, 6))
    source = {"parameters": {name: torch.ones_like(value) for name, value in model.named_parameters()}}
    source["parameters"]["daily_state_embedding.weight"] = torch.arange(8).reshape(4, 2).float()
    install_rolling_weights(model, source, old, new)
    torch.testing.assert_close(model.daily_state_embedding.weight, torch.tensor([[4., 5.], [6., 7.], [6., 7.], [6., 7.]]))
    torch.testing.assert_close(model.raw_decoder.weight, torch.ones_like(model.raw_decoder.weight))
