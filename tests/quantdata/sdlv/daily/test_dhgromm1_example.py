"""Standalone example: causal features, frozen references, OCE and CPU training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.sdlv.dhgromm1.datasets.q1_20250829_100k.dataset import SCHEMA, sha256
from quantdata.sdlv.dhgromm1.datasets.q1_20250829_100k.example_deep_hedging import (
    Config,
    actions,
    estimate,
    evaluate,
    initialize,
    load_sample,
    make_step,
    objective,
    prepare,
    profile_cash,
    utility,
)


def raw_sample() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(6)
    prices = rng.uniform(.01, .1, (8, 21, 21))
    prices[:, :, 0] = np.exp(rng.normal(0, .03, (8, 21)))
    return dict(prices=prices, gains=rng.normal(0, .01, (8, 20, 21)),
                payoff=np.abs(prices[:, 20, 0]/prices[:, 10, 0]-1))


def shared_fixture(root: Path) -> None:
    (root/"chunks").mkdir(parents=True)
    # Accessing this member with allow_pickle=False would fail. The small
    # example must read only prices, gains and payoff, not the optional marks.
    path = root/"chunks/000000.npz"
    np.savez_compressed(path, **raw_sample(), changes_packed=np.array([object()], dtype=object))
    (root/"MANIFEST.json").write_text(json.dumps(dict(schema=SCHEMA, paths=8,
        chunks=[dict(file="chunks/000000.npz", start=0, count=8, sha256=sha256(path))])))


def test_prefix_loader_preserves_pairs_without_reading_marks(tmp_path):
    shared_fixture(tmp_path)
    arrays, identity = load_sample(tmp_path, paths=4)
    assert identity["independent_antithetic_pairs"] == 2
    for key, values in raw_sample().items():
        np.testing.assert_array_equal(arrays[key], values[:4])
    with pytest.raises(ValueError):
        load_sample(tmp_path, paths=3)


def test_prepare_has_global_centering_and_excludes_future_labels():
    raw = raw_sample()
    sample, _ = prepare(raw)
    np.testing.assert_allclose(sample["returns"].mean(0), 0, atol=1e-9)
    assert sample["features"].shape == (8, 20, 22)
    altered = {key: value.copy() for key, value in raw.items()}
    altered["prices"][:, 20] *= 100
    altered["payoff"] *= 100
    altered["gains"] *= 100
    np.testing.assert_array_equal(sample["features"], prepare(altered)[0]["features"])


def test_zero_actions_live_hidden_gradients_frozen_reference_and_date_locality():
    config = Config(width=4)
    sample, _ = prepare(raw_sample())
    params = initialize(config)
    reference = jax.tree.map(lambda x: x.copy(), params)
    np.testing.assert_array_equal(actions(params, reference, sample["features"]), 0)
    loss = lambda p, r: objective(p, r, *(sample[k] for k in ("features", "returns", "payoff")), risk_aversion=100.)
    gradient = jax.grad(loss)(params, reference)
    assert np.linalg.norm(gradient["hidden"]) > 0
    frozen = jax.grad(loss, argnums=1)(params, reference)
    assert all(not np.asarray(v).any() for v in jax.tree.leaves(frozen))
    changed = dict(params, linear_bias=params["linear_bias"]+.1)
    future = sample["features"].copy()
    future[:, 8:] += 20
    np.testing.assert_array_equal(actions(changed, reference, future)[:, :8],
                                  actions(changed, reference, sample["features"])[:, :8])


def test_utility_and_profiled_cash_are_cash_invariant_and_correct_at_zero():
    assert float(jax.grad(lambda x: utility(x, 100.))(jnp.asarray(0.))) == pytest.approx(1.)
    x = jnp.asarray([-1e5, -.01, 0., .01, 100.])
    assert np.isfinite(np.asarray(utility(x, 100.))).all()
    net = np.array([[-.03, .1], [.01, .1], [.04, .1], [0., .1]])
    _, contribution, derivative = profile_cash(net, risk_aversion=100.)
    np.testing.assert_allclose(derivative.mean(0), 1, atol=1e-12)
    np.testing.assert_allclose(contribution[:, 1], .1, atol=1e-12)
    shifted = profile_cash(net+.3, risk_aversion=100.)[1]
    np.testing.assert_allclose(shifted, contribution+.3, atol=1e-12)


def test_oce_signs_empty_subtraction_and_paired_mc_errors():
    raw = raw_sample()
    sample, normal = prepare(raw)
    config = Config(width=4)
    params = initialize(config)
    results = evaluate(params, params, sample, raw, normal, config=config, batch=4)
    assert results["raw_oce"]["empty"]["value"] == pytest.approx(0., abs=1e-15)
    assert results["raw_oce"]["empty"]["three_mc_standard_errors"] < 1e-15
    assert results["bid"]["value"] <= results["q_expectation"]["value"] <= results["ask"]["value"]
    assert results["long_minus_empty"]["value"] == pytest.approx(results["raw_oce"]["long"]["value"])
    paired = np.array([1., 2., 4., 5.])
    expected = 3*np.std([1.5, 4.5], ddof=1)/np.sqrt(2)
    assert estimate(paired)["three_mc_standard_errors"] == pytest.approx(expected)


def test_compiled_adam_matches_eager_values_and_updates_parameters():
    config = Config(width=4)
    params = initialize(config)
    zero = jax.tree.map(jnp.zeros_like, params)
    sample, _ = prepare(raw_sample())
    values = tuple(sample[k] for k in ("features", "returns", "payoff"))
    expected = objective(params, params, *values, risk_aversion=100.)
    state, actual = make_step(params, config)((params, zero, zero, jnp.asarray(0)), *values)
    assert float(actual) == pytest.approx(float(expected), rel=2e-6)
    assert int(state[3]) == 1
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(state))
    assert not np.array_equal(params["hidden"], state[0]["hidden"])


def test_standalone_cli_does_not_need_project_imports_and_saves_all_cases(tmp_path):
    data = tmp_path/"dataset"
    shared_fixture(data)
    script = Path(__file__).parents[4]/"quantdata/sdlv/dhgromm1/datasets/q1_20250829_100k/example_deep_hedging.py"
    environment = dict(os.environ, PYTHONPATH="", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    run = subprocess.run([sys.executable, str(script), "--data", str(data), "--paths", "8", "--epochs", "2",
                          "--width", "4", "--batch-size", "4", "--output-root", str(tmp_path/"runs")],
                         cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr
    outputs = list((tmp_path/"runs").glob("q1-simple-*"))
    assert len(outputs) == 1
    result = json.loads((outputs[0]/"RESULTS.json").read_text())
    assert set(result["results"]["raw_oce"]) == {"empty", "long", "short"}
    assert len((outputs[0]/"history.jsonl").read_text().splitlines()) == 2
    with np.load(outputs[0]/"checkpoint.npz", allow_pickle=False) as checkpoint:
        assert int(checkpoint["adam_step"]) == 4
        assert {"params_hidden", "adam_first_hidden", "adam_second_hidden", "reference_hidden"} <= set(checkpoint.files)
    checkout = tmp_path/"checkout"
    checkout.mkdir()
    (checkout/".git").write_text("gitdir: /does/not/matter")
    refused = subprocess.run([sys.executable, str(script), "--output-root", str(checkout/"outputs")],
                             cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 2 and "outside every Git working tree" in refused.stderr
