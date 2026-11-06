"""Small CPU-only deep-hedging example; see DEEP_HEDGING.md beside this file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from .dataset import Dataset, sha256
else:
    from dataset import Dataset, sha256

LABELS = ("empty", "long", "short")
SIGNS = np.array([0., 1., -1.])
EXAMPLE_SCHEMA = "dhgromm1-simple-price-time-v1"


@dataclass(frozen=True, kw_only=True)
class Config:
    """A deliberately small, non-recurrent teaching model."""

    width: int = 16
    risk_aversion: float = 100.
    learning_rate: float = .0001
    seed: int = 17

    def __post_init__(self) -> None:
        if self.width < 1 or not np.isfinite([self.risk_aversion, self.learning_rate]).all():
            raise ValueError("width must be positive and utility/learning-rate parameters finite")
        if self.risk_aversion <= 0 or self.learning_rate <= 0:
            raise ValueError("risk aversion and learning rate must be positive")


def load_sample(root: Path, *, paths: int) -> tuple[dict[str, np.ndarray], dict]:
    """Read an original-order prefix of complete pairs, without unpacking marks."""
    data = Dataset(root)
    if paths < 4 or paths % 2 or paths > data.manifest["paths"]:
        raise ValueError("paths must be even, at least four, and no larger than the shared sample")
    parts = {key: [] for key in ("prices", "gains", "payoff")}
    for record in data.manifest["chunks"]:
        if record["start"] >= paths:
            break
        path = data.root / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"checksum mismatch for {path.name}; fetch the data with git lfs pull")
        count = min(record["count"], paths-record["start"])
        # Zip members are lazy: do not decompress changes_packed for this policy.
        with np.load(path, allow_pickle=False) as saved:
            for key in parts:
                parts[key].append(saved[key][:count].copy())
        print(f"loaded {record['start']+count}/{paths} paths", flush=True)
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite training inputs")
    p, h = arrays["prices"], arrays["payoff"]
    if p.shape != (paths, 21, 21) or arrays["gains"].shape != (paths, 20, 21) or np.any(p[:, :, 0] <= 0):
        raise ValueError("invalid shared training geometry or nonpositive spot")
    np.testing.assert_allclose(h, np.abs(p[:, 20, 0]/p[:, 10, 0]-1), atol=2e-12, rtol=1e-12)
    return arrays, dict(schema=data.manifest["schema"], manifest_sha256=sha256(data.root/"MANIFEST.json"),
                        paths=paths, independent_antithetic_pairs=paths//2)


def prepare(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit constants on the selected training sample, never on a validation set."""
    prices = arrays["prices"][:, :20]  # Day 20 never enters a decision feature.
    spot = prices[:, :, :1]
    current = np.concatenate((spot, prices[:, :, 1:]/spot), axis=-1)
    mean, scale = current.mean((0, 1)), np.maximum(current.std((0, 1)), 1e-4)
    clock = np.broadcast_to(np.arange(20)[None, :, None]/20, (len(prices), 20, 1))
    normal = dict(price_mean=mean, price_scale=scale, return_mean=arrays["gains"].mean(0),
                  payoff_mean=arrays["payoff"].mean())
    sample = dict(features=np.concatenate((clock, (current-mean)/scale), axis=-1).astype(np.float32),
                  returns=(arrays["gains"]-normal["return_mean"]).astype(np.float32),
                  payoff=(arrays["payoff"]-normal["payoff_mean"]).astype(np.float32))
    return sample, normal


def initialize(config: Config) -> dict[str, jax.Array]:
    """Independent random affine-plus-deep heads for Empty, Long and Short."""
    rng = np.random.default_rng(config.seed)
    shapes = {"linear": (3, 22, 21), "hidden": (3, 22, config.width), "out": (3, config.width, 21)}
    params = {key: jnp.asarray(rng.normal(size=shape).astype(np.float32)/np.sqrt(shape[1]), dtype=jnp.float32)
              for key, shape in shapes.items()}
    params["linear"] *= .02
    params["out"] *= .02
    params.update(hidden_bias=jnp.zeros((3, config.width), dtype=jnp.float32), out_bias=jnp.zeros((3, 21), dtype=jnp.float32),
                  linear_bias=jnp.zeros((3, 21), dtype=jnp.float32), cash=jnp.zeros(3, dtype=jnp.float32))
    return params


def actions(params: dict, reference: dict, features: jax.Array) -> jax.Array:
    """Executed floating quantities, (path, day, policy, instrument); no cap."""
    def head(weights):
        hidden = jnp.tanh(jnp.einsum("ntf,kfw->ntkw", features, weights["hidden"])+weights["hidden_bias"])
        return (jnp.einsum("ntf,kfi->ntki", features, weights["linear"])+weights["linear_bias"]
                + jnp.einsum("ntkw,kwi->ntki", hidden, weights["out"])+weights["out_bias"])
    # Only the reference weights are frozen. Hidden-layer gradients are alive
    # at initialization, unlike zeroing the final layer of a deep network.
    return head(params)-head(jax.tree.map(jax.lax.stop_gradient, reference))


def utility(value: jax.Array, risk_aversion: float) -> jax.Array:
    """Power-exponential u(0)=0, u'(0)=1, with a safe unused exp branch."""
    # maximum(x,0) would give a half-gradient at zero in JAX. A selected
    # branch preserves u'(0)=1 as well as avoiding overflow for large losses.
    positive = -jnp.expm1(-risk_aversion*jnp.where(value >= 0, value, 0))/risk_aversion
    return jnp.where(value >= 0, positive, value-.5*risk_aversion*value*value)


def objective(params: dict, reference: dict, features, returns, payoff, *, risk_aversion: float) -> jax.Array:
    """Three independent OCE objectives, optimized by minimizing their negative sum."""
    gain = jnp.einsum("ntki,nti->nk", actions(params, reference, features), returns)
    net = gain+payoff[:, None]*jnp.asarray(SIGNS, dtype=gain.dtype)
    contribution = utility(net+params["cash"], risk_aversion)-params["cash"]
    return -contribution.mean(0).sum()


def make_step(reference: dict, config: Config) -> Callable:
    """Compiled value/gradient and Adam update; days and instruments are batched."""
    def loss(params, features, returns, payoff):
        return objective(params, reference, features, returns, payoff, risk_aversion=config.risk_aversion)

    @jax.jit
    def step(state, features, returns, payoff):
        params, first, second, count = state
        value, gradient = jax.value_and_grad(loss)(params, features, returns, payoff)
        count = count+1
        first = jax.tree.map(lambda m, g: .9*m+.1*g, first, gradient)
        second = jax.tree.map(lambda v, g: .999*v+.001*g*g, second, gradient)
        updated = jax.tree.map(lambda p, m, v: p-config.learning_rate*(m/(1-.9**count))
                               /(jnp.sqrt(v/(1-.999**count))+1e-8), params, first, second)
        return (updated, first, second, count), value
    return step


def profile_cash(net: np.ndarray, *, risk_aversion: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize y for fixed actions using E[u'(y+X)]=1, in float64."""
    bound = np.max(np.abs(net), axis=0)+1
    lo, hi = -bound, bound
    for _ in range(65):
        mid = (lo+hi)/2
        x = net+mid
        derivative = np.where(x >= 0, np.exp(-risk_aversion*np.maximum(x, 0)), 1-risk_aversion*x)
        lower = derivative.mean(0) > 1
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    cash = (lo+hi)/2
    x = net+cash
    derivative = np.where(x >= 0, np.exp(-risk_aversion*np.maximum(x, 0)), 1-risk_aversion*x)
    u = np.where(x >= 0, -np.expm1(-risk_aversion*np.maximum(x, 0))/risk_aversion,
                 x-.5*risk_aversion*x*x)
    return cash, u-cash, derivative


def estimate(values: np.ndarray) -> dict[str, float]:
    """Averages and three sample-based MC SEs across independent pair averages."""
    paired = values.reshape(-1, 2).mean(1)
    return dict(value=float(values.mean()), three_mc_standard_errors=float(3*paired.std(ddof=1)/np.sqrt(len(paired))))


def evaluate(params: dict, reference: dict, sample: dict, raw: dict, normal: dict, *, config: Config, batch: int) -> dict:
    """Training-only prices, paired errors, and explicit mean-centering influence."""
    predict = jax.jit(lambda x: actions(params, reference, x))
    gain = np.empty((len(sample["payoff"]), 3))
    for start in range(0, len(gain), batch):
        sl = slice(start, start+batch)
        a = np.asarray(predict(sample["features"][sl]))
        gain[sl] = np.einsum("ntki,nti->nk", a, sample["returns"][sl], dtype=np.float64)
    net = gain+sample["payoff"][:, None]*SIGNS
    cash, contribution, derivative = profile_cash(net, risk_aversion=config.risk_aversion)
    coefficient = np.zeros((20, 3, 21))
    for start in range(0, len(gain), batch):
        sl = slice(start, start+batch)
        a = np.asarray(predict(sample["features"][sl]))
        coefficient += np.einsum("nk,ntki->tki", derivative[sl], a, dtype=np.float64)/len(gain)
    h = raw["payoff"]-normal["payoff_mean"]
    # Delta-method correction for the estimated global return/payoff means.
    # Policy fitting and feature-normalization estimation are held fixed.
    influence = (contribution-np.einsum("nti,tki->nk", raw["gains"]-normal["return_mean"], coefficient)
                 - h[:, None]*SIGNS*derivative.mean(0))
    def result(value, uncertainty):
        return dict(value=float(np.mean(value)), three_mc_standard_errors=estimate(uncertainty)["three_mc_standard_errors"])
    return dict(q_expectation=estimate(raw["payoff"]),
        raw_oce={key: result(contribution[:, k], influence[:, k]) for k, key in enumerate(LABELS)},
        long_minus_empty=result(contribution[:, 1]-contribution[:, 0], influence[:, 1]-influence[:, 0]),
        short_minus_empty=result(contribution[:, 2]-contribution[:, 0], influence[:, 2]-influence[:, 0]),
        bid=result(normal["payoff_mean"]+contribution[:, 1]-contribution[:, 0], h+influence[:, 1]-influence[:, 0]),
        ask=result(normal["payoff_mean"]+contribution[:, 0]-contribution[:, 2], h+influence[:, 0]-influence[:, 2]),
        profiled_cash=dict(zip(LABELS, cash.tolist(), strict=True)),
        mc_scope="antithetic pairs; return/payoff centering included; policy/feature fitting and selection excluded",
        sample="training only; no claim of convergence or guaranteed bid/EQ/ask bracketing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent)
    parser.add_argument("--paths", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error("epochs must be positive; batch size must be positive and even")
    root = args.output_root
    if root is None:
        if not os.environ.get("QUANTDATA_ROOT"):
            parser.error("set --output-root outside Git, or set QUANTDATA_ROOT")
        root = Path(os.environ["QUANTDATA_ROOT"])/"experiments/quant_data/sdlv/dhgromm1-examples"
    root = root.expanduser().resolve()
    # Empty .git guard directories are not repositories. Real checkouts have
    # .git/HEAD, or a .git file pointing to a linked worktree's metadata.
    if any((p/".git").is_file() or (p/".git/HEAD").is_file() for p in (root, *root.parents)):
        parser.error("generated runs must be outside every Git working tree")
    jax.config.update("jax_platforms", "cpu")  # Never claim or consume a shared GPU.
    config = Config(width=args.width, learning_rate=args.learning_rate, seed=args.seed)
    raw, provenance = load_sample(args.data, paths=args.paths)
    sample, normal = prepare(raw)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="q1-simple-", dir=root))
    params = initialize(config)
    reference = jax.tree.map(lambda x: x.copy(), params)
    zero = jax.tree.map(jnp.zeros_like, params)
    state = (params, zero, zero, jnp.asarray(0, dtype=jnp.int32))
    step = make_step(reference, config)
    rng = np.random.default_rng(config.seed)
    started = time.monotonic()
    print(f"output: {output}", flush=True)
    # Move complete antithetic pairs together; centering remains global, not per batch.
    with (output/"history.jsonl").open("x") as history:
        for epoch in range(1, args.epochs+1):
            order = (2*rng.permutation(args.paths//2)[:, None]+np.arange(2)).reshape(-1)
            total = 0.
            for start in range(0, args.paths, args.batch_size):
                idx = order[start:start+args.batch_size]
                state, loss = step(state, *(sample[k][idx] for k in ("features", "returns", "payoff")))
                value = float(loss)
                if not np.isfinite(value):
                    raise FloatingPointError("nonfinite objective; lower --learning-rate")
                total += len(idx)*value
            elapsed = time.monotonic()-started
            record = dict(epoch=epoch, epochs=args.epochs, minibatch_loss=total/args.paths,
                          elapsed_seconds=elapsed, remaining_seconds=elapsed*(args.epochs-epoch)/epoch)
            history.write(json.dumps(record)+"\n")
            history.flush()
            print(json.dumps(record), flush=True)
    final = evaluate(state[0], reference, sample, raw, normal, config=config, batch=args.batch_size)
    metadata = dict(schema=EXAMPLE_SCHEMA, config=asdict(config), data=provenance, epochs=args.epochs,
                    batch_size=args.batch_size, numpy_version=np.__version__, jax_version=jax.__version__,
                    rng_state=rng.bit_generator.state, results=final)
    # Retain optimizer moments, update count, random state and normalization as
    # well as weights; this teaching CLI does not implement a resume command.
    saved = {f"{label}_{key}": np.asarray(value) for label, tree in
             zip(("params", "adam_first", "adam_second"), state[:3], strict=True) for key, value in tree.items()}
    saved.update({f"reference_{k}": np.asarray(v) for k, v in reference.items()})
    saved.update({f"normal_{k}": np.asarray(v) for k, v in normal.items()})
    np.savez_compressed(output/"checkpoint.npz", **saved, adam_step=np.asarray(state[3]), metadata=json.dumps(metadata))
    (output/"RESULTS.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
