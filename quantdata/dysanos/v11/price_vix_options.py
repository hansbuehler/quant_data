"""Conditional P-payoff values of VIX options from the frozen neural models.

Daily simulations begin at the fitted as-of state and its decoded historical
context, not at the invariant distribution. The payoff index is a continuous
30-calendar-day log-contract proxy, not Cboe's discrete opening-auction SOQ.
Prior-close and expiry-close horizons show daily-model timing sensitivity;
neither is represented as exact morning settlement. No Q tilt, forward
matching, recentering or calibration to the VIX option quotes is performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import polars as pl
import torch
from cdxcore.err import verify_inp

from quantdata.biz_calendar import count_business_days_between, get_business_days
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import _decode_surface, load_presentation_model
from quantdata.dysanos.v11.cpu_pricing import enable_fast_cpu_decoder
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.variance_indices import BUSINESS_DTES, log_contract_total_variance
from quantdata.dysanos.v11.vix_sequence import VixSequenceConfig, VixSequenceModel
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig
from quantdata.widb import interpolate_zero_curve

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LABELS = {"lstm": "LSTM", "selective_ssm": "Mamba-style SSM"}
COLORS = {"lstm": "#d95f02", "selective_ssm": "#7a5195"}


def calendar_index_from_total_variance(
    total_variances: torch.Tensor, *, model_business_days: np.ndarray,
    residual_business_days: int, calendar_days: int = 30,
) -> torch.Tensor:
    """Interpolate total log-contract variance, then annualize in calendar time."""
    nodes = np.asarray(model_business_days, dtype=np.float64)
    verify_inp(nodes.ndim == 1 and len(nodes) >= 2 and np.all(np.diff(nodes) > 0), "Need at least two strictly increasing model tenors")
    verify_inp(total_variances.ndim == 2 and total_variances.shape[1] == len(nodes), "Variance/tenor shape mismatch")
    verify_inp(nodes[0] <= residual_business_days <= nodes[-1] and calendar_days > 0, "No tenor extrapolation is permitted")
    upper = min(int(np.searchsorted(nodes, residual_business_days, side="right")), len(nodes) - 1)
    lower = max(upper - 1, 0)
    alpha = (residual_business_days - nodes[lower]) / (nodes[upper] - nodes[lower])
    variance = (1 - alpha) * total_variances[:, lower] + alpha * total_variances[:, upper]
    if not bool(torch.all(torch.isfinite(variance) & (variance > 0))):
        raise RuntimeError("Invalid decoded log-contract variance")
    return 100 * torch.sqrt(variance * (365. / calendar_days))


def paired_payoff_statistics(
    terminal_index: np.ndarray, *, strike: float, is_call: bool, discount_factor: float,
) -> dict[str, float | int | bool]:
    """Correct MC errors for positive/negative shock pairs, not individual paths."""
    values = np.asarray(terminal_index, dtype=np.float64)
    verify_inp(values.ndim == 1 and len(values) >= 4 and len(values) % 2 == 0, "Need at least two antithetic pairs")
    verify_inp(np.all(np.isfinite(values)) and np.all(values > 0), "Terminal index levels must be positive and finite")
    verify_inp(np.isfinite(strike) and strike > 0 and np.isfinite(discount_factor) and discount_factor > 0,
               "Positive finite strike and discount factor required")
    signed = values - strike if is_call else strike - values
    payoffs = discount_factor * np.maximum(signed, 0)
    half = len(values) // 2
    pair_means = .5 * (payoffs[:half] + payoffs[half:])
    value = float(pair_means.mean())
    error = float(pair_means.std(ddof=1) / np.sqrt(half))
    nonzero = int(np.count_nonzero(pair_means))
    return {"model_value": value, "mc_se": error, "mc_low_95": value - 1.96 * error,
            "mc_high_95": value + 1.96 * error, "independent_pairs": half,
            "positive_payoff_pairs": nonzero, "mc_tail_resolved": nonzero >= 10}


def parity_forward(quotes: pl.DataFrame, *, discount_factor: float) -> float:
    """Infer the market forward at the closest matched call/put premium pair."""
    clean = quotes.filter((pl.col("best_bid") >= 0) & (pl.col("best_offer") >= pl.col("best_bid")))
    calls = clean.filter(pl.col("cp_flag") == "C").select("strike_price", ((pl.col("best_bid") + pl.col("best_offer")) / 2).alias("call"))
    puts = clean.filter(pl.col("cp_flag") == "P").select("strike_price", ((pl.col("best_bid") + pl.col("best_offer")) / 2).alias("put"))
    paired = calls.join(puts, on="strike_price").with_columns((pl.col("call") - pl.col("put")).abs().alias("distance")).sort("distance")
    verify_inp(paired.height > 0, "No matched call/put pair to identify the forward")
    nearest = paired.row(0, named=True)
    return float(nearest["strike_price"] + (nearest["call"] - nearest["put"]) / discount_factor)


def _decode(model: object, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    surface = _decode_surface(model, states)
    tv = log_contract_total_variance(surface.support_strikes, surface.densities, surface.backbone_total_vols.square())
    indices = [list(model.config.geometry.model_business_dtes).index(float(t)) for t in BUSINESS_DTES]
    log_indices = torch.log(100 * torch.sqrt(tv[:, indices] / surface.expiries_y[:, indices]))
    if not bool(torch.all(torch.isfinite(log_indices))):
        raise RuntimeError("Nonfinite recurrent input")
    return tv, log_indices


def run(*, output: Path, paths: int = 10_000, batch_pairs: int = 1024, threads: int = 8, seed: int = 20261029) -> dict:
    verify_inp(paths >= 4 and paths % 2 == 0 and batch_pairs > 0 and threads > 0, "Invalid simulation dimensions")
    torch.set_num_threads(threads)
    market_dir = output / "market"
    inputs = json.loads((market_dir / "manifest.json").read_text())
    asof = date.fromisoformat(inputs["as_of"])
    source = Path(inputs["source_model"])
    sequence = Path(inputs["sequence_run"])
    checkpoint = source / "checkpoints/joint-direct-200.pt"
    source_sha = checkpoint_digest(checkpoint)
    results_dir = output / f"simulation-{paths}-{seed}"
    results_dir.mkdir(parents=True, exist_ok=False)
    print(f"Loading frozen models; CPU threads={threads}, paths={paths}, pairs/batch={batch_pairs}", flush=True)
    _, decoder = load_presentation_model(checkpoint, device="cpu", precision=64)
    enable_fast_cpu_decoder(decoder)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    fit_manifest = json.loads((source / "reports/run-manifest.json").read_text())
    verify_inp(fit_manifest["as_of"] == str(asof), "Market and model dates differ")
    with np.load(market_dir / "spot_history.npz") as data:
        spot_dates = list(data["dates"])
        spots = data["spots"]
    verify_inp(spot_dates == fit_manifest["training_dates"][-21:], "Spot history not aligned with the fitted states")
    # Reconstruct the same front-PIT convention used to profile the original spot loading.
    _, initial = load_presentation_model(checkpoint, device="cpu", precision=64)
    affine_path = source / "checkpoints/direct-200.pt"
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    verify_inp(checkpoint_digest(affine_path) == original["source_decoder_sha256"], "Original affine initializer changed")
    initial.load_state_dict(torch.load(affine_path, map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
    enable_fast_cpu_decoder(initial)
    scores = _historical_spot_scores(initial, initial.daily_states.detach().numpy()[-21:-1], spots[1:] / spots[:-1], batch_size=32)
    del initial, original
    historical = pl.read_parquet(inputs["historic_targets"])
    all_log_indices = np.log(np.column_stack([historical[f"vix_{t}d"].to_numpy() for t in BUSINESS_DTES]))
    center = torch.tensor(np.nanmean(all_log_indices, axis=0), dtype=torch.float64)
    scale = torch.tensor(np.nanstd(all_log_indices, axis=0), dtype=torch.float64)
    with np.load(inputs["gaussian_recovery"]) as recovery:
        beta = torch.tensor(recovery["spot_loading"], dtype=torch.float64)
        score_mean = float(recovery["spot_score_mean"])
    transition = decoder.weekly_transition().detach()  # This checkpoint was fitted on daily pairs.
    mean = decoder.equilibrium_mean.detach()
    factor = decoder.innovation_cholesky().detach()
    nodes = np.asarray(decoder.config.geometry.model_business_dtes)

    def feature(log_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat(((log_indices - center) / scale, z[:, None]), dim=-1)

    with torch.inference_mode():
        _, historical_indices = _decode(decoder, decoder.daily_states[-20:])
        context0 = feature(historical_indices, torch.tensor(scores, dtype=torch.float64))
        initial_state = decoder.daily_states[-1].detach().clone()
        initial_tv, _ = _decode(decoder, initial_state[None])
    expiries = [date.fromisoformat(x) for x in inputs["pricing_expiries"]]
    cases = []
    for expiry in expiries:
        prior = get_business_days(asof, expiry - timedelta(days=1))[-1]
        for timing, terminal in (("prior_close", prior), ("expiry_close", expiry)):
            cases.append({"expiry": str(expiry), "timing": timing, "terminal_date": str(terminal),
                          "sessions": int(count_business_days_between(asof, terminal)),
                          "index_business_days": int(count_business_days_between(terminal, terminal + timedelta(days=30)))})
    max_steps = max(c["sessions"] for c in cases)
    quote_frame = pl.read_parquet(market_dir / "vix_quotes.parquet")
    expiry_column = pl.col("exdate").str.to_date() if quote_frame.schema["exdate"] == pl.String else pl.col("exdate").cast(pl.Date)
    quote_frame = quote_frame.with_columns(expiry_column)
    verify_inp(not quote_frame.select("exdate", "strike_price", "cp_flag").is_duplicated().any(),
               "Duplicate option contracts require explicit settlement/contract filtering")
    rates = pl.read_parquet(market_dir / "zero_curve.parquet").to_pandas()
    curve = {str(t): float(interpolate_zero_curve(rates, np.array([(t - asof).days], dtype=float))[0]) for t in expiries}
    market_forwards = {str(t): parity_forward(quote_frame.filter(pl.col("exdate") == t), discount_factor=curve[str(t)]) for t in expiries}
    all_rows, summary = [], []
    started = perf_counter()
    half = paths // 2
    for architecture in LABELS:
        saved_path = sequence / architecture / "last.pt"
        saved = torch.load(saved_path, map_location="cpu", weights_only=False)
        verify_inp(saved["source"]["decoder_sha256"] == source_sha, "Sequence model has a different decoder")
        verify_inp(saved["source"]["target_sha256"] == checkpoint_digest(Path(inputs["historic_targets"])), "Sequence normalizers do not match")
        raw_config = dict(saved["model_config"])
        raw_config["readout"] = FeedForwardConfig(**raw_config["readout"])
        config = VixSequenceConfig(**raw_config)
        verify_inp(saved["model_version"] == config.version.unique_id48, "Sequence checkpoint code version is stale")
        network = VixSequenceModel(BaseEnvironment(device="cpu", precision=64, debug=False), config=config, white_basis=saved["model"]["white_basis"])
        network.load_state_dict(saved["model"])
        network.eval()
        terminals = np.empty((paths, len(cases)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for offset in range(0, half, batch_pairs):
            pairs = min(batch_pairs, half - offset)
            h = initial_state[None].expand(2 * pairs, -1).clone()
            context = context0[None].expand(2 * pairs, -1, -1).clone()
            batch_started = perf_counter()
            with torch.inference_mode():
                for step in range(1, max_steps + 1):
                    base_z = torch.randn(pairs, generator=generator, dtype=torch.float64)
                    base_epsilon = torch.randn(pairs, len(initial_state), generator=generator, dtype=torch.float64)
                    z = torch.cat((base_z, -base_z))
                    epsilon = torch.cat((base_epsilon, -base_epsilon))
                    white, _ = network(context, z, epsilon)
                    h = mean + (h - mean) @ transition + (z - score_mean)[:, None] * beta + white @ factor.T
                    tv, log_indices = _decode(decoder, h)
                    context = torch.cat((context[:, 1:], feature(log_indices, z)[:, None]), dim=1)
                    for j, case in enumerate(cases):
                        if step == case["sessions"]:
                            values = calendar_index_from_total_variance(tv, model_business_days=nodes,
                                                                       residual_business_days=case["index_business_days"]).numpy()
                            terminals[offset:offset + pairs, j] = values[:pairs]
                            terminals[half + offset:half + offset + pairs, j] = values[pairs:]
                    if step == 1 or step % 5 == 0 or step == max_steps:
                        completed = offset * max_steps + pairs * step
                        elapsed = perf_counter() - started
                        within_elapsed = perf_counter() - batch_started
                        remaining = within_elapsed / step * (max_steps * (half - offset) / pairs - step)
                        print(f"{architecture} pairs={offset}:{offset + pairs}/{half} step={step}/{max_steps} "
                              f"elapsed={elapsed:.1f}s model_remaining~{remaining:.1f}s", flush=True)
                        write_json_atomic({"architecture": architecture, "completed_pair_steps": completed,
                                           "total_pair_steps": half * max_steps, "elapsed_seconds": elapsed,
                                           "estimated_model_remaining_seconds": remaining}, results_dir / "progress.json")
            np.savez_compressed(results_dir / f"{architecture}-terminal-progress.npz", terminal_indices=terminals[:offset + pairs],
                                antithetic_terminal_indices=terminals[half:half + offset + pairs], completed_pairs=offset + pairs)
        np.savez_compressed(results_dir / f"{architecture}-terminal-indices.npz", values=terminals)
        for j, case in enumerate(cases):
            expiry = date.fromisoformat(case["expiry"])
            discount = curve[case["expiry"]]
            options = quote_frame.filter(pl.col("exdate") == expiry)
            for quote in options.iter_rows(named=True):
                stats = paired_payoff_statistics(terminals[:, j], strike=quote["strike_price"],
                                                 is_call=quote["cp_flag"] == "C", discount_factor=discount)
                all_rows.append({"model": LABELS[architecture], **case, "cp_flag": quote["cp_flag"],
                                 "strike": quote["strike_price"], "market_bid": quote["best_bid"],
                                 "market_ask": quote["best_offer"], "market_forward": market_forwards[case["expiry"]],
                                 "discount_factor": discount, **stats})
            summary.append({"model": LABELS[architecture], **case, "physical_expected_index": float(terminals[:, j].mean()),
                            "market_forward": market_forwards[case["expiry"]], "pairs": half,
                            "initial_index_proxy": float(calendar_index_from_total_variance(initial_tv, model_business_days=nodes,
                                residual_business_days=int(count_business_days_between(asof, asof + timedelta(days=30))))[0])})
        pl.DataFrame(all_rows).write_csv(results_dir / "option_values.csv")
        del network, saved
    frame = pl.DataFrame(all_rows).with_columns(
        ((pl.col("market_bid") + pl.col("market_ask")) / 2).alias("market_mid"),
        ((pl.col("market_bid") > 0) & (pl.col("market_ask") >= pl.col("market_bid"))).alias("valid_market_quote"),
    ).with_columns((pl.col("model_value") - pl.col("market_mid")).alias("difference_from_mid"))
    frame.write_csv(results_dir / "option_values.csv")
    pl.DataFrame(summary).write_csv(results_dir / "forward_comparison.csv")
    _plot(frame, results_dir)
    report = {"status": "complete", "measure": "P, without a measure change or market-forward adjustment", "as_of": str(asof),
              "paths_per_model": paths, "independent_pairs_per_model": half, "cases": cases,
              "seed": seed, "batch_pairs": batch_pairs, "cpu_threads": threads,
              "market_vix_close": inputs["vix_close"], "source_model": str(source), "sequence_run": str(sequence),
              "input_sha256": {str(path): checkpoint_digest(path) for path in (
                  market_dir / "vix_quotes.parquet", market_dir / "zero_curve.parquet", market_dir / "spot_history.npz",
                  sequence / "lstm/last.pt", sequence / "selective_ssm/last.pt")},
              "source_decoder_sha256": source_sha, "normalization": "none; actual VIX-point strikes and discounted payoff units",
              "initial_context": "last 20 fitted daily surfaces plus realized SPX front-PIT scores; no stationary burn-in",
              "index": "continuous-strike 30-calendar-day log-contract proxy, calendar-annualized; NOT Cboe SOQ",
              "settlement": "prior-close and expiry-close sensitivity, neither an exact AM-settlement simulation",
              "mc_error": "sample std of antithetic-pair payoff means / sqrt(number of pairs); pointwise approximate 95% intervals",
              "rare_payoffs": "Fewer than 10 nonzero payoff pairs flagged unresolved; zero sampled payoff is not proof of zero value",
              "market_quote_rows": frame.select("expiry", "cp_flag", "strike").unique().height,
              "option_value_rows": frame.height, "elapsed_seconds": perf_counter() - started}
    write_json_atomic(report, results_dir / "report.json")
    print(json.dumps({"status": "complete", "results": str(results_dir)}, indent=2), flush=True)
    return report


def _plot(frame: pl.DataFrame, output: Path) -> None:
    expiries = sorted(frame["expiry"].unique())
    fig, axes = plt.subplots(2, len(expiries), figsize=(12, 7.4))
    for col, expiry in enumerate(expiries):
        for row, cp in enumerate(("C", "P")):
            ax = axes[row, col]
            subset = frame.filter((pl.col("expiry") == expiry) & (pl.col("cp_flag") == cp) & pl.col("valid_market_quote")
                                  & (pl.col("strike") <= 40)).sort("strike")
            market = subset.unique("strike").sort("strike")
            ax.fill_between(market["strike"], market["market_bid"], market["market_ask"], alpha=.25, color="#348abd", label="WRDS bid–ask")
            ax.plot(market["strike"], market["market_mid"], color="#348abd", linewidth=1.3, label="WRDS midpoint")
            for architecture, label in LABELS.items():
                for timing in ("prior_close", "expiry_close"):
                    d = subset.filter((pl.col("model") == label) & (pl.col("timing") == timing)).sort("strike")
                    ax.plot(d["strike"], d["model_value"], color=COLORS[architecture], linestyle="-" if timing == "prior_close" else "--",
                            label=label + (" prior close" if timing == "prior_close" else " expiry close"))
                    if timing == "prior_close":
                        ax.fill_between(d["strike"], d["mc_low_95"].clip(lower_bound=0), d["mc_high_95"], color=COLORS[architecture], alpha=.13)
            ax.set_title(f"{expiry}: {'calls' if cp == 'C' else 'puts'}")
            ax.set(xlabel="Strike (VIX points)", ylabel="Option value (VIX points)")
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("VIX option payoff values: conditional LSTM / Mamba-style P models\n2025-08-29; continuous VIX proxy; no Q reweighting or forward matching", fontsize=12)
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.18, hspace=.38, wspace=.22)
    fig.savefig(output / "option_values_comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--batch-pairs", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261029)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
