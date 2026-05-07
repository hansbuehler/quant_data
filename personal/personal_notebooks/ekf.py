import ekf
import events
import jax
import jax.numpy as jnp
import polars as pl


def f(x):
    return x


def h(x):
    return x


def ekf_step(x, P, z, R_scalar, Q):
    F = jax.jacrev(f)(x)
    x_pred = f(x)
    P_pred = F @ P @ F.T + Q

    H = jax.jacrev(h)(x_pred)
    y = z - h(x_pred)
    S = H @ P_pred @ H.T + R_scalar
    K_T = jnp.linalg.solve(S, H @ P_pred)
    K = K_T.T

    x_new = x_pred + K @ y
    I = jnp.eye(x.shape[0], dtype=x.dtype)
    P_new = (I - K @ H) @ P_pred
    return x_new, P_new


@jax.jit
def run_ekf(prices, Rs):
    x0 = jnp.array([prices[0]], dtype=jnp.float32)
    P0 = jnp.array([[1.0]], dtype=jnp.float32)
    Q = jnp.array([[0.05]], dtype=jnp.float32)

    def scan_step(carry, obs):
        x, P = carry
        price, R_scalar = obs
        z = jnp.array([price], dtype=jnp.float32)
        R = jnp.array([[R_scalar]], dtype=jnp.float32)
        x_new, P_new = ekf_step(x, P, z, R, Q)
        return (x_new, P_new), x_new[0]

    (_, _), estimates = jax.lax.scan(scan_step, (x0, P0), (prices, Rs))
    return estimates


def estimate_true_price(df_events: pl.DataFrame) -> pl.DataFrame:
    prices = jnp.array(df_events["observed_price"].to_numpy(), dtype=jnp.float32)
    Rs = jnp.array(df_events["R"].to_numpy(), dtype=jnp.float32)

    estimates = run_ekf(prices, Rs)

    return df_events.with_columns(
        pl.Series(name="kalman_price", values=jax.device_get(estimates).tolist())
    )


def estimate_true_iv(df_events: pl.DataFrame) -> pl.DataFrame:
    prices = jnp.array(df_events["observed_iv"].to_numpy(), dtype=jnp.float32)
    Rs = jnp.array(df_events["R_iv"].to_numpy(), dtype=jnp.float32)

    estimates = run_ekf(prices, Rs)

    return df_events.with_columns(
        pl.Series(name="kalman_iv", values=jax.device_get(estimates).tolist())
    )


# Usage
# df_result = estimate_true_price(df_events)


def kalman_filters_price_from_events(df_events, trade_on_quotes=True):
    """
    trades on quotes false to be implemented
    """
    df_quotes_only = df_events.filter(pl.col("event_type") == "quote")

    prices_train = jnp.array(df_quotes_only["observed_price"].to_numpy(), dtype=jnp.float32)
    Rs_train = jnp.array(df_quotes_only["R"].to_numpy(), dtype=jnp.float32)

    estimates_train = ekf.run_ekf(prices_train, Rs_train)

    df_quotes_result = df_quotes_only.with_columns(
        pl.Series("kalman_price", jax.device_get(estimates_train).tolist())
    )

    # ── STEP 2: Test on trades ────────────────────────────────────────────────────
    # For each trade, get the Kalman estimate from the most recent preceding quote.
    # This is a pure out-of-sample evaluation: the filter never saw trades.

    df_trades_only = df_events.filter(pl.col("event_type") == "trade")

    df_trades_result = df_trades_only.join_asof(
        df_quotes_result.select(["ts_ns", "kalman_price"]).rename({"kalman_price": "kalman_prior"}),
        on="ts_ns",
        strategy="backward",
    )

    # Residual: how far is the trade price from the filter's quote-derived estimate?
    df_trades_result = df_trades_result.with_columns(
        (pl.col("observed_price") - pl.col("kalman_prior")).alias("trade_residual")
    )

    return df_trades_result, df_quotes_result


def kalman_filters_iv_from_events(df_events, trade_on_quotes=True):
    """
    trades on quotes false to be implemented
    """
    # ── STEP 1: Train the Kalman filter on quotes only ───────────────────────────
    df_iv_quotes_only = df_events.filter(pl.col("event_type") == "quote")

    # NaN R_iv (from null bid/ask on one-sided quotes) poisons the entire JAX
    # scan — once the state goes NaN it never recovers. Drop those rows before
    # running the EKF, then re-join kalman_iv back onto the full quote set.
    df_valid = df_iv_quotes_only.filter(
        pl.col("observed_iv").is_not_nan() & pl.col("R_iv").is_not_nan()
    )

    prices_train = jnp.array(df_valid["observed_iv"].to_numpy(), dtype=jnp.float32)
    Rs_train = jnp.array(df_valid["R_iv"].to_numpy(), dtype=jnp.float32)

    estimates_train = ekf.run_ekf(prices_train, Rs_train)

    df_valid = df_valid.with_columns(
        pl.Series("kalman_iv", jax.device_get(estimates_train).tolist())
    )

    # Re-join onto full quote set so rows with NaN inputs get null kalman_iv
    # (not a propagated NaN), then forward-fill from the last valid estimate.
    df_iv_quotes_result = df_iv_quotes_only.join(
        df_valid.select(["ts_ns", "kalman_iv"]), on="ts_ns", how="left"
    ).with_columns(pl.col("kalman_iv").fill_null(strategy="forward"))

    TRADE_LAG_NS = 250_000_000  # 250ms in nanoseconds

    df_iv_trades_only = df_events.filter(pl.col("event_type") == "trade")

    # Add window-end timestamp as join key
    df_iv_trades_only = df_iv_trades_only.with_columns(
        (pl.col("ts_ns") + pl.duration(nanoseconds=events.TRADE_LAG_NS)).alias("ts_ns_window_end")
    )

    # Keep quote ts as a data column so we can filter post-join
    df_quotes_for_join = (
        df_iv_quotes_result.select(["ts_ns", "kalman_iv"])
        .with_columns(pl.col("ts_ns").alias("quote_ts"))
        .rename({"kalman_iv": "kalman_matched"})
    )

    # join_asof backward on window_end → last quote with ts_ns <= trade_ts + 250ms
    df_iv_trades_result = (
        df_iv_trades_only.join_asof(
            df_quotes_for_join,
            left_on="ts_ns_window_end",
            right_on="ts_ns",
            strategy="backward",
        )
        # Keep only quotes that fall *after* the trade (not before it)
        # .filter(pl.col("quote_ts").is_not_null() & (pl.col("quote_ts") >= pl.col("ts_ns")))
        .drop("ts_ns_window_end")
    )

    df_iv_trades_result = df_iv_trades_result.with_columns(
        (pl.col("observed_iv") - pl.col("kalman_matched")).alias("trade_residual")
    )

    return df_iv_trades_result, df_iv_quotes_result
