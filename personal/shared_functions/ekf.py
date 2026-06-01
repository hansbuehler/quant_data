import jax
import jax.numpy as jnp
import polars as pl

import personal.shared_functions.events as events


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


def kalman_filters_price_from_events(df_events):
    df_quotes_only = df_events.filter(pl.col("event_type") == "quote")

    prices_train = jnp.array(df_quotes_only["observed_price"].to_numpy(), dtype=jnp.float32)
    Rs_train = jnp.array(df_quotes_only["R"].to_numpy(), dtype=jnp.float32)
    estimates_train = run_ekf(prices_train, Rs_train)

    df_quotes_result = df_quotes_only.with_columns(
        pl.Series("kalman_price", jax.device_get(estimates_train).tolist())
    )

    df_trades_only = df_events.filter(pl.col("event_type") == "trade")
    df_trades_result = df_trades_only.join_asof(
        df_quotes_result.select(["ts_ns", "kalman_price"]).rename({"kalman_price": "kalman_prior"}),
        on="ts_ns",
        strategy="backward",
    ).with_columns((pl.col("observed_price") - pl.col("kalman_prior")).alias("trade_residual"))

    return df_trades_result, df_quotes_result


def kalman_filters_iv_from_events(df_events):
    df_iv_quotes_only = df_events.filter(pl.col("event_type") == "quote")

    df_valid = df_iv_quotes_only.filter(
        pl.col("observed_iv").is_not_nan() & pl.col("R_iv").is_not_nan()
    )

    prices_train = jnp.array(df_valid["observed_iv"].to_numpy(), dtype=jnp.float32)
    Rs_train = jnp.array(df_valid["R_iv"].to_numpy(), dtype=jnp.float32)
    estimates_train = run_ekf(prices_train, Rs_train)

    df_valid = df_valid.with_columns(
        pl.Series("kalman_iv", jax.device_get(estimates_train).tolist())
    )

    df_iv_quotes_result = df_iv_quotes_only.join(
        df_valid.select(["ts_ns", "kalman_iv"]), on="ts_ns", how="left"
    ).with_columns(pl.col("kalman_iv").fill_null(strategy="forward"))

    df_iv_trades_only = df_events.filter(pl.col("event_type") == "trade").with_columns(
        (pl.col("ts_ns") + pl.duration(nanoseconds=events.TRADE_LAG_NS)).alias("ts_ns_window_end")
    )

    df_quotes_for_join = (
        df_iv_quotes_result.select(["ts_ns", "kalman_iv"])
        .with_columns(pl.col("ts_ns").alias("quote_ts"))
        .rename({"kalman_iv": "kalman_matched"})
    )

    df_iv_trades_result = (
        df_iv_trades_only.join_asof(
            df_quotes_for_join,
            left_on="ts_ns_window_end",
            right_on="ts_ns",
            strategy="backward",
        )
        .drop("ts_ns_window_end")
        .with_columns((pl.col("observed_iv") - pl.col("kalman_matched")).alias("trade_residual"))
    )

    return df_iv_trades_result, df_iv_quotes_result


def kalman_filters_price(
    df_quotes: pl.DataFrame,
    df_matched: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Train EKF on quote prices; test on matched trades.

    Parameters
    ----------
    df_quotes : pl.DataFrame
        Raw quotes (MassiveClient.SCHEMA_QUOTES).
    df_matched : pl.DataFrame
        Output of match_taq (SCHEMA_TAQ). Rows with null quote_ts_ns are skipped.

    Returns
    -------
    (df_trades_result, df_quotes_result)
        df_quotes_result : processed quotes with kalman_price added.
        df_trades_result : matched trades (non-null quote_ts_ns) with
                           kalman_prior and trade_residual added.
    """
    df_q = events.process_quotes(df_quotes)

    prices_train = jnp.array(df_q["observed_price"].to_numpy(), dtype=jnp.float32)
    Rs_train = jnp.array(df_q["R"].to_numpy(), dtype=jnp.float32)
    estimates = run_ekf(prices_train, Rs_train)

    df_quotes_result = df_q.with_columns(
        pl.Series("kalman_price", jax.device_get(estimates).tolist())
    )

    # For each trade, look up the Kalman state at its matched quote's timestamp.
    # This is a pure out-of-sample evaluation: the filter never saw any trade.
    df_trades_result = (
        df_matched.filter(pl.col("quote_ts_ns").is_not_null())
        .sort("quote_ts_ns")
        .join_asof(
            df_quotes_result.select(["ts_ns", "kalman_price"]).rename(
                {"kalman_price": "kalman_prior"}
            ),
            left_on="quote_ts_ns",
            right_on="ts_ns",
            strategy="backward",
        )
        .drop("ts_ns_right", strict=False)
        .with_columns((pl.col("price") - pl.col("kalman_prior")).alias("trade_residual"))
    )

    return df_trades_result, df_quotes_result


def kalman_filters_iv(
    df_quotes_with_iv: pl.DataFrame,
    df_matched_with_iv: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Train EKF on quote implied vols; test on matched trade IVs.

    Parameters
    ----------
    df_quotes_with_iv : pl.DataFrame
        Processed quotes with observed_iv and R_iv columns added.
    df_matched_with_iv : pl.DataFrame
        Matched trades (non-null quote_ts_ns) with observed_iv column added.

    Returns
    -------
    (df_iv_trades_result, df_iv_quotes_result)
    """
    # NaN IV/R poisons the JAX scan — drop then re-join to preserve all quote rows
    df_valid = df_quotes_with_iv.filter(
        pl.col("observed_iv").is_not_nan() & pl.col("R_iv").is_not_nan()
    )

    prices_train = jnp.array(df_valid["observed_iv"].to_numpy(), dtype=jnp.float32)
    Rs_train = jnp.array(df_valid["R_iv"].to_numpy(), dtype=jnp.float32)
    estimates = run_ekf(prices_train, Rs_train)

    df_valid = df_valid.with_columns(pl.Series("kalman_iv", jax.device_get(estimates).tolist()))

    # Forward-fill over gaps where IV/R was NaN so every quote row has a kalman_iv
    df_iv_quotes_result = df_quotes_with_iv.join(
        df_valid.select(["ts_ns", "kalman_iv"]), on="ts_ns", how="left"
    ).with_columns(pl.col("kalman_iv").fill_null(strategy="forward"))

    # For each trade, look up the Kalman IV at its matched quote's timestamp
    df_iv_trades_result = (
        df_matched_with_iv.sort("quote_ts_ns")
        .join_asof(
            df_iv_quotes_result.select(["ts_ns", "kalman_iv"]).rename(
                {"kalman_iv": "kalman_matched"}
            ),
            left_on="quote_ts_ns",
            right_on="ts_ns",
            strategy="backward",
        )
        .with_columns((pl.col("observed_iv") - pl.col("kalman_matched")).alias("trade_residual"))
    )

    return df_iv_trades_result, df_iv_quotes_result
