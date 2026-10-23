# `quantdata.vixdata`

`refresh_cboe_vix()` downloads Cboe's public official daily VIX OHLC history
and stores it beneath the native `quantdata.env.locations("vixdata").raw`
location.  On the production configuration this is the canonical AAE-NAS raw
data root.  No fixed local path is embedded in the module.

`load_cboe_vix(start, end)` reads the cache and returns dates, open, high, low,
and close.  VIX levels are annualized implied-volatility percentage points.

The source is Cboe's public VIX historical-data CSV.  The module keeps its
source URL and retrieval timestamp in native-cache metadata.
