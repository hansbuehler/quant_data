# Black delta-and-vega hedge

This deterministic daily baseline uses the public dataset's ATM volatility curve.
For each digital target, Black's formula supplies the target's spot delta and
annual-volatility vega. The hedge uses the eligible vanilla expiry matching the
target, or the first later eligible expiry with a strike bracket. It takes the two quoted OTM
options immediately surrounding the digital's **fixed strike**, with weights
proportional to the opposite strike distances. Their *total* notional cancels
the target's Black vega. A one-day forward-adjusted spot position cancels the
combined delta of the digital and **both** options.

The vega represents one parallel shift to the Black ATM-volatility curve, not
the eight DLV PCA factors of the SANOS-LV baseline. Pricing is Black; realized
marks, transaction costs, and P&L use the same SANOS market panel and accounting
as every other daily benchmark. The model reports the actual bracketing contract
IDs, interpolation weights, Greek residuals, and any position-cap hits.

Run the small unit test with
`pytest -q tests/quantdata/sdlv/daily/hedging/models/test_bs_greeks.py`.
Run the rolling 1D benchmark with
`python -m quantdata.sdlv.daily.front_spot.hedging.runners.run_daily --model bs-greeks --target-maturity 1 --output /path/out`.
For a comparison already normalized to one digital cash-payout unit, use
`python -m quantdata.sdlv.daily.front_spot.hedging.models.bs_greeks.evaluate_dates --dates-csv /path/common_1d_pnl_usd.csv --output /path/out`.
This standalone path fits no target notionals and keeps unit claim quantity.
