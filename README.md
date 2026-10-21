# Grains & Oilseeds Risk Workbench

Streamlit hedging-desk prototype for corporate commodities exposure management: futures and European options on futures in corn, wheat (Chicago SRW and KC HRW), the soybean complex (beans, meal, oil) and palm oil. Crude oil is excluded.

Pricing uses **Black-76**, i.e. the Black-Scholes-Merton model for options on futures (cost of carry b = 0, underlying = futures price).

## Features
- **Add a trade (main page):** manual entry panel (commodity, instrument, buy/sell, contracts, contract month, price, multiplier, option terms) that appends straight to the portfolio ledger.
- **1 · Pre-trade analytics:** Black-76 premium, Greeks, implied volatility from an observed premium, market-vs-model gap, premium-vs-futures curve, and one-click "add to ledger".
- **2 · Hedge sizing:** converts a physical exposure in bushels, thousand/million bushels, metric tons, short tons, pounds or kg into exchange contracts. Supports hedge ratio, basis/cross-hedge beta, rounding policy, contract-size override (mini contracts), notional and 1% P&L sensitivity, a hedge-ratio ladder, and a protective-option alternative sized on delta (long call for a buyer, long put for a seller) with premium outlay. The hedge can be booked into the ledger in one click.
- **3 · Trade ledger & CSV:** editable ledger, CSV import (replace or append), export, blank template, validation.
- **4 · Portfolio risk & stress:** P&L, delta, gamma, vega, theta, notional by trade and by product; full-revaluation price × volatility stress grid (live, no button needed) with a pivot view and CSV export.
- Aggregates are converted to **USD equivalent** (palm oil is quoted in MYR; set the FX rate in the sidebar). Delta is also shown in futures-equivalent contracts by product, because summing raw delta across products with different contract sizes is not meaningful.

## Efficiency changes
Pricing, Greeks and the stress grid are fully numpy-vectorised (`scipy.special.ndtr`); the whole ledger and the whole scenario grid are revalued in one broadcast instead of looping over every row and scenario with scalar scipy calls. The ledger lives in `st.session_state`, and stress results are cached.

## Run locally
```bash
conda create -n grains-risk-workbench python=3.11 -y
conda activate grains-risk-workbench
cd "C:\PATH\TO\grains_oilseeds_risk_workbench"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Run the tests with `python -m unittest test_models.py`.

## Deploy from GitHub (Streamlit Community Cloud)
1. Push `app.py`, `risk_engine.py`, `requirements.txt` (and the rest) to a GitHub repository.
2. Go to share.streamlit.io, choose **New app**, select the repo and branch, and set the main file to `app.py`.
3. Deploy. Every push to the branch redeploys the app.

The ledger is held in the browser session only. Export the CSV before closing the page, or re-import it next time.

## Contract specifications used for defaults
| Product | Exchange | Size | Quote | Multiplier |
|---|---|---|---|---|
| Corn | CBOT | 5,000 bu | US cents/bu | 50 |
| Chicago SRW Wheat | CBOT | 5,000 bu | US cents/bu | 50 |
| KC HRW Wheat | CME | 5,000 bu | US cents/bu | 50 |
| Soybeans | CBOT | 5,000 bu | US cents/bu | 50 |
| Soybean Meal | CBOT | 100 short tons | USD/short ton | 100 |
| Soybean Oil | CBOT | 60,000 lb | US cents/lb | 600 |
| Palm Oil | Bursa Malaysia (FCPO) | 25 MT | MYR/MT | 25 |

Bushel conversions: corn 56 lb/bu (39.368 bu per MT); wheat and soybeans 60 lb/bu (36.744 bu per MT). Placeholder prices in the app are illustrative only. **Verify every specification against the current exchange rulebook before use.**

## Ledger convention
- `signed_quantity`: positive long, negative short, in contracts.
- Options: `trade_price` is the entry premium per quote unit; `market_option_premium` is the current observed premium. **Current P&L uses the observed market premium** (Black-76 is used for Greeks and for stress revaluation, anchored to that observed premium).
- Futures and physical: `trade_price` is the entry price and `market_futures_price` the current price.
- `contract_multiplier` = currency P&L per 1 quote unit per contract.
- `risk_free_rate` and `implied_volatility` are decimals (e.g. 0.04 and 0.25).

## Important limitations
This is a prototype and not a production trading system. It does not automatically account for volatility smiles/skew, American exercise, contract-specific exercise/settlement details, calendar-spread or intercommodity correlation risk, local basis, margin, liquidity, bid/ask slippage, or physical inventory reconciliation. The hedge-sizing beta is a user input, not an estimate. Scenario grids are not probabilistic VaR or Expected Shortfall. Market-vs-model comparison is only meaningful when inputs and quote units match the exact contract. Validate calculations against a trusted pricing source and exchange specifications before using in a live decision process.
