# Grains & Oilseeds Risk Workbench

Streamlit desk prototype for corporate commodities exposure management. It is designed around futures and European options on futures in grains and the oilseed complex, excluding crude oil.

## Features
- Pre-trade Black-76 theoretical premium, Greeks, implied volatility from an observed premium, market-vs-model gap, and indicative checks.
- Trade-by-trade editable ledger, CSV import, full ledger CSV export, and a blank template.
- Portfolio P&L, delta-equivalent exposure, Gamma, Vega, Theta and gross futures notional.
- Full revaluation scenario grid for futures-price and implied-volatility shocks; downloadable results.
- Products pre-populated for Corn, Chicago SRW Wheat, KC HRW Wheat, Soybeans, Soybean Meal, Soybean Oil and Palm Oil.

## Install (Anaconda Prompt, Windows)
Extract the ZIP. Open Anaconda Prompt and run:

```bash
conda create -n grains-risk-workbench python=3.11 -y
conda activate grains-risk-workbench
cd "C:\PATH\TO\grains_oilseeds_risk_workbench"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Replace the `cd` path with the folder containing `app.py`.

## Ledger convention
- `signed_quantity`: positive long, negative short.
- For options, `trade_price` is entry premium per quote unit; `market_option_premium` is current observed market premium. Theoretical option mark-to-market P&L is computed using current futures price and volatility, minus entry premium.
- For futures and physical positions, `trade_price` is entry price and `market_futures_price` is current reference price.
- `contract_multiplier` is user-entered. Check the exchange's current specifications and quote units for each product before use.
- `risk_free_rate` and `implied_volatility` are decimals (e.g. 0.04 and 0.25).

## Important limitations
This is a prototype and not a production trading system. It uses Black-76 for European options on futures and does not automatically account for volatility smiles/skew, American exercise, contract-specific exercise/settlement details, calendar-spread or intercommodity correlation risk, local basis, margin, liquidity, bid/ask slippage, or physical inventory reconciliation. Scenario grids are not probabilistic VaR or Expected Shortfall. Market-vs-model comparison is only meaningful when inputs and quote units match the exact contract. Validate calculations against a trusted pricing source and exchange specifications before using in a live decision process.
