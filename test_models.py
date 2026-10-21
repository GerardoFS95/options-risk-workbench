import math
import unittest

import numpy as np
import pandas as pd

from risk_engine import (black76_greeks, black76_price, clean_ledger, implied_volatility,
                         ledger_issues, ledger_risk, signed_position_value, size_hedge,
                         stress_grid, to_native)


def make_ledger():
    base = dict(trade_date='2026-10-20', risk_free_rate=0.04, implied_volatility=0.25, notes='', contract_month='Dec-2026', purpose='Hedge')
    rows = [
        {**base, 'trade_id': 'A', 'product': 'Soybeans', 'instrument_type': 'Call', 'signed_quantity': 5.0,
         'trade_price': 45.0, 'market_futures_price': 1050.0, 'strike': 1050.0, 'days_to_expiry': 90.0,
         'market_option_premium': 48.0, 'contract_multiplier': 50.0},
        {**base, 'trade_id': 'B', 'product': 'Corn', 'instrument_type': 'Put', 'signed_quantity': -2.0,
         'trade_price': 12.0, 'market_futures_price': 455.0, 'strike': 440.0, 'days_to_expiry': 60.0,
         'market_option_premium': 11.0, 'contract_multiplier': 50.0},
        {**base, 'trade_id': 'C', 'product': 'Corn', 'instrument_type': 'Future', 'signed_quantity': -3.0,
         'trade_price': 450.0, 'market_futures_price': 455.0, 'strike': 0.0, 'days_to_expiry': 0.0,
         'market_option_premium': 0.0, 'contract_multiplier': 50.0},
        {**base, 'trade_id': 'D', 'product': 'Palm Oil', 'instrument_type': 'Future', 'signed_quantity': 4.0,
         'trade_price': 4000.0, 'market_futures_price': 4100.0, 'strike': 0.0, 'days_to_expiry': 0.0,
         'market_option_premium': 0.0, 'contract_multiplier': 25.0},
    ]
    return clean_ledger(pd.DataFrame(rows))


class Black76Tests(unittest.TestCase):
    def test_call_put_parity(self):
        F, K, T, r, v = 450, 460, 0.5, 0.04, 0.25
        c = black76_price(F, K, T, r, v, 'Call'); p = black76_price(F, K, T, r, v, 'Put')
        self.assertAlmostEqual(c - p, math.exp(-r * T) * (F - K), places=8)

    def test_iv_roundtrip(self):
        premium = black76_price(450, 460, 0.5, 0.04, 0.25, 'Call')
        self.assertAlmostEqual(implied_volatility(premium, 450, 460, 0.5, 0.04, 'Call'), 0.25, places=7)

    def test_greeks_present(self):
        g = black76_greeks(450, 450, 0.5, 0.04, 0.25, 'Call')
        self.assertGreater(g['Theoretical premium'], 0); self.assertGreater(g['Gamma'], 0)
        self.assertGreater(g['Vega (per 1 vol pt)'], 0)

    def test_delta_matches_finite_difference(self):
        F, K, T, r, v, h = 450, 460, 0.5, 0.04, 0.25, 0.01
        for typ in ('Call', 'Put'):
            fd = (black76_price(F + h, K, T, r, v, typ) - black76_price(F - h, K, T, r, v, typ)) / (2 * h)
            self.assertAlmostEqual(black76_greeks(F, K, T, r, v, typ)['Delta (per futures unit)'], fd, places=6)

    def test_vega_matches_finite_difference(self):
        F, K, T, r, v, h = 450, 460, 0.5, 0.04, 0.25, 1e-5
        fd = (black76_price(F, K, T, r, v + h) - black76_price(F, K, T, r, v - h)) / (2 * h)
        self.assertAlmostEqual(black76_greeks(F, K, T, r, v)['Vega (per 1.00 vol)'], fd, places=4)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            black76_price(450, 450, 0, 0.04, 0.25)


class PortfolioTests(unittest.TestCase):
    def test_zero_shock_matches_current_pnl(self):
        led = make_ledger()
        sd = stress_grid(led, [0], [0])
        self.assertAlmostEqual(sd.change_vs_current_pnl.iloc[0], 0.0, places=8)

    def test_vector_stress_matches_scalar_revaluation(self):
        led = make_ledger()
        sd = stress_grid(led, [-10, 10], [-5, 5], {'MYR': 0.23})
        row = sd[(sd['futures_price_shock_%'] == 10) & (sd['volatility_shock_pts'] == 5)].iloc[0]
        total = 0.0
        for _, r in led.iterrows():
            fx = 0.23 if r['product'] == 'Palm Oil' else 1.0
            newF = r.market_futures_price * 1.10
            if r.instrument_type in ('Future', 'Physical'):
                pnl = r.signed_quantity * r.contract_multiplier * (newF - r.trade_price)
            else:
                T = r.days_to_expiry / 365
                base = black76_price(r.market_futures_price, r.strike, T, r.risk_free_rate, r.implied_volatility, r.instrument_type)
                new = black76_price(newF, r.strike, T, r.risk_free_rate, r.implied_volatility + 0.05, r.instrument_type)
                pnl = r.signed_quantity * r.contract_multiplier * (r.market_option_premium + new - base - r.trade_price)
            total += pnl * fx
        self.assertAlmostEqual(row.stressed_portfolio_pnl, total, places=6)

    def test_ledger_risk_matches_row_pnl(self):
        led = make_ledger()
        risk = ledger_risk(led)
        expected = [signed_position_value(r) for _, r in led.iterrows()]
        np.testing.assert_allclose(risk.current_pnl.to_numpy(), expected)

    def test_fx_applied_to_palm_only(self):
        led = make_ledger()
        risk = ledger_risk(led, {'MYR': 0.25})
        palm = risk[risk['product'] == 'Palm Oil'].iloc[0]
        self.assertAlmostEqual(palm.current_pnl_usd, palm.current_pnl * 0.25)

    def test_blank_rows_dropped_and_issues_flagged(self):
        raw = make_ledger().copy()
        raw.loc[len(raw)] = [None] * len(raw.columns)
        self.assertEqual(len(clean_ledger(raw)), 4)
        bad = make_ledger().copy()
        bad.loc[0, 'strike'] = 0.0
        self.assertTrue(any('strike' in i for i in ledger_issues(bad)))
        self.assertEqual(ledger_issues(make_ledger()), [])


class HedgeTests(unittest.TestCase):
    def test_corn_bushels(self):
        self.assertAlmostEqual(size_hedge('Corn', 1_000_000, 'Bushels')['contracts_exact'], 200.0)

    def test_corn_metric_tons(self):
        # 10,000 MT / 25.4012 kg per bu = 393,680 bu -> 78.74 contracts of 5,000 bu
        self.assertAlmostEqual(size_hedge('Corn', 10_000, 'Metric tons')['contracts_exact'], 78.736, places=2)

    def test_wheat_and_soy_use_60lb_bushel(self):
        self.assertAlmostEqual(to_native('Soybeans', 1000, 'Metric tons'), 36743.7, places=0)
        self.assertAlmostEqual(to_native('Chicago SRW Wheat', 1000, 'Metric tons'), 36743.7, places=0)

    def test_palm_oil(self):
        self.assertAlmostEqual(size_hedge('Palm Oil', 1000, 'Metric tons')['contracts_exact'], 40.0)

    def test_soybean_oil_and_meal(self):
        self.assertAlmostEqual(size_hedge('Soybean Oil', 1000, 'Metric tons')['contracts_exact'], 36.74, places=2)
        self.assertAlmostEqual(size_hedge('Soybean Meal', 1000, 'Metric tons')['contracts_exact'], 11.023, places=2)

    def test_bushels_invalid_for_palm(self):
        with self.assertRaises(ValueError):
            size_hedge('Palm Oil', 1000, 'Bushels')

    def test_ratio_beta_rounding(self):
        r = size_hedge('Corn', 1_000_000, 'Bushels', hedge_ratio=0.75, beta=1.0, rounding='Round down (under-hedge)')
        self.assertEqual(r['contracts'], 150)
        r = size_hedge('Corn', 1_000_020, 'Bushels', hedge_ratio=1.0, rounding='Round up (over-hedge)')
        self.assertEqual(r['contracts'], 201)
        r = size_hedge('Corn', 1_000_000, 'Bushels', beta=0.8)
        self.assertAlmostEqual(r['contracts_exact'], 160.0)
        self.assertAlmostEqual(r['hedged_pct'], 1.0)

    def test_notional_and_option_sizing(self):
        r = size_hedge('Corn', 1_000_000, 'Bushels', futures_price=450.0, option_delta=-0.5)
        self.assertAlmostEqual(r['per_contract_notional'], 22_500.0)
        self.assertAlmostEqual(r['hedge_notional'], 4_500_000.0)
        self.assertEqual(r['option_contracts'], 400)


if __name__ == '__main__':
    unittest.main()
