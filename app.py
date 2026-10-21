"""Grains & Oilseeds Risk Workbench - Streamlit hedging-desk prototype."""
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from risk_engine import (
    COLUMNS, INSTRUMENTS, NUMERIC, OPTION_TYPES, PRODUCTS, ROUNDING_MODES,
    TEXT_COLS, black76_greeks, black76_price_vec, clean_ledger, ensure_columns,
    exposure_units, implied_volatility, ledger_issues, ledger_risk, multiplier_for,
    product_summary, size_hedge, spec_for, stress_grid, unit_equivalents)

st.set_page_config(page_title='Grains & Oilseeds Risk Workbench', layout='wide')
st.title('Grains & Oilseeds Risk Management Tool')
st.caption('Pre-trade option analytics • hedge sizing • trade ledger • portfolio Greeks • scenario risk')


# Illustrative placeholder prices in each contract's quote units: overwrite with live quotes.
DEFAULT_PRICES = {'Corn': 450.0, 'Chicago SRW Wheat': 550.0, 'KC HRW Wheat': 560.0,
                  'Soybeans': 1050.0, 'Soybean Meal': 320.0, 'Soybean Oil': 45.0,
                  'Palm Oil': 4000.0, 'Other': 100.0}
PURPOSES = ['Hedge', 'Hedge / optionality', 'Risk reduction', 'Trading / opportunistic']


def sample_ledger():
    today = str(date.today())
    base = dict(trade_date=today, risk_free_rate=0.04, implied_volatility=0.25, notes='Example only')
    return pd.DataFrame([
        {**base, 'trade_id': 'DEMO-001', 'product': 'Soybeans', 'contract_month': 'Nov-2026',
         'instrument_type': 'Call', 'signed_quantity': 5.0, 'trade_price': 45.0,
         'market_futures_price': 1050.0, 'strike': 1050.0, 'days_to_expiry': 90.0,
         'market_option_premium': 45.0, 'contract_multiplier': 50.0, 'purpose': 'Hedge / optionality'},
        {**base, 'trade_id': 'DEMO-002', 'product': 'Corn', 'contract_month': 'Dec-2026',
         'instrument_type': 'Future', 'signed_quantity': -3.0, 'trade_price': 450.0,
         'market_futures_price': 455.0, 'strike': 0.0, 'days_to_expiry': 0.0,
         'market_option_premium': 0.0, 'contract_multiplier': 50.0, 'purpose': 'Hedge'},
    ], columns=COLUMNS)


def typed(df):
    out = df.copy()
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    for c in TEXT_COLS:
        out[c] = out[c].fillna('').astype(str)
    return out[COLUMNS]


def next_trade_id(ledger):
    stamp = date.today().strftime('%Y%m%d')
    existing = set(ledger['trade_id'].astype(str))
    n = len(ledger) + 1
    while f'TR-{stamp}-{n:03d}' in existing:
        n += 1
    return f'TR-{stamp}-{n:03d}'


def add_trade(row):
    ledger = st.session_state.ledger
    full = {c: row.get(c, '' if c in TEXT_COLS else 0.0) for c in COLUMNS}
    full.setdefault('trade_date', str(date.today()))
    if not full['trade_id']:
        full['trade_id'] = next_trade_id(ledger)
    if not full['trade_date']:
        full['trade_date'] = str(date.today())
    st.session_state.ledger = typed(pd.concat([ledger, pd.DataFrame([full])], ignore_index=True))
    return full['trade_id']


if 'ledger' not in st.session_state:
    st.session_state.ledger = sample_ledger()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header('Pre-trade option ticket')
    prod = st.selectbox('Commodity', PRODUCTS, index=0)
    px0 = DEFAULT_PRICES[prod]
    f = st.number_input('Underlying futures price', min_value=0.0001, value=px0, step=1.0, key=f'sb_f_{prod}')
    k = st.number_input('Strike', min_value=0.0001, value=px0, step=1.0, key=f'sb_k_{prod}')
    days = st.number_input('Days to expiry', min_value=1, value=90, step=1)
    opt_type = st.selectbox('Option type', ['Call', 'Put'])
    vol_pct = st.number_input('Volatility assumption (%)', min_value=0.01, value=25.0, step=0.5)
    rate_pct = st.number_input('Discount rate (%)', value=4.0, step=0.25)
    market_prem = st.number_input('Observed market premium (quote units)', min_value=0.0,
                                  value=round(px0 * 0.04, 2), step=0.25, key=f'sb_prem_{prod}')
    mult = st.number_input('Contract multiplier', min_value=0.0001, value=float(multiplier_for(prod)),
                           step=1.0, key=f'sb_mult_{prod}',
                           help='Currency P&L per 1 quote unit per contract (e.g. 50 for corn quoted in cents/bu).')
    qty = st.number_input('Proposed signed quantity (+ long / − short)', value=1.0, step=1.0)
    st.divider()
    usd_per_myr = st.number_input('USD per 1 MYR (palm oil → USD)', min_value=0.0001, value=0.23,
                                  step=0.005, format='%.4f',
                                  help='Placeholder: update with the current rate. Used only to aggregate USD-equivalent risk.')
    vol, rate, T = vol_pct / 100, rate_pct / 100, days / 365
FX = {'USD': 1.0, 'MYR': usd_per_myr}

kpi_box = st.container()

# ------------------------------------------------- manual trade entry panel
with st.expander('➕ Add a trade to the portfolio', expanded=True):
    c = st.columns([1.1, 1.5, 1.1, 1.2, 1.0, 1.2])
    t_date = c[0].date_input('Trade date', value=date.today(), key='add_date')
    t_prod = c[1].selectbox('Commodity', PRODUCTS, key='add_prod')
    t_inst = c[2].selectbox('Instrument', list(INSTRUMENTS), key='add_inst')
    t_side = c[3].selectbox('Side', ['Buy (long)', 'Sell (short)'], key='add_side')
    t_lots = c[4].number_input('Contracts', min_value=1, value=1, step=1, key='add_lots')
    t_month = c[5].text_input('Contract month', placeholder='Dec-2026', key='add_month')
    t_spec = spec_for(t_prod)
    t_opt = t_inst in OPTION_TYPES
    t_px = DEFAULT_PRICES[t_prod]
    d = st.columns(4)
    t_entry = d[0].number_input('Entry premium (quote units)' if t_opt else 'Entry price (quote units)',
                                min_value=0.0001, value=round(t_px * 0.04, 2) if t_opt else t_px, step=0.25,
                                key=f'add_entry_{t_prod}_{t_inst}', help=f'Quote unit: {t_spec["quote"]}')
    t_mkt = d[1].number_input('Current futures price', min_value=0.0001, value=t_px, step=1.0,
                              key=f'add_mkt_{t_prod}')
    t_mult = d[2].number_input('Contract multiplier', min_value=0.0001, value=float(multiplier_for(t_prod)),
                               step=1.0, key=f'add_mult_{t_prod}',
                               help='Currency P&L per 1 quote unit per contract. For Physical rows, express '
                                    'the position in futures-equivalent contracts.')
    t_purpose = d[3].selectbox('Purpose', PURPOSES, key='add_purpose')
    t_strike, t_days, t_vol, t_rate, t_mop = 0.0, 0.0, 25.0, 4.0, 0.0
    if t_opt:
        o = st.columns(5)
        t_strike = o[0].number_input('Strike', min_value=0.0001, value=t_px, step=1.0, key=f'add_k_{t_prod}')
        t_days = float(o[1].number_input('Days to expiry', min_value=1, value=90, step=1, key='add_days'))
        t_vol = o[2].number_input('Implied vol (%)', min_value=0.01, value=25.0, step=0.5, key='add_vol')
        t_rate = o[3].number_input('Discount rate (%)', value=4.0, step=0.25, key='add_rate')
        t_mop = o[4].number_input('Current market premium', min_value=0.0, value=round(t_px * 0.04, 2),
                                  step=0.25, key=f'add_mop_{t_prod}_{t_inst}')
        try:
            mdl = black76_greeks(t_mkt, t_strike, t_days / 365, t_rate / 100, t_vol / 100, t_inst)
            msg = f'Model premium {mdl["Theoretical premium"]:,.4f} • delta {mdl["Delta (per futures unit)"]:.3f}'
            if t_mop > 0:
                msg += f' • implied vol at current premium {implied_volatility(t_mop, t_mkt, t_strike, t_days / 365, t_rate / 100, t_inst):.2%}'
            st.caption(msg)
        except Exception as e:  # noqa: BLE001
            st.caption(f'Model check unavailable: {e}')
    t_notes = st.text_input('Notes', key='add_notes')
    if st.button('Add trade to ledger', type='primary', key='add_btn'):
        sign = 1.0 if t_side.startswith('Buy') else -1.0
        tid = add_trade({
            'trade_date': str(t_date), 'product': t_prod, 'contract_month': t_month,
            'instrument_type': t_inst, 'signed_quantity': sign * float(t_lots), 'trade_price': t_entry,
            'market_futures_price': t_mkt, 'strike': t_strike if t_opt else 0.0,
            'days_to_expiry': t_days if t_opt else 0.0, 'risk_free_rate': t_rate / 100,
            'implied_volatility': t_vol / 100, 'market_option_premium': (t_mop if t_mop > 0 else t_entry) if t_opt else 0.0,
            'contract_multiplier': t_mult, 'purpose': t_purpose, 'notes': t_notes})
        st.success(f'Added {tid}: {t_side.split()[0]} {int(t_lots)} × {t_prod} {t_inst}.')

pre, hedge_tab, ledger_tab, risk_tab = st.tabs(
    ['1 · Pre-trade analytics', '2 · Hedge sizing', '3 · Trade ledger & CSV', '4 · Portfolio risk & stress'])

# ----------------------------------------------------------- 1 pre-trade
with pre:
    st.subheader(f'{prod}: {opt_type} option pre-trade check')
    try:
        g = black76_greeks(f, k, T, rate, vol, opt_type)
        model = g['Theoretical premium']
        diff = market_prem - model
        c = st.columns(4)
        c[0].metric('Model premium', f'{model:,.4f}')
        c[1].metric('Market − model', f'{diff:,.4f}', delta=f'{(diff / model * 100 if model else 0):.2f}% vs model')
        c[2].metric('Delta-adjusted exposure', f'{qty * mult * g["Delta (per futures unit)"]:,.2f}')
        c[3].metric('Position vega / 1 vol pt', f'{qty * mult * g["Vega (per 1 vol pt)"]:,.4f}')
        st.dataframe(pd.DataFrame({'Metric': list(g.keys()), 'Value': list(g.values())}), hide_index=True)
        if market_prem > 0:
            try:
                iv = implied_volatility(market_prem, f, k, T, rate, opt_type)
                st.metric('Implied volatility from observed premium', f'{iv:.2%}',
                          delta=f'{(iv - vol) * 100:+.2f} vol points vs assumption')
                if iv > vol + .03:
                    st.warning('Market premium implies volatility materially above your assumption. Check event risk, skew, expiry and quote quality before trading.')
                elif iv < vol - .03:
                    st.info('Market premium implies volatility materially below your assumption. Confirm the bid/ask, liquidity and model inputs.')
                else:
                    st.success('Observed premium is broadly consistent with the entered volatility assumption under Black-76. This is not, by itself, a trade recommendation.')
            except Exception as e:  # noqa: BLE001
                st.warning(str(e))
        grid = np.linspace(f * 0.7, f * 1.3, 61)
        st.line_chart(pd.DataFrame({'Model premium vs futures price': black76_price_vec(grid, k, T, rate, vol, opt_type == 'Call')},
                                   index=pd.Index(grid.round(2), name='Futures price')))
        st.caption('Premium comparison is meaningful only when market premium, futures, strike, expiry, quote units and multiplier refer to the same contract. Use a volatility surface/skew-aware input for production decisions.')
        st.subheader('Proposed option position sensitivity')
        st.write(f'Approximate premium value = quantity × multiplier × model premium = **{qty * mult * model:,.2f}** quote-currency units.')
        if st.button('Add this option to the portfolio ledger', key='pre_add'):
            if qty == 0:
                st.warning('Proposed quantity is zero: nothing to add.')
            else:
                entry = market_prem if market_prem > 0 else model
                tid = add_trade({'product': prod, 'instrument_type': opt_type, 'signed_quantity': qty,
                                 'trade_price': entry, 'market_futures_price': f, 'strike': k,
                                 'days_to_expiry': float(days), 'risk_free_rate': rate, 'implied_volatility': vol,
                                 'market_option_premium': entry, 'contract_multiplier': mult,
                                 'purpose': 'Hedge / optionality', 'notes': 'Added from pre-trade ticket'})
                st.success(f'Added {tid} at premium {entry:,.4f}.')
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

# ------------------------------------------------------- 2 hedge sizing
with hedge_tab:
    st.subheader('Hedge sizing: physical exposure → number of contracts')
    st.caption('Convert an exposure in bushels, metric tons, short tons or pounds into exchange contracts, '
               'with an optional protective-option hedge sized on delta.')
    h1, h2, h3 = st.columns(3)
    h_prod = h1.selectbox('Commodity', PRODUCTS, key='h_prod')
    hs = spec_for(h_prod)
    h_long = h1.radio('Physical exposure', ['Long (need to buy: processor / buyer)',
                                            'Short (own or will sell: grower / inventory)'],
                      key='h_dir').startswith('Long')
    units = exposure_units(h_prod)
    default_unit = 'Bushels' if hs.get('lb_per_bu') else 'Metric tons'
    h_qty = h2.number_input('Exposure quantity', min_value=0.0, step=1000.0, key=f'h_qty_{h_prod}',
                            value=1_000_000.0 if hs.get('lb_per_bu') else 10_000.0)
    h_unit = h2.selectbox('Measure', units, index=units.index(default_unit), key=f'h_unit_{h_prod}')
    h_ratio = h3.slider('Hedge ratio (%)', 0, 150, 100, 5, key='h_ratio')
    h_beta = h3.number_input('Basis / cross-hedge beta', min_value=0.01, value=1.0, step=0.05, key='h_beta',
                             help='Futures volume per unit of exposure. 1.00 = same commodity; use a '
                                  'regression/min-variance beta when cross-hedging (e.g. palm vs soybean oil).')
    r4 = st.columns(4)
    h_price = r4[0].number_input(f'Futures price ({hs["quote"]})', min_value=0.0001,
                                 value=DEFAULT_PRICES[h_prod], step=1.0, key=f'h_px_{h_prod}')
    h_size = r4[1].number_input(f'Contract size ({hs["native"]})', min_value=0.0001, value=float(hs['size']),
                                step=1.0, key=f'h_size_{h_prod}',
                                help='Override for mini / alternative contracts.')
    h_round = r4[2].selectbox('Rounding', list(ROUNDING_MODES), key='h_round')
    h_inst = r4[3].selectbox('Hedge instrument', ['Futures', 'Protective option (long)'], key='h_inst')
    h_opt = h_inst.startswith('Protective')
    h_otype = 'Call' if h_long else 'Put'
    h_delta, h_prem, h_k, h_days = None, None, h_price, 90
    if h_opt:
        oc = st.columns(4)
        h_k = oc[0].number_input('Strike', min_value=0.0001, value=DEFAULT_PRICES[h_prod], step=1.0, key=f'h_k_{h_prod}')
        h_days = oc[1].number_input('Days to expiry', min_value=1, value=90, step=1, key='h_days')
        h_vol = oc[2].number_input('Volatility (%)', min_value=0.01, value=25.0, step=0.5, key='h_vol')
        h_rate = oc[3].number_input('Discount rate (%)', value=4.0, step=0.25, key='h_rate')
        try:
            hg = black76_greeks(h_price, h_k, h_days / 365, h_rate / 100, h_vol / 100, h_otype)
            h_delta, h_prem = hg['Delta (per futures unit)'], hg['Theoretical premium']
        except ValueError as e:
            st.error(str(e))
    st.caption(f'{hs["exchange"]} • contract size {h_size:,.0f} {hs["native"]} • quoted in {hs["quote"]} • '
               f'currency {hs["currency"]}. Verify against the current exchange rulebook.')
    try:
        res = size_hedge(h_prod, h_qty, h_unit, h_ratio / 100, h_beta, h_round, h_size, h_price, h_delta)
        h_mult = h_size / hs['quote_div']
        nat = res['native_unit']
        if h_opt and h_delta is not None:
            n_opt = res['option_contracts']
            headline = (f'**Buy {n_opt:,} {h_otype} options** (strike {h_k:,.2f}, delta {h_delta:+.3f}) '
                        f'to cover {res["contracts_exact"]:,.2f} futures-equivalent contracts.')
        else:
            headline = (f'**{"BUY" if h_long else "SELL"} {res["contracts"]:,} futures contracts** '
                        f'({res["contracts_exact"]:,.2f} exact before rounding).')
        st.success(headline)
        m = st.columns(4)
        m[0].metric(f'Exposure ({nat})', f'{res["exposure_native"]:,.0f}')
        m[1].metric(f'Hedge volume required ({nat})', f'{res["target_native"]:,.0f}')
        m[2].metric('Futures contracts', f'{res["contracts"]:,}', delta=f'{res["contracts_exact"]:,.2f} exact', delta_color='off')
        m[3].metric(f'Unhedged residual ({nat})', f'{res["residual_native"]:,.0f}', delta=f'{res["hedged_pct"]:.1%} hedged', delta_color='off')
        v = st.columns(4)
        v[0].metric(f'Notional per contract ({hs["currency"]})', f'{res["per_contract_notional"]:,.0f}')
        v[1].metric(f'Hedge notional ({hs["currency"]})', f'{res["hedge_notional"]:,.0f}')
        v[2].metric(f'Hedge P&L per 1% move ({hs["currency"]})', f'{res["pnl_per_1pct"]:,.0f}')
        if h_opt and h_prem is not None:
            v[3].metric(f'Premium outlay ({hs["currency"]})', f'{res["option_contracts"] * h_mult * h_prem:,.0f}',
                        delta=f'{h_prem:,.4f} per unit', delta_color='off')
        else:
            v[3].metric(f'Exposure value ({hs["currency"]})', f'{res["exposure_value"]:,.0f}')
        e1, e2 = st.columns(2)
        with e1:
            st.markdown('**Exposure in other measures**')
            eq = unit_equivalents(h_prod, res['exposure_native'])
            st.dataframe(pd.DataFrame({'Measure': list(eq), 'Quantity': list(eq.values())}), hide_index=True)
        with e2:
            st.markdown('**Hedge-ratio ladder**')
            rows = []
            for pct in sorted({25, 50, 75, 100, 125, h_ratio}):
                r_ = size_hedge(h_prod, h_qty, h_unit, pct / 100, h_beta, h_round, h_size, h_price, h_delta)
                rows.append({'Hedge ratio %': pct, 'Futures contracts': r_['contracts'],
                             'Exact': round(r_['contracts_exact'], 2),
                             f'Residual ({nat})': round(r_['residual_native'])})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown('**Book this hedge**')
        b = st.columns([1.2, 1.2, 1])
        b_month = b[0].text_input('Contract month', placeholder='Dec-2026', key='h_month')
        b_entry = b[1].number_input('Entry premium' if h_opt else 'Entry futures price', min_value=0.0001,
                                    value=float(h_prem) if (h_opt and h_prem) else float(h_price), step=0.25,
                                    key=f'h_entry_{h_prod}_{h_inst}')
        if b[2].button('Add hedge to ledger', type='primary', key='h_add'):
            n_book = res['option_contracts'] if h_opt and h_delta is not None else res['contracts']
            if n_book == 0:
                st.warning('Contract count is zero: nothing to add.')
            else:
                row = {'product': h_prod, 'contract_month': b_month, 'trade_price': b_entry,
                       'market_futures_price': h_price, 'contract_multiplier': h_mult,
                       'purpose': 'Hedge', 'notes': f'Hedge sizing: {h_qty:,.0f} {h_unit} at {h_ratio}%'}
                if h_opt and h_delta is not None:
                    row.update(instrument_type=h_otype, signed_quantity=float(n_book), strike=h_k,
                               days_to_expiry=float(h_days), risk_free_rate=h_rate / 100,
                               implied_volatility=h_vol / 100, market_option_premium=b_entry)
                else:
                    row.update(instrument_type='Future', signed_quantity=float(n_book) * (1 if h_long else -1))
                st.success(f'Added {add_trade(row)} to the ledger.')
    except ValueError as e:
        st.error(str(e))

# ------------------------------------------------------------ 3 ledger
with ledger_tab:
    st.subheader('Trade-by-trade portfolio ledger')
    st.write('One row per trade. Signed quantity: positive = long, negative = short. For options, trade_price is the '
             'premium paid/received per quote unit; market_option_premium is the current observed premium. For '
             'futures/physical rows, trade_price and market_futures_price are entry and current price.')
    up = st.file_uploader('Import existing portfolio CSV (optional)', type=['csv'], key='ledger_upload')
    if up is not None:
        imp_mode = st.radio('Import mode', ['Replace ledger', 'Append to ledger'], horizontal=True)
        if st.button('Import file'):
            try:
                imp = typed(ensure_columns(pd.read_csv(up)))
                imp['instrument_type'] = imp['instrument_type'].str.strip().str.title()
                base = st.session_state.ledger if imp_mode.startswith('Append') else sample_ledger().iloc[0:0]
                st.session_state.ledger = typed(pd.concat([base, imp], ignore_index=True))
                st.success(f'Imported {len(imp)} rows.')
            except Exception as e:  # noqa: BLE001
                st.error(f'CSV import error: {e}')
    extra = sorted(set(st.session_state.ledger['product'].astype(str)) - set(PRODUCTS) - {''})
    cfg = {'product': st.column_config.SelectboxColumn('product', options=PRODUCTS + extra),
           'instrument_type': st.column_config.SelectboxColumn('instrument_type', options=list(INSTRUMENTS))}
    edited = st.data_editor(st.session_state.ledger, num_rows='dynamic', column_config=cfg)
    st.session_state.ledger = edited
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.download_button('Export full portfolio CSV', edited.to_csv(index=False).encode('utf-8'),
                           'commodities_trade_ledger.csv', 'text/csv', type='primary')
    with b2:
        st.download_button('Download blank CSV template', pd.DataFrame(columns=COLUMNS).to_csv(index=False).encode('utf-8'),
                           'commodities_ledger_template.csv', 'text/csv')
    with b3:
        if st.button('Validate ledger'):
            try:
                found = ledger_issues(clean_ledger(edited))
                if found:
                    st.error('\n\n'.join(found))
                else:
                    st.success(f'Basic validation passed for {len(clean_ledger(edited))} rows. This checks fields, not economic suitability or exchange contract rules.')
            except ValueError as e:
                st.error(str(e))
    with b4:
        if st.button('Clear all rows'):
            st.session_state.ledger = sample_ledger().iloc[0:0]
            st.rerun()

# --------------------------------------------- shared ledger computations
try:
    clean = clean_ledger(st.session_state.ledger)
    issues = ledger_issues(clean)
    risk = ledger_risk(clean, FX) if not issues else pd.DataFrame()
except ValueError as e:
    clean, issues, risk = pd.DataFrame(), [str(e)], pd.DataFrame()

with kpi_box:
    if issues:
        st.warning(f'{len(issues)} ledger issue(s) block portfolio metrics: see the Ledger tab.')
    elif clean.empty:
        st.info('Portfolio is empty: add a trade below.')
    else:
        k5 = st.columns(5)
        k5[0].metric('Portfolio P&L (USD eq.)', f'{risk.current_pnl_usd.sum():,.0f}')
        k5[1].metric('Net delta notional (USD eq.)', f'{risk.delta_notional_usd.sum():,.0f}')
        k5[2].metric('Gross underlying notional (USD eq.)', f'{risk.gross_notional_usd.sum():,.0f}')
        k5[3].metric('Net vega / vol pt (USD eq.)', f'{risk.vega_usd.sum():,.0f}')
        k5[4].metric('Net theta / day (USD eq.)', f'{risk.theta_usd.sum():,.0f}')


@st.cache_data(show_spinner=False)
def run_stress(df, price_shocks, vol_shocks, usd_per_myr_):
    return stress_grid(df, price_shocks, vol_shocks, {'MYR': usd_per_myr_})


# ------------------------------------------------------- 4 risk & stress
with risk_tab:
    st.subheader('Portfolio Greeks, P&L and scenario stress')
    st.caption('Position-level results depend on signed quantity, contract multiplier and consistent exchange quote '
               'units. Options are revalued with Black-76; futures and physical rows use linear price P&L. '
               'Aggregates are converted to USD so products with different quote units and currencies can be added.')
    if issues:
        st.error('Fix these ledger issues first:\n\n' + '\n'.join(f'- {i}' for i in issues[:25]))
    elif clean.empty:
        st.info('Nothing to analyse yet.')
    else:
        st.markdown('**Exposure by product** (delta in futures-equivalent contracts)')
        st.dataframe(product_summary(risk), hide_index=True)
        st.markdown('**Trade-level risk** (native quote units; *_usd columns converted)')
        st.dataframe(risk, hide_index=True)
        st.download_button('Export portfolio risk CSV', risk.to_csv(index=False).encode('utf-8'),
                           'commodities_portfolio_risk.csv', 'text/csv')
        st.subheader('Full-revaluation stress scenarios')
        st.write('Independent parallel shocks are applied to each line’s underlying futures price and option '
                 'volatility. This is a first-pass stress grid; it does not model cross-commodity correlations, '
                 'calendar spreads, basis or liquidity.')
        px = st.multiselect('Futures price shock (%)', [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25],
                            default=[-15, -10, -5, 0, 5, 10, 15])
        vx = st.multiselect('Volatility shock (absolute vol points)', [-15, -10, -5, 0, 5, 10, 15, 25],
                            default=[-10, 0, 10, 15])
        if not px or not vx:
            st.info('Select at least one price shock and one volatility shock.')
        else:
            sd = run_stress(clean, tuple(px), tuple(vx), usd_per_myr)
            worst = sd.loc[sd.change_vs_current_pnl.idxmin()]
            st.metric('Worst scenario in selected grid (USD eq.)', f'{worst.change_vs_current_pnl:,.2f}',
                      delta=f"{worst['futures_price_shock_%']:+.0f}% futures / {worst['volatility_shock_pts']:+.0f} vol pts")
            st.markdown('**Change vs current P&L: price shock (rows) × vol shock (columns)**')
            st.dataframe(sd.pivot(index='futures_price_shock_%', columns='volatility_shock_pts',
                                  values='change_vs_current_pnl'))
            st.dataframe(sd.sort_values('change_vs_current_pnl'), hide_index=True)
            st.download_button('Export stress scenarios CSV', sd.to_csv(index=False).encode('utf-8'),
                               'commodities_stress_scenarios.csv', 'text/csv')

st.divider()
st.caption('Suggested next enhancements: market-data feed, tick values and per-contract-month specs, historical VaR/Expected '
           'Shortfall, crop-calendar and event scenarios, curve/calendar-spread risk, basis exposure, margin/liquidity '
           'estimates, and corporate physical exposure reconciliation.')
