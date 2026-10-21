"""Black-76 pricing, vectorised portfolio risk and hedge-sizing helpers.

Black-76 is the Black-Scholes-Merton model for European options on futures
(cost of carry b = 0, underlying = futures price F). Every pricing helper is
numpy-vectorised, so a whole ledger, or a whole scenario grid, is revalued in a
single call instead of looping over scalar scipy calls.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

# --------------------------------------------------------------------------
# Ledger schema
# --------------------------------------------------------------------------
INSTRUMENTS = ('Future', 'Physical', 'Call', 'Put')
OPTION_TYPES = ('Call', 'Put')
COLUMNS = ['trade_id', 'trade_date', 'product', 'contract_month', 'instrument_type',
           'signed_quantity', 'trade_price', 'market_futures_price', 'strike',
           'days_to_expiry', 'risk_free_rate', 'implied_volatility',
           'market_option_premium', 'contract_multiplier', 'purpose', 'notes']
REQUIRED = ['product', 'instrument_type', 'signed_quantity', 'trade_price',
            'market_futures_price', 'strike', 'days_to_expiry', 'risk_free_rate',
            'implied_volatility', 'market_option_premium', 'contract_multiplier']
NUMERIC = REQUIRED[2:]
TEXT_COLS = ['trade_id', 'trade_date', 'product', 'contract_month',
             'instrument_type', 'purpose', 'notes']
OPTION_ONLY = ['strike', 'days_to_expiry', 'risk_free_rate', 'implied_volatility',
               'market_option_premium']

# --------------------------------------------------------------------------
# Contract specifications (verify against the exchange before production use)
# multiplier = contract size / quote_div  -> currency P&L per 1 quote unit
# --------------------------------------------------------------------------
LB_KG = 0.45359237
SHORT_TON_KG = 907.18474
SPECS = {
    'Corn': {'exchange': 'CBOT (ZC)', 'native': 'bushels', 'size': 5000.0,
             'quote': 'US cents/bu', 'quote_div': 100.0, 'currency': 'USD', 'lb_per_bu': 56.0},
    'Chicago SRW Wheat': {'exchange': 'CBOT (ZW)', 'native': 'bushels', 'size': 5000.0,
                          'quote': 'US cents/bu', 'quote_div': 100.0, 'currency': 'USD', 'lb_per_bu': 60.0},
    'KC HRW Wheat': {'exchange': 'CME / KC HRW (KE)', 'native': 'bushels', 'size': 5000.0,
                     'quote': 'US cents/bu', 'quote_div': 100.0, 'currency': 'USD', 'lb_per_bu': 60.0},
    'Soybeans': {'exchange': 'CBOT (ZS)', 'native': 'bushels', 'size': 5000.0,
                 'quote': 'US cents/bu', 'quote_div': 100.0, 'currency': 'USD', 'lb_per_bu': 60.0},
    'Soybean Meal': {'exchange': 'CBOT (ZM)', 'native': 'short tons', 'size': 100.0,
                     'quote': 'USD/short ton', 'quote_div': 1.0, 'currency': 'USD'},
    'Soybean Oil': {'exchange': 'CBOT (ZL)', 'native': 'pounds', 'size': 60000.0,
                    'quote': 'US cents/lb', 'quote_div': 100.0, 'currency': 'USD'},
    'Palm Oil': {'exchange': 'Bursa Malaysia (FCPO)', 'native': 'metric tons', 'size': 25.0,
                 'quote': 'MYR/MT', 'quote_div': 1.0, 'currency': 'MYR'},
    'Other': {'exchange': 'User defined', 'native': 'metric tons', 'size': 1.0,
              'quote': 'quote unit', 'quote_div': 1.0, 'currency': 'USD'},
}
PRODUCTS = list(SPECS)
DEFAULT_FX = {'USD': 1.0, 'MYR': 0.23}  # USD per 1 unit of currency (placeholder: update)

UNIT_KG = {'Pounds': LB_KG, 'Kilograms': 1.0, 'Short tons': SHORT_TON_KG,
           'Metric tons': 1000.0, 'Thousand metric tons': 1_000_000.0}
BUSHEL_UNITS = {'Bushels': 1.0, 'Thousand bushels': 1e3, 'Million bushels': 1e6}
ROUNDING_MODES = ('Nearest', 'Round down (under-hedge)', 'Round up (over-hedge)')


def spec_for(product):
    return SPECS.get(product, SPECS['Other'])


def multiplier_for(product):
    s = spec_for(product)
    return s['size'] / s['quote_div']


# --------------------------------------------------------------------------
# Black-76 core (vectorised)
# --------------------------------------------------------------------------
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _is_call(option_type):
    return str(option_type).strip().lower().startswith('c')


def _prep(F, K, T, r, sigma, is_call):
    F, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, r, sigma))
    if not all(np.all(np.isfinite(a)) for a in (F, K, T, r, sigma)):
        raise ValueError("Futures price, strike, expiry and volatility must be finite.")
    if np.any(F <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("Futures price, strike, time to expiry and volatility must be greater than zero.")
    return F, K, T, r, sigma, np.asarray(is_call, dtype=bool)


def _d1_d2(F, K, T, sigma):
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return d1, d1 - sigma * sqT, sqT


def black76_price_vec(F, K, T, r, sigma, is_call):
    F, K, T, r, sigma, is_call = _prep(F, K, T, r, sigma, is_call)
    d1, d2, _ = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    call = disc * (F * ndtr(d1) - K * ndtr(d2))
    put = disc * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def black76_greeks_vec(F, K, T, r, sigma, is_call):
    """Raw Greeks: delta per futures unit, gamma, vega per 1.00 vol,
    theta per year, rho per 1.00 rate."""
    F, K, T, r, sigma, is_call = _prep(F, K, T, r, sigma, is_call)
    d1, d2, sqT = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    call = disc * (F * ndtr(d1) - K * ndtr(d2))
    put = disc * (K * ndtr(-d2) - F * ndtr(-d1))
    price = np.where(is_call, call, put)
    delta = np.where(is_call, disc * ndtr(d1), -disc * ndtr(-d1))
    return {'price': price, 'delta': delta,
            'gamma': disc * pdf / (F * sigma * sqT),
            'vega': disc * F * pdf * sqT,
            'theta': r * price - disc * F * pdf * sigma / (2.0 * sqT),
            'rho': -T * price}


# --------------------------------------------------------------------------
# Scalar wrappers (same API as before)
# --------------------------------------------------------------------------
def black76_price(F, K, T, r, sigma, option_type='Call'):
    return float(black76_price_vec(F, K, T, r, sigma, _is_call(option_type)))


def black76_greeks(F, K, T, r, sigma, option_type='Call'):
    g = {k: float(v) for k, v in black76_greeks_vec(F, K, T, r, sigma, _is_call(option_type)).items()}
    return {'Theoretical premium': g['price'],
            'Delta (per futures unit)': g['delta'], 'Gamma': g['gamma'],
            'Vega (per 1 vol pt)': g['vega'] / 100, 'Vega (per 1.00 vol)': g['vega'],
            'Theta (daily)': g['theta'] / 365, 'Rho (per 1 rate pt)': g['rho'] / 100}


def implied_volatility(premium, F, K, T, r, option_type='Call'):
    if premium <= 0:
        raise ValueError('Market premium must be positive.')
    call = _is_call(option_type)
    f = lambda v: float(black76_price_vec(F, K, T, r, v, call)) - premium
    lo, hi = 1e-5, 5.0
    if f(lo) * f(hi) > 0:
        raise ValueError('Cannot solve implied volatility: premium may be outside model bounds or inputs may be inconsistent.')
    return brentq(f, lo, hi, maxiter=500)


def signed_position_value(row, futures_price=None, vol=None, rate=None):
    """Single-row P&L (kept for API compatibility; the ledger uses ledger_risk)."""
    typ = str(row['instrument_type']).lower()
    qty = float(row['signed_quantity'])
    mult = float(row['contract_multiplier'])
    mark = float(row['market_futures_price'] if futures_price is None else futures_price)
    entry = float(row['trade_price'])
    if typ in ('future', 'physical'):
        return qty * mult * (mark - entry)
    if typ in ('call', 'put'):
        strike = float(row['strike'])
        T = max(float(row['days_to_expiry']) / 365, 1e-6)
        r = float(row['risk_free_rate'] if rate is None else rate)
        v = max(0.0001, float(row['implied_volatility'] if vol is None else vol))
        if futures_price is None and vol is None and rate is None:
            premium = float(row['market_option_premium'])
        else:
            premium = black76_price(mark, strike, T, r, v, typ.title())
        return qty * mult * (premium - entry)
    raise ValueError('instrument_type must be Future, Physical, Call or Put')


# --------------------------------------------------------------------------
# Ledger cleaning / validation
# --------------------------------------------------------------------------
def ensure_columns(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    out = df.copy()
    for c in TEXT_COLS:
        if c not in out.columns:
            out[c] = ''
    return out


def clean_ledger(df):
    """Drop blank rows, normalise instrument names, coerce numbers."""
    out = ensure_columns(df)
    blank = out[['product', 'instrument_type', 'signed_quantity', 'trade_price']].isna().all(axis=1)
    out = out.loc[~blank].copy()
    out['instrument_type'] = out['instrument_type'].astype(str).str.strip().str.title()
    out['product'] = out['product'].astype(str).str.strip()
    out['trade_id'] = out['trade_id'].fillna('').astype(str)
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    opt = out['instrument_type'].isin(OPTION_TYPES)
    out.loc[~opt, OPTION_ONLY] = out.loc[~opt, OPTION_ONLY].fillna(0.0)
    return out.reset_index(drop=True)


def ledger_issues(df):
    """Row-level problems in a cleaned ledger (empty list = OK)."""
    issues = []
    ids = df['trade_id'].astype(str)

    def add(mask, msg):
        for i in df.index[np.asarray(mask, dtype=bool)]:
            issues.append(f'Row {i + 1} ({ids[i] or "no id"}): {msg}')

    add(~df['instrument_type'].isin(INSTRUMENTS), 'invalid instrument_type (use Future, Physical, Call or Put)')
    for c in NUMERIC:
        add(~np.isfinite(df[c].to_numpy(float)), f'{c} is missing or not a number')
    add(df['contract_multiplier'] <= 0, 'contract_multiplier must be positive')
    add(df['market_futures_price'] <= 0, 'market_futures_price must be positive')
    opt = df['instrument_type'].isin(OPTION_TYPES)
    add(opt & (df['strike'] <= 0), 'option strike must be positive')
    add(opt & (df['days_to_expiry'] <= 0), 'option days_to_expiry must be positive')
    add(opt & (df['implied_volatility'] <= 0), 'option implied_volatility must be positive')
    add(opt & (df['market_option_premium'] < 0), 'market_option_premium cannot be negative')
    return issues


# --------------------------------------------------------------------------
# Portfolio risk (vectorised)
# --------------------------------------------------------------------------
def _fx_vector(df, fx):
    fx = {'USD': 1.0, **(fx or {})}
    ccy = df['product'].map(lambda p: spec_for(p)['currency'])
    return ccy, ccy.map(lambda c: float(fx.get(c, 1.0))).to_numpy(float)


def _arrays(df):
    t = df['instrument_type'].to_numpy()
    f = lambda c: df[c].to_numpy(dtype=float)
    return {'q': f('signed_quantity'), 'm': f('contract_multiplier'),
            'F': f('market_futures_price'), 'entry': f('trade_price'),
            'K': f('strike'), 'T': np.maximum(f('days_to_expiry') / 365.0, 1e-6),
            'r': f('risk_free_rate'), 'v': f('implied_volatility'),
            'mop': f('market_option_premium'),
            'is_opt': np.isin(t, OPTION_TYPES), 'is_call': t == 'Call'}


def ledger_risk(df, fx=None):
    a = _arrays(df)
    n = len(df)
    ccy, fxv = _fx_vector(df, fx)
    scale = a['q'] * a['m']
    mark = np.where(a['is_opt'], a['mop'], a['F'])
    pnl = scale * (mark - a['entry'])

    delta_contracts = a['q'].copy()
    gamma = np.zeros(n)
    vega = np.zeros(n)
    theta = np.zeros(n)
    model = np.full(n, np.nan)
    idx = a['is_opt']
    if idx.any():
        g = black76_greeks_vec(a['F'][idx], a['K'][idx], a['T'][idx], a['r'][idx],
                               a['v'][idx], a['is_call'][idx])
        model[idx] = g['price']
        delta_contracts[idx] = a['q'][idx] * g['delta']
        gamma[idx] = scale[idx] * g['gamma']
        vega[idx] = scale[idx] * g['vega'] / 100.0
        theta[idx] = scale[idx] * g['theta'] / 365.0
    net_delta = delta_contracts * a['m']
    gross = np.abs(scale * a['F'])
    mop = np.where(idx, a['mop'], np.nan)

    return pd.DataFrame({
        'trade_id': df['trade_id'].to_numpy(), 'product': df['product'].to_numpy(),
        'instrument_type': df['instrument_type'].to_numpy(), 'signed_quantity': a['q'],
        'ccy': ccy.to_numpy(), 'current_pnl': pnl, 'gross_futures_notional': gross,
        'model_premium': model, 'market_option_premium': mop,
        'premium_model_gap': mop - model, 'net_delta': net_delta, 'net_gamma': gamma,
        'net_vega_per_vol_pt': vega, 'net_theta_daily': theta,
        'delta_contracts': delta_contracts,
        'current_pnl_usd': pnl * fxv, 'gross_notional_usd': gross * fxv,
        'delta_notional_usd': net_delta * a['F'] * fxv,
        'vega_usd': vega * fxv, 'theta_usd': theta * fxv})


def product_summary(risk):
    if risk.empty:
        return pd.DataFrame()
    return (risk.groupby('product', sort=False)
            .agg(ccy=('ccy', 'first'), trades=('trade_id', 'count'),
                 delta_contracts=('delta_contracts', 'sum'),
                 pnl_usd=('current_pnl_usd', 'sum'),
                 delta_notional_usd=('delta_notional_usd', 'sum'),
                 gross_notional_usd=('gross_notional_usd', 'sum'),
                 vega_usd=('vega_usd', 'sum'), theta_usd=('theta_usd', 'sum'))
            .reset_index())


def stress_grid(df, price_shocks, vol_shocks, fx=None):
    """Full revaluation of every line for every (price shock %, vol shock pts)
    pair in one broadcast (scenarios x lines). Results in USD equivalent."""
    ps = np.asarray(sorted(price_shocks), dtype=float) / 100.0
    vs = np.asarray(sorted(vol_shocks), dtype=float) / 100.0
    if np.any(ps <= -1.0):
        raise ValueError('Price shocks must be greater than -100%.')
    a = _arrays(df)
    _, fxv = _fx_vector(df, fx)
    P = np.repeat(ps, len(vs))
    V = np.tile(vs, len(ps))
    scale = (a['q'] * a['m'])[None, :]
    newF = a['F'][None, :] * (1.0 + P[:, None])
    pnl = scale * (newF - a['entry'][None, :])
    base_mark = np.where(a['is_opt'], a['mop'], a['F'])
    base = float((a['q'] * a['m'] * (base_mark - a['entry']) * fxv).sum())

    idx = a['is_opt']
    if idx.any():
        row = lambda x: x[idx][None, :]
        newvol = np.maximum(1e-4, row(a['v']) + V[:, None])
        newprem = black76_price_vec(newF[:, idx], row(a['K']), row(a['T']), row(a['r']),
                                    newvol, row(a['is_call']))
        base_model = black76_price_vec(a['F'][idx], a['K'][idx], a['T'][idx], a['r'][idx],
                                       a['v'][idx], a['is_call'][idx])
        # Anchor to the observed premium; model changes drive the incremental stress.
        scen_mark = row(a['mop']) + newprem - base_model[None, :]
        pnl[:, idx] = scale[:, idx] * (scen_mark - row(a['entry']))
    total = (pnl * fxv[None, :]).sum(axis=1)
    return pd.DataFrame({'futures_price_shock_%': P * 100.0, 'volatility_shock_pts': V * 100.0,
                         'stressed_portfolio_pnl': total, 'change_vs_current_pnl': total - base})


# --------------------------------------------------------------------------
# Hedge sizing
# --------------------------------------------------------------------------
def exposure_units(product):
    units = list(BUSHEL_UNITS) if spec_for(product).get('lb_per_bu') else []
    return units + list(UNIT_KG)


def native_kg(product):
    s = spec_for(product)
    if s['native'] == 'bushels':
        return s['lb_per_bu'] * LB_KG
    return {'short tons': SHORT_TON_KG, 'pounds': LB_KG, 'metric tons': 1000.0}[s['native']]


def to_native(product, quantity, unit):
    """Convert an exposure to the contract's native unit (bu, short tons, lb or MT)."""
    if unit in BUSHEL_UNITS:
        lb = spec_for(product).get('lb_per_bu')
        if not lb:
            raise ValueError(f'Bushels are not a valid measure for {product}.')
        kg = quantity * BUSHEL_UNITS[unit] * lb * LB_KG
    elif unit in UNIT_KG:
        kg = quantity * UNIT_KG[unit]
    else:
        raise ValueError(f'Unknown measure: {unit}')
    return kg / native_kg(product)


def unit_equivalents(product, native_qty):
    kg = native_qty * native_kg(product)
    out = {'Metric tons': kg / 1000.0, 'Short tons': kg / SHORT_TON_KG, 'Pounds': kg / LB_KG}
    lb = spec_for(product).get('lb_per_bu')
    if lb:
        out = {'Bushels': kg / (lb * LB_KG), **out}
    return out


def round_contracts(x, mode='Nearest'):
    if mode.startswith('Round down'):
        return int(math.floor(x + 1e-9))
    if mode.startswith('Round up'):
        return int(math.ceil(x - 1e-9))
    return int(math.floor(x + 0.5))


def size_hedge(product, quantity, unit, hedge_ratio=1.0, beta=1.0, rounding='Nearest',
               contract_size=None, futures_price=None, option_delta=None):
    """Contracts needed to hedge a physical exposure.

    hedge_ratio: fraction of exposure to hedge (1.0 = 100%).
    beta: basis / cross-hedge ratio (futures volume per unit of exposure).
    option_delta: if given, also size an option hedge (futures-equivalent / |delta|).
    """
    spec = spec_for(product)
    size = float(contract_size) if contract_size else float(spec['size'])
    if quantity < 0 or hedge_ratio < 0:
        raise ValueError('Exposure and hedge ratio cannot be negative.')
    if size <= 0 or beta <= 0:
        raise ValueError('Contract size and beta must be positive.')
    exposure = to_native(product, quantity, unit)
    target = exposure * hedge_ratio * beta
    exact = target / size
    contracts = round_contracts(exact, rounding)
    hedged_equiv = contracts * size / beta
    out = {'native_unit': spec['native'], 'contract_size': size,
           'exposure_native': exposure, 'target_native': target,
           'contracts_exact': exact, 'contracts': contracts,
           'hedged_native': contracts * size, 'residual_native': exposure - hedged_equiv,
           'hedged_pct': hedged_equiv / exposure if exposure > 0 else 0.0}
    if futures_price:
        per_contract = size * futures_price / spec['quote_div']
        out.update(per_contract_notional=per_contract,
                   exposure_value=exposure * futures_price / spec['quote_div'],
                   hedge_notional=contracts * per_contract,
                   pnl_per_1pct=contracts * per_contract * 0.01)
    if option_delta:
        d = abs(option_delta)
        if d <= 0:
            raise ValueError('Option delta must be non-zero.')
        out['option_contracts_exact'] = exact / d
        out['option_contracts'] = round_contracts(exact / d, rounding)
    return out
