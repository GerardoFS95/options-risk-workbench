"""Black-76 pricing and risk helpers for European options on futures."""
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def _check(F,K,T,sigma):
    if not all(np.isfinite(x) for x in [F,K,T,sigma]):
        raise ValueError("Futures price, strike, expiry and volatility must be finite.")
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Futures price, strike, time to expiry and volatility must be greater than zero.")

def black76_price(F,K,T,r,sigma,option_type='Call'):
    _check(F,K,T,sigma)
    d1=(math.log(F/K)+0.5*sigma*sigma*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    disc=math.exp(-r*T)
    if str(option_type).lower().startswith('c'):
        return disc*(F*norm.cdf(d1)-K*norm.cdf(d2))
    return disc*(K*norm.cdf(-d2)-F*norm.cdf(-d1))

def black76_greeks(F,K,T,r,sigma,option_type='Call'):
    _check(F,K,T,sigma)
    d1=(math.log(F/K)+0.5*sigma*sigma*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    disc=math.exp(-r*T)
    is_call=str(option_type).lower().startswith('c')
    delta=disc*norm.cdf(d1) if is_call else -disc*norm.cdf(-d1)
    gamma=disc*norm.pdf(d1)/(F*sigma*math.sqrt(T))
    vega=disc*F*norm.pdf(d1)*math.sqrt(T)
    if is_call:
        theta=(r*black76_price(F,K,T,r,sigma,'Call') - disc*F*norm.pdf(d1)*sigma/(2*math.sqrt(T)))
        rho=-T*black76_price(F,K,T,r,sigma,'Call')
    else:
        theta=(r*black76_price(F,K,T,r,sigma,'Put') - disc*F*norm.pdf(d1)*sigma/(2*math.sqrt(T)))
        rho=-T*black76_price(F,K,T,r,sigma,'Put')
    return {'Theoretical premium':black76_price(F,K,T,r,sigma,option_type),
            'Delta (per futures unit)':delta,'Gamma':gamma,'Vega (per 1 vol pt)':vega/100,
            'Vega (per 1.00 vol)':vega,'Theta (daily)':theta/365,'Rho (per 1 rate pt)':rho/100}

def implied_volatility(premium,F,K,T,r,option_type='Call'):
    if premium <= 0: raise ValueError('Market premium must be positive.')
    f=lambda v:black76_price(F,K,T,r,v,option_type)-premium
    lo,hi=1e-5,5.0
    if f(lo)*f(hi)>0: raise ValueError('Cannot solve implied volatility: premium may be outside model bounds or inputs may be inconsistent.')
    return brentq(f,lo,hi,maxiter=500)

def signed_position_value(row, futures_price=None, vol=None, rate=None):
    typ=str(row['instrument_type']).lower()
    qty=float(row['signed_quantity']); mult=float(row['contract_multiplier'])
    mark=float(row['market_futures_price'] if futures_price is None else futures_price)
    entry=float(row['trade_price'])
    if typ=='future':
        return qty*mult*(mark-entry)
    if typ=='physical':
        return qty*mult*(mark-entry)
    if typ in ('call','put'):
        strike=float(row['strike']); T=max(float(row['days_to_expiry'])/365,1e-6)
        r=float(row['risk_free_rate'] if rate is None else rate)
        v=max(0.0001,float(row['implied_volatility'] if vol is None else vol))
        if futures_price is None and vol is None and rate is None:
            premium=float(row['market_option_premium'])
        else:
            premium=black76_price(mark,strike,T,r,v,typ.title())
        return qty*mult*(premium-entry)
    raise ValueError('instrument_type must be Future, Physical, Call or Put')
