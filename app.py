import io
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from risk_engine import black76_price, black76_greeks, implied_volatility, signed_position_value

st.set_page_config(page_title='Grains & Oilseeds Risk Workbench',layout='wide')
st.title('Grains & Oilseeds Risk Workbench')
st.caption('Pre-trade option analytics • trade ledger • portfolio Greeks • scenario risk')
st.warning('Designed for European options on futures using Black-76. Validate exchange contract specifications, quote units, exercise style and settlement terms before production use. This is decision support, not an automated trade approval system.')
PRODUCTS=['Corn','Chicago SRW Wheat','KC HRW Wheat','Soybeans','Soybean Meal','Soybean Oil','Palm Oil','Other']
COLUMNS=['trade_id','trade_date','product','contract_month','instrument_type','signed_quantity','trade_price','market_futures_price','strike','days_to_expiry','risk_free_rate','implied_volatility','market_option_premium','contract_multiplier','purpose','notes']
SAMPLE=pd.DataFrame([
 {'trade_id':'DEMO-001','trade_date':str(date.today()),'product':'Soybeans','contract_month':'Nov-2026','instrument_type':'Call','signed_quantity':5,'trade_price':45.0,'market_futures_price':1050.0,'strike':1050.0,'days_to_expiry':90,'risk_free_rate':0.04,'implied_volatility':0.25,'market_option_premium':45.0,'contract_multiplier':50,'purpose':'Hedge / optionality','notes':'Example only'},
 {'trade_id':'DEMO-002','trade_date':str(date.today()),'product':'Corn','contract_month':'Dec-2026','instrument_type':'Future','signed_quantity':-3,'trade_price':450.0,'market_futures_price':455.0,'strike':0.0,'days_to_expiry':0,'risk_free_rate':0.04,'implied_volatility':0.25,'market_option_premium':0.0,'contract_multiplier':50,'purpose':'Hedge','notes':'Example only'}
])

with st.sidebar:
 st.header('Pre-trade option ticket')
 prod=st.selectbox('Commodity',PRODUCTS,index=0)
 f=st.number_input('Underlying futures price',min_value=0.0001,value=450.0,step=1.0)
 k=st.number_input('Strike',min_value=0.0001,value=450.0,step=1.0)
 days=st.number_input('Days to expiry',min_value=1,value=90,step=1)
 typ=st.selectbox('Option type',['Call','Put'])
 vol_pct=st.number_input('Volatility assumption (%)',min_value=0.01,value=25.0,step=0.5)
 rate_pct=st.number_input('Discount rate (%)',value=4.0,step=0.25)
 market_prem=st.number_input('Observed market premium (quote units)',min_value=0.0,value=15.0,step=0.25)
 mult=st.number_input('Contract multiplier',min_value=0.0001,value=50.0,step=1.0)
 qty=st.number_input('Proposed signed quantity (+ long / − short)',value=1.0,step=1.0)
 vol=vol_pct/100; rate=rate_pct/100; T=days/365

pre, ledger_tab, risk_tab = st.tabs(['1 · Pre-trade analytics','2 · Trade ledger & CSV','3 · Portfolio risk & stress'])
with pre:
 st.subheader(f'{prod}: {typ} option pre-trade check')
 try:
  g=black76_greeks(f,k,T,rate,vol,typ)
  model=g['Theoretical premium']; diff=market_prem-model
  c=st.columns(4)
  c[0].metric('Model premium',f'{model:,.4f}')
  c[1].metric('Market − model',f'{diff:,.4f}',delta=f'{(diff/model*100 if model else 0):.2f}% vs model')
  c[2].metric('Delta-adjusted exposure',f'{qty*mult*g["Delta (per futures unit)"]:,.2f}')
  c[3].metric('Position vega / 1 vol pt',f'{qty*mult*g["Vega (per 1 vol pt)"]:,.4f}')
  st.dataframe(pd.DataFrame({'Metric':list(g.keys()),'Value':list(g.values())}),hide_index=True,use_container_width=True)
  if market_prem>0:
   try:
    iv=implied_volatility(market_prem,f,k,T,rate,typ)
    st.metric('Implied volatility from observed premium',f'{iv:.2%}',delta=f'{(iv-vol)*100:+.2f} vol points vs assumption')
    if iv > vol + .03: st.warning('Market premium implies volatility materially above your assumption. Check event risk, skew, expiry and quote quality before trading.')
    elif iv < vol - .03: st.info('Market premium implies volatility materially below your assumption. Confirm the bid/ask, liquidity and model inputs.')
    else: st.success('Observed premium is broadly consistent with the entered volatility assumption under Black-76. This is not, by itself, a trade recommendation.')
   except Exception as e: st.warning(str(e))
  st.caption('Premium comparison is meaningful only when market premium, futures, strike, expiry, quote units and multiplier refer to the same contract. Use a volatility surface/skew-aware input for production decisions.')
  st.subheader('Proposed option position sensitivity')
  st.write(f'Approximate premium value = quantity × multiplier × model premium = **{qty*mult*model:,.2f}** quote-currency units.')
 except Exception as e: st.error(str(e))

with ledger_tab:
 st.subheader('Trade-by-trade portfolio ledger')
 st.write('Enter one row per trade. Signed quantity: positive = long, negative = short. For options, trade_price is the premium paid/received per quote unit; market_option_premium is the current observed premium. For futures/physical rows, trade_price and market_futures_price are entry and current price.')
 uploaded=st.file_uploader('Import existing portfolio CSV (optional)',type=['csv'],key='ledger_upload')
 try: initial=pd.read_csv(uploaded) if uploaded else SAMPLE.copy()
 except Exception as e: st.error(f'CSV import error: {e}'); initial=SAMPLE.copy()
 edited=st.data_editor(initial,num_rows='dynamic',use_container_width=True,key='trade_ledger')
 b1,b2,b3=st.columns(3)
 with b1:
  st.download_button('Export full portfolio CSV',edited.to_csv(index=False).encode('utf-8'),'commodities_trade_ledger.csv','text/csv',type='primary')
 with b2:
  st.download_button('Download blank CSV template',pd.DataFrame(columns=COLUMNS).to_csv(index=False).encode('utf-8'),'commodities_ledger_template.csv','text/csv')
 with b3:
  if st.button('Validate ledger'):
   missing=[x for x in COLUMNS if x not in edited.columns]
   if missing: st.error('Missing columns: '+', '.join(missing))
   else:
    issues=[]
    for i,row in edited.iterrows():
     if str(row.instrument_type).title() not in ['Future','Physical','Call','Put']: issues.append(f'Row {i+1}: invalid instrument_type')
     if float(row.contract_multiplier)<=0: issues.append(f'Row {i+1}: contract_multiplier must be positive')
     if str(row.instrument_type).title() in ['Call','Put'] and (float(row.strike)<=0 or float(row.days_to_expiry)<=0 or float(row.implied_volatility)<=0): issues.append(f'Row {i+1}: option strike, expiry and volatility must be positive')
    if issues: st.error('\n'.join(issues))
    else: st.success(f'Basic validation passed for {len(edited)} rows. This checks fields, not economic suitability or exchange contract rules.')

with risk_tab:
 st.subheader('Portfolio Greeks, P&L and scenario stress')
 st.caption('All position-level results depend on signed quantity, contract multiplier and consistent exchange quote units. Options are revalued with Black-76; futures and physical rows use linear price P&L.')
 try:
  needed=['product','instrument_type','signed_quantity','trade_price','market_futures_price','strike','days_to_expiry','risk_free_rate','implied_volatility','market_option_premium','contract_multiplier']
  if any(c not in edited.columns for c in needed): st.error('Ledger is missing required columns. Validate or correct it on the ledger tab.')
  else:
   rows=[]
   for _,row in edited.iterrows():
    typ0=str(row.instrument_type).title(); q0=float(row.signed_quantity); m0=float(row.contract_multiplier); F0=float(row.market_futures_price)
    pnl=signed_position_value(row)
    result={'trade_id':row.trade_id,'product':row.product,'instrument_type':typ0,'signed_quantity':q0,'current_pnl':pnl,'gross_futures_notional':abs(q0*m0*F0)}
    if typ0 in ['Call','Put']:
     T0=max(float(row.days_to_expiry)/365,1e-6); g0=black76_greeks(F0,float(row.strike),T0,float(row.risk_free_rate),float(row.implied_volatility),typ0)
     scale=q0*m0
     result.update({'model_premium':g0['Theoretical premium'],'market_option_premium':float(row.market_option_premium),'premium_model_gap':float(row.market_option_premium)-g0['Theoretical premium'],'net_delta':scale*g0['Delta (per futures unit)'],'net_gamma':scale*g0['Gamma'],'net_vega_per_vol_pt':scale*g0['Vega (per 1 vol pt)'],'net_theta_daily':scale*g0['Theta (daily)']})
    else:
     result.update({'model_premium':np.nan,'market_option_premium':np.nan,'premium_model_gap':np.nan,'net_delta':q0*m0,'net_gamma':0.0,'net_vega_per_vol_pt':0.0,'net_theta_daily':0.0})
    rows.append(result)
   risk=pd.DataFrame(rows)
   m1,m2,m3,m4,m5=st.columns(5)
   m1.metric('Current aggregate P&L',f'{risk.current_pnl.sum():,.2f}')
   m2.metric('Net futures delta-equivalent',f'{risk.net_delta.sum():,.2f}')
   m3.metric('Net gamma',f'{risk.net_gamma.sum():,.5f}')
   m4.metric('Net vega / vol point',f'{risk.net_vega_per_vol_pt.sum():,.2f}')
   m5.metric('Gross futures notional',f'{risk.gross_futures_notional.sum():,.2f}')
   st.dataframe(risk,hide_index=True,use_container_width=True)
   st.download_button('Export portfolio risk CSV',risk.to_csv(index=False).encode('utf-8'),'commodities_portfolio_risk.csv','text/csv')
   st.subheader('Full-revaluation stress scenarios')
   st.write('Independent parallel shocks are applied to each line’s underlying futures price and option volatility. This is a first-pass stress grid; it does not automatically model cross-commodity correlations, calendar spreads, basis or liquidity.')
   px=st.multiselect('Futures price shock (%)',[-25,-20,-15,-10,-5,0,5,10,15,20,25],default=[-15,-10,-5,0,5,10,15])
   vx=st.multiselect('Volatility shock (absolute vol points)',[-15,-10,-5,0,5,10,15,25],default=[-10,0,10,15])
   if st.button('Run full portfolio stress',type='primary'):
    basep=float(risk.current_pnl.sum()); scen=[]
    for ps in px:
     for vs in vx:
      total=0.0
      for _,row in edited.iterrows():
       typ0=str(row.instrument_type).title(); qty0=float(row.signed_quantity); mult0=float(row.contract_multiplier)
       newF=float(row.market_futures_price)*(1+ps/100)
       if typ0 in ['Future','Physical']:
        total += qty0*mult0*(newF-float(row.trade_price))
       else:
        newvol=max(0.0001,float(row.implied_volatility)+vs/100)
        T0=max(float(row.days_to_expiry)/365,1e-6)
        base_model=black76_price(float(row.market_futures_price),float(row.strike),T0,float(row.risk_free_rate),float(row.implied_volatility),typ0)
        newprem=black76_price(newF,float(row.strike),T0,float(row.risk_free_rate),newvol,typ0)
        # Anchor scenario P&L to the observed market premium; model price changes drive incremental stress.
        scenario_mark=float(row.market_option_premium) + (newprem-base_model)
        total += qty0*mult0*(scenario_mark-float(row.trade_price))
      scen.append({'futures_price_shock_%':ps,'volatility_shock_pts':vs,'stressed_portfolio_pnl':total,'change_vs_current_pnl':total-basep})
    sd=pd.DataFrame(scen).sort_values('change_vs_current_pnl')
    worst=sd.iloc[0]
    st.metric('Worst scenario in selected grid',f'{worst.change_vs_current_pnl:,.2f}',delta=f"{worst['futures_price_shock_%']:+.0f}% futures / {worst['volatility_shock_pts']:+.0f} vol pts")
    st.dataframe(sd,hide_index=True,use_container_width=True)
    st.download_button('Export stress scenarios CSV',sd.to_csv(index=False).encode('utf-8'),'commodities_stress_scenarios.csv','text/csv')
 except Exception as e: st.error(f'Risk calculation error: {e}')

st.divider()
st.caption('Suggested next enhancements: market-data feed, product-specific contract multipliers and tick values, historical VaR/Expected Shortfall, crop-calendar and event scenarios, curve/calendar-spread risk, basis exposure, margin/liquidity estimates, and corporate physical exposure reconciliation.')
