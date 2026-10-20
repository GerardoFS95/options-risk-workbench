import unittest
from risk_engine import black76_price, black76_greeks, implied_volatility

class Black76Tests(unittest.TestCase):
 def test_call_put_parity(self):
  F,K,T,r,v=450,460,0.5,0.04,0.25
  c=black76_price(F,K,T,r,v,'Call'); p=black76_price(F,K,T,r,v,'Put')
  self.assertAlmostEqual(c-p, __import__('math').exp(-r*T)*(F-K), places=8)
 def test_iv_roundtrip(self):
  premium=black76_price(450,460,0.5,0.04,0.25,'Call')
  self.assertAlmostEqual(implied_volatility(premium,450,460,0.5,0.04,'Call'),0.25,places=7)
 def test_greeks_present(self):
  g=black76_greeks(450,450,0.5,0.04,0.25,'Call')
  self.assertGreater(g['Theoretical premium'],0); self.assertGreater(g['Gamma'],0); self.assertGreater(g['Vega (per 1 vol pt)'],0)
if __name__=='__main__': unittest.main()
