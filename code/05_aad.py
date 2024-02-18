import numpy as np
import tensorflow as tf

from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import newton

from finmarkets import DiscountCurve, ForwardRateCurve
from finmarkets import Swap, InterestRateSwap, FixedRateBond

n = 6
today = date.today()
dates = [today+relativedelta(years=i) for i in range(n)]
dfs = [1/(1+0.05)**i for i in range(n)]
dc = DiscountCurve(today, dates, dfs)

irs = InterestRateSwap(1, today, "5y", 0.055, "3m")
print ("IRS BPV: {:.4f}".format(irs.bpv(dc)))

###################################################

bond = FixedRateBond(today, 0.055, "5y", "3m", 1)

print (f"Bond ytom: {bond.yield_to_maturity(dc):.3f}")
print (f"Bond duration (mod duration): {bond.duration(dc):.2f} ({bond.mod_duration(dc):.3f})")
print (f"Bond NPV: {bond.npv(dc)}")

print ("BPV approx.: {:.4f}".format(bond.mod_duration(dc)*bond.npv(dc)*0.0001))

###################################################

n = 6
today = date.today()
dates = [today+relativedelta(years=i) for i in range(n)]

vals = []
dr = 0.01
rates_up = np.array([0.1]*n) + dr
rates_down = np.array([0.1]*n) - dr
libor_up = ForwardRateCurve(today, dates, rates_up)
libor_down = ForwardRateCurve(today, dates, rates_down)
dfs_up = [1/(1+rates_up[i])**i for i in range(n)]
dfs_down = [1/(1+rates_down[i])**i for i in range(n)]
dc_up = DiscountCurve(today, dates[1:], dfs_up[1:])
dc_down = DiscountCurve(today, dates[1:], dfs_down[1:])

irs = InterestRateSwap(1, today, "5y", 0.02, "3m")
npv_up = irs.npv(dc_up, libor_up)
npv_down = irs.npv(dc_down, libor_down)
dv01 = (irs.npv(dc_up, libor_up) - irs.npv(dc_down, libor_down))/2
print ("DV01 IRS: {:.4f}".format(dv01))

#####################################################

zero_rate   = 0.015
fixed_rate  = 0.05
tau         = 1.0
terms       = [1.0, 2.0, 3.0, 4.0, 5.0]
float_rates = [0.01, 0.01, 0.01, 0.01, 0.01]

swap = Swap(1000000, fixed_rate, tau,terms, float_rates, zero_rate)

price, dv01 = swap.swap_price(0.0001, 0.0001)
price_manual, dv01_manual = swap.swap_price_tangent_mode_manual(0.0001, 0.0001)

print (f"Swap price {price:.2f}")
print (f"DV01 (DV01 'manual'): {dv01:.2f} ({dv01_manual:.2f})")
