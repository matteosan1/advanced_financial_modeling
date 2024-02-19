import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import newton

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve, ForwardRateCurve, dt_from_str
from finmarkets import CapFloorLet, CapFloor, CapFloorType

valuation_date = date.today()
rates_data = pd.read_csv("../input_files/rates_data.csv")
pillar_dates = [valuation_date + relativedelta(months=i*12) for i in rates_data['dt']]

fc = ForwardRateCurve(valuation_date, pillar_dates, rates_data['rates'])
dc = DiscountCurve(valuation_date, pillar_dates, rates_data['dfs'])

cap_data = pd.read_csv("../input_files/cap_data.csv")
vol_interp = interp1d(cap_data['cap_maturities'], cap_data['cap_volatilities'], fill_value='extrapolate', kind='slinear')

ttm = ['6m', "1y", '18m', "2y", '30m', "3y", '42m', "4y", '54m', "5y"]
K = 0.013

def obj_func(sigma, caplet, dc, fc, target_price):
    return caplet.npv(sigma, dc, fc)-target_price

sigmas = []
for i in range(len(ttm)-1):
    ttm_val = dt_from_str(ttm[i])
    if ttm_val < cap_data['cap_maturities'][0]:
        sigmas.append(cap_data['cap_volatilities'][0])
    else:
        cap0 = CapFloor(1, valuation_date, ttm[i], "6m", K, CapFloorType.Cap)
        cap1 = CapFloor(1, valuation_date, ttm[i+1], "6m", K, CapFloorType.Cap)
        
        P1 = cap1.npv(vol_interp(dt_from_str(ttm[i+1])), dc, fc)
        P0 = cap0.npv(vol_interp(ttm_val), dc, fc)
        dP = P1 - P0
        caplet = CapFloorLet(1, cap0.dates[-1], "6m", K, CapFloorType.Cap)
        sigma = newton(obj_func, 1, args=(caplet, dc, fc, dP))
        sigmas.append(sigma)

plt.rcParams['figure.figsize'] = (8, 6)
ts = [dt_from_str(ttm[i]) for i in range(len(ttm)-1)]

plt.plot(ts, sigmas, marker="o", color="xkcd:blue")
plt.xticks(ts, ttm[:-1], fontsize=14)
plt.grid(True)
plt.xlabel("Maturity", fontsize=14)
plt.ylabel("Spot Volatility", fontsize=14)
plt.show()
