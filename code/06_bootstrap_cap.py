import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import newton

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve, ForwardRateCurve, dt_from_str
from finmarkets import CapFloorLet, CapFloor, CapFloorType

valuation_date = date.today()
rates = np.array([0.02, 0.02, 0.07, 0.25, 0.70, 1.10, 1.62, 2.00])/100
dfs = [1/(1+rates[i])**t for i, t in enumerate([0, 0.25, 0.5, 1, 2, 3, 5, 7])]
pillar_dates = [valuation_date + relativedelta(months=i*12) for i in [0, 0.25, 0.5, 1, 2, 3, 5, 7]]
fc = ForwardRateCurve(valuation_date, pillar_dates, rates)
dc = DiscountCurve(valuation_date, pillar_dates, dfs)

cap_maturities = [dt_from_str(m) for m in ["1y", "2y", "3y", "4y", "5y"]]
cap_volatilities = [0.44, 0.45, 0.44, 0.41, 0.39]
vol_interp = interp1d(cap_maturities, cap_volatilities, fill_value='extrapolate', kind='slinear')

ttm = ['6m', "1y", '18m', "2y", '30m', "3y", '42m', "4y", '54m', "5y"]
K = 0.013

def obj_func(sigma, caplet, dc, fc, target_price):
    return caplet.npv(sigma, dc, fc)-target_price

sigmas = []
for i in range(len(ttm)-1):
    ttm_val = dt_from_str(ttm[i])
    if ttm_val < cap_maturities[0]:
        sigmas.append(cap_volatilities[0])
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
