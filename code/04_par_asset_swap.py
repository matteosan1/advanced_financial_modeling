import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import ForwardRateCurve, DiscountCurve, FixedRateBond
from finmarkets import ParAssetSwap, CreditCurve, CreditDefaultSwap

today = date.today()

dfs = [1, 0.9510991280247174, 0.9047134761940828, 0.8605900792981379, 0.8186186058617915, 0.7785874422199597]
dates = [today+relativedelta(years=i) for i in range(len(dfs))]
dc = DiscountCurve(today, dates[1:], dfs[1:])

libor = ForwardRateCurve(today, dates[1:], [0.05]*(len(dates)-1))

bond = FixedRateBond(today, 0.055, "1y", "3m", 100)
print ("Bond price: {:.2f}".format(bond.npv(dc)))

market_price_bond = bond.npv_flat_default(dc, pd=0.05, R=0.4)
print ("Bond price: {:.2f}".format(market_price_bond))

asw = ParAssetSwap(market_price_bond, bond, "3m", dc, libor)
print ("ASW spread: {:.3f}".format(asw.spread))

dates = [today+relativedelta(years=i) for i in range(2)]
cc = CreditCurve(today, dates, [1.0, 0.95])
cds = CreditDefaultSwap(1, today, "1y", 0.05, "3m")
print (f"CDS spread: {cds.breakevenRate(dc, cc):.3f}")


pds = np.arange(0, 1.0, 0.01)

bond = FixedRateBond(today, 0.055, "5y", "3m", 100)

spreads = []
prices = []
for pd in pds:
  market_price_bond = bond.npv_flat_default(dc, pd=pd, R=0.4)
  prices.append(market_price_bond)
  asw = ParAssetSwap(market_price_bond, bond, "3m", dc, libor)
  spreads.append(asw.spread)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Probability of Default')
ax1.set_ylabel('Asset Swap Spread', color=color)
ax1.plot(pds, spreads, color=color)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.25)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Risky Bond Value', color=color)
ax2.plot(pds, prices, color=color)
ax2.set_ylim(0, 105)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
