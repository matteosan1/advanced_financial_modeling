import matplotlib.pyplot as plt
import pandas as pd, numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve, dt_from_str
from finmarkets.credit import FloatingRateNote

rates = pd.read_csv("../input_files/libor.csv")

frn = FloatingRateNote(date.today(), "3Y", "3m")
pillars = [date.today()+relativedelta(months=dt_from_str(f"{t}M")) for t in rates['T']]
prices = []

d = date.today()
for i in range(0, 360*2):
  prices.append(frn.price(d, pillars, rates['r'].values))
  d = date.today()+relativedelta(days=i)

plt.plot(prices)
plt.ylim(1.0, 1.002)
plt.xlabel("days")
plt.ylabel("Bond Value")
plt.show()
