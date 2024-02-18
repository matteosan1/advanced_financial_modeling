import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import newton

from finmarkets.options.vanilla import BS, OptionType

vols = []
prices = []

def implied_vol(sigma, S0, K, r, T, market_price, flag=OptionType.Call):
  bs_price = BS(S0, K, r, sigma, T, flag)
  vols.append(sigma)
  prices.append(bs_price)
  return bs_price - market_price

S0, K, T, r, sigma = 30, 28, "3m", 0.025, 0.3
market_price = 4.311109723696367
imp_sigma = newton(implied_vol, 0.1, args=(S0, K, r, T, market_price))
print (f"{imp_sigma:.4f}")

bs_prices = []
xvals = np.arange(0.01, 1.5, 0.01)
for sigma in np.arange(0.01, 1.5, 0.01):
  bs_prices.append(BS(S0, K, r, sigma, T, OptionType.Call))

fig, ax = plt.subplots()
plt.title('Newton-Raphson Method for Option Implied Volatility')
plt.ylabel('Call Price')
plt.xlabel('Implied Volatility')
ax.scatter(0.54, market_price, s=80, marker='o', color='xkcd:blue', label = 'Market Price')
ax.plot(xvals, bs_prices, label = 'Black Scholes Price')
ax.scatter(vols, prices, s=80, facecolors='none', edgecolors='xkcd:red', label='Newton Guesses')

ax.legend(loc='upper left')

a = plt.axes([.65, .2, .2, .2], facecolor=None)
a.scatter(0.54, market_price, s=80,  marker='o', color='xkcd:blue', label = 'Market Price')
a.plot(xvals, bs_prices, label = 'Black Scholes Price')
a.scatter(vols, prices, s=80, facecolors='none', edgecolors='xkcd:red')
a.set_xlim(0.535, 0.545)
a.set_ylim(4.29, 4.33)

plt.show()

