import numpy as np
import matplotlib.pyplot as plt

from finmarkets import GBM
from finmarkets.options.vanilla import BS, OptionType
from finmarkets.options.algos import longstaff_schwartz

S0    = 36.0   # initial stock level
K     = 40.0   # strike price
T     = 1.00   # time-to-maturity
r     = 0.06   # short rate
sigma = 0.20   # volatility

N = 10000
tsteps = 100
dt = T/tsteps
df = np.exp(-r * dt)

S = GBM(r, sigma, S0, T, tsteps, N)

t = np.linspace(0, T, tsteps+1)
paths = S[:,1:100]

expiry = S[-1,:]
hist = np.histogram(expiry, 100)
index = np.arange(100)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(paths)

plt.subplot(122)
plt.bar(index, hist[0])

plt.show()

###########################################

n = 25
maturity  = S[50,:]
reference = S[50-n,:]
payoff = np.maximum(K - maturity, 0)*np.exp(-r *n*dt)
plt.plot(reference, payoff, '.', color='xkcd:blue')
plt.show()

###########################################

C = BS(reference, K, r, sigma, "6m", OptionType.Put)

regr = np.polyfit(reference, payoff, 5)
poly_points = np.polyval(regr, reference)

plt.plot(reference, payoff, '.', color="xkcd:blue")
plt.plot(reference, C, '.', color='xkcd:red')
plt.plot(reference, poly_points, '.', color='xkcd:green')
plt.show()

############################################

V0 = longstaff_schwartz(S0, r, sigma, K, tsteps, T, N, side=OptionType.Put)
print (f"bermudan put option value {V0:5.3f}")
