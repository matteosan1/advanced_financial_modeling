import numpy as np
import matplotlib.pyplot as plt

from finmarkets import GBMShifted, shifted_lognormal
from finmarkets.options.vanilla import BSShifted, OptionType

N = 10000
tsteps = 500
T = 3.0
sigma = 0.2
L0 = -0.05
shift = 0.1

np.random.seed(4)
paths = GBMShifted(0.0, sigma, shift, L0, T, tsteps, N)


plt.plot(paths[:, 0:20])
plt.xlim(0, tsteps)
plt.grid()
plt.show()

############################################

shiftV = [1.0, 2.0, 3.0, 4.0, 5.0]
legend = []
for shift in shiftV:
  x = np.arange(-shift, 5, 0.0001)
  pdf_x = shifted_lognormal(x, L0, sigma, T, shift)
  plt.plot(x, pdf_x)
  legend.append(f'shift={shift}')

plt.legend(legend)
plt.xlabel('x')
plt.ylabel('pdf')
plt.title('shifted lognormal density')
plt.ylim(0, 1.5)
plt.xlim(-5, 5)
plt.grid()
plt.show()

#############################################

np.random.seed(4)

N = 10000
tsteps = 500
T = "3y"
sigma = 0.2
L0 = -0.05
shift = 0.1

K = np.linspace(-shift, np.abs(L0)*3, 25)
optPriceMCV = np.zeros([len(K), 1])
for idx in range(len(K)):
  optPriceMCV[idx] = np.mean(np.maximum(paths[-1, :] - K[idx], 0.0))

optPriceExact = BSShifted(L0, K, shift, 0.0, sigma, T, OptionType.Call)
plt.plot(K, optPriceMCV)
plt.plot(K, optPriceExact, '--r')
plt.grid(True)
plt.xlabel('strike, K')
plt.ylabel('option price')
plt.legend(['Monte Carlo', 'Exact'])
plt.show()

#################################################

legend = []
for shift in [0.2, 0.3, 0.4, 0.5]:
  K = np.linspace(-shift, np.abs(L0)*10.0, 25)
  optPriceExact = BSShifted(L0, K, shift, 0.0, sigma, T, OptionType.Call)
  plt.plot(K, optPriceExact)
  legend.append(f'shift={shift}')

plt.grid(True)
plt.xlabel('strike, K')
plt.ylabel('option price')
plt.legend(legend)
plt.show()
