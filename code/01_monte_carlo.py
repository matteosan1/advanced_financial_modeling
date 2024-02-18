import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
N = 1000
nflips = [20, 40, 60]

results = np.zeros(shape=(N, len(nflips)))
for j in range(len(nflips)):
  for i in range(N):
    results[i, j] = np.sum(np.random.randint(0, 2, size=nflips[j]))/nflips[j]

for j, f in enumerate(nflips):
  plt.hist(results[:, j], range=(0, 1), bins=100, density=True, label=f'{f} flips')

plt.legend()
plt.show()

#####################################################

from scipy.stats import norm

for j in range(len(nflips)):
  fit_result = norm.fit(results[:, j])
  print (f"n. of flips {nflips[j]}: avg. {fit_result[0]:.3f} +- {fit_result[1]:.3f}")

##########################################

np.random.seed(10)
def random_walk_1d(startpoint, nsteps, p):
  X = np.zeros(shape=(nsteps,))
  X[0] = startpoint
  games = np.random.random(size=nsteps-1)
  X[1:] = np.where(games>p, 1, -1)
  return np.cumsum(X)

X = random_walk_1d(0, 100, 0.5)

plt.plot(X)
plt.show()

##########################################

# It should be implemented an exit strategy against infinite loop
# The gambler cannot play for an infinite amount of time

def random_walk_with_stop(startpoint, p, N):
  X = [startpoint]
  while True:
    game = np.random.random()
    if game < p:
      X.append(X[-1] + 1)
    else:
      X.append(X[-1] - 1)
    if X[-1] <= 0:
      return 0
    elif X[-1] >= N:
      return 1

simulations = 10000
outcomes = []
for _ in range(simulations):
  outcomes.append(random_walk_with_stop(100, 0.5, 120))

print (f"Probability to hit upper limit: {sum(outcomes)/simulations}")

#############################################

def random_walk_with_stop_for_steps(startpoint, p, N):
  games = 0
  X = [startpoint]
  while True:
    game = np.random.random()
    games += 1
    if game < p:
      X.append(X[-1] + 1)
    else:
      X.append(X[-1] - 1)
    if X[-1] <= 0 or X[-1] >= N:
      return games

simulations = 10000
games = []
for _ in range(simulations):
  games.append(random_walk_with_stop_for_steps(100, 0.5, 120))

print (np.mean(games))

##############################################

K0 = 10
simulations = 100000
games = 500

np.random.seed(1)
K = np.zeros(shape=(simulations, games))
for s in range(simulations):
  #np.random.seed(s)
  K[s, :] = np.random.randint(1, 7, size=games)
  K[s, :] = np.where(K[s, :]<3, -1, K[s, :])
  K[s, :] = np.where(K[s, :]>4, 1, K[s, :])
  K[s, :] = np.where((K[s, :]<5)&(K[s, :]>2), 0, K[s, :])
  K[s, :] = np.cumsum(K[s, :])

for s in range(10):
  p = plt.plot(K0 + K[s, :])

mean = np.mean(K[:, -1])
std = 1.96*np.std(K[:, -1])/np.sqrt(simulations)
plt.text(20, 42, f"Expected gain: {mean:.2f} +- {std:.2f}", fontsize=20)
plt.grid(True)
plt.xlim(0, games)
plt.xlabel("games", fontsize=14)
plt.show()

##########################################

plt.rcParams['figure.figsize'] = (8, 8)

n = 100000

x = np.zeros(n)
y = np.zeros(n)

for i in range(1, n):
    val = np.random.randint(1, 5)
    if val == 1:
        x[i] = x[i-1] + 1
        y[i] = y[i-1]
    elif val == 2:
        x[i] = x[i-1] - 1
        y[i] = y[i-1]
    elif val == 3:
        x[i] = x[i-1]
        y[i] = y[i-1] + 1
    else:
        x[i] = x[i-1]
        y[i] = y[i-1] - 1

plt.title("Random Walk ($n = " + str(n) + "$ steps)")
plt.plot(x, y)
plt.show()

#########################################

from finmarkets import BM

np.random.seed(10)
y = BM(0, 1, 0, 3, 365, 2)

plt.rcParams['figure.figsize'] = (9, 6)
plt.plot(y[:, 0], label="realization 1", color='xkcd:blue')
plt.plot(y[:, 1], label="realization 2", color='xkcd:red')
plt.grid(True)
plt.xlabel("time")
plt.xlim(0, 365)
plt.legend()
plt.show()

#########################################

from finmarkets import GBM

np.random.seed(1)

S = GBM(0.005, 0.05, 100, 120, 120, 1)
Sdet = GBM(0.005, 0.0, 100, 120, 120, 1)

plt.plot(S, label='Stochastic Path')
plt.plot(Sdet, linestyle='--', color='red', label='Deterministic Path')
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(fontsize='16')
plt.show()

##########################################
#
#path = np.DataSource("https://github.com/matteosan1/advanced_financial_modeling/raw/master/input_files")
#S = np.load(path.open("stock_2023.npy", "rb"))
#
#plt.plot(S[50, :], label='Stochastic Path')
#plt.grid(True)
#plt.xlabel('Time')
#plt.ylabel('Stock Price')
#plt.xlim(0,100)
#plt.legend(fontsize='16')
#plt.show()
#
#for i in range(50):
#  plt.plot(S[i, :])
#plt.grid(True)
#plt.xlabel('Time')
#plt.ylabel('Stock Price')
#plt.xlim(0,100)
#plt.show()
#
#mean = np.mean(S, axis=0)
#
#plt.plot(mean)
#plt.grid(True)
#plt.xlabel('Time')
#plt.ylabel('Avg Realized Stock Price')
#plt.xlim(0,100)
#plt.show()
#
#var = np.var(S, axis=0)
#
#plt.plot(var)
#plt.grid(True)
#plt.xlabel('Time')
#plt.ylabel(r'Stock Price $\sigma$')
#plt.xlim(0,100)
#plt.show()
#
#X = np.zeros_like(S)
#for t in range(S.shape[1]-1):
#  X[:, t] = np.log(S[:, t+1]/S[:, t])
#
#mean = np.mean(X, axis=0)
#
#plt.plot(mean)
#plt.grid(True)
#plt.xlabel('Time')
#plt.ylabel('Avg Realized Stock Price')
#plt.xlim(0,100)
#plt.show()
#
#var = np.var(X, axis=0)
#
#plt.plot(var)
#plt.grid(True)
#plt.xlabel('Time')
#plt.ylabel(r'Stock Price $\sigma$')
#plt.xlim(0,98)
#plt.show()
#
##########################################

def MCHeston(St, mu, T, nu0, kappa, theta, csi, rho, simulations=10000, timeStepsPerYear=100):
  timesteps = T * timeStepsPerYear
  dt = 1/timeStepsPerYear

  S_t = np.zeros(shape=(timesteps, simulations))
  V_t = np.zeros(shape=(timesteps, simulations))

  V_t[0, :] = nu0
  S_t[0, :] = St

  means = [0, 0]
  covs = [[1  , rho],
          [rho, 1  ]]
  Z = np.random.multivariate_normal(mean=means, cov=covs, size=(simulations, timesteps)).T
  Z1 = Z[0]
  Z2 = Z[1]

  for i in range(1, timesteps):
    V_t[i, :] = np.maximum(V_t[i-1, :] + kappa*(theta-V_t[i-1, :])*dt + csi*np.sqrt(V_t[i-1, :]*dt)*Z2[i, :], 0)
    S_t[i, :] = S_t[i-1, :] + mu*S_t[i-1, :]*dt + np.sqrt(V_t[i, :]*dt)*S_t[i-1, :]*Z1[i, :]

  return S_t, V_t


S0 = 100
mu = 0.25
T = 3
nu0 = 0.05
kappa = 1.2
theta = 0.10
csi = 0.593
rho = -0.5

np.random.seed(1000)
S, nu = MCHeston(S0, mu, T, nu0, kappa, theta, csi, rho)
S_mean = np.mean(S, axis=1)
nu_mean = np.mean(nu, axis=1)


plt.rcParams['figure.figsize'] = (14,6)
fig, ax = plt.subplots(1, 2)
ax[0].plot(nu[:, :30])
ax[0].plot(nu_mean, color="black")
ax[0].hlines(theta, 0, 300, linestyle="--", color='black')
ax[0].grid(True)
ax[0].set_xlabel("time")
ax[0].set_ylabel("Volatility")
ax[0].set_ylim(0, 1.0)
ax[0].set_xlim(0, 300)

ax[1].plot(S[:, :30])
ax[1].plot(S_mean, color="black")
ax[1].grid(True)
ax[1].set_xlabel("time")
ax[1].set_ylabel("Asset Value")
ax[1].set_xlim(0, 300)
ax[1].set_ylim(0, 380)
plt.show()

