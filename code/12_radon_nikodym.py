import numpy as np


T = 1
M = 252
dt = T/M
N = 1000
mu = 0.0125/dt

np.random.seed(1000)
epsilon = np.zeros(shape=(M, N))
for i in range(N):
  epsilon[1:, i] = np.random.normal(size=M-1)

dW = np.zeros_like(epsilon)
dW[1:, :] = np.sqrt(dt)*epsilon[1:, :]
W = np.cumsum(dW, axis=0)
dY = np.zeros_like(epsilon)
dY[1:, :] = mu*dt + np.sqrt(dt)*epsilon[1:, :]
Y = np.cumsum(dY, axis=0)


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8,4)

plt.subplot(1,2,1)
plt.plot(W[:, :30])
plt.grid(True)
plt.ylim(-4, 4)
plt.xlim(0, M)
plt.title("Realizations of $W$")

plt.subplot(1,2,2)
plt.plot(Y[:, :30])
plt.grid(True)
plt.ylim(-2, 6)
plt.xlim(0, M)
plt.title("Realizations of $Y$")

plt.show()


############################################
#
#import numpy as np
#from scipy.stats import norm
#import matplotlib.pyplot as plt
#
#def radon_nikodym(x, P, Q):
#  return Q.pdf(x)/P.pdf(x)
#
#mu = 0.05
#r = 0.025
#sigma = 0.2
#P = norm(mu, sigma)
#Q = norm(r, sigma)
#
#plt.rcParams['figure.figsize'] = (8,4)
#plt.subplot(1, 2, 1)
#x = np.arange(-1, 1, 0.01)
#plt.plot(x, P.pdf(x), label="$\mathbb{P}$", color='xkcd:red')
#plt.plot(x, Q.pdf(x), label="$\mathbb{Q}$", color='xkcd:blue')
#plt.xlabel("$x$")
#plt.legend()
#plt.xlim(-.8, .8)
#plt.ylim(0, )
#plt.grid(True)
#
#plt.subplot(1, 2, 2)
#x = np.arange(-6, 6, 0.01)
#y = radon_nikodym(x, P, Q)
#plt.plot(x, y, color='xkcd:blue', label=r"$\frac{d\mathbb{Q}}{d\mathbb{P}}$")
#plt.fill_between(x, 0, y, color='xkcd:baby blue')
#plt.grid(True)
#plt.xlim(-6, 6)
#plt.ylim(0, 25)
#plt.xlabel("$x$")
#plt.legend(fontsize=14)
#plt.title("Radon-Nikodym density")
#plt.show()
#
#
#############################################
#
## Radon-Nikodym process
#RN = np.ones_like(epsilon)
#RN[1:, :] = np.exp(-mu*dY[1:, :] - 0.5*mu**2*dt)
#
## process Y under the new measure
#YQ = np.cumsum(dY*RN, axis=0)
#
#Y_mean_P = np.mean(Y, axis=1)
#Y_std_P = np.std(Y, axis=1)
#Y_std_Q = np.std(YQ, axis=1)
#Y_mean_Q = np.mean(YQ, axis=1)
#
#
#import matplotlib.pyplot as plt
#
#plt.rcParams['figure.figsize'] = (5, 5)
#plt.plot(Y_std_P, color="black", label='$dW^{P}$')
#plt.plot(Y_std_Q, color="xkcd:red", label='$dW^{Q}$')
#plt.plot(Y_mean_Q, color="xkcd:blue", label='$\mathbb{E}[YQ]$')
#plt.grid(True)
#plt.legend()
#plt.show()
#
#idx = 100
##plt.plot(W[:, idx], color='xkcd:red', label="$W^Q$")
#plt.plot(Y[:, idx], color='black', label="$Y$")
#plt.plot(YQ[:, idx], color='red', label="$YQ$")
#
#plt.xlim(0, M)
#plt.grid(True)
#plt.legend()
#plt.show()
#
#########################################
#
#import numpy as np
#
#mu = 0.05
#r = 0.01
#sigma = 0.20
#l = (mu-r)/sigma
#S0 = 100
#K = 120
#T = 1
#M = 365
#dt = T/M
#N = 1000
#
#np.random.seed(1)
#
#epsilon = np.zeros(shape=(M, N))
#for i in range(N):
#  epsilon[:, i] = np.random.normal(size=M)
#dW = np.sqrt(dt)*epsilon
#
#SP = np.zeros_like(dW)
#SP[0, :] = S0
#for sim in range(N):
#  for t in range(1, M):
#    SP[t, sim] = SP[t-1, sim] + mu*dt*SP[t-1, sim] + sigma*SP[t-1, sim]*dW[t-1, sim]
#
#import matplotlib.pyplot as plt
#
#SP_exp = np.mean(SP, axis=1)
#
#plt.rcParams['figure.figsize'] = (8,4)
#plt.subplot(1,2,1)
#plt.plot(SP[:, :20])
#plt.grid(True)
#plt.xlim(0, M)
#plt.title("Realizations of $S^P$")
#
#plt.subplot(1,2,2)
#plt.plot(SP_exp)
#plt.grid(True)
#plt.xlim(0, M)
#plt.title("$\mathbb{E}^P[S]$")
#plt.show()
#
#
#SQ = np.zeros_like(epsilon)
#SQ[0, :] = S0
#for sim in range(N):
#  for t in range(1, M):
#    SQ[t, sim] = SQ[t-1, sim] + r*dt*SQ[t-1, sim] + sigma*SQ[t-1, sim]*dW[t-1, sim]
#
#
#SQ_exp = np.mean(SQ, axis=1)
#
#plt.rcParams['figure.figsize'] = (8,4)
#plt.subplot(1,2,1)
#plt.plot(SQ[:, :20])
#plt.grid(True)
#plt.xlim(0, M)
#plt.title("Realizations of $S^Q$")
#
#plt.subplot(1,2,2)
#plt.plot(SQ_exp)
#plt.grid(True)
#plt.xlim(0, M)
#plt.title("$\mathbb{E}^Q[S]$")
#plt.show()
#
## the Radon-Nikodym derivative is defined according to Girsanov theorem
#RN = np.ones_like(epsilon)
#for sim in range(N):
#  for t in range(1, M):
#    RN[t, sim] = np.exp(-l*np.sqrt(dt)*epsilon[t, sim] - 0.5*l**2*dt)
#
#plt.rcParams['figure.figsize'] = (4,4)
#plt.plot(np.mean(np.cumprod(RN, axis=0), axis=1), color='black')
#plt.ylim(0, 2)
#plt.xlim(0, 365)
#plt.grid(True)
#plt.show()
#
#CQ = np.maximum(np.exp(-r*T)*(SQ[-1, :]-K), 0)
#print (f"Price under Q: {np.mean(CQ):.2f}")
#
#CP_RN = np.maximum(np.exp(-r*T)*(SP[-1, :]-K), 0)*np.cumprod(RN, axis=0)[-1, :]
#print (f"Price under Q (via RN): {np.mean(CP_RN):.2f}")
#
