from finmarkets.short_rates.vasicek import VasicekModel
from finmarkets import SwaptionShortRate, SwapSide

notional = 1
expiry = 1
tenor = 2
K = 0.02
r = 0.03
theta = 0.005
kappa = 0.03
sigma = 0.04

vasicek = VasicekModel(theta, kappa, sigma)
swaption = SwaptionShortRate(notional, expiry, tenor, K, vasicek, SwapSide.Receiver)
r_star = swaption.rstar()
print (r_star)
print (swaption.npv(r, r_star))
