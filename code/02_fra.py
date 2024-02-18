import matplotlib.pyplot as plt

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import InterestRateSwap, SwapSide, DiscountCurve

def makePFS(rates):
    vals = []
    today = date.today()
    dfs = [1/(1+rates[i])**i for i in range(1, len(rates))]
    dc = DiscountCurve(today,
                       [today+relativedelta(years=i) for i in range(1, len(rates))],
                       dfs)

    irs = InterestRateSwap(1, today, "5y", 0.137, "1y")
    swap_rate = irs.swap_rate_single_curve(dc)
    print ("Swap Rate: {:.3f}".format(swap_rate))
    irs = InterestRateSwap(1, today, "5y", swap_rate, "1y", side=SwapSide.Payer)
    val, vals = irs.npv_with_FRA(dc)
    plt.plot(range(len(rates)), rates, linestyle="--", label=r"Interest rates $r$")
    plt.scatter(range(1, len(vals)+1), vals, color='red', label="FRA")
    plt.hlines(0, 0, 6, color='black', linestyle=":")
    plt.xlim(0, 6)
    plt.ylim(-0.1, 0.2)
    plt.xlabel("Year")
    plt.legend()
    plt.show()
    
makePFS([0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17])
