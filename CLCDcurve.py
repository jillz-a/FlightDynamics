from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt

alpha =np.array([1.7, 2.4, 3.6, 8.7, 10.6])
V = np.array([128.0967, 113.6922, 98.7733, 66.8777, 60.70444])
CLgraph = W /(0.5 * V**2 * rho * S)

CDgraph = CD0 + (CLgraph * alpha) ** 2 / (pi * A * e)
print(CDgraph)

# xt = np.polyfit(alpha, CLgraph,1)
# yt = np.poly1d(xt)
# plt.grid()
# fig = plt.plot(xt,yt(xt))
# plt.show()