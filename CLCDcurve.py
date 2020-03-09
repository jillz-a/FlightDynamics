from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import csv

with open('AOA_VTAS.csv', encoding='UTF-8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    alpha = []
    V = []
    for row in readCSV:
        alpha.append(float(row[0]))
        V.append(float(row[1])*0.51444444444444) #convert knots to m/s
AOA = np.array(alpha)
Vtas = np.array(V)

# alpha =np.array([1.7, 2.4, 3.6, 8.7, 10.6])
# V = np.array([128.0967, 113.6922, 98.7733, 66.8777, 60.70444])
CLgraph = W /(0.5 * Vtas**2 * rho * S)
CDgraph = CD0 + (CLgraph * AOA * (pi/180)) ** 2 / (pi * A * e)
print(np.argmax(CLgraph),max(CLgraph))
print(np.argmin(Vtas),min(Vtas))

# def test_func(x,a,b):
#     return a*x**4 + b
# params, params_covariance = optimize.curve_fit(test_func, alpha, CDgraph)

plt.grid()
plt.scatter(AOA,CLgraph)
plt.show()
# # xt = np.polyfit(alpha, CLgraph,1)
# # yt = np.poly1d(xt)
# # fig = plt.plot(xt,yt(xt))
# # plt.plot(alpha,CDgraph)
# plt.scatter(alpha,CDgraph)
# plt.plot(alpha,test_func(alpha,params[0],params[1]))
# plt.ylim(0.0395,0.0418)
# plt.show()

