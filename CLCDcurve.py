from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ReadMeas import *

##READ DATA AND CREATE ARRAY##
time = np.array(pd.read_csv('flight_data/time.csv', delimiter=',', header=None))
time = np.array([time[i][0] for i in range(len(time))])
AOA1 = np.array(pd.read_csv('flight_data/AOA.csv', delimiter=' ', header=None))
TAS = np.array(pd.read_csv('flight_data/TAS.csv', delimiter=' ', header=None))      #TAS in knots
TAS2 = TAS * 0.51444444444444
TAT = np.array(pd.read_csv('flight_data/TAT.csv', delimiter=' ', header=None))
Mach = np.array(pd.read_csv('flight_data/Mach.csv', delimiter=' ', header=None))
# alt = np.array(pd.read_csv('flight_data/alt.csv', delimiter=' ', header=None))
# bcAlt = np.array(pd.read_csv('flight_data/bcAlt.csv', delimiter=' ', header=None))
de = np.array(pd.read_csv('flight_data/delta_e.csv', delimiter=' ', header=None))
xcg = np.array(pd.read_csv('x_cg.csv', delimiter=' ', header=None))

AT = np.column_stack([AOA1,TAS2,de,xcg])
cut_off = 10
AT_trimmed = AT[AT[:,1] > cut_off]
print(AT_trimmed.shape)

##Calculate CL and CLalpha##
AOA = AT_trimmed[:,0]
V = AT_trimmed[:,1]
CLgraph = W/(0.5 * V**2 * rho * S)
t, m = np.polyfit(AOA,CLgraph,1)
CLline = t*AOA + m
print('Cl_alpha =', t)

##Calculate CD##
CDgraph = CD0 + (CLgraph) ** 2 / (pi * A * e)

#Plots##
# plt.grid()
# plt.scatter(AOA,CLgraph)
# plt.plot(AOA,t*AOA+m,c='red')
# plt.title('CL-alpha curve')
# plt.show()

# plt.grid()
# plt.scatter(CDgraph,CLgraph)
# plt.title('CD-CL curve')
# plt.show()

##Calculate Reynolds Range with Sutherland Equation##
b = 1.458*10**(-6)  #kg/msK^1/2
S = 110.4 #K
T = TAT + 273.15
mu = (b * T ** (3/2))/(T + S)
print('Reynoldsnumber Range =', max(mu), min(mu))

##Cmalpha and Cmdelta Calculations##
de = AT_trimmed[:,2]
deda, q = np.polyfit(AOA,de,1)
line = deda*AOA+q
print('deda =', deda)
plt.grid()
plt.scatter(AOA,de)
plt.plot(AOA,line, c='red')
plt.ylim(7,-7)
plt.ylabel('-delta_e')
plt.xlabel('AOA')
plt.show()

dde = [i.de for i in CGshift]
print(dde)
xcg = AT_trimmed[:,3]
dxcg = np.array([[xcg[i] - xcg[i-1]] for i in range(1,len(xcg))])
xcgd = min(dxcg)
# Cmdelta = -(1/dde) *CLline * xcgd/c
# Cmalpha = -deda * Cmdelta
# print('Cmdelta =', np.average(Cmdelta))
# print('Cmalpha =', np.average(Cmalpha))

####-------------------------Comments----------------------------------#####

# dde = np.array([[de[i] - de[i-1]] for i in range(1,len(de))])
# dde = np.vstack([dde,0.0001])
# for i in range(len(dde)):
#     if dde[i] <= 0.000001 and dde[i] >= -0.000001:
#         dde[i] = 0.001
#     else:
#         dde[i] = dde[i]
# print(dde, len(dde))
# Cmdelta = np.array([(1/dde[i]) * CLline[i] * (dxcg[i]/c) for i in range(len(dde))])