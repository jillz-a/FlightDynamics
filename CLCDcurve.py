from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##READ DATA AND CREATE ARRAY##
AOA = np.array(pd.read_csv('flight_data/AOA.csv', delimiter=' ', header=None))
TAS = np.array(pd.read_csv('flight_data/TAS.csv', delimiter=' ', header=None))      #TAS in knots
TAS2 = TAS * 0.51444444444444
TAT = np.array(pd.read_csv('flight_data/TAT.csv', delimiter=' ', header=None))
Mach = np.array(pd.read_csv('flight_data/Mach.csv', delimiter=' ', header=None))
# alt = np.array(pd.read_csv('flight_data/alt.csv', delimiter=' ', header=None))
# bcAlt = np.array(pd.read_csv('flight_data/bcAlt.csv', delimiter=' ', header=None))
de = np.array(pd.read_csv('flight_data/delta_e.csv', delimiter=' ', header=None))

AT = np.column_stack([AOA,TAS2,de])
cut_off = 39
AT_trimmed = AT[AT[:,1] > cut_off]
print(AT_trimmed.shape)

##Calculate CL and CLalpha##
CLgraph = W/(0.5 * AT_trimmed[:,1]**2 * rho * S)
t, m = np.polyfit(AT_trimmed[:,0],CLgraph,1)
print('Cl_alpha =', t)

##Calculate CD##
CDgraph = CD0 + (CLgraph) ** 2 / (pi * A * e)

#Subplots##
plt.subplot(121)
plt.grid()
plt.scatter(AT_trimmed[:,0],CLgraph)
plt.plot(AT_trimmed[:,0],t*AT_trimmed[:,0]+m,c='red')
plt.title('CL-alpha curve')
# plt.show()
plt.subplot(122)
plt.grid()
plt.scatter(CDgraph,CLgraph)
plt.title('CD-CL curve')
plt.show()

##Calculate Reynolds Range with Sutherland Equation##
b = 1.458*10**(-6)  #kg/msK^1/2
S = 110.4 #K
T = TAT + 273.15
mu = (b * T ** (3/2))/(T + S)
print('Reynoldsnumber Range =', max(mu), min(mu))

##Cmalpha and Cmdelta Calculations##
deda, q = np.polyfit(AT_trimmed[:,0],-AT_trimmed[:,2],1)
line = deda*AT_trimmed[:,0]+q
print('deda =', -deda)
plt.grid()
plt.scatter(AT_trimmed[:,0],-AT_trimmed[:,2])
plt.plot(AT_trimmed[:,0],line, c='red')
plt.ylabel('-delta_e')
plt.xlabel('AOA')
plt.show()

# Cmdelta = - (1/de) * CLgraph * (dxcg/c)
# Cmalpha = - deda *Cmdelta