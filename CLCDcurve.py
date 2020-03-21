from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ReadMeas import *
from ClCdRef import passmass, Vequi

##READ DATA AND CREATE ARRAY##
time = np.array(pd.read_csv('flight_data/time.csv', delimiter=',', header=None))
time = np.array([time[i][0] for i in range(len(time))])
AOA1 = np.array(pd.read_csv('flight_data/AOA.csv', delimiter=' ', header=None))
TAS = np.array(pd.read_csv('flight_data/TAS.csv', delimiter=' ', header=None))      #TAS in knots
TAS2 = TAS * 0.51444444444444
TAT = np.array(pd.read_csv('flight_data/TAT.csv', delimiter=' ', header=None))
Mach = np.array(pd.read_csv('flight_data/Mach.csv', delimiter=' ', header=None))
de = np.array(pd.read_csv('flight_data/delta_e.csv', delimiter=' ', header=None))
xcg = np.array(pd.read_csv('x_cg.csv', delimiter=' ', header=None))
alt = np.array(pd.read_csv('flight_data/bcAlt.csv', delimiter=' ', header = None))
alt2 = alt * 0.3048   #ft to meters

AT = np.column_stack([AOA1,TAS2,de,xcg,alt2,TAT])
cut_off = 70
AT_trimmed = AT[AT[:,1] > cut_off]
# print(AT_trimmed.shape)

##Calculate CL and CLalpha##
AOA = AT_trimmed[:,0]
V = AT_trimmed[:,1]
h = AT_trimmed[:,4]
rho1 = rho0 * pow((1 + (Tempgrad*h)/Temp0),(-g/(R*Tempgrad) - 1))
CLgraph = W/(0.5 * V**2 * rho1 * S)
t, ma = np.polyfit(AOA,CLgraph,1)
CLline = t*AOA + ma
print('Cl_alpha =', t, t *(180/pi))
##Calculate CD##
CDgraph = CD0 + (CLline) ** 2 / (pi * A * e)

#Plots##
# plt.grid()
# scatter = plt.scatter(AOA,CLgraph,marker= '.', label='Measure point')
# line = plt.plot(AOA,CLline,c='red', label= 'Least squares')
# plt.title('CL-alpha curve')
# plt.legend()
# plt.show()

# plt.grid()
# plt.scatter(CDgraph,CLline)
# plt.title('CD-CL curve')
# plt.show()

##Calculate Reynolds Range with Sutherland Equation##
b = 1.458*10**(-6)  #kg/msK^1/2
S = 110.4 #K
T = AT_trimmed[:,5] + 273.15
mu = (b * T ** (3/2))/(T + S)
Reyn = np.array([(rho1[i] * V[i] * c/mu[i]) for i in range(len(mu))])
print('Reynoldsnumber Range =', max(Reyn), min(Reyn))

##Cmalpha and Cmdelta Calculations##
de = AT_trimmed[:,2]
deda, q = np.polyfit(AOA,de,1)
line = deda*AOA+q
print('deda =', deda)
# plt.grid()
# plt.scatter(AOA,de)
# plt.plot(AOA,line, c='red')
# plt.ylim(7,-7)
# plt.ylabel('-delta_e')
# plt.xlabel('AOA')
# plt.show()

##------------Calculate Cmdelta and Cmalpha using Post Flight Data-------------------------##
dde1 = [i.de for i in CGshift]
dde = (dde1[1] - dde1[0])*(pi/180)
dde2 = -0.15 *(pi/180)          #from first version data sheet excel
xcg = AT_trimmed[:,3]
dxcg = np.array([[xcg[i] - xcg[i-1]] for i in range(1,len(xcg))])
xcgd = min(dxcg)
hp = CGshift[1].height
Vias = CGshift[1].IAS
Tm = float(CGshift[1].TAT) + 273.15
VTAS, rhoTAS = Vequi(hp,Vias,Tm)[0:2]
Fused = CGshift[1].Fused
Weight = (m + passmass + fuelblock - Fused)*9.81
CN = Weight /(0.5*rhoTAS*(VTAS**2)*S)
print(CN)
Cmdelta = -(1/dde2) * CN * xcgd/c
Cmalpha = -deda * Cmdelta
print('Cmdelta =', Cmdelta)                 #ongeveer factor 2 te klein
print('Cmalpha =', Cmalpha)
####-------------------------Comments----------------------------------#####