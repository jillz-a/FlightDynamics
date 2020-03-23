from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ReadMeas import *
from ClCdRef import passmass, Vequi, totalthrustele, totalthrustelestand

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
shiftxcg = np.array(pd.read_csv('cg_shift.csv', delimiter=' ', header=None))
alt = np.array(pd.read_csv('flight_data/bcAlt.csv', delimiter=' ', header = None))
alt2 = alt * 0.3048   #ft to meters
FUl = np.array(pd.read_csv('flight_data/FUl.csv', delimiter=' ', header = None))
FUr = np.array(pd.read_csv('flight_data/FUr.csv', delimiter=' ', header = None))
FUtot = (FUl + FUr) * 0.453592 #lbs to kg

AT = np.column_stack([AOA1,TAS2,de,xcg,alt2,TAT,FUtot])
cut_off = 70
AT_trimmed = AT[AT[:,1] > cut_off]
# print(AT_trimmed.shape)

##Calculate CL and CLalpha##
AOA = AT_trimmed[:,0]
V = AT_trimmed[:,1]
h = AT_trimmed[:,4]
FU = AT_trimmed[:,6]
rho1 = rho0 * pow((1 + (Tempgrad*h)/Temp0),(-g/(R*Tempgrad) - 1))
masstot = mass + passmass + fuelblock
Weight = [(masstot - FU[i])*g for i in range(len(FU))]
CLgraph = Weight/(0.5 * V**2 * rho1 * S)
t, ma = np.polyfit(AOA,CLgraph,1)
CLline = t*AOA + ma
print('Cl_alpha =', t, t*(180/pi))
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
St = 110.4 #K
T = AT_trimmed[:,5] + 273.15
mu = (b * T ** (3/2))/(T + St)
Reyn = np.array([(rho1[i] * V[i] * c/mu[i]) for i in range(len(mu))])
print('Reynoldsnumber Range =', max(Reyn), min(Reyn))

##------------Calculate Cmdelta and Cmalpha using Post Flight Data-------------------------##
dde1 = [i.de for i in CGshift]
dde = (dde1[1] - dde1[0])*(pi/180)
dxcg = shiftxcg[1]-shiftxcg[0]
hp = CGshift[1].height
Vias = CGshift[1].IAS
Tm = float(CGshift[1].TAT) + 273.15
VTAS = Vequi(hp,Vias,Tm)[0]
rhoTAS = Vequi(hp,Vias,Tm)[1]
Fused = CGshift[1].Fused
Weight = (mass + passmass + fuelblock - Fused)*g
CN = Weight/(0.5*rhoTAS*(VTAS**2)*S)
print('CN =', CN)
Cmdelta = -(1/dde) * CN * dxcg/c
print('Cmdelta =', Cmdelta)

##_______________________________________Stationary Flight Data_________________________________________##

##--------------Elevator Trim Curve Ve-----------------##
height = np.array([i.height for i in EleTrimCurve])
V_ias = np.array([i.IAS for i in EleTrimCurve])
Temp = np.array([(i.TAT + 273.15) for i in EleTrimCurve])
Vtasele = Vequi(height,V_ias,Temp)[0]
rhoele = Vequi(height,V_ias,Temp)[1]
V_e = Vequi(height,V_ias,Temp)[2]
Fusedele = np.array([i.Fused for i in EleTrimCurve])
mtot_el = mass + passmass + fuelblock - Fusedele
Wele = mtot_el * g
Ws = 60500 #N
Ve_e = V_e * np.sqrt(Ws/Wele)

##-----------Elevator Trim Curve Ele defl eq-----------##
Cmtc = -.0064  #reader appendix
eledefl = np.array([i.de for i in EleTrimCurve])
aoa = np.array([i.AoA for i in EleTrimCurve])
d_eng = 0.686 #m
Tc = totalthrustele/(0.5*rhoele*Ve_e**2*S)
Tcs = totalthrustelestand/(0.5*rhoele*Ve_e**2*d_eng**2)
deleq = eledefl - (1/Cmdelta *Cmtc * (Tcs - Tc))

##-------Plotting AoA against Ele delfection and determine Cmalpha------##
deda, q = np.polyfit(aoa,deleq,1)
line = deda*aoa+q
print('deda =', deda)
plt.grid()
plt.scatter(aoa,deleq)
plt.plot(aoa,line, c='red')
plt.ylim(2,-2)
plt.ylabel('-delta_e')
plt.xlabel('aoa')
plt.show()

Cmalpha = -deda * Cmdelta
print('Cmalpha =', Cmalpha)

#-------------------Plotting Ele defl against Ve----##
plt.grid()
plt.scatter(Ve_e,deleq)
plt.ylim(1.5,-1)
plt.ylabel('-delta_e')
plt.xlabel('Ve_e')
plt.show()

##------------Reduced Elevator control Curve----------##
Femea = np.array([i.Fe for i in EleTrimCurve])
Fe = Femea * (Ws/Wele)
plt.grid()
plt.scatter(Ve_e,Fe)
plt.ylim(70,-40)
plt.ylabel('-Fe')
plt.xlabel('Ve_e')
plt.show()

##_______________________________________Flight test DATA_______________________________________##
time_ele = time[29910:32911]
# print(time_ele)
AOA_ele = AOA1[29910:32911]
de_ele = de[29910:32911]


####-------------------------Comments----------------------------------#####
# xcg = AT_trimmed[:,3]
# dxcg = np.array([[xcg[i] - xcg[i-1]] for i in range(1,len(xcg))])
# xcgd = min(dxcg)