from Cit_par import mass,rho0,Tempgrad,R,g,Temp0,S,c,A
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ReadMeas import *
from ClCdRef import passmass, Vequi, totalthrustele, totalthrustelestand, totalthrustele_mat, totalthrustele_matstand, CLalpha, b,e,CD0, AOAlist, CLlist,CL2

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
Fele = np.array(pd.read_csv('flight_data/Fele.csv', delimiter=' ', header = None))
SAT1 = np.array(pd.read_csv('flight_data/SAT.csv', delimiter=' ', header = None))
SAT = SAT1 + 273.15

##Calculate CL and CLalpha##
time_CL = time[16710:23911] #to check
AOA_CL = np.array(AOA1[16710:23911])
VTAS_CL = np.array(TAS2[16710:23911])
h_CL = np.array(alt2[16710:23911])
FU_CL = np.array(FUtot[16710:23911])
rho1_CL = rho0 * pow((1 + (Tempgrad*h_CL)/Temp0),(-g/(R*Tempgrad) - 1))
masstot = mass + passmass + fuelblock
Weight_CL = [(masstot - FU_CL[i])*g for i in range(len(FU_CL))]
CLgraph_mat = Weight_CL/(0.5 * VTAS_CL**2 * rho1_CL * S)

#find linear relation for CL measurements
clalpha_mat,ma_mat  = np.polyfit(AOA_CL[:,0],CLgraph_mat[:,0],1)
CLline_CL = clalpha_mat*AOA_CL[:,0] + ma_mat
print('Cl_alpha =', clalpha_mat, clalpha_mat*(180/pi))
##Calculate CD##
CDgraph_mat = CD0 + (CLline_CL) ** 2 / (pi * A * e)

#From Numerical Model#
AOAstat = np.array(AOAlist)
linecl_stat = CLalpha*AOAstat + b
CDstat = CD0 + linecl_stat/(pi * A * e)

#Plots CL and CD##
# plt.grid()
# plt.plot(AOAstat,linecl_stat, label='Stationary Flight Measurements')
# plt.plot(AOA_CL[:,0],CLline_CL,c='darkorange', label= 'Least Squares of Flightdata')
# plt.ylabel('Lift Coefficient [-]')
# plt.xlabel('Angle of Attack [deg]')
# plt.legend()
# # plt.savefig('CLalphacompare.jpg')
# plt.show()
#
# plt.grid()
# plt.scatter(CDgraph_mat,CLline_CL, marker='.', label='Measure Point Flightdata')
# plt.plot(CDstat,CLlist,c='orange', label='Stationary Flight Measurements')
# plt.ylabel('Lift Coefficient [-]')
# plt.xlabel('Drag Coefficient [-]')
# plt.legend()
# plt.savefig('CLCDcompare.jpg')
# plt.show()

##------------Reynolds Number Range-----------##
# b = 1.458*10**(-6)  #kg/msK^1/2
# St = 110.4 #K
# Tst = np.array(SAT[16710:23911])
# mu = (b * Tst ** (3/2))/(Tst + St)
# Reyn = np.array([(rho1_CL[i] * VTAS_CL[i] * c/mu[i]) for i in range(len(mu))])
# print('Reynoldsnumber Range =', max(Reyn), min(Reyn))

##_______________________________________Stationary Flight Data_________________________________________##

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

##--------------Elevator Trim Curve Ve-----------------##
height = np.array([i.height for i in EleTrimCurve])
V_ias = np.array([i.IAS for i in EleTrimCurve])
Temp = np.array([(i.TAT + 273.15) for i in EleTrimCurve])
# Vtasele = Vequi(height,V_ias,Temp)[0]
# rhoele = Vequi(height,V_ias,Temp)[1]
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
Tc = totalthrustele/(0.5*rho0*Ve_e**2*S)
print(Tc)
Tcs = totalthrustelestand/(0.5*rho0*Ve_e**2*d_eng**2)
print(Tcs)
deleq = eledefl - (1/Cmdelta *Cmtc * (Tcs - Tc))
##-------Plotting AoA against Ele delfection and determine Cmalpha------##
deda, q = np.polyfit(aoa,deleq,1)
line = deda*aoa+q
print('deda =', deda)
# plt.grid()
# plt.scatter(aoa,deleq, label='Measure Point')
# plt.plot(aoa,line, c='orange', label='Least Squares')
# plt.ylim(1.2,-0.5)
# plt.ylabel('Reduced Elevator Deflection [deg]')
# plt.xlabel('Angle of Attack [deg]')
# plt.legend()
# plt.savefig('DedAOA.jpg')
# plt.show()

Cmalpha = -deda * Cmdelta
print('Cmalpha =', Cmalpha)

#-------------------Plotting Ele defl against Ve----##
Ve_e_dde1 = np.column_stack([Ve_e,deleq])
Ve_e_dde = Ve_e_dde1[Ve_e_dde1[:,0].argsort()]
d, f, j = np.polyfit(Ve_e_dde[:,0],Ve_e_dde[:,1],2)
line_eleV = d*Ve_e_dde[:,0]**2 + f*Ve_e_dde[:,0] + j
# plt.grid()
# plt.scatter(Ve_e_dde[:,0],Ve_e_dde[:,1], label='Measure Point')
# plt.plot(Ve_e_dde[:,0],d*Ve_e_dde[:,0]**2 + f*Ve_e_dde[:,0] + j, c='orange', label='Least Squares')
# plt.ylim(1.2,-0.4)
# plt.ylabel('Reduced Elevator Deflection [deg]')
# plt.xlabel('Reduced Equivalent Airspeed [m/s]')
# plt.legend()
# plt.savefig('DedV.jpg')
# plt.show()

##------------Reduced Elevator control Curve----------##
Femea = np.array([i.Fe for i in EleTrimCurve])
Fe = Femea * (Ws/Wele)
Ve_e_Fe1 = np.column_stack([Ve_e,Fe])
Ve_e_Fe = Ve_e_Fe1[Ve_e_Fe1[:,0].argsort()]
d, f, j = np.polyfit(Ve_e_Fe[:,0],Ve_e_Fe[:,1],2)
line_feele = d*Ve_e_Fe[:,0]**2 + f*Ve_e_Fe[:,0] + j
# plt.grid()
# plt.scatter(Ve_e_Fe[:,0],Ve_e_Fe[:,1], label='Measure Point')
# plt.plot(Ve_e_Fe[:,0],d*Ve_e_Fe[:,0]**2 + f*Ve_e_Fe[:,0] + j, c='orange', label='Least Squares')
# plt.ylim(70,-40)
# plt.ylabel('Reduced Force on Elevator Control Wheel [N]')
# plt.xlabel('Reduced Equivalent Speed [m/s]')
# plt.legend()
# plt.savefig('FeV.jpg')
# plt.show()

##_______________________________________Flight test DATA_______________________________________##

##---------------------Cmdelta determination of matlab data-----------------------------------##
time_cg = time[33510:35911]
xcg_cg = np.array(xcg[33510:35911])
dxcg_cg1 = np.array([xcg_cg[i] - xcg_cg[i-1] for i in range(1,len(xcg_cg))])
dxcg_cg = min(dxcg_cg1)
de_cg = np.array(de[33510:35911])
dde_cg = (de_cg[2000] - de_cg[399]) * pi/180    #determined by exact time of interval stationary data
FUtot_cg = FUtot[33510:35911]
index = np.where(dxcg_cg1 == np.amin(dxcg_cg1))
W_cg = (masstot - FUtot_cg[2000])*g
h_cg = alt2[35512]
rho_cg = rho0 * pow((1 + (Tempgrad*h_cg)/Temp0),(-g/(R*Tempgrad) - 1))
Vtas_cg = TAS2[35512]
CN_cg = W_cg/(0.5*rho_cg*Vtas_cg**2*S)
Cmdelta_mat = -(1/dde_cg) * CN_cg * (dxcg_cg/c)
print('Cmdelta matlab =', Cmdelta_mat)

##-------------------------------Elevator Trim Curve Of Matlab Data------------------##
time_ele = time[29910:33511]
AOA_ele = np.array(AOA1[29910:33511])
de_ele = np.array(de[29910:33511])
Vtas_ele = np.array(TAS2[29910:33511])
h_ele = np.array(alt2[29910:33511])
rho_ele = rho0 * pow((1 + (Tempgrad*h_ele)/Temp0),(-g/(R*Tempgrad) - 1))
Ve_ele = Vtas_ele * np.sqrt(rho_ele/rho0)
FUtot_ele = np.array(FUtot[29910:33511])
W_ele = np.array([(masstot-FUtot_ele[i])*g for i in range(len(FUtot_ele))])
Ve_graph = Ve_ele * np.sqrt(Ws/W_ele)
# print(len(Ve_graph))
Tc = totalthrustele_mat/(0.5*rho0*Ve_graph**2*S)
Tcs = totalthrustele_matstand/(0.5*rho0*Ve_graph**2*d_eng**2)
de_elemat = de_ele - (1/Cmdelta_mat * Cmtc) * (Tcs - Tc)
# print(len(de_elemat))

# f , k, v = np.polyfit(Ve_graph[:,0],de_elemat[:,0],2)
# plt.grid()
# plt.scatter(Ve_graph[:,0],de_elemat[:,0], marker='.', label='Measure Point')
# plt.plot(Ve_graph[:,0], f*Ve_graph[:,0]**2 + k *Ve_graph[:,0] + v, c='orange', label='Least Squares')
# plt.plot(Ve_graph[:,0], f*Ve_graph[:,0]**2 + k *Ve_graph[:,0] + v, c='orange', label='Least Squares of Flightdata')
# plt.plot(Ve_e_dde[:,0],line_eleV, label='Stationary Flight Measurements')
# plt.ylim(1.25,-0.7)
# plt.ylabel('Reduced Elevator Deflection [deg]')
# plt.xlabel('Reduced Equivalent Airspeed [m/s]')
# plt.legend()
# plt.savefig('DedVcompare.jpg')
# plt.show()

deda_mat, b_mat = np.polyfit(AOA_ele[:,0], de_elemat[:,0],1)
# plt.grid()
# # plt.scatter(AOA_ele[:,0],de_elemat[:,0],marker='.', label='Measure Point')
# plt.plot(aoa,line, label='Stationary Flight Measurements')
# plt.plot(AOA_ele[:,0], deda_mat*AOA_ele[:,0] + b_mat, c='orange', label='Least Squares of Flightdata')
# # plt.plot(AOA_ele[:,0], deda_mat*AOA_ele[:,0] + b_mat, c='orange', label='Least Squares')
# plt.ylim(1.25,-0.7)
# plt.ylabel('Reduced Elevator Deflection [deg]')
# plt.xlabel('Angle of Attack [deg]')
# plt.legend()
# plt.savefig('DedAOAcompare.jpg')
# plt.show()

Cmalpha_mat = -deda_mat * Cmdelta_mat
print('Cmalpha matlab =', Cmalpha_mat)

Femea_mat = np.array(Fele[29910:33511])
Fele_mat = Femea_mat * Ws/W_ele
# w , s, v = np.polyfit(Ve_graph[:,0],Fele_mat[:,0],2)
# plt.grid()
# # plt.scatter(Ve_graph[:,0],Fele_mat[:,0], marker='.', label='Measure Point')
# # plt.plot(Ve_graph[:,0],w*Ve_graph[:,0]**2 + s * Ve_graph[:,0] + v, c='orange', label='Least Squares')
# plt.plot(Ve_graph[:,0],w*Ve_graph[:,0]**2 + s * Ve_graph[:,0] + v, c='orange', label='Least Squares of Flightdata')
# plt.plot(Ve_e_Fe[:,0],line_feele, label='Stationary Flight Measurements')
# plt.ylim(70,-50)
# plt.ylabel('Reduced Force on Elevator Control Wheel [N]')
# plt.xlabel('Reduced Equivalent Airspeed [m/s]')
# plt.legend()
# plt.savefig('FeVcompare.jpg')
# plt.show()

####-------------------------Old versions----------------------------------#####
# AT = np.column_stack([AOA1,TAS2,de,xcg,alt2,TAT,FUtot])
# cut_off = 70
# AT_trimmed = AT[AT[:,1] > cut_off]
# print(AT_trimmed.shape)

##Calculate CL and CLalpha##
# AOA = AT_trimmed[:,0]
# V = AT_trimmed[:,1]
# h = AT_trimmed[:,4]
# FU = AT_trimmed[:,6]
# rho1 = rho0 * pow((1 + (Tempgrad*h)/Temp0),(-g/(R*Tempgrad) - 1))
# masstot = mass + passmass + fuelblock
# Weight = [(masstot - FU[i])*g for i in range(len(FU))]
# CLgraph = Weight/(0.5 * V**2 * rho1 * S)
# t, ma = np.polyfit(AOA,CLgraph,1)
# CLline = t*AOA + ma
# print('Cl_alpha =', t, t*(180/pi))
# ##Calculate CD##
# CDgraph = CD0 + (CLline) ** 2 / (pi * A * e)

#Plots CL and CD##
# plt.grid()
# scatter = plt.scatter(AOA,CLgraph,marker= '.', label='Measure point')
# line = plt.plot(AOA,CLline,c='darkorange', label= 'Least squares')
# plt.ylabel('Lift Coefficient [-]')
# plt.xlabel('Angle of Attack [deg]')
# plt.legend()
# plt.show()
#
# plt.grid()
# plt.scatter(CDgraph,CLline)
# plt.ylabel('Lift Coefficient [-]')
# plt.xlabel('Drag Coefficient [-]')
# plt.show()

##Calculate Reynolds Range with Sutherland Equation##
# b = 1.458*10**(-6)  #kg/msK^1/2
# St = 110.4 #K
# T = AT_trimmed[:,5] + 273.15
# mu = (b * T ** (3/2))/(T + St)
# Reyn = np.array([(rho1[i] * V[i] * c/mu[i]) for i in range(len(mu))])
# print('Reynoldsnumber Range =', max(Reyn), min(Reyn))
