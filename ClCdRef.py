import numpy as np
from Cit_par import S, m, Tempgrad, Temp0, g, rho0, R, A
from ReadMeas import *
import matplotlib.pyplot as plt

p0 = 101325
gamma = 1.4

passmass = 0
for passenger in passlist:
    passmass = passmass + passenger.weight

def Vequi(hp,Vias,Tm):
    p =  p0*(1+Tempgrad*hp/Temp0)**(-g/(Tempgrad*R))
    M = np.sqrt( 2/(gamma-1) * ((1+p0/p*((1+ (gamma-1)/(2*gamma) * rho0/p0 * Vias**2)**(gamma/(gamma-1))-1))**((gamma-1)/gamma)-1))
    T = Tm/(1+(gamma-1)/2*M**2)
    a = np.sqrt(gamma*R*T)
    Vt = M*a
    rho = p/(R*T)
    Ve = Vt*np.sqrt(rho/rho0) 
    return Vt, rho

# thrust = open("Thrust//thrustCLCD1Meas.dat", "r")
# thrustarray = thrust.readlines()
# leftthrust = []
# rightthrust = []
# for i in thrustarray:
#     i = i.split('\t')
#     i[1] = i[1].replace('\n','')
#     leftthrust.append(float(i[0]))
#     rightthrust.append(float(i[1]))
#
# totalthrust = np.add(leftthrust,rightthrust)
# for i in range(len(CLCD1)):
#     CLCD1[i].thrust = totalthrust[i]

totalthrust = np.add(leftthrust,rightthrust)
for i in range(len(CLCD1)):
    CLCD1[i].thrust = totalthrust[i]

    
CLlist = []
CDlist = []
AOAlist = []
for i in CLCD1: 
    hp = i.height
    Vias = i.IAS
    Tm = float(i.TAT) + 273.15
#    type(Tm)
    Fused = i.Fused
    D = i.thrust
    
    mtot = m + passmass + fuelblock - Fused
    Ve = Vequi(hp,Vias,Tm)
    aero = 0.5*rho0*Ve**2*S
    Cl = mtot*g/aero
    Cd = D/aero
    
    CLlist.append(Cl)
    CDlist.append(Cd)
    AOAlist.append(float(i.AoA))

#plt.subplot(211)
#Am = np.vstack([AOAlist, np.ones(len(AOAlist))]).T
#a,b = np.linalg.lstsq(Am,CLlist,rcond=None)[0]
#plt.scatter(AOAlist,CLlist)
#plt.plot(AOAlist, np.array(AOAlist)*a + b)
#plt.ylabel("Lift coefficient [-]")
#plt.xlabel("Angle of Attack [deg]")
#plt.xlim(0,11)
#plt.ylim(0,1.3)
#
#plt.subplot(212)
#B = np.vstack([AOAlist, np.ones(len(AOAlist))]).T
#c,d = np.linalg.lstsq(B,CDlist,rcond=None)[0]
#plt.scatter(AOAlist,CDlist)
#plt.plot(AOAlist, np.array(AOAlist)*c + d)
#plt.ylabel("Drag coefficient [-]")
#plt.xlabel("Angle of Attack [deg]")
#plt.xlim(0,11)
#plt.ylim(0,0.1)
#plt.show()

# CL2 = np.array(CLlist)**2
# C = np.vstack([CL2, np.ones(len(CL2))]).T
# piAe,CD0 = np.linalg.lstsq(C,CDlist,rcond=None)[0]
#plt.scatter(CL2,CDlist)
#plt.plot(CL2, CL2*piAe + CD0)
#plt.xlabel('Lift coefficient squared [-]')
#plt.ylabel('Drag coefficient [-]')
#plt.plot()

# e = 1/(piAe*np.pi*A)
# print('CD0 = ', CD0,'  e = ',e)
#
# plt.scatter(CDlist,CLlist)
# plt.plot(CD0 + CL2/(np.pi*A*e), CLlist)
# plt.xlabel('Drag coefficient [-]')
# plt.ylabel('Lift coefficient [-]')
# plt.plot()
