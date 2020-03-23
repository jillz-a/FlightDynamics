# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:25:49 2020

@author: daanv
"""

from ReadMeas import *
from Cit_par import Tempgrad, Temp0, g, R, rho0
import numpy as np 

p0 = 101325
gamma = 1.4


matlab = open("matlab1.dat", "w")
for i in CLCD1:

    hp = i.height
    Vias = i.IAS
    Tempm = float(i.TAT) + 273.15
    FFl = i.FFl/3600
    FFr = i.FFr/3600

    p =  p0*(1+Tempgrad*hp/Temp0)**(-g/(Tempgrad*R))
    M = np.sqrt( 2/(gamma-1) * ((1+p0/p*((1+ (gamma-1)/(2*gamma) * rho0/p0 * Vias**2)**(gamma/(gamma-1))-1))**((gamma-1)/gamma)-1))
    Tstat = Tempm/(1+(gamma-1)/2*M**2)
    TempISA = Temp0 + Tempgrad*hp
    Tempdiff = Tstat -TempISA

    hp = round(hp, 2)
    M = round(M,6)
    Tempdiff = round(Tempdiff, 5)
    FFl = round(FFl, 3)
    FFr = round(FFr, 3)    
    
    print(hp, M, Tempdiff, FFl, FFr)
    matlab.write(str(hp) + " " + str(M) + " " + str(Tempdiff) + " " + str(FFl) + " " + str(FFr) + "\n") 
matlab.close()

