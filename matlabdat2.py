from Cit_par import Temp0
import numpy as np
import pandas as pd

p0 = 101325
gamma = 1.4

Mach = np.array(pd.read_csv('flight_data/Mach.csv', delimiter=' ', header=None))
SAT = np.array(pd.read_csv('flight_data/SAT.csv', delimiter=' ', header=None))
alt = np.array(pd.read_csv('flight_data/alt.csv', delimiter=' ', header= None)) * 0.3048 #meters
FMFL = np.array(pd.read_csv('flight_data/FMF_eng1.csv', delimiter=',', header=None)) * 0.000125998 #1 kg/s
FMFR = np.array(pd.read_csv('flight_data/FMF_eng2.csv', delimiter=',', header=None)) * 0.000125998 #2 kg/s
T = [((SAT[i] + 273.15) - Temp0) for i in range(len(SAT))]

trust = np.column_stack([alt,Mach,T,FMFL,FMFR])

matlab = open("matlab.dat", "w")
for i in range(len(trust)):
    hp = round(trust[i,0],3)
    M = round(trust[i,1],3)
    T = round(trust[i,2],3)
    FFl = round(trust[i,3],3)
    FFr = round(trust[i,4],3)
    matlab.write(str(hp) + " " + str(M) + " " + str(T) + " " + str(FFl) + " " + str(FFr) + "\n")
matlab.close()