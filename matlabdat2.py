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
FMFL2 = np.ones([3601,1])*0.048
FMFR2 = np.ones([3601,1])*0.048
trust = np.column_stack([alt[29910:33511],Mach[29910:33511],T[29910:33511],FMFL2,FMFR2])

matlab = open("matlab3.dat", "w")
for i in range(len(trust)):
    hp = round(trust[i,0],0)
    M = round(trust[i,1],3)
    T = round(trust[i,2],3)
    FFl = round(0.048,3)
    FFr = round(0.048,3)
    matlab.write(str(hp) + " " + str(M) + " " + str(T) + " " + str(FFl) + " " + str(FFr) + "\n")
matlab.close()