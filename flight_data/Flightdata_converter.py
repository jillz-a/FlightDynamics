from mat4py import loadmat
import csv
import numpy as np

data = loadmat('matlab.mat')
l = len(data["flightdata"]["vane_AOA"]['data'])

AOA = []
for i in range(l):
    a = data["flightdata"]["vane_AOA"]['data'][:][i]
    AOA.append(a[0])

np.savetxt("AOA.csv", AOA, delimiter=",")
#
# elevator_def = []
# for i in range(l):
#     a = data["flightdata"]["elevator_dte"]['data'][:][i]
#     elevator_def.append(a[0])
#
# np.savetxt("elevator_def", elevator_def, delimiter=",")
#
#
# FMF_eng1 = []
# for i in range(l):
#     a = data["flightdata"]["lh_engine_FMF"]['data'][:][i]
#     FMF_eng1.append(a[0])
#
# np.savetxt("FMF_eng1", FMF_eng1, delimiter=",")

TAS = []
for i in range(l):
    a = data["flightdata"]["Dadc1_tas"]['data'][:][i]
    TAS.append(a[0])

np.savetxt("TAS.csv", TAS, delimiter=",")

Mach = []
for i in range(l):
    a = data["flightdata"]["Dadc1_mach"]['data'][:][i]
    Mach.append(a[0])

np.savetxt("Mach.csv", Mach, delimiter=",")
#
TAT = []
for i in range(l):
    a = data["flightdata"]["Dadc1_tat"]['data'][:][i]
    TAT.append(a[0])

np.savetxt("TAT.csv", TAT, delimiter=",")

bcAlt = []
for i in range(l):
    a = data["flightdata"]["Dadc1_bcAlt"]['data'][:][i]
    bcAlt.append(a[0])

np.savetxt("bcAlt.csv", bcAlt, delimiter=",")
#
alt = []
for i in range(l):
    a = data["flightdata"]["Dadc1_alt"]['data'][:][i]
    alt.append(a[0])

np.savetxt("alt.csv", alt, delimiter=",")

#test
#TAS, MACH, TAT, bcALT, alt



