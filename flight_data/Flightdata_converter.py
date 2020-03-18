from mat4py import loadmat
import csv
import numpy as np

data = loadmat('Flightdata.mat')
l = len(data["flightdata"]["vane_AOA"]['data'])

# AOA = []
# for i in range(l):
#     a = data["flightdata"]["vane_AOA"]['data'][:][i]
#     AOA.append(a[0])
#
# np.savetxt("AOA.csv", AOA, delimiter=",")
#
elevator_def = []
for i in range(l):
    a = data["flightdata"]["delta_e"]['data'][:][i]
    elevator_def.append(a[0])

np.savetxt("elevator_def", elevator_def, delimiter=",")

elevator_trim = []
for i in range(l):
    a = data["flightdata"]["elevator_dte"]['data'][:][i]
    elevator_def.append(a[0])

#
np.savetxt("elevator_trim", elevator_trim, delimiter=",")
#
#
# FMF_eng2 = []
# for i in range(l):
#     a = data["flightdata"]["rh_engine_FMF"]['data'][:][i]
#     FMF_eng2.append(a[0])
#
# np.savetxt("FMF_eng2.csv", FMF_eng2, delimiter=",")
#
# FMF_eng1 = []
# for i in range(l):
#     a = data["flightdata"]["lh_engine_FMF"]['data'][:][i]
#     FMF_eng1.append(a[0])
#
# np.savetxt("FMF_eng1.csv", FMF_eng1, delimiter=",")

# TAS = []
# for i in range(l):
#     a = data["flightdata"]["Dadc1_tas"]['data'][:][i]
#     TAS.append(a[0])
#
# np.savetxt("TAS.csv", TAS, delimiter=",")
#
# Mach = []
# for i in range(l):
#     a = data["flightdata"]["Dadc1_mach"]['data'][:][i]
#     Mach.append(a[0])
#
# np.savetxt("Mach.csv", Mach, delimiter=",")
# #
# TAT = []
# for i in range(l):
#     a = data["flightdata"]["Dadc1_tat"]['data'][:][i]
#     TAT.append(a[0])
#
# np.savetxt("TAT.csv", TAT, delimiter=",")
#
# bcAlt = []
# for i in range(l):
#     a = data["flightdata"]["Dadc1_bcAlt"]['data'][:][i]
#     bcAlt.append(a[0])
#
# np.savetxt("bcAlt.csv", bcAlt, delimiter=",")
# #
# alt = []
# for i in range(l):
#     a = data["flightdata"]["Dadc1_alt"]['data'][:][i]
#     alt.append(a[0])
#
# np.savetxt("alt.csv", alt, delimiter=",")

# time = []
# for i in range(l):
#     a = data["flightdata"]["time"]['data'][:][i]
#     b = []
#     b.append(a)
#     time.append(b[0])
#
#
# np.savetxt("time.csv", time, delimiter=",")

#test
#TAS, MACH, TAT, bcALT, alt


# delta_e = []
# for i in range(l):
#     a = data["flightdata"]["delta_e"]['data'][:][i]
#     delta_e.append(a[0])
#
#np.savetxt("delta_e.csv", delta_e, delimiter=",")
