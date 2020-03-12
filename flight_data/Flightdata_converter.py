from mat4py import loadmat
import csv

data = loadmat('matlab.mat')


AOA = []
for i in range(len(data["flightdata"]["vane_AOA"]['data'])):
    a = data["flightdata"]["vane_AOA"]['data'][:][i]
    AOA.append(a[0])

with open("AOA.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(AOA)
