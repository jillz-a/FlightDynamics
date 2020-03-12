from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import csv
##READ DATA AND CREATE ARRAY##
variables = ['flightdata/AOA','flightdata/VTAS','flightdata/MACH',]
for i in variables:
    f = open(i,'r')
    i = []
    for line in f.readlines():
        i.append(float(line))
    i = np.array(i)

##Calculate CLalpha##
CLgraph = W/(0.5 * data2[1]**2 * rho * S)
t, m = np.polyfit(AOA,CLgraph,1)
print('Cl_alpha =', t*(180/pi))

##Calculate CD##
CDgraph = CD0 + (CLgraph) ** 2 / (pi * A * e)

#Subplots##
plt.subplot(121)
plt.grid()
plt.scatter(AOA,CLgraph)
plt.plot(AOA,t*AOA+m,c='red')
plt.title('CL-alpha curve')

plt.subplot(122)
plt.grid()
plt.scatter(CDgraph,CLgraph)
plt.title('CD-CL curve')

plt.show()

##Calculate Reynolds##


##COMMENTS##
# with open('AOA_VTAS.csv', encoding='UTF-8') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     alpha = []
#     V = []
#     for row in readCSV:
#         if float(row[0])>75:
#             alpha.append(float(row[1]))
#             V.append(float(row[0])*0.51444444444444) #convert knots to m/s
#
# data = np.array([alpha,V])
# data2 = data[np.argsort(data[:, 0])]