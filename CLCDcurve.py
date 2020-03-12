from Cit_par import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##READ DATA AND CREATE ARRAY##
AOA = np.array(pd.read_csv('flight_data/AOA.csv', delimiter=' ', header=None))
TAS = np.array(pd.read_csv('flight_data/TAS.csv', delimiter=' ', header=None))      #TAS in knots
TAS = TAS * 0.51444444444444
TAT = np.array(pd.read_csv('flight_data/TAT.csv', delimiter=' ', header=None))
Mach = np.array(pd.read_csv('flight_data/Mach.csv', delimiter=' ', header=None))
alt = np.array(pd.read_csv('flight_data/alt.csv', delimiter=' ', header=None))
bcAlt = np.array(pd.read_csv('flight_data/bcAlt.csv', delimiter=' ', header=None))

AT = np.column_stack([AOA,TAS])
# AT = [[[AOA[x],TAS[x]] for x in AOA if TAS[x] > 39] for x in TAS]     #filter the outliers for the plots
AT = [AT[i,:] for i in AT if AT[i,1] > 39.00]
print(AT)


##Calculate CLalpha##
CLgraph = W/(0.5 * AT[1]**2 * rho * S)
t, m = np.polyfit(AT[0],CLgraph,1)
print('Cl_alpha =', t*(180/pi))

##Calculate CD##
CDgraph = CD0 + (CLgraph) ** 2 / (pi * A * e)

#Subplots##
plt.subplot(121)
plt.grid()
plt.scatter(AT[0],CLgraph)
plt.plot(AT[0],t*AT[0]+m,c='red')
plt.title('CL-alpha curve')

plt.subplot(122)
plt.grid()
plt.scatter(CDgraph,CLgraph)
plt.title('CD-CL curve')

plt.show()

##Calculate Reynolds##

##COMMENTS##
    # alpha = []
    # V = []
    # for row in readCSV:
    #     if float(row[0])>75:
    #         alpha.append(float(row[1]))
    #         V.append(float(row[0])*0.51444444444444) #convert knots to m/s
