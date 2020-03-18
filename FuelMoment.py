import CGLoc as cg
import numpy as np

fuel =  cg.fuel_data

#splitting up the array into x and y arrays
#make empty arrays
fuelx = np.zeros(len(fuel)) #weight, in pounds
fuely = np.zeros(len(fuel)) #moment, in pounds-inch

#replace entries with fuel and moment
for i in range(len(fuel)):
    fuelx[i] = fuel[i,0]
#from pounds to kg
fuelx = fuelx * 0.453592

for i in range(len(fuel)):
    fuely[i] = fuel[i,1] * 100
#from pounds-inch to kg-m
fuely = fuely * 0.0254 * 0.453592

#set up points where to interpolate
x = np.arange(fuelx[0],fuelx[49],5)


fuel_int = np.zeros(len(x))
for i in range(len(x)):
    fuel_int[i] = np.interp(x[i],fuelx,fuely)
print(fuel_int) #moment of the fuel, in kg-m

