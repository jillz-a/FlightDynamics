#test
import numpy as np
from CGLoc import time, fuelx, fuely, fuel_t
import matplotlib.pyplot as plt
#vo

#lol

#ola
# 
M_fuel_i = np.interp(fuelx, fuely, 1300)

print(M_fuel_i)

plt.plot(fuelx, fuely)
plt.show()