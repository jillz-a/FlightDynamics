import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

time = np.array(pd.read_csv('flight_data/time.csv', delimiter=',', header=None))
time = np.array([time[i][0] for i in range(len(time))])
diff = [(time[i] - time[i - 1]) for i in range(1,len(time))]
ave_diff = np.average(diff)

#======================================initial values (non metric)======================================================
OEW = 9165 #pounds

Fuel_block = 4050 #pounds

x_s1 = 131
m_s1 = 150

x_s2 = 131
m_s2 = 150

x_s3 = 214
m_s3 = 150

x_s4 = 214
m_s4 = 150

x_s5 = 251
m_s5 = 150

x_s6 = 251
m_s6 = 150

x_s7 = 288
m_s7 = 150

x_s8 = 288
m_s8 = 150

x_s10 = 170
m_s10 = 150

m_payload = m_s1 + m_s2 + m_s2 + m_s3 + m_s4 + m_s5 + m_s6 + m_s7 + m_s8 + m_s10 #pounds
#initial values (metric)
Fuel_block = Fuel_block * 0.453592 #kg
OEW = OEW*0.453592 #kg

x_s1 = x_s1 * 0.0254 #m
m_s1 = m_s1 * 0.0254 #kg

x_s2 = x_s2 * 0.0254 #m
m_s2 = m_s2 * 0.453592 #kg

x_s3 = x_s3 * 0.0254 #m
m_s3 = m_s3 * 0.453592 #kg

x_s4 = x_s4 * 0.0254 #m
m_s4 = m_s4 * 0.453592 #kg

x_s5 = x_s5 * 0.0254 #m
m_s5 = m_s5 * 0.453592 #kg

x_s6 = x_s6 * 0.0254 #m
m_s6 = m_s6 * 0.453592 #kg

x_s7 = x_s7 * 0.0254 #m
m_s7 = m_s7 * 0.453592 #kg

x_s8 = x_s8 * 0.0254 #m
m_s8 = m_s8 * 0.453592 #kg

x_s10 = x_s10 * 0.0254 #m
m_s10 = m_s10 * 0.453592 #kg

m_payload = m_payload * 0.453592 #kg
#================================================Moment contributions==============================================================
#Empty mass contributions
M_empty = 2672953.5 * 0.453592 * 0.0254 #kgm
M_empty_t = np.ones(len(time)) * M_empty #kgm per time step

#Payload contribution
M_pay = x_s1 * m_s1 + x_s2 * m_s2 + x_s3 * m_s3 + x_s4 * m_s4 + x_s5 * m_s5 + x_s6 * m_s6 + x_s7 * m_s7 + x_s8 * m_s8 + x_s10 * m_s10 #kgm
M_pay_t = np.ones(len(time)) * M_pay

#Fuel contribution
#Data from weighing form
fuel_data = np.array(pd.read_csv('fuel_variation.csv', delimiter=',', header=None)) #moment arm/100 in inches for every 100 pounds

flow_eng1 = np.array(pd.read_csv('flight_data/FMF_eng1.csv', delimiter=',', header=None))
flow_eng1 = np.array([flow_eng1[i][0] for i in range(len(flow_eng1))]) #pounds per hour

flow_eng2 = np.array(pd.read_csv('flight_data/FMF_eng2.csv', delimiter=',', header=None))
flow_eng2= np.array([flow_eng2[i][0] for i in range(len(flow_eng2))]) #pounds per hour

FMF = flow_eng1 + flow_eng2 #total fuel mass flow in pounds per hour
FMF = FMF * (0.453592 / 3600.) #kg/s

m_fuel_t = [] #fuel mass per for every time step
for i in range(len(time)):
    Fuel_block = Fuel_block - FMF[i]*ave_diff #Fuel mass for every time step in kg, average was taken of difference due to outliers
    m_fuel_t.append(Fuel_block)

#Using moment arm data for fuel
fuel = fuel_data

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


M_fuel_t = []
for i in range(len(m_fuel_t)):
    f_fuel = interp1d(fuelx, fuely)
    M_fuel_i = f_fuel(m_fuel_t[i])
    M_fuel_t.append(M_fuel_i)

M_fuel_t = np.array(M_fuel_t) #moment of fuel for every time step in kgm

#Total Moment Contribution
M_total_t = M_empty_t + M_pay_t + M_fuel_t #total moment in kg

#Devide by weight to get the x_cg for every time step

OEW_t = np.ones(len(time))*OEW #OEW for every time step
m_payload_t = np.ones(len(time))*m_payload #Payload weight for every time step

# x_cg_t = M_total_t / (OEW_t + m_payload_t + m_fuel_t)

x_cg_t = np.divide(M_total_t, np.add(np.add(OEW_t, m_payload), m_fuel_t))

# plt.plot(time, M_fuel_t)
# plt.plot(time, M_empty_t)
# plt.plot(time, M_pay_t)
plt.plot(time, x_cg_t)
plt.show()
