import numpy as np
import pandas as pd
#======================================initial values (non metric)======================================================
Ramp_mass = 0 #pounds
x_cg_ramp = 0 #inches

Fuel_block = 4050 #pounds

x_s1 = 131
m_s1 = 0

x_s2 = 131
m_s2 = 0

x_s3 = 214
m_s3 = 0

x_s4 = 214
m_s4 = 0

x_s5 = 251
m_s5 = 0

x_s6 = 251
m_s6 = 0

x_s7 = 288
m_s7 = 0

x_s8 = 288
m_s8 = 0

x_s10 = 170
m_s10 = 0

#initial values (metric)
Ramp_mass = Ramp_mass*0.453592 #kg
x_cg_ramp = x_cg_ramp*0.0254 #m

x_s1 = x_s1 * 0.453592 #m
m_s1 = m_s1 * 0.0254 #kg

x_s2 = x_s2 * 0.453592 #m
m_s2 = m_s2 * 0.0254 #kg

x_s3 = x_s3 * 0.453592 #m
m_s3 = m_s3 * 0.0254 #kg

x_s4 = x_s4 * 0.453592 #m
m_s4 = m_s4 * 0.0254 #kg

x_s5 = x_s5 * 0.453592 #m
m_s5 = m_s5 * 0.0254 #kg

x_s6 = x_s6 * 0.453592 #m
m_s6 = m_s6 * 0.0254 #kg

x_s7 = x_s7 * 0.453592 #m
m_s7 = m_s7 * 0.0254 #kg

x_s8 = x_s8 * 0.453592 #m
m_s8 = m_s8 * 0.0254 #kg

x_s10 = x_s10 * 0.453592 #m
m_s10 = m_s10 * 0.0254 #kg

#================================================Moment contributions==============================================================
#Empty mass contributions
M_empty = 2672953.5 * 0.453592 * 0.0254 #Nm

#Payload contribution
M_pay = x_s1 * m_s1 + x_s2 * m_s2 + x_s3 * m_s3 + x_s4 * m_s4 + x_s5 * m_s5 + x_s6 * m_s6 + x_s7 * m_s7 + x_s8 * m_s8 + x_s10 * m_s10

#Fuel contribution
#Data from weighing form
fuel_data = np.array(pd.read_csv('fuel_variation.csv', delimiter=',', header=None))
time = pd.read_csv('flight_data/time.csv', delimiter=',', header=None)
time2 = time.values




#Steps to calculate ramp mass and x_cg_datum:
#Using table E.2, calculate aircraft empty mass (this is basic empty weight (no fuel no payload) + fuel weight)
#From this, when the fuel weight is known (this is given during the flight test) the basic empty weight can be calculated
#By filling in table E.1 for the person weights and baggage weights, so the payload, the zero fuel mass can be calculated
#Adding the fuel weight to this the ramp mass is calculated
#In between these steps all the x_cg's can be calculated
#Varying fuel mass is implemented later

