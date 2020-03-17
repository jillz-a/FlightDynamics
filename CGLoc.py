import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#import data
time = np.array(pd.read_csv('flight_data/time.csv', delimiter=',', header=None)) #sec

fuel_data = np.array(pd.read_csv('fuel_variation.csv', delimiter=',', header=None)) #moment arm/100 in inches for every 100 pounds

flow_eng1 = np.array(pd.read_csv('flight_data/FMF_eng1.csv', delimiter=',', header=None)) #pounds/hour

flow_eng2 = np.array(pd.read_csv('flight_data/FMF_eng2.csv', delimiter=',', header=None)) #pounds / hour

def x_cg(time, fuel_data, flow_eng1, flow_eng2):
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

    x_s1 = x_s1 * 0.0254 #m pilot 1
    m_s1 = m_s1 * 0.0254 #kg

    x_s2 = x_s2 * 0.0254 #m pilot 2
    m_s2 = m_s2 * 0.453592 #kg

    x_s3 = x_s3 * 0.0254 #m observer 1L
    m_s3 = m_s3 * 0.453592 #kg

    x_s4 = x_s4 * 0.0254 #m observer 1R
    m_s4 = m_s4 * 0.453592 #kg

    x_s5 = x_s5 * 0.0254 #m observer 2L
    m_s5 = m_s5 * 0.453592 #kg

    x_s6 = x_s6 * 0.0254 #m observer 2R
    m_s6 = m_s6 * 0.453592 #kg

    x_s7 = x_s7 * 0.0254 #m observer 3L
    m_s7 = m_s7 * 0.453592 #kg

    x_s8 = x_s8 * 0.0254 #m observer 3R
    m_s8 = m_s8 * 0.453592 #kg

    x_s10 = x_s10 * 0.0254 #m co-coordinator
    m_s10 = m_s10 * 0.453592 #kg

    m_payload = m_payload * 0.453592 #kg
    #================================================Moment contributions==============================================================
    #add time list
    time = np.array([time[i][0] for i in range(len(time))])
    diff = [(time[i] - time[i - 1]) for i in range(1,len(time))]
    ave_diff = np.average(diff)

    #Empty mass contributions
    M_empty = 2672953.5 * 0.453592 * 0.0254 #kgm
    M_empty_t = np.ones(len(time)) * M_empty #kgm per time step

    #Payload contribution
    #account for observer 3L moving to cockpit at 58 min 19 sec: so at t[34990]
    x_s7_t = []
    for i in range(len(time)):
        if i < 34990:
            x_s7_t.append(x_s7)
        else:
            x_s7_t.append(x_s1)
    #change of payload moment per time step
    M_pay_t = np.array([x_s1 * m_s1 + x_s2 * m_s2 + x_s3 * m_s3 + x_s4 * m_s4 + x_s5 * m_s5 + x_s6 * m_s6 + x_s7 * m_s7 + x_s8 * m_s8 + x_s10 * m_s10 for x_s7 in x_s7_t])#kgm


    #Fuel contribution
    #Data from weighing form

    flow_eng1 = np.array([flow_eng1[i][0] for i in range(len(flow_eng1))]) #pounds per hour

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
    fuelx = np.array([fuel[i,0] for i in range(len(fuel))]) #weight, in pounds
    fuely = np.array([fuel[i,1] * 100 for i in range(len(fuel))]) #moment, in pounds-inch

    #from pounds to kg
    fuelx = fuelx * 0.453592

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

    #=========================x_cg location in m========================
    x_cg_t = np.divide(M_total_t, np.add(np.add(OEW_t, m_payload), m_fuel_t))


    plt.plot(time, x_cg_t)
    plt.xlabel('time [s]')
    plt.ylabel('x_cg [m]')
    plt.show()

    np.savetxt('x_cg.csv', x_cg_t, delimiter=',')

    return x_cg_t

x_cg_t = x_cg(time, fuel_data, flow_eng1, flow_eng2)
