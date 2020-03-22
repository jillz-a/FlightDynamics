import CGLoc as cg
import numpy as np

flow_eng_1 = 500 * np.ones((53431,1))
test  = np.ones((53431,1))
flow_eng_2 = flow_eng_1
time = cg.time
fuel_data = cg.fuel_data

x_cg_test = cg.x_cg(time, fuel_data, flow_eng_1, flow_eng_2)
