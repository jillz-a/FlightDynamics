from Cit_par import *
import numpy as np

'''
Short Period Oscillation.
See https://brightspace.tudelft.nl/d2l/le/content/213481/viewContent/1474787/View

'''

cbar = c
V = V0

SPO_C1 = np.array([  [cbar/V*(CZadot-2*muc), 0],
                    [cbar/V*Cmadot, -2*muc*KY2*(cbar/V)**2]])

SPO_C2 = np.array([  [CZa, cbar/V*(CZq + 2*muc)],
                    [Cma, cbar/V*Cmq]])

SPO_A = -np.matmul(np.linalg.inv(SPO_C1),SPO_C2)

print("Eigenvalues SPO: \n", np.linalg.eigvals(SPO_A))


## Phugoid simplification ##
PHU_C1 = np.array([ [-1/V*2*muc*cbar/V, 0, 0],
                    [0, 0, 0],
                    [0, -cbar/V, 0]])

PHU_C2 = np.array([ [CXu/V, CZ0, 0],
                    [CZu/V, 0, cbar/V*2*muc],
                    [0, 0, cbar/V]])

#PHU_A = -np.matmul(np.linalg.inv(PHU_C1),PHU_C2)
#print("Eigenvalues PHU: \n", np.linalg.eigvals(PHU_A))