import control as ctrl
import numpy as np
from Cit_par.py import *

## Composite Matrices from appendix D
# These have been rewritten in terms of State variable u,a,theta,q.
# C1* xdot + C2* x + C3 * u

cbar = c
V = V0

C1 = np.array([ [(-2*muc*cbar/V**2), 0, 0,0],
                [0, (CZadot -2*muc), 0, 0],
                [0, 0, (-cbar/V), 0],
                [0,(cbar/V*Cmadot)]])

C2 = np.array([ [1/V*CXa, CXa, CZ0, cbar/V*CXa],
                [1/V*CZa, Cza, -CX0, -cbar/V*(CZa + 2*muc)],
                [0, 0, 0, cbar/V],
                [1/V*Cma, Cma, 0,cbar/V*Cmq]])

C3 = np.array([ [CXde],
                [CZde],
                [0],
                [Cmde]])

## Define matrices A,B,C,D ##

# State matrix A
A = -np.matmul(np.linalg.inv(C1),C2)
B = -np.matmul(np.linalg.inv(C1),C3)

C = np.array([  [1/V, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, cbar/V]])
D = np.zeros((4,4))


## Create state space system ##

sys = ctrl.StateSpace(A,B,C,D)