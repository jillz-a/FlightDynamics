import control as ctrl
import numpy as np
import Cit_par.py

## Composite Matrices from appendix D
# These have been rewritten in terms of State variable u,a,theta,q.
# C1* xdot + C2* x + C3 * u

cbar = c
V = V0

C1 = np.array([  [(-2*muc*cbar/V**2), 0, 0,0],
                [0, (CZadot -2*muc), 0, 0],
                [0, 0, (-cbar/V), 0],
                [0,(cbar/V*Cmadot)]])

C2 = np.array()

## Define matrices A,B,C,D ##

# State matrix A

# Input matrix B

# Output matrix C

# Feedforward matrix D

## Create state space system ##

sys = ctrl.StateSpace(A,B,C,D)