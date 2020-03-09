import control as ctrl
from Cit_par import *
import numpy as np

cbar = c
V = V0

def GenSymmetricStateSys():
    '''
    Generates the state-space system for the symmetric case. Requires Cit_par to be imported.

    Returns
    ------
    sys: State-space system
    '''

    ## Composite Matrices from appendix D
    # These have been rewritten in terms of State variable u,a,theta,q.
    # C1* xdot + C2* x + C3 * u
    C1 = np.array([ [(-2*muc*cbar/V**2), 0., 0., 0.],
                    [0., (CZadot -2*muc), 0., 0.],
                    [0., 0., (-cbar/V), 0.],
                    [0., (cbar/V*Cmadot), 0., (-2*muc*KY2*(cbar/V)**2)]])

    C2 = np.array([ [1/V*CXa, CXa, CZ0, cbar/V*CXa],
                    [1/V*CZa, CZa, -CX0, -cbar/V*(CZa + 2*muc)],
                    [0., 0., 0., cbar/V],
                    [1/V*Cmu, Cma, 0., cbar/V*Cmq]])

    C3 = np.array([ [CXde],
                    [CZde],
                    [0.],
                    [Cmde]])

    ## Define matrices A,B,C,D #

    # State matrix A
    A = -np.matmul(np.linalg.inv(C1),C2)
    B = -np.matmul(np.linalg.inv(C1),C3)

    C = np.array([  [1/V, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, cbar/V]])
    D = np.zeros((4,1))


    ## Create state space system ##

    sys = ctrl.StateSpace(A,B,C,D)
    return sys

def GenAsymmetricStateSys():
    
    C1 = np.array([ [(CYbdot -2*mub)*b/V, 0, 0, 0],
                    [0, -.5*b/V, 0, 0],
                    [0, 0, -2*mub*KX2*(b/V)**2, 2*mub*KXZ*(b/V)**2],
                    [0, 0, 2*mub*KXZ*(b/V)**2, -2*mub*KX2*(b/V)**2]])

    C2 = np.array([ [CYb, CL, CYp*b/(2*V), (CYr - 4*mub)],
                    [0, 0, b/(2*V), 0],
                    [Clb, 0, Clp*b/(2*V), Clr*b/(2*V)]
                    [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]])
