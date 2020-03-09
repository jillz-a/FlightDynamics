import control as ctrl
from Cit_par import *
import numpy as np
from matplotlib import pyplot as plt

cbar = c
V = V0

def GenSymmetricStateSys():
    '''
    Generates the state-space system for the symmetric case. Requires Cit_par to be imported. 
    Uses global variable instances, takes no input.

    Returns
    ------
    sys: State-space system. 
        Inputs: [u, alpha, theta, q]
        Outputs: [udakje, alpha, theta, qding]
    '''

    ## Composite Matrices from appendix D
    # These have been rewritten in terms of State variable u,a,theta,q.
    # C1* xdot + C2* x + C3 * u
    C1 = np.array([ [(-2*muc*cbar/V**2), 0., 0., 0.],
                    [0., (CZadot -2*muc) * (cbar/V), 0., 0.],
                    [0., 0., (-cbar/V), 0.],
                    [0., ((cbar/V)*Cmadot), 0., (-2*muc*KY2*(cbar/V)**2)]])

    C2 = np.array([ [(1/V)*CXa, CXa, CZ0, (cbar/V)*CXa],
                    [(1/V)*CZa, CZa, -CX0, (cbar/V)*(CZq + 2*muc)],
                    [0., 0., 0., cbar/V],
                    [(1/V)*Cmu, Cma, 0., (cbar/V)*Cmq]])

    C3 = np.array([ [CXde],
                    [CZde],
                    [0.],
                    [Cmde]])

    ## Define matrices A,B,C,D 
    A = -np.matmul(np.linalg.inv(C1),C2)
    B = -np.matmul(np.linalg.inv(C1),C3)

    C = np.array([  [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    D = np.zeros((4,1))


    ## Create state space system ##
    sys = ctrl.ss(A,B,C,D)

    ## Calculate eigenvalues and vectors of A
    eigs = np.linalg.eig(A)

    return sys, eigs

def GenAsymmetricStateSys():
    '''
    Generates the state-space system for the asymmetric case. Requires Cit_par to be imported. 
    Uses global variable instances, takes no input.

    Returns
    ------
    sys: State-space system
    '''
    
    C1 = np.array([ [(CYbdot -2*mub)*(b/V), 0, 0, 0],
                    [0, -.5*(b/V), 0, 0],
                    [0, 0, -2*mub*KX2*(b/V)**2, 2*mub*KXZ*(b/V)**2],
                    [0, 0, 2*mub*KXZ*(b/V)**2, -2*mub*KZ2*(b/V)**2]])

    C2 = np.array([ [CYb, CL, CYp*(b/(2*V)), (CYr - 4*mub)*(b/(2*V))],
                    [0, 0, b/(2*V), 0],
                    [Clb, 0, Clp*(b/(2*V)), Clr*(b/(2*V))],
                    [Cnb, 0, Cnp*(b/(2*V)), Cnr*(b/(2*V))]])

    C3 = np.array([ [CYda, CYdr],
                    [0 , 0],
                    [Clda, Cldr],
                    [Cnda, Cndr]])

    ## Define matrices A,B,C,D
    A = -np.matmul(np.linalg.inv(C1),C2)
    B = -np.matmul(np.linalg.inv(C1),C3)

    C = np.array([  [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    D = np.zeros((4,2))


    ## Create state space system ##
    sys = ctrl.ss(A,B,C,D)

    ## Calculate eigenvalues and vectors of A
    eigs = np.linalg.eig(A)

    return sys, eigs


######### Symmetric response ###############
symmsys, symmsysEig = GenAsymmetricStateSys()
## General System information

print(symmsys)

symmsyspoles = symmsys.damp()
print("Pole information.\n wn: ",symmsyspoles[0],"\n Zeta: ",symmsyspoles[1],"\n Poles: ",symmsyspoles[2])
print("Eigenvalues: ", symmsysEig)


# Pole and zeroes map #
plt.scatter(symmsys.pole().real, symmsys.pole().imag)
plt.grid()
plt.show()


## System Response ##
initials = [V0,alpha0,th0,0]
t, y = ctrl.impulse_response(symmsys,X0=initials)
plt.plot(t,y[0])
plt.show()
