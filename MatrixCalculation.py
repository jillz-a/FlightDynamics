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
        u is the deviation of speed to V0, alpha is the AoA to alpha0, theta is the flight path angle to theta0, q is the real pitch rate.
        Outputs: [udakje, alpha, theta, qding]
    '''

    ## Composite Matrices from appendix D
    # These have been rewritten in terms of State variable u,a,theta,q.
    # C1* xdot + C2* x + C3 * u
    C1 = np.array([ [(-2*muc*cbar/V**2), 0., 0., 0.],
                    [0., (CZadot -2*muc) * (cbar/V), 0., 0.],
                    [0., 0., (-cbar/V), 0.],
                    [0., ((cbar/V)*Cmadot), 0., (-2*muc*KY2*(cbar/V)**2)]])

    C2 = np.array([ [(1/V)*CXu, CXa, CZ0, (cbar/V)*CXa],
                    [(1/V)*CZu, CZa, -CX0, (cbar/V)*(CZq + 2*muc)],
                    [0., 0., 0., cbar/V],
                    [(1/V)*Cmu, Cma, 0., (cbar/V)*Cmq]])

    C3 = np.array([ [CXde],
                    [CZde],
                    [0.],
                    [Cmde]])

    ## Define matrices A,B,C,D 
    A = -np.matmul(np.linalg.inv(C1),C2)
    B = -np.matmul(np.linalg.inv(C1),C3)

    C = np.array([  [1/V, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, cbar/V]])
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

    C = np.array([  [1/V, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, cbar/V]])
    D = np.zeros((4,2))

    ## Create state space system ##
    sys = ctrl.ss(A,B,C,D)

    ## Calculate eigenvalues and vectors of A
    eigs = np.linalg.eig(A)

    return sys, eigs


def CalcResponse(mode):
    '''
    Calculates the response of the specified system and reports by showing initial response, impulse response, and step response. Also shows a pole-zero map and prints in the log the pole and system information.

    Input
    ------
    mode:
        "symmetric" or "symm" calculates using the symmetric case
        "asymmetric" or "asymm" calculates using the asymmetric configuration

    Returns
    ------
    Success on completion
    '''
    if mode.lower == "symmetric" or "symm" or "sym":
        sys, sysEig = GenSymmetricStateSys()
    elif mode.lower == "asymmetric" or "asymm" or "asym":
        sys, sysEig = GenAsymmetricStateSys()
    else:
        print("Please fill in either \"Symmetric\" or \"Asymmetric\" as a paramter for the CalcResponse")

    sys, sysEig = GenSymmetricStateSys()
    print(sys)

    # Pole and zeroes map #
    plt.scatter(sys.pole().real, sys.pole().imag)
    #plt.scatter(sys.zero().real, sys.zero.imag)
    plt.suptitle("Pole-Zero map")
    plt.grid()
    syspoles = sys.damp()
    print("Pole information.\n wn: ",syspoles[0],"\n Zeta: ",syspoles[1],"\n Poles: ",syspoles[2])
    print("Eigenvalues: ", sysEig)

    ## System Responses ##
    initials = [5,alpha0,th0,0]
    T = np.linspace(0,100,2000)
    (time,yinit) = ctrl.initial_response(sys, T, initials)
    _, y_impulse  = ctrl.impulse_response(sys,T, initials)
    _, y_step = ctrl.step_response(sys, T, initials)

    fig1, axs1 = plt.subplots(4, sharex=True)
    fig1.suptitle("Initial Condition Response")
    axs1[0].plot(time,yinit[0])
    axs1[0].set_title("u response")
    axs1[1].plot(time,yinit[1])
    axs1[1].set_title("alpha response")
    axs1[2].plot(time,yinit[2])
    axs1[2].set_title("theta response")
    axs1[3].plot(time,yinit[3])
    axs1[3].set_title("q response")

    fig2, axs2 = plt.subplots(4, sharex=True)
    fig2.suptitle("Impulse Response")
    axs2[0].plot(time,y_impulse[0])
    axs2[0].set_title("u response")
    axs2[1].plot(time,y_impulse[1])
    axs2[1].set_title("alpha response")
    axs2[2].plot(time,y_impulse[2])
    axs2[2].set_title("theta response")
    axs2[3].plot(time,y_impulse[3])
    axs2[3].set_title("q response")

    fig3, axs3 = plt.subplots(4, sharex=True)
    fig3.suptitle("Step Response")
    axs3[0].plot(time,y_step[0])
    axs3[0].set_title("u response")
    axs3[1].plot(time,y_step[1])
    axs3[1].set_title("alpha response")
    axs3[2].plot(time,y_step[2])
    axs3[2].set_title("theta response")
    axs3[3].plot(time,y_step[3])
    axs3[3].set_title("q response")

    plt.show()
    return 1

CalcResponse("symm")