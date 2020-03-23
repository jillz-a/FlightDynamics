import numpy as np
from Cit_par import c as cbar
from Cit_par import b
import MatrixCalculation
import control as ctrl
from matplotlib import pyplot as plt

dt = 0.1 #seconds



## Importing all data for symmetric motion
TAS = np.genfromtxt('flight_data/TAS.csv')
AOA = np.genfromtxt('flight_data/AOA.csv')
th = np.genfromtxt('flight_data/theta.csv')
#qdak = np.diff(theta)/dt*cbar/V0                           # Old method, is also given directly
q = np.genfromtxt('flight_data/pitchrate.csv')
elevatordef = np.genfromtxt('flight_data/delta_e.csv')
ailerondef = np.genfromtxt('flight_data/delta_a.csv')
rudderdef = np.genfromtxt('flight_data/delta_r.csv')
phiab = np.genfromtxt('flight_data/roll.csv')
p = np.genfromtxt('flight_data/rollrate.csv')
r = np.genfromtxt('flight_data/yawrate.csv')


def CheckData(tstart, tend):
    istart = tstart*10
    iend = tend*10
    T = np.linspace(tstart, tend, (tend-tstart)*10 + 1)
    V0 = TAS[istart]
    udak = ((TAS - V0)/V0)[istart:(iend+1)]
    alpha = (AOA - AOA[istart])[istart:(iend+1)]
    theta = (th - th[istart])[istart:(iend+1)]
    qdak = (q*cbar/V0)[istart:(iend+1)]

    # beta does not exist in the data...
    phi = (phiab - phiab[istart])[istart:(iend+1)]
    prel = (p*b/(2*V0))[istart:(iend+1)]
    rrel = (r*b/(2*V0))[istart:(iend+1)]


    stVec = ["u [airpseed, TAS]", "alpha [AoA]", "theta [flight path]", "q [pitch rate, nondimensional]"]
    stVec2 = ["beta [sideslip]", "phi [roll]", "p [roll rate]", "r [yaw rate]"]

    fig1, axs1 = plt.subplots(7, sharex=True)
    fig1.suptitle("Flight Data as provided ")
    axs1[0].plot(T,udak)
    axs1[0].set_title(stVec[0])
    axs1[1].plot(T,alpha)
    axs1[1].set_title(stVec[1])
    axs1[2].plot(T,theta)
    axs1[2].set_title(stVec[2])
    axs1[3].plot(T,qdak)
    axs1[3].set_title(stVec[3])

    axs1[4].plot(T,phi)
    axs1[4].set_title(stVec2[1])
    axs1[5].plot(T,prel)
    axs1[5].set_title(stVec2[2])
    axs1[6].plot(T,rrel)
    axs1[6].set_title(stVec2[3])

    plt.show()

    return True

CheckData(3600,3660)

def ValidateModel(mode):
    MatrixCalculation.GenSymmetricStateSys()
    inputparam = 1
    sys, sysEig, inputindex, stVec, inputtitle = MatrixCalculation.ResponseInputHandler(mode, inputparam)

    if inputindex == 0:
        # elevator deflection
        forcedFunction = elevatordef 
    elif inputindex == 1:
        # aileron input
        forcedFunction = np.vstack((ailerondef, rudderdef))
    else:
        print("Please input a valid mode.")

    initials = [0,0,0,0]
    T = np.linspace(0, 5343, 53431)
    
    time, y_forced, _ = ctrl.forced_response(sys,T, forcedFunction, initials)
    

    fig1, axs1 = plt.subplots(4, sharex=True)
    fig1.suptitle("Forced Response"+inputtitle)
    axs1[0].plot(time,yinit[0])
    axs1[0].set_title(stVec[0] + " response")
    axs1[1].plot(time,yinit[1])
    axs1[1].set_title(stVec[1]+ " response")
    axs1[2].plot(time,yinit[2])
    axs1[2].set_title(stVec[2]+ " response")
    axs1[3].plot(time,yinit[3])
    axs1[3].set_title(stVec[3]+" response")
    
