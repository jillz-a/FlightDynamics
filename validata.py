import numpy as np
from Cit_par import c as cbar
from Cit_par import b
import MatrixCalculation
from MatrixCalculation import Mode
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


def convert(list): 
    return tuple(i for i in list)


class FLContainer():
    def __init__(self, T, V0, udak, alpha, theta, qdak, beta,  phi, prel, rrel, delta_a, delta_e, delta_r):
        self.T = T
        self.V0 = V0
        self.udak = udak
        self.alpha = alpha
        self.theta = theta
        self.qdak = qdak
        self.beta = beta
        self.phi = phi
        self.prel = prel
        self.rrel = rrel
        self.delta_a = delta_a
        self.delta_e = delta_e
        self.delta_r = delta_r


def TrimData(tstart, tend):
    istart = tstart*10
    iend = tend*10
    T = np.linspace(tstart, tend, (tend-tstart)*10 + 1)
    V0 = TAS[istart]
    udak = ((TAS - V0)/V0)[istart:(iend+1)]
    alpha = (AOA - AOA[istart])[istart:(iend+1)]
    theta = (th - th[istart])[istart:(iend+1)]
    qdak = (q*cbar/V0)[istart:(iend+1)]
    beta = 0

    # beta does not exist in the data...
    phi = (phiab - phiab[istart])[istart:(iend+1)]
    prel = (p*b/(2*V0))[istart:(iend+1)]
    rrel = (r*b/(2*V0))[istart:(iend+1)]
    delta_a = ailerondef[istart:(iend+1)]
    delta_e = elevatordef[istart:(iend+1)]
    delta_r = rudderdef[istart:(iend+1)]

    return FLContainer(T, V0, udak, alpha, theta, qdak, beta,  phi, prel, rrel, delta_a, delta_e, delta_r)

def CheckData(tstart, tend, instashow=True, title="Flight Data as provided "):
    '''
    Checks the data from the flight test and plots it between the times as input

    ------
    Input:
        tstart: time to start evaluating in seconds.
        tend: time to end evaluation in seconds.
        instashow: Plot straight away or wait for collection?
        title: image title. 
    Note that both values must be in the range of the flight duration, 0 ->  5343.2


    '''
    trimmedData = TrimData(tstart, tend)

    stVec = ["u [airpseed, TAS]", "alpha [AoA]", "theta [flight path]", "q [pitch rate, nondimensional]"]
    stVec2 = ["beta [sideslip]", "phi [roll]", "p [roll rate]", "r [yaw rate]"]

    fig1, axs1 = plt.subplots(7, sharex=True)
    fig1.suptitle(title)
    axs1[0].plot(trimmedData.T,trimmedData.udak)
    axs1[0].set_title(stVec[0])
    axs1[1].plot(trimmedData.T,trimmedData.alpha)
    axs1[1].set_title(stVec[1])
    axs1[2].plot(trimmedData.T,trimmedData.theta)
    axs1[2].set_title(stVec[2])
    axs1[3].plot(trimmedData.T,trimmedData.qdak)
    axs1[3].set_title(stVec[3])

    axs1[4].plot(trimmedData.T,trimmedData.phi)
    axs1[4].set_title(stVec2[1])
    axs1[5].plot(trimmedData.T,trimmedData.prel)
    axs1[5].set_title(stVec2[2])
    axs1[6].plot(trimmedData.T,trimmedData.rrel)
    axs1[6].set_title(stVec2[3])


    if instashow == True:
        plt.show()

    return fig1,axs1

def DisplayEigenmotionData():
    fig1, axs1 = CheckData(3600, 3780, instashow=False, title="Phugoid Motion")
    fig2, axs2 = CheckData(3780, 3900, instashow=False, title="Short Period")
    fig3, axs3 = CheckData(3900, 4020, instashow=False, title="Aperiodic Roll")
    fig4, axs4 = CheckData(4080, 4110, instashow=False, title="Dutch Roll")
    fig5, axs5 = CheckData(4110, 4160, instashow=False, title="Dutch Roll YD")
    fig6, axs6 = CheckData(4200, 4320, instashow=False, title="Spiral")             # Assuming it took approximately 2 minutes

    plt.show()




def CompareData(mode, tstart, tend, instashow=True, title="Flight Data as provided "):
    '''
    Almost exactly like CheckData(), but now it also plots the system response.
    '''
    initials = [0,0,0,0]
    FD = TrimData(tstart, tend)                                                                         # This represents the flight data

    if mode == Mode.Symmetric:
        sys, eigs = MatrixCalculation.GenSymmetricStateSys()                                            # Generates a state-space system like in MatrixCalculation
        _, SDList, _ = ctrl.forced_response(sys,FD.T, FD.delta_a, initials)                             # Generates the model data under the same input as the flight data
        SD = FLContainer(FD.T, FD.V0, SDList[0], SDList[1], SDList[2], SDList[3], 0, 0, 0, 0, 0, 0, 0)
    elif mode == Mode.Asymmetric:
        sys, eigs = MatrixCalculation.GenAsymmetricStateSys()
        _, SDList, _ = ctrl.forced_response(sys,FD.T, np.vstack((FD.delta_e, FD.delta_r)), initials)
        SD = FLContainer(FD.T, FD.V0, 0, 0, 0, 0, SDList[0], SDList[1], SDList[2], SDList[3], 0, 0 , 0)

    else: 
        print("Please select a valid mode form the Mode enumerator")
        return


    stVec = ["u [airpseed, TAS]", "alpha [AoA]", "theta [flight path]", "q [pitch rate, nondimensional]"]
    stVec2 = ["beta [sideslip]", "phi [roll]", "p [roll rate]", "r [yaw rate]"]

    fig1, axs1 = plt.subplots(4, sharex=True)
    fig1.suptitle(title)

    if mode == Mode.Symmetric:
        # Symmetric, Flight Data
        axs1[0].plot(FD.T,FD.udak, label="FD")
        axs1[0].set_title(stVec[0])
        axs1[1].plot(FD.T,FD.alpha, label="FD")
        axs1[1].set_title(stVec[1])
        axs1[2].plot(FD.T,FD.theta, label="FD")
        axs1[2].set_title(stVec[2])
        axs1[3].plot(FD.T,FD.qdak, label="FD")
        axs1[3].set_title(stVec[3])

    if mode == Mode.Asymmetric:
        # Asymmetric, Flight Data
        axs1[1].plot(FD.T,FD.phi, label="FD")
        axs1[1].set_title(stVec2[1])
        axs1[2].plot(FD.T,FD.prel, label="FD")
        axs1[2].set_title(stVec2[2])
        axs1[3].plot(FD.T,FD.rrel, label="FD")
        axs1[3].set_title(stVec2[3])
    
    if mode == Mode.Symmetric:
        # Symmetric, simulation data
        axs1[0].plot(SD.T,SD.udak, label="SD")
        axs1[0].set_title(stVec[0])
        axs1[1].plot(SD.T,SD.alpha, label="SD")
        axs1[1].set_title(stVec[1])
        axs1[2].plot(SD.T,SD.theta, label="SD")
        axs1[2].set_title(stVec[2])
        axs1[3].plot(SD.T,SD.qdak, label="SD")
        axs1[3].set_title(stVec[3])

    if mode == Mode.Asymmetric:
        # Asymmetric, simulation data
        axs1[1].plot(SD.T,SD.phi, label="SD")
        axs1[1].set_title(stVec2[1])
        axs1[2].plot(SD.T,SD.prel, label="SD")
        axs1[2].set_title(stVec2[2])
        axs1[3].plot(SD.T,SD.rrel, label="SD")
        axs1[3].set_title(stVec2[3])

    if instashow == True:
        plt.show()

    return fig1,axs1


def ValidateModel():
    fig1, axs1 = CompareData(Mode.Symmetric,  3600, 3780, instashow=False, title="Phugoid Motion")
    fig2, axs2 = CompareData(Mode.Symmetric,  3780, 3900, instashow=False, title="Short Period")
    fig3, axs3 = CompareData(Mode.Asymmetric, 3900, 4020, instashow=False, title="Aperiodic Roll")
    fig4, axs4 = CompareData(Mode.Asymmetric, 4080, 4110, instashow=False, title="Dutch Roll")
    fig5, axs5 = CompareData(Mode.Asymmetric, 4110, 4160, instashow=False, title="Dutch Roll YD")
    fig6, axs6 = CompareData(Mode.Asymmetric, 4200, 4320, instashow=False, title="Spiral")             # Assuming it took approximately 2 minutes

    plt.legend()
    plt.show()

############# Main ##################

DisplayEigenmotionData()
#CompareData(Mode.Symmetric, 3600, 3780)
#ValidateModel()