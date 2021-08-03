import numpy as np
from scipy.linalg import expm
import math
from qiskit import IBMQ
from qiskit.tools.jupyter import *

# pauli-matrices initialization 
sx = np.mat([[0, 1],[ 1, 0]], dtype=complex)
sy = np.mat([[0, -1j],[1j, 0]], dtype=complex)
sz = np.mat([[1, 0],[0, -1]], dtype=complex)

# psi_0 = np.mat([[0, 1],[1 , 0]], dtype=complex)
psi_0 = np.array(np.identity(2, dtype=complex))
# psi_target = 1/math.sqrt(2) * np.mat([[1, 1],[1, -1]], dtype=complex)
psi_target = sy
qubit_freq = 4.71 # will load this from qiskit soon
unitary = psi_0


class Env:
    def __init__(self, dt=0.001, N=20): # two values in action space
        super(Env, self).__init__()
        self.n_actions = N
        self.n_features = 4 # correlated to neural network code
        self.state = np.array([1,0,0,0])
        self.nstep = 0 
        self.dt=dt

    def reset(self):
        self.state = np.array([1,0,0,0])
        self.nstep = 0 
        return self.state

    def step(self, action, coefficient):
        # understand what psi does
        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi= np.array(psi)

        U = np.array(np.identity(2, dtype=complex)) # [[1,0][0,1]]
        H =  qubit_freq*sz/2 + coefficient*sx
        print(f"H: {H}")
        U = expm(-1j * H * self.dt) 
        print(f"U: {U}")
        global unitary
        unitary = U * unitary
        print(f"unitary: {unitary}")
        psi = unitary * psi
        print(f"target * unitary: {psi_target.H*unitary}")
        print(f"trace_value: {np.trace(psi_target.H*unitary) / 2}")
        err = 1-abs(np.trace(psi_target.H*unitary) / 2)**2 # gate fidelity error calculation
        rwd = 10 * (err<0.5)+100 * (err<0.1)+5000*(err < 10e-3)   

        done =( (err < 10e-3) or self.nstep>=np.pi/self.dt ) 
        self.nstep += 1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, 1 - err
