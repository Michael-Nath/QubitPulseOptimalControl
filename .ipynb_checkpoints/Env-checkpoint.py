import numpy as np
from scipy.linalg import expm
import math
from qiskit import IBMQ
from qiskit.tools.jupyter import *
from numpy import ndarray

class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T
'''
provider = IBMQ.load_account()
armonk = provider.get_backend("ibmq_armonk")
armonk
armonk.configuration()
'''

# pauli-matrices initialization 
sx = np.mat('0, 1;1, 0')
sy = np.mat('0, -1j;1j, 0')
sz = np.mat('1, 0;0, -1')

# set target and initial state --> future: want to make user-input target
# currently --> going from Pauli-X to Hadamard
psi_target = np.mat([[0],[1]], dtype=complex)
psi_0 = np.mat([[1],[0]], dtype=complex)

qubit_freq = 4.71# load this from qiskit
U_0 = np.array(np.identity(2,dtype=complex))


class Env(object):
    def __init__(self, dt=0.1): # two values in action space
        super(Env, self).__init__()
        self.n_features = 4 # correlated to neural network code
        self.state = np.array([1,0,0,0])
        self.nstep = 0 
        self.dt=dt
        self.N = 10
        
    def reset(self):
        self.state = np.array([1,0,0,0])
        self.nstep = 0 
        return self.state

    def step(self, action, coefficient):
        
        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.array(psi)
        
        H =  qubit_freq*sz/2 + coefficient*sx
        U = expm(-1j * H * self.dt)
        
        global U_0
        if (self.nstep == 0):
            U_0 = U
        else:
            U_0 = U * U_0
        psi = U_0 * psi
        print(psi)
        err = math.log10(1 - (np.abs(psi.view(myarray).H * psi_target) ** 2).item(0).real)  # state fidelity error calculation
        if (self.nstep % 10 == 9):
            rwd = 10 * (err<0.5)+100 * (err<0.1)+5000*(err < 10e-3)   
        else:
            rwd = None
            
        done =( (err < 10e-3) or self.nstep>=np.pi/self.dt ) 
        self.nstep +=1  

        return self.state, rwd, done, 1-err
