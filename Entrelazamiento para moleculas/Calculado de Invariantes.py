import numpy as np
from numpy import sqrt
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, IGate, CXGate
from qiskit.quantum_info import Statevector, partial_trace, concurrence
import os

def CalculoInvariantes(psi):  # psi es el estado como un StateVector de qiskit
    #  Callculo de los inariantes polinomiales
    x = psi.data
    h1 = x[-1] *x[0] - x[1]* x[-2] - x[2] *x[-3] + x[3]* x[-4] -  x[4] *x[-5] + x[5] *x[-6] + x[6] *x[-7] - x[7] *x[-8]
    h2 = np.linalg.det(np.array([
        [x[0],x[4],x[8],x[12]],
        [x[1],x[5],x[9],x[13]],
        [x[2],x[6],x[10],x[14]],
        [x[3],x[7],x[11],x[15]]
    ]))

    h3 = np.linalg.det(np.array([
        [x[0], x[8], x[2], x[10]],
        [x[1], x[9], x[3], x[11]],
        [x[4], x[12], x[6], x[14]],
        [x[5], x[13], x[7], x[15]]
    ] ))
    h4 = np.linalg.det(np.array([
        [x[0] * x[3] - x[1] * x[2], x[3] * x[4] - x[2] * x[5] - x[1] * x[6] + x[0] * x[7], -x[5] * x[6] + x[4] * x[7]],
        [x[3] * x[8] - x[2] * x[9] - x[1] * x[10] + x[0] * x[11], x[7] * x[8] - x[6] * x[9] - x[5] * x[10] + x[4] * x[11] + x[3] * x[12] - x[2] * x[13] - x[1] * x[14] + x[0] * x[15],
            x[7] * x[12] - x[6] * x[13] - x[5] * x[14] + x[4] * x[15]],
        [ -x[9] * x[10] + x[8] * x[11], x[11] * x[12] - x[10] * x[13] - x[9] * x[14] + x[8] * x[15], -x[13] * x[14] + x[12] * x[15]
        ]
    ]))

    #  calcular los invariantes de entrelazamiento
    permutations = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    Cab = concurrence(partial_trace(psi, [2,3]).data)
    Cac = concurrence(partial_trace(psi, [1,3]).data)
    Cad = concurrence(partial_trace(psi, [1,2]).data)
    Cbc = concurrence(partial_trace(psi, [0,3]).data)
    Cbd = concurrence(partial_trace(psi, [0,2]).data)
    Ccd = concurrence(partial_trace(psi, [0,1]).data)
    Cgm = min([sqrt(2- 2 * np.trace(partial_trace(psi, i).data @ partial_trace(psi, i).data)) for i in permutations])

    return [h1.item().real, h2, h3, h4, Cab, Cac, Cad, Cbc, Cbd, Ccd, Cgm]