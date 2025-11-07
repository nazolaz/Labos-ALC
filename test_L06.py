# Test L06-metpot2k, Aval

import numpy as np
from modulo.moduloALC import *
from modulo.moduloALCaux import*

np.random.seed(2)

def diagRH(A, tol = 1e-15, K = 1000):
    n = len(A)
    v1, l1, _ = metpot2k(A, tol, K)

    resta = normalizarVector(restaVectorial(colCanonico(n,0), v1),2)
    producto = productoExterno(resta, resta)
     
    matrizResta = productoMatricial(producto, traspuesta(producto))
    
    Hv1 = restaMatricial(nIdentidad(n), productoEscalar(matrizResta, 2))

    mid = productoMatricial(Hv1,productoMatricial(A,traspuesta(Hv1)))

    if n == 2:
        return Hv1, mid
    
    Amoño = submatriz(mid, 2, n)
    Smoño, Dmoño = diagRH(Amoño, tol, K)

    D = extenderConIdentidad(Dmoño, n)
    D[0][0] = l1

    S = productoMatricial(Hv1, extenderConIdentidad(Smoño, n))
    
    return S, D


# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
print(D)
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
print(DRH)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)

# Pedimos que pase el 95% de los casos
exitos = 0
sumaError = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
print("EXITOS: ", exitos)
assert exitos >= 95


