import numpy as np
from moduloALC import *
from moduloALCaux import *
# Tests L08
def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """

    m, n = A.shape
    k = min(m,n) if k == "max" else k


    AtA = A.T @ A
    VHat, SigmaHat = diagRH(AtA, tol=1e-15, K=2000)

    SigmaHatVector = vectorValoresSingulares(SigmaHat, k)

    B = productoMatricial(A, VHat)
    UHat = np.array(traspuesta(normaliza(traspuesta(B), 2)))

    return UHat, SigmaHatVector, VHat[:n, :k]







def vectorValoresSingulares(SigmaHat, k):
    SigmaHatVector = list()
    for i in range(k):
            SigmaHatVector.append(np.sqrt(np.abs(SigmaHat[i][i])))
    return SigmaHatVector

# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)


# atest = genera_matriz_para_test(2,5)
# hU,hS,hV = svd_reducida(atest)
# nU,nS,nVT = np.linalg.svd(atest)


# print('V nuestra', hV.T)
# print('V real', nVT)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    print('nuestra')
    print(hU)
    
    print('real')
    print(nU[:12, :4])


    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))


for m in [2,5,10,20]:
    for n in [2,5, 10, 20]:
        for i in range(10):
            print(f'iteración {i} con n={n} y m={m}')
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for i in range(10):
        print(f'iteración {i} con tamaño de nucleo={tam_nucleo} y m={m}')

        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) == k, 'Tamaño de hS incorrecto'
