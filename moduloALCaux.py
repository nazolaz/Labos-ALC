import numpy as np
import moduloALC

def calcularAx(A, x):
    res = np.zeros(A.shape[0])  #A.shape[0] devuelve la cantidad de filas de A
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            res[i] += value * x[j][0] 
    return res


def normaInf(A):
    sumatorias = []
    for i, row in enumerate(A):
        sumatorias.append(sum(abs(row)))
    
    return max(sumatorias)


def esSimetrica(A):
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            if A[j][i] != A[i][j]:
                return False
    return True
    


def multiplicacion_matrices(A, M):
    res = np.empty((A.shape[0], M.shape[1]))
    n = A.shape[1]
    for i in range(len(A)):
        for j in range(M.shape[1]):
            value = 0
            for k in range(n):
                value += A[i][k] * M[k][j]
            res[i][j] = value
    return res  


def matricesIguales(A, B, atol = 1e-8):
    if A.size != B.size and A[0].size != B[0].size:
        return False
    for i, fila in enumerate(A):
        for j, valor in enumerate(fila):
           if moduloALC.error(np.float64(valor), np.float64(B[i][j])) > atol:
                return False
    return True


def colIdentidad(dimension, col):
    columna = np.zeros(dimension)
    columna[col] = 1
    return columna



def triangSup(A):
    ATriangSup = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j < i:
                ATriangSup[i,j] = 0
    
    return ATriangSup

def triangL(A):
    L = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j > i:
                L[i][j] = 0 
            if j == i:
                L[i][i] = 1
    
    return L