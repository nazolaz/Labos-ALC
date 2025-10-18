import numpy as np
import modulo.moduloALC as alc
from collections.abc import Iterable

def calcularAx(A, x):
    res = np.zeros(cantFilas(A))  
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            res[i] += value * x[j][0] 
    return res

def normaInf(A):
    sumatorias = []
    for i, row in enumerate(A):
        sumatorias.append(sum(abs(row)))
    
    return max(sumatorias)

def esSimetrica(A, tol = 1e-8):
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            if alc.error_relativo(A[j][i], A[i][j]) > tol:
                return False
    return True

def productoMatricial(A, M):
    n = cantFilas(A)
    res = np.empty((n, cantColumnas(M)))
    for i in range(len(A)):
        for j in range(cantColumnas(M)):
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
           if alc.error(np.float64(valor), np.float64(B[i][j])) > atol:
                return False
    return True


def filaIdentidad(dimension, i):
    fila = np.zeros(dimension)
    fila[i] = 1
    return fila

def colIdentidad(dimension, i):
    columna = np.zeros((dimension, 1))
    columna[i][0] = 1
    return columna

def normalizarVector(vector, p):
    vectorNormalizado = []

    normaVector = alc.norma(vector, p)
    for xi in vector:
        vectorNormalizado.append(xi/normaVector)

    return np.array(vectorNormalizado)

def traspuesta(A):
    if (isinstance(A[0], Iterable)):
        res = np.zeros((cantColumnas(A), cantFilas(A)))
        for i, row in enumerate(A):
            for j, value in enumerate(row):
                res[j][i] = A[i][j]

    else:
        res = np.zeros((len(A),1))
        for i, value in enumerate(A):
            res[i][0] = value 
            
    return res

def dimension(A):
    return cantFilas(A), cantColumnas(A)

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

def cantFilas(A):
    if isinstance(A[0], Iterable):
        return len(A)

    else:
        return 1

def cantColumnas(A):
    if isinstance(A[0], Iterable):
        return len(A[0])
    
    else:
        return len(A)

def longitudVector(v):
    return len(v)

def conseguirColumna(A, j):
    columna = []
    for k in range(cantColumnas(A)):
        columna.append(A[k][j])

    return np.array(columna)

def insertarColumna(A, b, j):
    for k in range(cantFilas(A)):
        A[k][j] = b[k]
    return A

def conseguirColumnaSufijo(A, j, k):
    columna = []
    for l in range(k, cantColumnas(A)):
        columna.append(A[l][j])
    
    return np.array(columna)

def productoInterno(u, v):
    sum = 0
    for ui, vi in zip(u, v):
        sum += ui*vi
    
    return sum

def productoEscalar(u, k):
    res = []
    for _, ui in enumerate(u):
        res.append(k*ui)
    
    return res

def restaVectorial(u, v):
    res = []
    for ui, vi in zip(u,v):
        res.append(ui - vi)

    return res

def nIdentidad(n):
    I = np.zeros((n,n))
    for k in range(n):
        I[k][k] = 1
    return I

def signo(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0