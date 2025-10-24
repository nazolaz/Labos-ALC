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

def esSimetrica(A): #podria hacerse con tol pero no corren los tests
    if not esCuadrada(A):
        return False
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            if A[j][i] != A[i][j]:
                return False
    return True

def esCuadrada(A):
    return cantColumnas(A) == cantFilas(A)

def productoMatricial(A, B):# A es nxp, B es pxm
    n = cantFilas(A)
    p = cantFilas(B)
    m = cantColumnas(B)
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            value = 0
            for k in range(p):
                value += A[i][k] * B[k][j]
            res[i][j] = value
    return res  

def productoVectorColumnaPorFila(u, v):
    n = cantFilas(u)
    res = np.zeros((n, n))
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            res[i][j] = ui[0] * vj
    return res



def matricesIguales(A, B, atol = 1e-8):
    if A.size != B.size and A[0].size != B[0].size:
        return False
    for i, fila in enumerate(A):
        for j, valor in enumerate(fila):
           if alc.error(np.float64(valor), np.float64(B[i][j])) > atol:
                return False
    return True


def filaCanonica(dimension, i):
    fila = np.zeros(dimension)
    fila[i] = 1
    return fila

def colCanonico(dimension, i):
    columna = np.zeros((dimension, 1))
    columna[i][0] = 1
    return columna

def normalizarVector(vector, p):
    vectorNormalizado = []

    normaVector = alc.norma(vector, p)
    if normaVector == 0:
        return vector 
    
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
    subtotal = 0
    for ui, vi in zip(u, v):
        subtotal += ui*vi
    
    return subtotal


def productoEscalar(A, k):
    if isinstance(A, Iterable):
        return [productoEscalar(ai, k) for ai in A]
    else:
        return A * k

def extenderConIdentidad(A, p): #solo para matrices cuadradas
    res = nIdentidad(p)
    n = cantFilas(A)
    for i in range(p - n, p):
        k = i - (p - n)
        for j in range(p - n, p):
            l = j - (p - n)
            res[i][j] = A[k][l]
    return res

def restaVectorial(u, v):
    res = []
    if isinstance(u[0], Iterable):
        return restaVectorial(traspuesta(u),traspuesta(v))

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
    
