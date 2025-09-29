### Funciones L05-QR

def norma (x, p):
    if p == 'inf':
        return max(map(abs ,x))
    
    res = 0 
    for xi in x:
        res += xi**p
    return res**(1/p)


def normaliza(Xs, p):
    res = []
    for x in Xs:
        res.append(x/norma(x,p))
    return res

def cantFilas(A):
    return len(A)

def cantColumnas(A):
    return len(A[0])

def conseguirColumna(A, j):
    columna = np.zeros(cantColumnas(A))
    for k in range(cantColumnas(A)):
        columna.append(A[k][0])

    return np.array(columna)

def insertarColumna(A, b, j):
    for k in range(cantColumnas(A)):
        A[k][j] = b[k]
    return A

def productoInterno(u, v):
    sum = 0
    for ui, vi in zip(u, v):
        sum += ui*vi
    
    return sum

def productoEscalar(u, k):
    res = ()
    for _, ui in enumerate(u):
        res.append(k*ui)
    
    return res

def restaVectorial(u, v):
    res = ()
    for ui, vi in zip(u,v):
        res.append(ui - vi)

    return res

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    Q = np.zeros(cantFilas(A),cantColumnas(A))
    R = np.zeros(cantFilas(A),cantColumnas(A))
    nops = 0

    a_1 = conseguirColumna(A, 0)
    insertarColumna(Q, normaliza(a_1, 2), 0)
    R[0][0] = norma(a_1, 2)

    for j in range(1, cantFilas(A)):
        qMoño_j = conseguirColumna(A, j)

        for k in range(0, j-1):
            q_k = conseguirColumna(Q, k)
            R[k][j] = productoInterno(q_k, qMoño_j)
            qMoño_j = restaVectorial(qMoño_j, productoEscalar(q_k, R[k][j]))
        
        R[j][j] = norma(qMoño_j, 2)
        insertarColumna(Q, productoEscalar(qMoño_j, 1/R[j][j]), j)

    if (retorna_nops):
        return Q, R, nops

    else:
        return Q, R






def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

# Tests L05-QR:

import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)