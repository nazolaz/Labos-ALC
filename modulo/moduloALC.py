import numpy as np
from modulo.moduloALCaux import *

# POR HACER/TERMINAR
#   - contar cantidad de operaciones en QR-GS

def error(x, y):
    return abs(np.float64(x) - np.float64(y))

def error_relativo(x,y):
    if x == 0:
        return abs(y)
    return error(x,y)/abs(x)

def sonIguales(x,y,atol=1e-08):
    
    return np.allclose(error(x,y),0,atol=atol)

def rota(theta: float):
    cos = np.cos(theta)
    sen = np.sin(theta)
    
    return np.array([[cos,-sen],[sen,cos]])

def escala(s):
    matriz = np.eye(len(s))

    for i in range(len(s)):
        matriz[i][i] = s[i]

    return matriz


def rota_y_escala(theta: float, s):
    return multiplicacion_matrices(escala(s), rota(theta))

def afin(theta, s, b):
    m1 = rota_y_escala(theta, s)
    return np.array([[m1[0][0],m1[0][1], b[0]],[m1[1][0], m1[1][1], b[1]],[0,0,1]])
    

def trans_afin(v, theta, s, b):
    casi_res = multiplicacion_matrices(afin(theta, s, b),np.array([[v[0]],[v[1]],[1]]))
    return np.array([casi_res[0][0], casi_res[1][0]])


def norma(Xs, p):
    if p == 'inf':
        return max(map(abs ,Xs))
    
    res = 0 
    for xi in Xs:
        res += xi**p
    return res**(1/p)


def normaliza(Xs, p):
    XsNormalizado = list()

    for vector in Xs:
        res = normalizarVector(vector, p)
        XsNormalizado.append(res)

    return XsNormalizado


def normaExacta(A, p = [1, 'inf']):
    if p == 1:
        return normaInf(A.T)
    
    elif p == 'inf':
        return normaInf(A)
    
    else:
        return None

def normaMatMC(A, q, p, Np):
    n = len(A)
    vectors = []

    ## generamos Np vectores random
    for _ in range(0,Np):
        vectors.append(np.random.rand(n,1)*2-1)
    
    ## normalizamos los vectores
    normalizados = normaliza(vectors, p)


    ## multiplicar A por cada Xs
    multiplicados = []
    for Xs in normalizados:
        multiplicados.append((calcularAx(A, Xs)))
    
    maximo = [0,0]
    for vector in multiplicados:
        
        if norma(vector, q) > maximo[0]:
            maximo[0] = norma(vector, q)
            maximo[1] = vector

    return maximo


def condMC(A, p, Np=1000000):
    AInv = np.linalg.inv(A)
    if AInv is None:
        return None
    
    normaAInv = normaMatMC(AInv, p, p, Np)[0]
    normaA = normaMatMC(A, p, p, Np)[0]
    

    return normaA * normaAInv


def condExacta(A, p):
    AInv = inversa(A)
    normaA = normaExacta(A, p)
    normaAInv = normaExacta(AInv, p)
    
    if normaA is None:
        return 0


    return normaA * normaAInv

def sustitucionHaciaAtras(A, b):
    valoresX = np.zeros(len(b))
    for i in range(len(A)-1, -1, -1):
        cocienteActual = A[i][i]
        sumatoria = 0
        for k in range(i + 1, len(b)):
            sumatoria += A[i][k] * valoresX[k]

        if cocienteActual == 0:
            valoresX[i] = np.nan  # o puedes lanzar una excepción si prefieres
        else:
            valoresX[i] = (b[i] - sumatoria)/cocienteActual
    return valoresX

def sustitucionHaciaDelante(A, b):
    valoresX = []
    for i, row in enumerate(A):
        cocienteActual = row[i]
        sumatoria = 0
        for k in range(i):
            sumatoria += A[i][k] * valoresX[k]
        valoresX.append((b[i] - sumatoria)/cocienteActual)
    return np.array(valoresX)



def res_tri(L, b, inferior=True):
    if(inferior):
        return sustitucionHaciaDelante(L,b)
    return sustitucionHaciaAtras(L,b)



def calculaLU(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        return None, None, 0
    
    for k in range(0, n-1):
        if A[k][k] == 0:
            return None, None, 0
        
        for i in range(k + 1, n):
            
            mi = Ac[i][k]/Ac[k][k]
            cant_op += 1
            Ac[i][k] = mi
            for j in range(k+1, m):
                Ac[i][j] = Ac[i][j] - mi * Ac[k][j]
                cant_op += 2 
    
    return triangL(Ac), triangSup(Ac), cant_op




def inversa(A):
    dim = len(A)

    L,U,_ = calculaLU(A)

    if (L is None or U is None):
        return None    

    Linv = np.zeros((dim,dim))
    Uinv = np.zeros((dim,dim))

    for i in range(dim):
        colInv = res_tri(L, filaIdentidad(dim, i), inferior=True)
        for j in range(dim):
            Linv[j][i] = colInv[j]

    for i in range(dim):
        if( U[i,i] == 0):
            return None

        colInv = res_tri(U, filaIdentidad(dim, i), inferior=False)
        for j in range(dim):
            Uinv[j][i] = colInv[j]

    return multiplicacion_matrices(Uinv, Linv)





def calculaLDV(A):
    L, U, nops1 = calculaLU(A)

    if(U is None):
        return None, None, None, 0

    Vt, D, nops2 = calculaLU(U.T)


    if Vt is None:
        return None, None, None, 0
    
    return L, D, Vt.T, nops1 + nops2



def esSDP(A, atol=1e-10):
    if( not (esSimetrica(A))):
        return False
    
    L, D, Lt, _ = calculaLDV(A)

    if( D is None):
        return False
    
    for i in range(len(D)):
        if (D[i,i] <= 0):
            return False
    return True




def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    Q = np.zeros((cantFilas(A),cantColumnas(A)))
    R = np.zeros((cantFilas(A),cantColumnas(A)))
    nops = 0

    a_1 = conseguirColumna(A, 0)
    insertarColumna(Q, normalizarVector(a_1, 2), 0)
    R[0][0] = norma(a_1, 2)

    for j in range(1, cantFilas(A)):
        qMoño_j = conseguirColumna(A, j)

        for k in range(0, j):
            q_k = conseguirColumna(Q, k)
            R[k][j] = productoInterno(q_k, qMoño_j)
            qMoño_j = restaVectorial(qMoño_j, productoEscalar(q_k, R[k][j]))
        
        R[j][j] = norma(qMoño_j, 2)
        insertarColumna(Q, productoEscalar(qMoño_j, 1/R[j][j]), j)

    if (retorna_nops):
        return Q, R, nops

    return Q, R
