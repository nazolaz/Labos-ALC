import numpy as np
from modulo.moduloALCaux import *


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
    return escala(s)@rota(theta)

def afin(theta, s, b):
    m1 = rota_y_escala(theta, s)
    return np.array([[m1[0][0],m1[0][1], b[0]],[m1[1][0], m1[1][1], b[1]],[0,0,1]])
    

def trans_afin(v, theta, s, b):
    casi_res = afin(theta, s, b)@np.array([[v[0]],[v[1]],[1]])
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

def normalizarVector(vector, p):
    vectorNormalizado = list()

    normaVector = norma(vector, p)
    for xi in vector:
        vectorNormalizado.append(xi/normaVector)
    return vectorNormalizado


def normaExacta(A, p = [1, 'inf']):
    if p == 1:
        return normaInf(A.T)
    
    elif p == 'inf':
        return normaInf(A)
    
    else:
        return None

def normaMatMC(A, q, p, Np):
    m = len(A[0])
    vectors = []
    for i in range(0,Np):
        vectors.append(np.random.rand(m,1))
    
    normalizados = normaliza(vectors, p)

    multiplicados = []
    for vector in normalizados:
        multiplicados.append(calcularAx(A, vector))
    
    
    for i, vector in enumerate(multiplicados):
        multiplicados[i] = (norma(vector, q), normalizados[i])
    
    return max(multiplicados, key=lambda t: t[0])


def condMC(A, p, Np=1000000):
    AInv = inversa(A)
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
        colInv = res_tri(L, colIdentidad(dim, i), inferior=True)
        for j in range(dim):
            Linv[j][i] = colInv[j]

    for i in range(dim):
        if( U[i,i] == 0):
            return None

        colInv = res_tri(U, colIdentidad(dim, i), inferior=False)
        for j in range(dim):
            Uinv[j][i] = colInv[j]

    return Uinv @ Linv





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
