import numpy as np
from moduloALCaux import *
from tqdm import tqdm

# POR HACER/TERMINAR
#   - hacer nuestro propio inverso y reemplazar linalg.inv
#   - terminar SVD

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
    return productoMatricial(escala(s), rota(theta))

def afin(theta, s, b):
    m1 = rota_y_escala(theta, s)
    return np.array([[m1[0][0],m1[0][1], b[0]],[m1[1][0], m1[1][1], b[1]],[0,0,1]])

def trans_afin(v, theta, s, b):
    casi_res = productoMatricial(afin(theta, s, b),np.array([[v[0]],[v[1]],[1]]))
    return np.array([casi_res[0][0], casi_res[1][0]])

def norma(Xs, p):
    if p == 'inf':
        return max(map(abs ,Xs))
    
    res = 0 
    for xi in Xs:
        res += xi**p
    return res**(1/p)

def normaliza(Xs, p):
    XsNormalizado = []

    for vector in Xs:
        res = normalizarVector(vector, p)
        XsNormalizado.append(res)

    return XsNormalizado

def normaExacta(A, p = [1, 'inf']):
    if p == 1:
        return normaInf(traspuesta(A))
    
    elif p == 'inf':
        return normaInf(A)
    
    else:
        return None

def normaMatMC(A, q, p, Np):
    n = cantFilas(A)
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
    for i in range(cantFilas(A)-1, -1, -1):
        cocienteActual = A[i][i]
        sumatoria = 0
        for k in range(i + 1, len(b)):
            sumatoria += A[i][k] * valoresX[k]

        if cocienteActual == 0:
            valoresX[i] = np.nan  
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
    m, n = dimension(A)
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
    n = cantFilas(A)

    L,U,_ = calculaLU(A)

    if (L is None or U is None):
        return None    

    Linv = np.zeros((n,n))
    Uinv = np.zeros((n,n))

    for i in range(n):
        colInv = res_tri(L, filaCanonica(n, i), inferior=True)
        for j in range(n):
            Linv[j][i] = colInv[j]

    for i in range(n):
        if( U[i,i] == 0):
            return None

        colInv = res_tri(U, filaCanonica(n, i), inferior=False)
        for j in range(n):
            Uinv[j][i] = colInv[j]

    return productoMatricial(Uinv, Linv)

def calculaLDV(A):
    L, U, nops1 = calculaLU(A)

    if(U is None):
        return None, None, None, 0

    Vt, D, nops2 = calculaLU(traspuesta(U))


    if Vt is None:
        return None, None, None, 0
    
    return L, D, traspuesta(Vt), nops1 + nops2

def esSDP(A, atol=1e-10):
    if(not (esSimetrica(A, atol))):
        return False
    
    L, D, Lt, _ = calculaLDV(A)

    if( D is None):
        return False
    
    for i in range(len(D)):
        if (D[i,i] <= 0):
            return False
    return True


def metpot2k(A, tol=1e-15, K=1000.0):
    n = len(A[0])
    v = np.random.rand(n,1)
    vmoñotemp = f_A(A, v)
    vmoño = f_A(A, vmoñotemp)
    e = float(productoInterno(vmoño, v))
    k = 0
    while( abs(e - 1) > tol and k < K):

        v = vmoño
        vmoñotemp = f_A(A, v)
        vmoño = f_A(A, vmoñotemp)
        e = float(productoInterno(vmoño, v))
        k = k + 1
    
    ax = calcularAx(A, vmoño)
    autovalor = productoInterno(vmoño, ax)

    return vmoño, autovalor, k


def f_A(A, v):

    wprima = A @ v

    if norma(wprima, 2) > 0:
        return normalizarVector(wprima, 2) 


    return 0

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    n = cantFilas(A)
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    nops = 0

    a_1 = conseguirColumna(A, 0)
    insertarColumna(Q, normalizarVector(a_1, 2), 0)
    R[0][0] = norma(a_1, 2)
    nops += 2*n + 1

    for j in range(1, n):
        qMoño_j = conseguirColumna(A, j)

        for k in range(0, j):
            q_k = conseguirColumna(Q, k)
            R[k][j] = productoInterno(q_k, qMoño_j)
            nops += 2*n - 1
            qMoño_j = restaVectorial(qMoño_j, productoEscalar(q_k, R[k][j]))
            nops += 2*n
        
        R[j][j] = norma(qMoño_j, 2)
        nops += 2*n - 1
        insertarColumna(Q, productoEscalar(qMoño_j, 1/R[j][j]), j)
        nops += 1

    if (retorna_nops):
        return Q, R, nops

    return Q, R

def QR_con_HH (A, tol = 1e-12):
    m = cantFilas(A)
    n = cantColumnas(A)

    if m < n:
        return None, None
    
    Q = nIdentidad(m)

    for k in tqdm(range(n)):
        x = A[k:, k]
        a = (-1)*signo(x[0])*alc.norma(x, 2)
        u = x - productoEscalar(a, filaCanonica(n - k, 0))
        
        if alc.norma(u, 2) > tol:
            u_n = normalizarVector(u, 2)
            uut = productoExterno(traspuesta(u_n), u_n)
            dosuut = productoEscalar(2, uut)
            H_k = nIdentidad(n - k) - dosuut
            H_k_ext = extenderConIdentidad(H_k, m)
            A = productoMatricial(H_k_ext, A)
            Q = productoMatricial(Q, traspuesta(H_k_ext))

    return Q, A

def calculaQR(A, metodo = 'RH', tol = 1e-12, nops = False):
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    
    elif metodo == 'GS':
        if nops:
            return QR_con_GS(A, tol, True)
        else:
            return QR_con_GS(A, tol)
    
    else: 
        return None, None, None
    
def diagRH(A, tol = 1e-15, K = 1000):
    n = len(A)
    v1, l1, _ = metpot2k(A, tol, K)

    resta = normalizarVector(restaVectorial(colCanonico(n,0), v1),2)
    producto = productoExterno(resta, resta)    
    Hv1 = restaMatricial(nIdentidad(n), productoEscalar(producto, 2))
    mid = productoMatricial(Hv1,productoMatricial(A,traspuesta(Hv1)))

    if n == 2:
        return Hv1, mid
    
    Amoño = submatriz(mid, 2, n)
    Smoño, Dmoño = diagRH(Amoño, tol, K)

    D = extenderConIdentidad(Dmoño, n)
    D[0][0] = l1

    S = productoMatricial(Hv1, extenderConIdentidad(Smoño, n))

    return S, D

def transiciones_al_azar_continuas(n):
    t = []
    for i in range(n):
        randvec = np.random.uniform(0, 1, n)
        t.append(randvec)
    tnormalizado = normaliza(t, 1)
    return traspuesta(np.array(tnormalizado))


def transiciones_al_azar_uniformes(n,thres):
    if n == 1:
        return np.array([[1]])

    t = []
    for i in range(n):
        randvec = np.random.uniform(0, 1, n)
        t.append(randvec)

    for i in range(len(t)):
        for j in range(len(t[0])):
            if t[i][j] < thres:
                t[i][j] = 1
            else:
                t[i][j] = 0
            if i == j:
                t[i][i] = 1
    tnormalizado = normaliza(t, 1)
    return np.array(traspuesta(tnormalizado))

#funciona pq λi es autovalor sii σi es valor singular
def nucleo(A,tol=1e-15):
    normalA = productoMatricial(traspuesta(A), A)
    SA, DA = diagRH(normalA)
    nucleo = []
    
    #consigo la columna respectiva del autovalor 0 
    for i in range(len(DA)):
            if DA[i][i] <= tol:
                nucleo.append(conseguirColumna(SA, i))
                
    return traspuesta(np.array(nucleo))


def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    if len(listado) == 0:
        #cualquiera pero los tests esperan esto
        return [], (m_filas, n_columnas)
    
    aristas = {}

    for i in range(len(listado[0])):
        ij_valor = listado[2][i]
        if ij_valor > tol:
            ij = ((listado[0][i]),listado[1][i])
            aristas[ij] =  ij_valor
    return aristas, (m_filas, n_columnas)

def multiplica_rala_vector(A,v):
    w = np.zeros(v.shape)
    ijs = A.keys()
    
    for parIj in ijs:
        w[parIj[0]] += A[parIj] * v[parIj[1]]

    return w
