from moduloALC import *
from moduloALCaux import *
import numpy as np
from tqdm import tqdm

def pinvEcuacionesNormales(X, Y):
    n, p = X.shape
    rangoX = min(n, p)

    if rangoX == p and rangoX < n:
        XtX = productoMatricial(traspuesta(X), X)
        L = cholesky(XtX)
        Utraspuesta = np.zeros((n,p))
        
        for i in range(n):
        
            y_i = sustitucionHaciaDelante(L, X[i]) # iesima columna de X traspuesta
            u_i = sustitucionHaciaAtras(traspuesta(L), y_i)
            Utraspuesta[i] = u_i
        U = traspuesta(Utraspuesta)
        W = productoMatricial(Y, U)


    elif rangoX == n and rangoX < p:
        XXt = productoMatricial(X, traspuesta(X))
        L = cholesky(XXt)
        V = np.zeros((p,n))
        Xtraspuesta = traspuesta(X)

        for i in tqdm(range(n), desc='eq normales'):
            y_i = sustitucionHaciaDelante(L, Xtraspuesta[i]) # iesima columna de X
            V[i] = sustitucionHaciaAtras(traspuesta(L), y_i)

        W = productoMatricial(Y, V)


    elif rangoX == p and p == n:
        Xinv = inversa(X)
        W = productoMatricial(Y, Xinv)

    return W


def svdFCN(X, Y, tol = 1e-15):
    n, p = X.shape
    

    # SVD REDUCIDA ME DA V1 DIRECTAMENTE
    U, S, V1 = svd_reducida(X, tol=tol)

    S_inv_diag = np.zeros((n, n))
    for i in range(len(S)):
        S_inv_diag[i,i] = 1.0 / S[i]
    
    X_plus = productoMatricial(V1, productoMatricial(S_inv_diag, traspuesta(U)))
    
    W = productoMatricial(Y, X_plus)
    
    return W

def qrFCN(Q, R, Y):

    #despejamos V haciendo R * V.T = Q.T
    m_r, n_r = R.shape # shape R (2000, 1536)
    m_p, n_p = Q.shape # shape Q (2000, 2000)

    V = np.zeros((m_p, n_r)) # shape V.T (1536, 2000) -> shape V (2000, 1536) 
 
    
    for i in tqdm(range(m_p)):
        b = Q[i] # esto es equivalente a conseguirColumna(traspuesta(Q))
        V[i] = sustitucionHaciaAtras(R, b)

    return productoMatricial(Y,  V)


    
def esPseudoInversa(X, pX, tol= 1e-8):
    X_pX = productoMatricial(X, pX)
    pX_X = productoMatricial(pX, X)

    condicion1 = matricesIguales(X, productoMatricial(X,pX_X), tol)
    condicion2 = matricesIguales(pX, productoMatricial(pX_X,pX), tol)
    condicion3 = esSimetrica(X_pX, tol)
    condicion4 = esSimetrica(pX_X, tol)

    return condicion1 & condicion2 & condicion3 & condicion4


