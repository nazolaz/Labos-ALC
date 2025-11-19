from moduloALC import *
from moduloALCaux import *
import numpy as np
from tqdm import tqdm

def fully_connected_lineal(X, Y, tol=1e-15, method = "QR"):
    match method:
        case "Cholesky":
            return pinvEcuacionesNormales(X, Y, tol)
        case "SVD":
            return svdFCN(X, Y, tol)
        case "QR-HH":
            return qrFCN(X, Y, tol)
        case "QR-GS":
            return ""
    

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
        for i in range(n):
            y_i = sustitucionHaciaDelante(L, Xtraspuesta[i]) # iesima columna de X
            V[i] = sustitucionHaciaAtras(traspuesta(L), y_i)

        W = productoMatricial(Y, V)


    elif rangoX == p and p == n:
        Xinv = inversa(X)
        W = productoMatricial(Y, Xinv)

    return W


def svdFCN(X, Y, tol = 1e-15):
    n, p = X.shape
    
    U, S, Vh = svd_reducida(X, tol=tol)
    # U, S, Vh = np.linalg.svd(X)

    S_inv_diag = np.zeros((len(S), len(S)))
    for i in range(len(S)):
        S_inv_diag[i,i] = 1.0 / S[i]


    V1 = traspuesta(Vh)  # Dimensiones: (p x n) o (2000 x 1536)
    
    X_plus = productoMatricial(V1, productoMatricial(S_inv_diag, traspuesta(U)))
    
    W = productoMatricial(Y, X_plus)
    
    return W

def qrFCN(Q, R, Y):
    #despejamos V haciendo R* V.T = Q.T
    n = R.shape[0] #1536
    p = Q.shape[0] #2000
    
    V = np.zeros((p, n))

    for i in tqdm(range(p)):
        b = Q[i] # esto es igual a conseguirColumna(traspuesta(Q))
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


