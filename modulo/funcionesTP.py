from moduloALC import *
from moduloALCaux import *
import numpy as np


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
    

def pinvEcuacionesNormales(X, Y, tol=1e-15):
    n, p = X.shape
    _, Sigma, _ = np.linalg.svd(X)
    rangoX = 0
    for valorSingular in Sigma:
        if valorSingular > tol:
            rangoX += 1
    print('rango ', rangoX)

    if rangoX == p and rangoX < n:
        XtX = productoMatricial(traspuesta(X), X)
        print('producto hecho!')
        
        
        L, Lt = cholesky(XtX)
        Utraspuesta = np.array((n,p))
        
        for i in range(n):
            print('sustitución...')
        
            y_i = sustitucionHaciaDelante(L, X[i]) # iesima columna de X traspuesta
            u_i = sustitucionHaciaAtras(Lt, y_i)
            Utraspuesta[i] = u_i
        U = traspuesta(Utraspuesta)
        W = productoMatricial(Y, U)


    elif rangoX ==n and rangoX < p:
        XXt = productoMatricial(X, traspuesta(X))
        print('producto hecho!')



        L, Lt = cholesky(XXt)
        V = np.array((n,p))
        Xtraspuesta = traspuesta(X)
        for i in range(n):
            print('sustitución...')
            y_i = sustitucionHaciaDelante(L, Xtraspuesta[i]) # iesima columna de X
            v_i = sustitucionHaciaAtras(Lt, y_i)
            V[i] = v_i
        W = productoMatricial(Y, V)


    elif rangoX == p and p == n:
        Xinv = inversa(X)
        print('inversa!')
        W = productoMatricial(Y, Xinv)

    return W


def svdFCN(X, Y, tol = 1e-15):
    pass


def qrFCN(Q, R, Y):
    n = R.shape[0]
    p = Q.shape[0]
    
    V = np.zeros((p, n))
    
    for i in range(p):
        b = conseguirColumna(traspuesta(Q),i)
        V[i] = sustitucionHaciaAtras(R, b) # SI PONGO LAS SOLS EN FILAS CONSIGO V DE UNA (+1000 DE AURA)

    return productoMatricial(Y,  V)

    
def esPseudoInversa(X, pX, tol= 1e-8):
    X_pX = productoMatricial(X, pX)
    pX_X = productoMatricial(pX, X)

    condicion1 = matricesIguales(X, productoMatricial(X,pX_X))
    condicion2 = matricesIguales(pX, productoMatricial(pX_X,pX))
    condicion3 = esSimetrica(productoMatricial(X, pX))
    condicion4 = esSimetrica(productoMatricial(pX, X))

    return condicion1 & condicion2 & condicion3 & condicion4