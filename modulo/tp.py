from moduloALC import *
from moduloALCaux import *

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

    if rangoX == p and rangoX < n:
        XtX = productoMatricial(traspuesta(X), X)
        L, Lt = cholesky(XtX)
        Utraspuesta = np.array((n,p))
        for i in range(n):
            y_i = sustitucionHaciaDelante(L, X[i]) # iesima columna de X traspuesta
            u_i = sustitucionHaciaAtras(Lt, y_i)
            Utraspuesta[i] = u_i
        U = traspuesta(Utraspuesta)
        W = productoMatricial(Y, U)


    if rangoX ==n and rangoX < p:
        XXt = productoMatricial(X, traspuesta(X))
        L, Lt = cholesky(XXt)
        V = np.array((n,p))
        Xtraspuesta = traspuesta(X)
        for i in range(n):
            y_i = sustitucionHaciaDelante(L, Xtraspuesta[i]) # iesima columna de X
            v_i = sustitucionHaciaAtras(Lt, y_i)
            V[i] = v_i
        W = productoMatricial(Y, V)


    if rangoX == p and p == n:
        Xinv = inversa(X)
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

    