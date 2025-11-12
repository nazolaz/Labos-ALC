from moduloALC import *
from moduloALCaux import *


def busquedaW(X, Y, tol=1e-15):
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