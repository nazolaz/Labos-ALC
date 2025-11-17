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
    

def pinvEcuacionesNormales(X, Y, tol=1e-15):
    n, p = X.shape
    _, Sigma, _ = np.linalg.svd(X)
    rangoX = 0
    for valorSingular in Sigma:
        if valorSingular > tol:
            rangoX += 1

    if rangoX == p and rangoX < n:
        XtX = productoMatricial(traspuesta(X), X)
        
        
        L = cholesky(XtX)
        Utraspuesta = np.zeros((n,p))
        
        for i in range(n):
        
            y_i = sustitucionHaciaDelante(L, X[i]) # iesima columna de X traspuesta
            u_i = sustitucionHaciaAtras(traspuesta(L), y_i)
            Utraspuesta[i] = u_i
        U = traspuesta(Utraspuesta)
        W = productoMatricial(U, Y)


    elif rangoX ==n and rangoX < p:
        # XXt = productoMatricial(X, traspuesta(X))
        XXt = X @ X.T

        L = cholesky(XXt)
        
        
        
        V = np.zeros((p,n))
        Xtraspuesta = traspuesta(X)
        for i in tqdm(range(n)):
            y_i = sustitucionHaciaDelante(L, Xtraspuesta[i]) # iesima columna de X
            V[i] = sustitucionHaciaAtras(traspuesta(L), y_i)

        W = productoMatricial(V, Y)


    elif rangoX == p and p == n:
        Xinv = inversa(X)
        W = productoMatricial(Xinv, Y)

    return W


def svdFCN(X, Y, tol = 1e-15):
    n, p = X.shape
    U, S, V = np.linalg.svd(X)
 
    sigma_plus = np.zeros((p,n))
    for i in range(len(S)):
        sigma_plus[i,i] = 1 / S[i]







# TEST BORRAR
from pathlib import Path
import numpy as np
from funcionesTP import *

def cargarDataset(carpeta: Path):
    pathCats = carpeta.joinpath('cats/efficientnet_b3_embeddings.npy')
    pathDogs = carpeta.joinpath('dogs/efficientnet_b3_embeddings.npy')

    embeddingsCats = np.load(pathCats)
    embeddingsDogs = np.load(pathDogs)

    embeddings = np.concatenate((embeddingsCats, embeddingsDogs), axis=1)
    _, m = embeddings.shape

    Y = np.zeros((2,m))
    for i in range(0, int(m/2)):
        Y[0,i] = 1

    for i in range(int(m/2), m):
        Y[1,i] = 1

    return embeddings, Y

Xt, Yt = cargarDataset(Path('../TP/template-alumnos/dataset/cats_and_dogs/train'))

svdFCN(Xt, Yt)











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