import numpy as np
from moduloALC import *
from moduloALCaux import *

def QR_con_HH (A, tol = 1e-12):
    m = cantFilas(A)
    n = cantColumnas(A)

    if m < n:
        return None, None
    
    R = A.copy()
    Q = nIdentidad(m)

    for k in range(n):
        x = conseguirColumnaSufijo(R, k, k)
        a = (-1)*signo(x[0])*alc.norma(x, 2)
        u = x - productoEscalar(a, filaCanonica(m - k, 0))
        
        if alc.norma(u, 2) > tol:
            u_n = normalizarVector(u, 2)
            uut = productoExterno(traspuesta(u_n), u_n)
            dosuut = productoEscalar(2, uut)
            H_k = nIdentidad(m - k) - dosuut
            H_k_ext = extenderConIdentidad(H_k, m)
            R = productoMatricial(H_k_ext, R)
            Q = productoMatricial(Q, traspuesta(H_k_ext))

    return Q, R

def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

A2 = np.array([[1., 2.],
            [3., 4.]])

A3 = np.array([[1., 0., 1.],
            [0., 1., 1.],
            [1., 1., 0.]])

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)
