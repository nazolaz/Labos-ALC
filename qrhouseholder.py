import numpy as np
import modulo.moduloALC as alc
from modulo.moduloALCaux import *

def QR_con_HH (A, tol = 1e-12):
    n = cantFilas(A)
    m = cantColumnas(A)

    if m < n:
        return None, None
    
    R = A
    Q = nIdentidad(m)

    for k in range(n):
        x = conseguirColumnaSufijo(R, k, k)
        a = (-1)*signo(x[0])*alc.norma(x, 2)
        u = x - productoEscalar(a, filaIdentidad(n - k + 1, 1))
        
        if alc.norma(u, 2) > tol:
            u_n = normalizarVector(u, 2)
            H_k = nIdentidad(m - 1 + k)

    

