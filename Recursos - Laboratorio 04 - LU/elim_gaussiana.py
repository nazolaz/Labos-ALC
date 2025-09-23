#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
a_matrix = np.array([[ 1,  1,  2],
                    [ 4,  5,  6],
                    [ 3,  1, 4]])

def triangSup(A):
    ATriangSup = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j < i:
                ATriangSup[i,j] = 0
    
    return ATriangSup

def triangL(A):
    L = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j > i:
                L[i][j] = 0 
            if j == i:
                L[i][i] = 1
    
    return L


def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    Ac[0] = A[0]
    for k in range(0, n-1):
        if A[k][k] != 0:
            for i in range(k + 1, n):
                mi = A[i][k]/A[k][k]
                cant_op += 1
                Ac[i][k] = mi
                for j in range(k+1, m):
                    Ac[i][j] = Ac[i][j] - mi * Ac[k][j]
                    cant_op += 2 
        else: 
            print('Matriz no tiene LU')
            return
    
    return triangL(Ac), triangSup(Ac), cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
