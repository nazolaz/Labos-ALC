import unittest
import numpy as np
from modulo.moduloALC import *

class TestModuloALC(unittest.TestCase):

    def test_sonIguales(self):
        self.assertFalse(sonIguales(1,1.1))
        self.assertTrue(sonIguales(1,1 + np.finfo('float64').eps))
        self.assertFalse(sonIguales(1,1 + np.finfo('float32').eps))
        self.assertFalse(sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
        self.assertTrue(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

    def test_error_relativo(self):
        self.assertTrue(np.allclose(error_relativo(1,1.1),0.1))
        self.assertTrue(np.allclose(error_relativo(2,1),0.5))
        self.assertTrue(np.allclose(error_relativo(-1,-1),0))
        self.assertTrue(np.allclose(error_relativo(1,-1),2))

    def test_matricesIguales(self):
        self.assertTrue(matricesIguales(np.diag([1,1]),np.eye(2)))
        self.assertTrue(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
        self.assertFalse(matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

    def test_rota(self):
        self.assertTrue(np.allclose(rota(0), np.eye(2)))
        self.assertTrue(np.allclose(
            rota(np.pi / 2),
            np.array([[0, -1],
                      [1,  0]])
        ))
        self.assertTrue(np.allclose(
            rota(np.pi),
            np.array([[-1,  0],
                      [ 0, -1]])
        ))

    def test_escala(self):
        self.assertTrue(np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]])))
        self.assertTrue(np.allclose(escala([1, 1, 1]), np.eye(3)))
        self.assertTrue(np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]])))

    def test_rota_y_escala(self):
        def rota_y_escala(theta: float, s):
            return escala(s)@rota(theta)
        self.assertTrue(np.allclose(
            rota_y_escala(0, [2, 3]),
            np.array([[2, 0],
                      [0, 3]])
        ))
        self.assertTrue(np.allclose(
            rota_y_escala(np.pi / 2, [1, 1]),
            np.array([[0, -1],
                      [1,  0]])
        ))
        self.assertTrue(np.allclose(
            rota_y_escala(np.pi, [2, 2]),
            np.array([[-2,  0],
                      [ 0, -2]])
        ))
    def test_afin(self):
        self.assertTrue(np.allclose(
            afin(0, [1, 1], [1, 2]),
            np.array([
                [1, 0, 1],
                [0, 1, 2],
                [0, 0, 1]
            ])
        ))
        self.assertTrue(np.allclose(
            afin(np.pi / 2, [1, 1], [0, 0]),
            np.array([
                [0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]
            ])
        ))
        self.assertTrue(np.allclose(
            afin(0, [2, 3], [1, 1]),
            np.array([
                [2, 0, 1],
                [0, 3, 1],
                [0, 0, 1]
            ])
        ))

    def test_trans_afin(self):
        self.assertTrue(np.allclose(
            trans_afin(np.array([1, 0]), np.pi / 2, [1, 1], [0, 0]),
            np.array([0, 1])
        ))
        self.assertTrue(np.allclose(
            trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]),
            np.array([2, 3])
        ))
        self.assertTrue(np.allclose(
            trans_afin(np.array([1, 0]), np.pi / 2, [3, 2], [4, 5]),
            np.array([4, 7])  
        ))

    def test_norma(self):
        self.assertTrue(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
        self.assertTrue(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
        self.assertLessEqual(norma(np.random.rand(10),2),np.sqrt(10))
        self.assertGreaterEqual(norma(np.random.rand(10),2),0)

    def test_normaliza(self):
        for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
            self.assertTrue(np.allclose(norma(x,2),1))
        for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
            self.assertFalse(np.allclose(norma(x,2),1) )
        for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
            self.assertTrue(np.allclose(norma(x,'inf'),1))

    def test_normaMatMC(self):
        nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
        self.assertTrue(np.allclose(nMC[0],1,atol=1e-3))
        self.assertTrue(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
        self.assertTrue(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

        nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
        self.assertTrue(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
        self.assertTrue(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

        A = np.array([[1,2],[3,4]])
        nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
        self.assertTrue(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

    def test_normaExacta(self):
        self.assertTrue(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
        self.assertTrue(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
        self.assertTrue(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
        self.assertIsNone(normaExacta(np.array([[1,-2],[-3,-4]]),2))
        self.assertLessEqual(normaExacta(np.random.random((10,10)),1),10)
        self.assertLessEqual(normaExacta(np.random.random((4,4)),'inf'),4)

    def test_condMC(self):
        A = np.array([[1,1],[0,1]])
        A_ = np.linalg.solve(A,np.eye(A.shape[0]))
        normaA = normaMatMC(A,2,2,10000)
        normaA_ = normaMatMC(A_,2,2,10000)
        condA = condMC(A,2,10000)
        assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

        A = np.array([[3,2],[4,1]])
        A_ = np.linalg.solve(A,np.eye(A.shape[0]))
        normaA = normaMatMC(A,2,2,10000)
        normaA_ = normaMatMC(A_,2,2,10000)
        condA = condMC(A,2,10000)
        assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

    def test_condExacta(self):
        A = np.random.rand(10,10)
        A_ = np.linalg.solve(A,np.eye(A.shape[0]))
        normaA = normaExacta(A,1)
        normaA_ = normaExacta(A_,1)
        condA = condExacta(A,1)
        self.assertTrue(np.allclose(normaA*normaA_,condA))

        A = np.random.rand(10,10)
        A_ = np.linalg.solve(A,np.eye(A.shape[0]))
        normaA = normaExacta(A,'inf')
        normaA_ = normaExacta(A_,'inf')
        condA = condExacta(A,'inf')
        self.assertTrue(np.allclose(normaA*normaA_,condA))

    def test_res_tri(self):
        A = np.array([[1,0,0],[1,1,0],[1,1,1]])
        b = np.array([1,1,1])
        self.assertTrue(np.allclose(res_tri(A,b),np.array([1,0,0])))
        b = np.array([0,1,0])
        self.assertTrue(np.allclose(res_tri(A,b),np.array([0,1,-1])))
        b = np.array([-1,1,-1])
        self.assertTrue(np.allclose(res_tri(A,b),np.array([-1,2,-2])))
        self.assertTrue(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))
        A = np.array([[3,2,1],[0,2,1],[0,0,1]])
        b = np.array([3,2,1])
        self.assertTrue(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))
        A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
        b = np.array([1,0,1])
        self.assertTrue(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))

    def test_inversa(self):
        ntest = 10
        iter = 0
        while iter < ntest:
            A = np.random.random((4,4))
            A_ = inversa(A)
            if not A_ is None:
                self.assertTrue(np.allclose(np.linalg.inv(A),A_))
                iter += 1
        # Matriz singular debería devolver None
        A = np.array([[1,2,3],[4,5,6],[7,8,9]])
        self.assertIsNone(inversa(A))

    def test_calculaLDV(self):
        L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
        D0 = np.diag([1,2,3])
        V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
        A =  L0 @ D0  @ V0
        L,D,V,nops = calculaLDV(A)
        self.assertTrue(np.allclose(L,L0))
        self.assertTrue(np.allclose(D,D0))
        self.assertTrue(np.allclose(V,V0))

        L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
        D0 = np.diag([3,2,1])
        V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
        A =  L0 @ D0  @ V0
        L,D,V,nops = calculaLDV(A)
        self.assertTrue(np.allclose(L,L0,1e-3))
        self.assertTrue(np.allclose(D,D0,1e-3))
        self.assertTrue(np.allclose(V,V0,1e-3))

        def test_esSDP(self):
            L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
            D0 = np.diag([1,1,1])
            A = L0 @ D0 @ L0.T
            self.assertTrue(esSDP(A))

            D0 = np.diag([1,-1,1])
            A = L0 @ D0 @ L0.T
            self.assertFalse(esSDP(A))

            D0 = np.diag([1,1,1e-16])
            A = L0 @ D0 @ L0.T
            self.assertFalse(esSDP(A))

            L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
            D0 = np.diag([1,1,1])
            V0 = np.array([[1,0,0],[1,1,0],[1,1+1e-10,1]]).T
            A = L0 @ D0 @ V0
            self.assertFalse(esSDP(A))
