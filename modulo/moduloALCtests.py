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


#COMENTO PQ TARDA UN MONTON
    # def test_normaMatMC(self):
    #     nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
    #     self.assertTrue(np.allclose(nMC[0],1,atol=1e-3))
    #     self.assertTrue(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
    #     self.assertTrue(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

    #     nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
    #     self.assertTrue(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
    #     self.assertTrue(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

    #     A = np.array([[1,2],[3,4]])
    #     nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
    #     self.assertTrue(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

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


    def test_metpot(self):
        S = np.vstack([
        np.array([2,1,0])/np.sqrt(5),
        np.array([-1,2,5])/np.sqrt(30),
        np.array([1,-2,1])/np.sqrt(6)
                ]).T

        # Pedimos que pase el 95% de los casos
        exitos = 0
        for i in range(100):
            D = np.diag(np.random.random(3)+1)*100
            A = S@D@S.T
            v,l,_ = metpot2k(A,1e-15,1e5)
            if np.abs(l - np.max(D))< 1e-8:
                exitos += 1
        assert exitos > 95


        #Test con HH
        exitos = 0
        for i in range(100):
            v = np.random.rand(9)
            #v = np.abs(v)
            #v = (-1) * v
            ixv = np.argsort(-np.abs(v))
            D = np.diag(v[ixv])
            I = np.eye(9)
            H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

            A = H@D@H.T
            v,l,_ = metpot2k(A, 1e-15, 1e5)
            #max_eigen = abs(D[0][0])
            if abs(l - D[0,0]) < 1e-8:         
                exitos +=1
        assert exitos > 95





    def test_QR(self):

        def check_QR(Q,R,A,tol=1e-10):
            # Comprueba ortogonalidad y reconstrucción
            assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
            assert np.allclose(Q @ R, A, atol=tol)

        A2 = np.array([[1., 2.],
                    [3., 4.]])

        A3 = np.array([[1., 0., 1.],
                    [0., 1., 1.],
                    [1., 1., 0.]])

        A4 = np.array([[2., 0., 1., 3.],
                    [0., 1., 4., 1.],
                    [1., 0., 2., 0.],
                    [3., 1., 0., 2.]])

        Q3h,R3h = QR_con_HH(A3)
        check_QR(Q3h,R3h,A3)

        Q4h,R4h = QR_con_HH(A4)
        check_QR(Q4h,R4h,A4)    

        Q2c,R2c = calculaQR(A2,metodo='RH')
        check_QR(Q2c,R2c,A2)

        Q3c,R3c = calculaQR(A3,metodo='GS')
        check_QR(Q3c,R3c,A3)

        Q4c,R4c = calculaQR(A4,metodo='RH')
        check_QR(Q4c,R4c,A4)