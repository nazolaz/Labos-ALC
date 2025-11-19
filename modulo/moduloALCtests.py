import unittest
import numpy as np
from moduloALC import *

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

    def test_diagRH(self):
        D = np.diag([1,0.5,0.25])
        S = np.vstack([
            np.array([1,-1,1])/np.sqrt(3),
            np.array([1,1,0])/np.sqrt(2),
            np.array([1,-1,-2])/np.sqrt(6)
                    ]).T

        A = S@D@S.T
        SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
        assert np.allclose(D,DRH)
        assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)

        # Pedimos que pase el 95% de los casos
        exitos = 0
        sumaError = 0
        for i in range(100):
            A = np.random.random((5,5))
            A = 0.5*(A+A.T)
            S,D = diagRH(A,tol=1e-15,K=1e5)
            ARH = S@D@S.T
            e = normaExacta(ARH-A,p='inf')
            if e < 1e-5: 
                exitos += 1
        print("EXITOS DIAGRH: ", exitos)
        assert exitos >= 95

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

    def test_Markov(self):
        def es_markov(T,tol=1e-6):
            """
            T una matriz cuadrada.
            tol la tolerancia para asumir que una suma es igual a 1.
            Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
            """
            n = T.shape[0]
            for i in range(n):
                for j in range(n):
                    if T[i,j]<0:
                        return False
            for j in range(n):
                suma_columna = sum(T[:,j])
                if np.abs(suma_columna - 1) > tol:
                    return False
            return True

        def es_markov_uniforme(T,thres=1e-6):
            """
            T una matriz cuadrada.
            thres la tolerancia para asumir que una entrada es igual a cero.
            Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
            """
            if not es_markov(T,thres):
                return False
            # cada columna debe tener entradas iguales entre si o iguales a cero
            m = T.shape[1]
            for j in range(m):
                non_zero = T[:,j][T[:,j] > thres]
                # all close
                close = all(np.abs(non_zero - non_zero[0]) < thres)
                if not close:
                    return False
            return True


        def esNucleo(A,S,tol=1e-5):
            """
            A una matriz m x n
            S una matriz n x k
            tol la tolerancia para asumir que un vector esta en el nucleo.
            Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
            """
            for col in S.T:
                res = A @ col
                if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
                    return False
            return True

        ## TESTS
        # transiciones_al_azar_continuas
        # transiciones_al_azar_uniformes
        for i in range(1,100):
            T = transiciones_al_azar_continuas(i)
            assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"
            
            T = transiciones_al_azar_uniformes(i,0.3)
            assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
            # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
            T = transiciones_al_azar_uniformes(i,0.01)
            assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
            T = transiciones_al_azar_uniformes(i,0.01)
            assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
            
        # nucleo
        A = np.eye(3)
        S = nucleo(A)
        assert S.shape[0]==0, "nucleo fallo para matriz identidad"
        A[1,1] = 0
        S = nucleo(A)
        msg = "nucleo fallo para matriz con un cero en diagonal"
        assert esNucleo(A,S), msg
        assert S.shape==(3,1), msg
        assert abs(S[2,0])<1e-2, msg
        assert abs(S[0,0])<1e-2, msg

        v = np.random.random(5)
        v = v / np.linalg.norm(v)
        H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
        S = nucleo(H)
        msg = "nucleo fallo para matriz de proyeccion ortogonal"
        assert S.shape==(5,1), msg
        v_gen = S[:,0]
        v_gen = v_gen / np.linalg.norm(v_gen)
        assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg

        # crea rala
        listado = [[0,17],[3,4],[0.5,0.25]]
        A_rala_dict, dims = crea_rala(listado,32,89)
        assert dims == (32,89), "crea_rala fallo en dimensiones"
        assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
        assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
        assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

        listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
        A_rala_dict, dims = crea_rala(listado,50,50)
        assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
        assert A_rala_dict.get((32,3)) == 7
        assert A_rala_dict[(16,4)] == 0.5
        assert A_rala_dict[(5,7)] == 0.25

        listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
        A_rala_dict, dims = crea_rala(listado,10,10)
        assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
        assert (1,4) not in A_rala_dict
        assert A_rala_dict[(2,5)] == 0.5
        assert A_rala_dict[(3,6)] == 0.25
        assert len(A_rala_dict) == 2

        # caso borde: lista vacia. Esto es una matriz de 0s
        listado = []
        A_rala_dict, dims = crea_rala(listado,10,10)
        assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
        assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

        # multiplica rala vector
        listado = [[0,1,2],[0,1,2],[1,2,3]]
        A_rala, _ = crea_rala(listado,3,3)
        v = np.random.random(3)
        v = v / np.linalg.norm(v)
        res = multiplica_rala_vector(A_rala,v)
        A = np.array([[1,0,0],[0,2,0],[0,0,3]])
        res_esperado = A @ v
        assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

        A = np.random.random((5,5))
        A = A * (A > 0.5) 
        listado = [[],[],[]]
        for i in range(5):
            for j in range(5):
                listado[0].append(i)
                listado[1].append(j)
                listado[2].append(A[i,j])
                
        A_rala, _ = crea_rala(listado,5,5)
        v = np.random.random(5)
        assert np.allclose(multiplica_rala_vector(A_rala,v), A @ v)

    def genera_matriz_para_test(self, m,n=2,tam_nucleo=0):
        if tam_nucleo == 0:
            A = np.random.random((m,n))
        else:
            A = np.random.random((m,tam_nucleo))
            A = np.hstack([A,A])
        return(A)
    
    def _test_svd_reducida_mn(self, A, tol=1e-15):

        m,n = A.shape
        hU,hS,hV = svd_reducida(A,tol=tol)
        nU,nS,nVT = np.linalg.svd(A, full_matrices=False)
        
        r = len(hS)+1
        assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
        assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
        assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
        assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

    def test_svd_reducida_mn(self):
        for m in [2,5,10,20]:
            for n in [2,5, 10, 20]:
                for _ in range(10):
                    A = self.genera_matriz_para_test(m,n)
                    self._test_svd_reducida_mn(A)        


    def test_svd_matrices_con_nucleo(self):

        m = 12
        for tam_nucleo in [2,4,6]:
            for i in range(10):
                print(f'iteración {i} con tamaño de nucleo={tam_nucleo} y m={m}')
                A = self.genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
                self._test_svd_reducida_mn(A)

    def test_svd_tamaños_de_las_reducidas(self):
        A = np.random.random((8,6))
        for k in [1,3,5]:
            hU,hS,hV = svd_reducida(A,k=k) # type: ignore
            assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
            assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
            assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
            assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
            assert len(hS) == k, 'Tamaño de hS incorrecto'




if __name__ == '__main__':
    unittest.main(verbosity=2)