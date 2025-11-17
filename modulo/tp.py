from pathlib import Path
import numpy as np
from funcionesTP import *
import unittest



class tpTests(unittest.TestCase):
    def cargarDataset(self, carpeta: Path):
        pathCats = carpeta.joinpath('./cats/efficientnet_b3_embeddings.npy')
        pathDogs = carpeta.joinpath('./dogs/efficientnet_b3_embeddings.npy')

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

    def cargarDatasetCompleto(self):
        Xt, Yt = self.cargarDataset(Path('TP/template-alumnos/dataset/cats_and_dogs/train'))
        Xv, Yv = self.cargarDataset(Path('TP/template-alumnos/dataset/cats_and_dogs/val'))

        return Xt, Yt, Xv, Yv


    def test_Cholesky(self):
        Xt, Yt, Xv, Yv = self.cargarDatasetCompleto()

        print('dataset cargado!')
        
        W = pinvEcuacionesNormales(Xt, Yt)

        print((W @ Xv) - Yv )


    def test_QRHH(self):
        Xt, Yt, Xv, Yv = self.cargarDatasetCompleto()

        print('dataset cargado!')
        Q, R = QR_con_HH(traspuesta(Xt))
        W = qrFCN(Q, R, Yt)

        print((W @ Xv) - Yv )

if __name__ == '__main__':
    unittest.main(verbosity=2)