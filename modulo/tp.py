from pathlib import Path
import numpy as np
from funcionesTP import *

def cargarDataset(carpeta: Path):
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

def cargarDatasetCompleto():
    Xt, Yt = cargarDataset(Path('TP/template-alumnos/dataset/cats_and_dogs/train'))
    Xv, Yv = cargarDataset(Path('TP/template-alumnos/dataset/cats_and_dogs/val'))

    return Xt, Yt, Xv, Yv





def testCholesky():
    Xt, Yt, Xv, Yv = cargarDatasetCompleto()

    
    
    W = pinvEcuacionesNormales(Xt, Yt)

    


