## Sintesis Final
Luego de utilizar diferentes métodos para encontrar la pseudoinversa de la matriz X, necesaria para realizar el calculo de W, matriz propia de los pesos de la ultima capa de la red neuronal, llegamos a las siguientes conclusiones:

En primer lugar, nuestra implementación de producto matricial usa el algoritmo standard de orden O(n^3), a diferencia del producto matricial de numpy que utiliza el algoritmo de Strassen de orden O(n^log₂(7)). 

Por ejemplo, nuestra implementacion de qrHH hace 3073 productos matriciales de tamaño 2000x1536. Usando nuestro algoritmo de multiplicacion matricial se traduce en 2,4584x 10^13 operaciones, mientras que el algoritmo de numpy realiza 5,6849×10¹² operaciones, 10 veces menos.

Esto sumado a todo el resto de funciones que implementamos siguiendo de manera directa el algoritmo matematico sin optimizar, resulta en tiempos de ejecucion inviables para los metodos . Por ejemplo el algoritmo de qrhh tarda casi 1 hora por cada iteracion, lo que resulta en +1500 horas de ejecucion.

## TO DO
- arreglar productoexterno
- corregir la sintesis porque ahora qrhh está optimizada a full
- hacer matrices de confusión
- terminar de agregar comentarios en el ipynb
- revisar todo que no haya IA!!!!!
- revisar por que eqnormales da 20 de norma en las matrices de entrenamiento
- mencionar que diagRH es super ineficiente. SVD con np.linalg.svd tarda >1min

## TIEMPOS DE EJECUCION DE CADA ALGORITMO
- QRHH: 11m 52s (np.outer)
- QRGS: 18m 59s
- eqnormales: 56m 33s
- SVD: 463m 12s