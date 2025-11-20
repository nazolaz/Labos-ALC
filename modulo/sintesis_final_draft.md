## Sintesis Final
Luego de utilizar diferentes métodos para encontrar la pseudoinversa de la matriz X, necesaria para realizar el calculo de W, matriz propia de los pesos de la ultima capa de la red neuronal, llegamos a las siguientes conclusiones:

En primer lugar, nuestra implementación de producto matricial usa el algoritmo standard de orden O(n^3), a diferencia del producto matricial de numpy que utiliza el algoritmo de Strassen de orden O(n^log₂(7)). 

Esto sumado a todo el resto de funciones que implementamos siguiendo de manera directa el algoritmo matematico sin optimizar, resulta en tiempos de ejecucion inviables para los metodos . Por ejemplo el algoritmo de qrhh tarda casi 1 hora por cada iteracion, lo que resulta en +1500 horas de ejecucion.

## TO DO
- terminar de agregar comentarios en el ipynb
- revisar todo que no haya IA!!!!!
- revisar por que eqnormales da 20 de norma en las matrices de entrenamiento

## TIEMPOS DE EJECUCION DE CADA ALGORITMO
- QRHH: 11m 52s
- QRGS: 18m 59s
- eqnormales: 56m 33s
- SVD: 463m 12s



## Propuesta
Tras la implementación y evaluación de los distintos métodos numéricos para el cálculo de la pseudoinversa de la matriz $X$ —paso fundamental para obtener la matriz de pesos $W$ de la red neuronal—, hemos observado diferencias notables en el rendimiento y estabilidad de cada algoritmo.

En primer lugar, el menor de los tiempos de ejecución fue conseguido por la factorización QR mediante Householder (QRHH), con un tiempo total de 11m 52s. Este resultado contrasta significativamente con la versión previa del código (que demoraba \approx 1 hora por cada iteracion, lo que resulta en +1500 horas de ejecucion); la mejora radica en la optimización al cálculo de QR con HouseHolder que se basa en la siguiente observación:
$$ 
H = I - 2 vv^t\\
H A = (I - 2 vv^t) A\\
H A = A - 2 v (v^tA)\\
$$
Esta optimización nos evita tener que construir y extender la matriz $H_k$ en cada iteración, calculando $HA$ directamente con 2 multiplicaciones matriz-vector.


Por otro lado, QR mediante Gram-Schmidt (QRGS), aunque teóricamente comparable, resultó casi un 60% más lento (18m 59s). Esto se debe a {RAZONES RAZONES RAZONES RAZONES RAZONES}.

El caso de Ecuaciones Normales (56m 33s) presenta una anomalía interesante. Si bien teóricamente suele ser el método más directo ($A^TAx = A^Tb$), nuestra implementación sufrió penalizaciones de precisión, arrojando una norma de error de 20.6, superior al 12.7 obtenido por QR y SVD. Esto se fundamenta teóricamente en el condicionamiento de las matrices: mientras que la condición de nuestra matriz de datos es $cond(X) \approx 363$, al operar con Ecuaciones Normales este valor se eleva al cuadrado ($cond(X^TX) \approx 1.3 \times 10^5$). Esta degradación en el condicionamiento magnifica los errores de punto flotante, impidiendo al algoritmo alcanzar el mínimo exacto que sí logran los métodos más estables (QR/SVD).

Finalmente, la descomposición en valores singulares (SVD) resultó computacionalmente inviable, con un tiempo de 463m 12s (más de 7 horas). La ineficiencia aquí recae en el cálculo de la SVD reducida utilizando el algoritmo recursivo de diagonalización (diagRH). Esto contrasta fuertemente con la función que ofrece numpy para la svd (np.linalg.svd), que desploma el tiempo de ejecución a 57 segundos.

En conclusión, para esta aplicación específica, QR Householder optimizado representa el mejor compromiso entre estabilidad numérica y eficiencia computacional, superando drásticamente a la implementación manual de SVD y evitando la inestabilidad inherente de las Ecuaciones Normales.