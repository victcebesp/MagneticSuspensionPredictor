# Magnetic Suspension Control System

### Introducción
El problema a solucionar que elegimos fue el de la Identificación o control de un Sistema de Suspensión Magnética. El objetivo principal de la práctica es crear una red neuronal supervisada de tipo backpropagation que tras el entrenamiento, sea capaz de reproducir el comportamiento que tiene un imán en suspensión cuya distancia a un electroimán va cambiando según se le va aplicando diferentes intensidades al electroimán. Dicho comportamiento puede ser representado mediante la siguiente ecuación diferencial:

<p align="center">
    <img width="413" alt="captura de pantalla 2018-11-17 a las 10 59 04" src="https://user-images.githubusercontent.com/15388747/48664428-5cef9800-ea96-11e8-886a-c9ab4d1d0bc6.png">
</p>

- Donde M es la masa del imán en suspensión.
- g es la constante de la gravedad.
- ß es el coeficiente de fricción del material donde se mueve el cuerpo magnético en suspensión.
- α es la fuerza del campo que produce el cuerpo electromagnético.

<p align="center">
    <img width="315" alt="captura de pantalla 2018-11-17 a las 11 02 06" src="https://user-images.githubusercontent.com/15388747/48664451-91fbea80-ea96-11e8-8ee1-b6204982bb38.png">
</p>

Tanto para generar la red, como para entrenarla y hacer las predicciones post-entrenamiento, hemos usado un framework de Python llamado Keras (https://keras.io). Se trata de un framework de alto nivel escrito en python que es capaz de ejecutarse por encima de Tensorflow o Theano entre otros.



Conjunto de datos y preprocesado
El conjunto de datos consiste en unos archivos excel con tres columnas. La primera columna indica el tiempo. La segunda columna indica la intensidad aplicada al electroimán. Por último, la última columna indica la altura a la que se encuentra el imán en suspensión.
En este problema en concreto, puede darse el caso de que para una misma intensidad, el imán se encuentre suspendido a la misma altura. Es por esto por lo que hemos tenido que realizar ciertas transformaciones al conjunto de datos para inyectarle a la red como conjunto de entrada, no solo la intensidad y la altura en un momento determinado, si no también un histórico de intensidades y alturas. El programa ha sido desarrollado para que pueda elegir el tamaño del intervalo de ese histórico.
Para aclarar más lo explicado en el párrafo anterior, en caso de que el tamaño del intervalo tenga el valor 5, cada línea del conjunto de datos final quedará de la siguiente manera:

i1-i5 + y1-y5 + y6
i2-i6 + y2-y6 + y7
i3-i7 + y3-y7 + y8
.
.
.

Como se puede observar, además de añadir tanto los intervalos de intensidades como los intervalos de alturas, se añade un último elemento, que es la altura a predecir. Finalmente se puede decir que la entrada será en el primer caso del ejemplo i1-i5 + y1-y5 y la salida a predecir será y6.

Cabe destacar que todos los datos son normalizados, de tal manera que sus valores se encuentran entre 0 y 1. Esto se realiza aplicando la siguiente ecuación:

<p align="center">
    <img width="521" alt="captura de pantalla 2018-11-17 a las 18 28 59" src="https://user-images.githubusercontent.com/15388747/48664463-ab049b80-ea96-11e8-82e5-ecff81e998d6.png">
</p>

Para llevar a cabo todas estas acciones, se ha decido crear una clase llamada DatasetManager. 


### Arquitectura de la red
La red está formada por 5 capas entre las cuales encontramos la capa de entrada, la capa de salida y tres capas ocultas con 20 neuronas cada una. En todas las neuronas de las capas ocultas, se utiliza una función de activación relu. En la capa de salida por lo contrario, se utiliza una función de activación sigmoid para que las salidas de la red estén entre 0 y 1 y así poder hacer correctamente los cálculos de errores con las salidas normalizadas en la fase de preprocesamiento. Cabe destacar que el método que genera la arquitectura de la red, tiene en cuenta el parámetro introducido por el usuario para el indicar el tamaño del intervalo del histórico, de forma que automáticamente, se genera la red con el número de neuronas en la capa de entrada correctas.

La función de coste es la función error cuadrático medio. Además, para optimizar el backpropagation, se utiliza el algoritmo Adam (https://arxiv.org/abs/1412.6980v8) con un learning rate de 0.001.

### Entrenamiento
Para llevar a cabo el entrenamiento de la red, se ha decidido elegir un tamaño de batch de 32. Con eso, aparte de añadir un poco de ruido para evitar que la red caiga en un mínimo local, consigue mejorar el tiempo de convergencia. Adicionalmente, se ha escogido un número de épocas de 600, de tal manera que se entrenará la red pasando en total 600 veces por el conjunto de datos.

A lo largo del entrenamiento, se utiliza un conjunto de datos para en cada iteración, evaluar el rendimiento de la red, de tal manera que al finalizar dicho entrenamiento, se construye una gráfica con el error en cada una de las iteraciones.

### Resultados
Tras haber llevado a cabo el entrenamiento de la red neuronal, hemos decido mostrar dos gráficas mediante las cuales podemos ver el resultado final conseguido. Por un lado se puede observar la gráfica del error.
Como se puede observar, rápidamente la gráfica converge pasando de más de un 60% de error a entre un 3.5% y 4%. Cabe resaltar el ruido que se observa en el error. Este ruido como ya ha sido comentado anteriormente, viene aparece por el tamaño del batch y la componente estocástica del algoritmos de optimización del backpropagation Adam.

<p align="center">
    <img width="612" alt="captura de pantalla 2018-11-17 a las 12 49 11" src="https://user-images.githubusercontent.com/15388747/48664477-d1c2d200-ea96-11e8-92aa-31fedb6e629a.png">
</p>

En cuanto a las predicciones que realiza la red neuronal, mostramos esta segunda gráfica:

<p align="center">
    <img width="610" alt="captura de pantalla 2018-11-17 a las 12 49 18" src="https://user-images.githubusercontent.com/15388747/48664484-ea32ec80-ea96-11e8-837b-a71c97c19b35.png">
</p>

Tal y como indica la leyenda, la serie naranja son los resultados previstos para la entrada del conjunto de datos de test y la serie azul son las salidas para dicho conjunto ya conocidas. Vemos que los resultados son bastantes buenos ya que la topología de las predicciones es bastante similar a la topología de las salidas ya conocidas.
