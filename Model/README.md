# Creación del Modelo

Para este modelo se decidió utilizar la arquitectura de ***InceptionV3*** sin incluir la capa del modelo original al querer que entrene en capas personalizadas.

Posterior a especificar que el modelo será secuencial y agregar el modelo base que contiene a ***InceptionV3***, se tiene lo siguiente:

1. Capa de reducción de dimensionalidad global: Se agrega una capa de GlobalAveragePooling2D que reduce la dimensionalidad de las características extraídas de la base InceptionV3 a través de un promedio global.
2. Capa *dense*: Posee 256 neuronas y activación ReLU.
3. Capa de salida: Se agrega una capa densa final con 5 neuronas (correspondientes a las 5 clases de salida) y activación softmax. Esto genera una distribución de probabilidad sobre las clases, indicando la probabilidad de que una imagen pertenezca a cada clase.

La decisión por utilizar ***InceptionV3*** se hizo por su gran capacidad para identificar correctamente los elementos que pertenecen a cada clase.

## Justificación

Si bien inicialmente se planeaba utilizar **MobileNetV2** como el artículo de *"Landscape Classification"*, la investigación de *"Classifying Tourists’ Photos"* me llevó a considerar hacer el cambio a **InceptionV3**.
Aunado a todo esto, decidí mantener una arquitectura similar a la de los ejemplos vistos en clase más que nada para no generar una arquitectura que esté o simplificada o sea robusta en exceso.

## Recursos Bibliográficos

* Buscombe D, Ritchie AC. Landscape Classification with Deep Neural Networks. Geosciences. 2018; 8(7):244. https://doi.org/10.3390/geosciences8070244
* Kim, J., Kang, Y., Cho, N., & Park, S. (2021). Classifying Tourists’ Photos and Exploring Tourism Destination Image Using a Deep Learning Model. Abstracts of the ICA, 3, 150.
