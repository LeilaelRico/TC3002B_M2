# TC3002B_M2

Proyecto para el Módulo 2 de "Desarrollo de aplicaciones avanzadas de ciencias computacionales". En este se realizará un modelo el cual se entrenará con imágenes de paisajes para que, posteriormente, pueda identificar imágenes pertenecientes a estos.

## Contenido del *Dataset*

El *dataset* a utilizar fue obtenido de [*Kaggle*](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images/data) y pertenece al usuario ***DeepNets***

Originalmente, este contiene imágenes de diferentes paisajes separados en 5 clases:

* Costa.
* Desierto.
* Montañas.
* Glaciar.
* Bosque.

Las imágenes son variadas en su forma, algunas contienen elementos como marcas de agua o bordes en las esquinas.

Estas imágenes se encontraban divididas en 3 subdirectorios conteniendo diferentes cantidades de archivos en cada uno; dentro del directorio *training* habían 2000 imágenes por cada clase mientras que en *testing* solamente habían 100 por cada una de las divisiones.

Para asegurar una correcta distribución de 80% a 20%, todas las imágenes contenidas en la carpeta *testing* fueron movidas a *training* para, posteriormente, renombrarlas y mover 420 imágenes por clase a *testing* terminando así con:

* *Testing*: 420 imágenes por clase.
* *Training*: 1680 imágenes por clase.

De momento, las 300 imágenes por clase almacenadas en la carpeta *validation* se reservarán para un futuro uso en caso de ser necesarias.

## Links de Interés

[*Dataset* de *Kaggle*](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images/data)

[Carpeta de *Drive* con imágenes del proyecto](https://drive.google.com/drive/folders/1MYSXEZ1Kj9biLE9t6nm71JLbR2NCDUnU?usp=sharing)
