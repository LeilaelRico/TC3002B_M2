"""
* Cristian Leilael Rico Espinosa
* El programa pasa un 20% de las imágenes contenidas de forma aleatoria a otra carpeta.
"""

import os
import random
import shutil

# Ruta de la carpetas de origen y destino
carpeta_origen = "...\\Landscape\\Landscape Classification\\Landscape Classification\\Training Data\\Forest"
carpeta_destino = "...\\Landscape Classification\\Landscape Classification\\Testing Data\\Forest"

# Obtener la lista de archivos de imagen en la carpeta de origen
archivos_imagen = [archivo for archivo in os.listdir(carpeta_origen) if archivo.endswith((".jpg", ".jpeg", ".png"))]

# Calcular el número de archivos a mover (20% de los archivos)
num_archivos_a_mover = int(len(archivos_imagen) * 0.20)

# Obtener una lista aleatoria de archivos a mover
archivos_a_mover = random.sample(archivos_imagen, num_archivos_a_mover)

# Mover los archivos a la carpeta de destino
for archivo in archivos_a_mover:
    ruta_origen = os.path.join(carpeta_origen, archivo)
    ruta_destino = os.path.join(carpeta_destino, archivo)
    shutil.move(ruta_origen, ruta_destino)

print(f"Se han movido {num_archivos_a_mover} imágenes de forma aleatoria a la carpeta de destino.")
