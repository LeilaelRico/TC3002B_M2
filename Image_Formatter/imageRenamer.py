"""
* Cristian Leilael Rico Espinosa
* El programa ayuda a rápidamente renombrar a las imágenes contenidas en el directorio especificado.
"""

import os

# Ruta al folder en el que se encuentran las imágenes
# directorio = 'ruta\\al\\directorio\\Landscape\\Landscape Classification\\Landscape Classification\\Training Data\\Carpeta-con-imágenes'
directorio = 'C:\\Users\\crisb\\Downloads\\Landscape\\Landscape Classification\\Landscape Classification\\Training Data\\Glacier'

# Obtener la lista de archivos en el directorio
carpeta = os.listdir(directorio)

# Contador para mantener la secuencia
contador = 1

# Recorrer cada archivo en el directorio
for imagen in carpeta:
    # Comprobar si el archivo es una imagen (puedes ajustar esta condición según el tipo de archivo)
    if imagen.endswith('.jpg') or imagen.endswith('.png') or imagen.endswith('.jpeg'):
        # Crear el nuevo nombre de archivo
        nuevo_nombre = f'G{contador}.jpeg'
        # Ruta completa del archivo original
        ruta_original = os.path.join(directorio, imagen)
        # Ruta completa del nuevo archivo
        ruta_nuevo = os.path.join(directorio, nuevo_nombre)
        # Renombrar el archivo
        os.rename(ruta_original, ruta_nuevo)
        # Incrementar el contador
        contador += 1

print("Archivos renombrados")
