# -*- coding: utf-8 -*-
"""A01707023__Proyecto.ipynb


Original file is located at
    https://colab.research.google.com/drive/1JcWZyBYjLjzBH2qKYVfGOY2bfh6C3qFM

# Proyecto
Cristian Leilael Rico Espinosa

Matrícula: A01707023

Proyecto para el Módulo 2 de "Desarrollo de aplicaciones avanzadas de ciencias computacionales". En este se realizará un modelo el cual se entrenará con imágenes de paisajes para que, posteriormente, pueda identificar imágenes pertenecientes a estos.
"""

from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/TC3002B_M2"
# !ls


base_dir = '.\\Model_V1\\images\\'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

"""## Preprocesado de Datos

Se comienza con la carga y preprocesamiento de imágenes desde directorios, escalando sus valores de píxeles al rango de 0 a 1 para, posteriormente, crear generadores de datos los cuales serán utilizados durante el entrenamiento y la prueba del modelo.
"""

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# """### Imágenes de *Train*

# Para comprobar si la conversión se hizo adecuadamente, se utiliza *plot* para mostrar visualizaciones de algunas de las imágenes.
# """

# plt.figure(figsize=(20, 6))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     batch = next(train_generator)
#     image = batch[0][0]
#     plt.imshow(image)
#     plt.axis('off')
# plt.show()

# """### Imágenes de *Test*

# Para comprobar si la conversión se hizo adecuadamente, se utiliza *plot* para mostrar visualizaciones de algunas de las imágenes.
# """

# plt.figure(figsize=(20, 6))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     batch = next(test_generator)
#     image = batch[0][0]
#     plt.imshow(image)
#     plt.axis('off')
# plt.show()


# ----- Modelo -----

base_model = InceptionV3(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False


# En caso de requerir fine tune, cambiar trainable a true
# fine_tune_at = 100


# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False


model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))  # 5 clases de salida

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

history = model.fit(
    train_generator,
    # steps_per_epoch=200,
    epochs=50
)

model.save('3rdrun.keras')

acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc)+1)

plt.figure()
# subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1, 2, figsize=(10, 3))
axarr[0].plot(epochs, acc, label='train accuracy')
axarr[0].legend()
axarr[1].plot(epochs, loss, label='train loss')
axarr[1].legend()

test_loss, test_acc = model.evaluate(test_generator)
print('\ntest acc :\n', test_acc)

predictions = model.predict(test_generator)
predict_class = (predictions > 0.5).astype("int32")
predict_class.shape

# Comparación de Resultados

test_imgs = test_generator[0][0]
test_labels = test_generator[0][1]


predictions = model.predict(test_imgs)
classes_x = np.argmax(predictions, axis=1)
classes_x

# ----- Matriz de Confusión -----

# Resultados reales
true_labels = np.argmax(test_labels, axis=1)

# Predicciones hechas por el modelo
predicted_labels = np.argmax(predictions, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualización
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5"], yticklabels=["Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5"])
plt.xlabel('Predicción')
plt.ylabel('Resultados Correctos')
plt.title('Matriz de Confusión')
plt.show()
