import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = '.\\images\\'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'validation')

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

validation_datagen = ImageDataGenerator(
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
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical'
)

class_names = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']

# Creación del modelo


layer_names = ['clre1', 'clre2', 'clre3', 'clre4', 'clre5']

base_model = InceptionV3(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = True

# En caso de requerir fine tune, cambiar trainable a true
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

inputs = layers.Input(shape=(150, 150, 3), name='input')

x = base_model(inputs)

x = layers.Flatten(name=layer_names[1])(x)
x = layers.Dense(512, activation='relu', name=layer_names[2])(x)
x = layers.Dropout(0.5, name=layer_names[3])(x)
outputs = layers.Dense(5, activation='softmax', name=layer_names[4])(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

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

# Graficar la precisión del entrenamiento y la validación
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Graficar la pérdida del entrenamiento y la validación
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Comparación de Resultados
test_imgs = test_generator[0][0]
test_labels = test_generator[0][1]

test_nV = np.argmax(test_labels, axis=1)


predictions = model.predict(test_imgs)
classes_x = np.argmax(predictions, axis=1)
classes_x

# Matriz de Confusión


# Obtener todas las etiquetas reales
all_true_labels = []
for i in range(len(train_generator)):
    batch = train_generator[i]
    true_labels_batch = np.argmax(batch[1], axis=1)
    all_true_labels.extend(true_labels_batch)

# Obtener todas las predicciones
all_predictions = model.predict(train_generator)

# Convertir las etiquetas a índices de clases
predicted_labels = np.argmax(all_predictions, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(all_true_labels, predicted_labels)

# Visualización
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=[
            "Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5"], yticklabels=["Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5"])
plt.xlabel('Predicción')
plt.ylabel('Resultados Correctos')
plt.title('Matriz de Confusión de todo el conjunto de entrenamiento')
plt.show()

# Muestreo de Predicciones


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(5))
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 10
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_nV, test_imgs)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_nV)
plt.tight_layout()
plt.show()

model.save('modelv2.keras')