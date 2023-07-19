import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import imgaug.augmenters as iaa

def cargar_imagenes(ruta):
    imagenes = []
    etiquetas = []

    for carpeta in os.listdir(ruta):
        ruta_carpeta = os.path.join(ruta, carpeta)
        if os.path.isdir(ruta_carpeta): 
            for archivo in os.listdir(ruta_carpeta):
                etiqueta = carpeta
                ruta_completa = os.path.join(ruta_carpeta, archivo)
                imagen = cv2.imread(ruta_completa)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagenes.append(imagen)
                etiquetas.append(etiqueta)

    return np.array(imagenes), np.array(etiquetas)

directorio_entrenamiento = 'img_preprocesadas2'

X, y = cargar_imagenes(directorio_entrenamiento)

# Definir las transformaciones de aumento de datos
seq = iaa.Sequential([
    iaa.Rotate((-10, 10)),  
    iaa.Flipud(0.5),  
    iaa.GaussianBlur(sigma=(0, 1.0)),  
])

X_augmented = seq.augment_images(X)

X_all = np.concatenate([X, X_augmented])
y_all = np.concatenate([y, y])

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Entrenar el modelo de reconocimiento de rostros
print('Entrenando el modelo...')
recognizer = SVC(C=1.0, kernel='linear', probability=True)
recognizer.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
accuracy = recognizer.score(X_test, y_test)
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))

# Guardar el modelo entrenado en un archivo pickle
modelo_entrenado = 'modelo2.pkl'
with open(modelo_entrenado, 'wb') as archivo:
    pickle.dump(recognizer, archivo)

