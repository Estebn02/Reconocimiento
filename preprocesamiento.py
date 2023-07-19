import os
import cv2
import numpy as np

def preprocesar_imagenes(ruta_origen, ruta_destino, tamano=(1000, 1000)):
    # Crear la carpeta de destino si no existe
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    # Cargar el clasificador de detección de rostros
    detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for persona in os.listdir(ruta_origen):
        ruta_persona_origen = os.path.join(ruta_origen, persona)
        if os.path.isdir(ruta_persona_origen):  # Comprobar si es una carpeta
            ruta_persona_destino = os.path.join(ruta_destino, persona)
            if not os.path.exists(ruta_persona_destino):
                os.makedirs(ruta_persona_destino)

            for archivo in os.listdir(ruta_persona_origen):
                ruta_completa_origen = os.path.join(ruta_persona_origen, archivo)

                imagen = cv2.imread(ruta_completa_origen)
                if imagen is None:
                    continue  # Saltar al siguiente archivo si no se puede leer

                # Detección de rostros
                rostros = detector_rostros.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(rostros) == 0:
                    continue  # Saltar al siguiente archivo si no se detectan rostros

                # Recorte de rostros
                (x, y, w, h) = rostros[0]
                rostro_recortado = imagen[y:y+h, x:x+w]

                # Normalización de iluminación (Ecualización del histograma en escala de grises)
                rostro_gris = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2GRAY)
                rostro_ecualizado = cv2.equalizeHist(rostro_gris)

                # Redimensionamiento
                rostro_redimensionado = cv2.resize(rostro_ecualizado, tamano)

                ruta_completa_destino = os.path.join(ruta_persona_destino, archivo)
                cv2.imwrite(ruta_completa_destino, rostro_redimensionado)

# Rutas de origen y destino
directorio_origen = 'img ML2'
directorio_destino = 'img_preprocesadas2'

# Realizar el preprocesamiento y guardar las imágenes preprocesadas
preprocesar_imagenes(directorio_origen, directorio_destino)

