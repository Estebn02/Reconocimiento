import cv2
import pickle

# Cargar el modelo entrenado desde el archivo pickle
modelo_entrenado = 'modelo2.85.19.pkl'  # Ruta del modelo entrenado
with open(modelo_entrenado, 'rb') as archivo:
    clf = pickle.load(archivo)

# Inicializar el detector de caras de OpenCV
cascada_cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la cámara
cam = cv2.VideoCapture(0)

while True:
    # Capturar el fotograma de la cámara
    ret, fotograma = cam.read()

    # Convertir a escala de grises
    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el fotograma
    caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

    # Recorrer cada cara detectada
    for (x, y, w, h) in caras:
        # Extraer la región de interés (ROI) que contiene la cara
        roi = gris[y:y+h, x:x+w]

        # Redimensionar la ROI a la misma dimensión que las imágenes de entrenamiento
        roi = cv2.resize(roi, (1000, 1000))

        # Aplanar la ROI para que tenga la misma forma que las imágenes de entrenamiento
        roi = roi.reshape(1, -1)

        # Realizar la predicción utilizando el modelo entrenado
        etiqueta = clf.predict(roi)[0]

        # Dibujar un rectángulo alrededor de la cara y mostrar la etiqueta
        cv2.rectangle(fotograma, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(fotograma, etiqueta, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma con las caras detectadas
    cv2.imshow('Reconocimiento Facial', fotograma)

    # Salir si se presiona la tecla 'p'
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Liberar la cámara y cerrar las ventanas
cam.release()
cv2.destroyAllWindows()
