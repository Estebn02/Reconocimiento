README - Proyecto de Machine Learning
Descripción del Proyecto
Este proyecto de Machine Learning está diseñado para reconocer a personas en imágenes utilizando técnicas de preprocesamiento y entrenamiento de un modelo de Machine Learning. El proyecto consta de tres archivos principales: preprocesamiento.py, entrenamiento.py, y reconocer.py. A continuación, se explica la funcionalidad de cada uno de ellos:

Archivos
1. preprocesamiento.py
Este archivo se encarga de realizar el preprocesamiento de las imágenes contenidas en la carpeta img_ML2. El preprocesamiento incluye varias técnicas para mejorar la calidad de las imágenes y prepararlas para el entrenamiento del modelo. Al finalizar el preprocesamiento, se creará una nueva carpeta llamada img_preprocesadas2, donde se almacenarán las imágenes preprocesadas.

2. entrenamiento.py
Una vez que las imágenes han sido preprocesadas, este archivo se encarga de entrenar un modelo de Machine Learning utilizando las imágenes almacenadas en la carpeta img_preprocesadas2. El modelo entrenado será guardado en un archivo llamado modelo_entrenado.pkl para su uso posterior.

3. reconocer.py
Este archivo utiliza el modelo entrenado (modelo_entrenado.pkl) para llevar a cabo el reconocimiento de personas en imágenes. El programa buscará en la carpeta especificada las imágenes que contienen a las personas que se desean reconocer. El resultado será la identificación de las personas reconocidas en cada imagen.

Instrucciones de Uso
Coloca las imágenes que deseas utilizar para el entrenamiento en la carpeta img_ML2.

Ejecuta el archivo preprocesamiento.py para realizar el preprocesamiento de las imágenes. Las imágenes preprocesadas se almacenarán en la carpeta img_preprocesadas2.

Ejecuta el archivo entrenamiento.py para entrenar el modelo de Machine Learning utilizando las imágenes preprocesadas. El modelo entrenado se guardará en el archivo modelo_entrenado.pkl.

Una vez que el modelo ha sido entrenado, puedes utilizar el archivo reconocer.py para reconocer a las personas en las imágenes. Asegúrate de colocar las imágenes que deseas analizar en una carpeta específica y modificar el código para apuntar a esa carpeta.

Requisitos
Asegúrate de tener instaladas las siguientes bibliotecas de Python en tu entorno:

Biblioteca1
Biblioteca2
Biblioteca3
Notas
Puedes ajustar los parámetros del preprocesamiento y del modelo de Machine Learning en los respectivos archivos .py para obtener mejores resultados.

Asegúrate de contar con suficientes imágenes de entrenamiento para lograr un buen rendimiento del modelo.

Si encuentras algún problema o tienes preguntas, no dudes en contactar al autor del proyecto.

¡Buena suerte con tu proyecto de Machine Learning!
