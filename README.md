# TFM-ML-Privacy-Comparative - Files

## Ficheros ejecutables

**execute_attack_MIA_art**: Ejecuta el ataque MIA utilizando la librería ART para cada modelo localizado en la carpeta "models" (ficheros finalizados en la extensión hdf5).

**execute_attack_MIA**: Ejecuta el ataque MIA utilizando la librería Tensorflow Privacy para cada modelo localizado en la carpeta "models" (ficheros finalizados en la extensión hdf5).

**execute_drawEpsilons**: Obtiene los valores de epsilon para los datasets predefinidos (variable "datasets").

**execute_trainModels**: Entrena los modelos y los almacena en el directorio models. Entrena utilizando Differencial Privacy y sin ella. 

**execute_modelsDataToCsv**: Transorma la información sobre los modelos entrenados (ficheros finalizados en la extension hdf5) encontrados en el directorio "models" y la muestra en formato csv.

## Ficheros de apoyo

**loadDatasets**: Carga y devuelve los datasets pasados como parámetro.

**loadFlags**: Devuelve los flags de entrenamiento oportunos para el dataset pasado como parámetro.

**loadModel**: Crea y devuelve el modelo para el dataset pasado como parámetro.

**loadOptimizer**: Crea y devuelve el optimizador y funcion error para el dataset y los flags pasados como parámetros.

