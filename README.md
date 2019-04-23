# ChatBots

Chat Bot creado usando nltk , lematizadores y Machine Learning (Regresion Logística)
El Chat Bot incluye puesta en producción ( con Flask)

Primer Modelo
-------------
Con con un enconder , codificamos las respuesta y preguntas
Tenemos respuestas predefinidas  ,codificadas
Se hace un modelo regresionando las respuestas codificadas y el codigo de la respuesta codificada

Segundo Modelo
--------------

En este script incluimos el algoritmo "bolsa de frases"
Con con un enconder , codificamos las respuesta y preguntas
Con K means se hace cluster de frases de respuesta
Se hace un modelo regresionando las respuestas codificadas y el codigo del cluster
La respuesta es el cluster, de ese cluster (bolsa) elegimos una respuesta al azar
Es mucho mas dinamico que el primer modelo , se alimenta tanto de las preguntas y las respuestas

