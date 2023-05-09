# Humath-Curie-ECG

Algoritmo para la detección de arritmias en señales de electrocardiograma

Los datos son extraídos de la base de datos MIMIC, donde podemos ver lo siguiente:

- ecg_arritmias_sanos.json: allí se encuentran los ultimos 20 segundos de las señales de ecg de pacientes en UCI (728 personas), con 6 etiquetas:
  - 0: No presenta arritmia
  - 1: Bradicardia
  - 2: Taquicardia
  - 3: Aleteo ventricular
  - 4: Taquicardia ventricular
  - 5 : Asistolia

- ecg_arritmias_sanos_corru70.json: allí se encuentran los mismos datos de ecg_arritmias_sanos.json pero solo aquellos cuyos porcentjas de datos corruptos es menor al 70%, siendo un total de 705 pacientes.
- fiducal_normalized_arritmias_sanos.txt: puntos fiduciales normalizados usando los datos ecg_arritmias_sanos_corru70.json, allí se incluyen arritmias y casos de no arritmias.
- fiducal_normalized_arritmias.txt: puntos fiduciales normalizados usando los datos ecg_arritmias_sanos_corru70.json, allí se incluyen solo arritmias.


