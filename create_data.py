import pandas as pd
import numpy as np
from Humath.Functions.AnomalyDetection_ECG import anomalydetection_ecg


path_signals = 'ecg_arritmias_sanos.json'
total_signals = pd.read_json(path_signals)
fs = 250
# contar el numero de señales por clase
etiqueta_counts = total_signals['Arrhythmia'].value_counts().sort_index()
print(etiqueta_counts)

proportions = []
for person in range(len(total_signals)):
    ecg = total_signals.loc[person,'ECG_II']
    ecg = np.array(ecg,dtype=float).reshape((len(ecg[0])))

    tvent = 0.8 # duración de la ventana en segundos
    t = np.linspace(0,20,fs*20)
    AnoDet = anomalydetection_ecg(Fs = fs, tenvt = tvent)
    corru = AnoDet.fit_transform(ecg)
    proportion = corru.sum()*0.02
    # Guardar la proporción en la lista
    proportions.append(proportion)

# Crear una nueva columna 'proportion' en el dataframe signals
total_signals['proportion'] = proportions

# Filtrar el dataframe para mantener solo las filas con 'proportion' menor o igual a 70%
final_signals = total_signals[total_signals['proportion'] <= 70]
# Eliminar la columna 'proportion' del dataframe signals
final_signals = final_signals.drop('proportion', axis=1)
# Resetear los índices del dataframe
final_signals = final_signals.reset_index(drop=True)
final_signals.to_json('ecg_arritmias_sanos_corru70.json')
