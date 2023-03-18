import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from ecg_feature_extraction.ecg_taxonomy import taxonomy, temporal_ecg_features
from ecg_feature_extraction.fiducial_point_detection import ecg_delineation, vector_fiducial
from ecg_feature_extraction.visualization_ecg import plot_original_ecg,plot_ecg_fiducial_points
sys.path.append('Humath/Functions')
from Humath.Functions.AnomalyDetection_ECG import anomalydetection_ecg



def main_features_ecg(signal, fs):
    ecg_fiducial_R, ecg_fiducial_points_nk, signal_filtered = ecg_delineation(signal=signal, fs=fs)
    delineation={'algoritmo R':ecg_fiducial_R, 'neurokit2': ecg_fiducial_points_nk, 'signal filtered':signal_filtered}
    signal_fiducial_zeros = np.zeros(signal.shape, dtype=int)
    vector_ecg_fiducial = vector_fiducial(ecg_fiducial_R, signal_fiducial_zeros)
    features = temporal_ecg_features(ecg_fiducial_R,signal_filtered,fs)
    return vector_ecg_fiducial,features, delineation


def tipo_arritmia_mimic(id,y):
    Arrhythmia = {'Bradycardia': 1, 'Tachycardia': 2, 'Ventricular_Flutter_Fib': 3, 'Ventricular_Tachycardia': 4,
              'Asystole': 5}

    if int(y) == 0:
        print(f'El paciente {id} segun la MIMIC no presenta arritmia \n')

    else:
        print(f'El paciente {id} Segun la MIMIC presenta {list(Arrhythmia.keys())[list(Arrhythmia.values()).index(int(y))]}\n')

def validar_taxonomia(ecg,t_start,t_end):
    fs=250
    tvent = 0.8 # duración de la ventana en segundos
    t = np.linspace(0,20,fs*20)
    AnoDet = anomalydetection_ecg(Fs = fs, tenvt = tvent)
    corru = AnoDet.fit_transform(ecg)

    if corru.sum() < 0.7*len(ecg):
        vector, features, fiducial = main_features_ecg(ecg, fs)
        print('\nTaxonomia con los puntos fiduciales encontrados con Neurokit2:\n')
        taxonomy(fiducial['neurokit2'],ecg,fs)
        print('\n')
        print('-'*60)
        print('\nTaxonomia con los puntos fiduciales encontrados en R:\n')
        taxonomy(fiducial['algoritmo R'],ecg,fs)
    else:
        print('más del 70% de la señal es corrupta')

    print('\n')
    print('='*60)
    print('SEÑAL ELECTROCARDIOGRÁFICA ANALIZADA')
    print('-'*60)
    print('\nSeñal de ECG original')
    plot_original_ecg(ecg,t_start,t_end,fs)
    plt.show()
    print('-'*60)
    print('\nSeñal de ECG con los puntos fiduciales encontrados con  neurokit2')
    plot_ecg_fiducial_points(fiducial['neurokit2'], t_start,t_end,fs,'Puntos fiduciales con neurokit2')
    plt.show()
    print('-'*60)
    print('\nSeñal de ECG con los puntos fiduciales encontrados con R')
    plot_ecg_fiducial_points(fiducial['algoritmo R'], t_start,t_end,fs,'Puntos fiduciales con algoritmo R')
    plt.show()








def main_validacion(id,t_start, t_end,df):
    ecg = df.loc[id,'ECG_II']
    y = df.loc[id,'Arrhythmia']
    ecg= np.array(ecg,dtype=float).reshape((len(ecg[0])))

    # si se desea ver el tipo de arritmia
    print('='*60)
    print('INFORMACIÓN DE LA MIMIC')
    print('-'*60)
    tipo_arritmia_mimic(id,y)
    print('\n')

    print('='*60)
    print('TAXONOMIA PARA IDENTIFICAR TIPOS DE ARRITMIAS')
    print('-'*60)
    validar_taxonomia(ecg,t_start,t_end)






if __name__ == '__main__':
    df = pd.read_json('ecg_ii_arrhythmia.json')
    main_validacion(2,0,5,df)

