import pandas as pd
import numpy as np
import sys
sys.path.append('/ecg_feature_extraction')
from ecg_feature_extraction.ecg_taxonomy import temporal_ecg_features, taxonomy
from ecg_feature_extraction.fiducial_point_detection import ecg_delineation, vector_fiducial
from Humath.Functions.AnomalyDetection_ECG import anomalydetection_ecg


def main(signal, fs):
    ecg_fiducial_R, ecg_fiducial_points_nk, signal_filtered = ecg_delineation(signal=signal, fs=fs)
    fiducial={'algoritmo R':ecg_fiducial_R, 'neurokit2': ecg_fiducial_points_nk}
    signal_fiducial_zeros = np.zeros(signal.shape, dtype=int)
    vector_ecg_fiducial = vector_fiducial(ecg_fiducial_R, signal_fiducial_zeros)
    features = temporal_ecg_features(ecg_fiducial_R,signal_filtered,fs)
    return vector_ecg_fiducial,features, fiducial


if __name__ == '__main__':
    import pandas as pd
    # Estos tiene una fs= 250,  Wn_low = 60 y Wn_high = 0.5
    #path_arritmia = 'D:/Bioseñales/MIMIC/MIMIC_arritmia.txt'
    # signals = load_data_arrhythmia(path_arritmia)
    #signal = load_data_arrhythmia(path_arritmia)
    path= 'D:/Joven Investigador/ecg_interpretation/humat_curie/ecg_8.csv'
    path_signals = 'ecg_ii_arrhythmia.json'

    ecg = pd.read_csv(path, sep=",", index_col=0)
    signals = pd.read_json(path_signals)

    # caso taquicardia
    ecg_taquicardia = signals.loc[265,'ECG_II']
    ecg_taquicardia = np.array(ecg_taquicardia,dtype=float).reshape((len(ecg_taquicardia[0])))

    ecg_nue=ecg.to_numpy().reshape((len(ecg)))
    fs = 250

    tvent = 0.8 # duración de la ventana en segundos
    t = np.linspace(0,20,fs*20)
    AnoDet = anomalydetection_ecg(Fs = fs, tenvt = tvent)
    corru = AnoDet.fit_transform(ecg_taquicardia)


    if corru.sum() < 0.7*len(ecg_taquicardia):
        vector, features, fiducial = main(ecg_taquicardia, fs)
        taxonomy(fiducial['neurokit2'],ecg_taquicardia,fs)
        taxonomy(fiducial['algoritmo R'],ecg_taquicardia,fs)
    else:
        print('más del 70% de la señal esta corrupta')






