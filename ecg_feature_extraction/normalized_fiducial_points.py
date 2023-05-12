# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:25:41 2023

@author: Melissa
"""

import numpy as np
from collections import defaultdict
from fiducial_point_detection import ecg_delineation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib;
from scipy.stats import norm, kurtosis, skew

matplotlib.use('Qt5Agg')


# funcion para normalizar el tiempo, donde se obtiene el max, min y el tiempo normalizado
def normalizar(lista):
    xmin = min(lista)
    xmax = max(lista)
    for i, x in enumerate(lista):
        lista[i] = (x - xmin) / (xmax - xmin)
    return xmin, xmax, lista


# funcion para normalizar la ubicación de los puntos fiduciales
def normalizar2(xmin, xmax, x):
    x = (x - xmin) / (xmax - xmin)
    return x


def normalized_segment(fiducial_points, fs, cycle: int):
    """
    Normaliza puntos fiduciales de un ciclo de señal
    Parameters
    ----------
    fiducial_points:
    fs
    cycle

    Returns
    -------

    """
    segment = []
    for key, value in fiducial_points.items():
        segment.append(value[cycle])
        segment_t = [index / fs for index in segment]
        tmin, tmax, segment_t_normalized = normalizar(segment_t)
    return segment_t_normalized


def normalized_fiducial(fiducial_points, fs):
    """

    Parameters
    ----------
    fiducial_points
    fs

    Returns
    -------

    """
    segments = []
    del fiducial_points['ECG']
    del fiducial_points['Tiempo']

    for cycle in range(len(fiducial_points['ECG_T_Offsets'])):
        segment_t_normalized = normalized_segment(fiducial_points, fs, cycle)
        segments.append(segment_t_normalized)

    segments_dic = {}
    segments_n = np.array(segments).T

    for key, y in zip(fiducial_points.keys(), segments_n):
        segments_dic[key] = y

    return segments_dic


# Calculo de los puntos fiduciales normalizados
def normalized_fiducial2(Paciente, fs):
    """Esta función permite normalizar los puntos fiduciales en tiempo y amplitud
        ----------
        Paciente:dict
                  diccionario con los puntos fiduciales del paciente a analizar
        fs: frecuencia de muestreo
                
        Return
        ----------
        P1_data, P_data, P2_data, Q_data, R_data, S_data, T1_data, T_data, T2_data: list
                   puntos fiduciales normalizados
                
    """
    normalized_data = dict()
    P1_data = []
    P_data = []
    P2_data = []
    Q_data = []
    R_data = []
    S_data = []
    T1_data = []
    T_data = []
    T2_data = []
    J_data = []

    for i in range(len(Paciente['ECG_T_Offsets'])):
        P1 = Paciente['ECG_P_Onsets'][i]
        P = Paciente['ECG_P_Peaks'][i]
        P2 = Paciente['ECG_P_Offsets'][i]
        Q = Paciente['ECG_Q_Peaks'][i]
        R = Paciente['ECG_R_Peaks'][i]
        S = Paciente['ECG_S_Peaks'][i]
        T1 = Paciente['ECG_T_Onsets'][i]
        T = Paciente['ECG_T_Peaks'][i]
        T2 = Paciente['ECG_T_Offsets'][i]
        J = Paciente['ECG_R_Offsets'][i]

        # tiempo=Paciente['Tiempo'][P1:T2]
        tmin, tmax = P1 / fs, T2 / fs

        ecg = Paciente['ECG'][P1:T2]
        t = np.linspace(0, 1, T2 - P1)

        P1n = normalizar2(tmin, tmax, P1 / fs)
        Pn = normalizar2(tmin, tmax, P / fs)
        P2n = normalizar2(tmin, tmax, P2 / fs)
        Qn = normalizar2(tmin, tmax, Q / fs)
        Rn = normalizar2(tmin, tmax, R / fs)
        Sn = normalizar2(tmin, tmax, S / fs)
        T1n = normalizar2(tmin, tmax, T1 / fs)
        Tn = normalizar2(tmin, tmax, T / fs)
        T2n = normalizar2(tmin, tmax, T2 / fs)
        Jn = normalizar2(tmin, tmax, J / fs)

        # Puntos fiduciales normalizados del paciente
        P1_data.append(P1n)
        P_data.append(Pn)
        P2_data.append(P2n)
        Q_data.append(Qn)
        R_data.append(Rn)
        S_data.append(Sn)
        T1_data.append(T1n)
        T_data.append(Tn)
        T2_data.append(T2n)
        J_data.append(Jn)

        """
        plt.plot(t,ecg)
        plt.scatter(P1n, Paciente['ECG'][P1],c='red')
        plt.scatter(Pn, Paciente['ECG'][P],c='blue')
        plt.scatter(P2n, Paciente['ECG'][P2],c='cyan')
        plt.scatter(Qn, Paciente['ECG'][Q],c='orange')
        plt.scatter(Rn, Paciente['ECG'][R],c='yellow')
        plt.scatter(Sn, Paciente['ECG'][S],c='green')
        plt.scatter(T1n, Paciente['ECG'][T1],c='gray')
        plt.scatter(Tn, Paciente['ECG'][T],c='pink')
        plt.scatter(T2n, Paciente['ECG'][T2],c='purple')
        """

        normalized_data = {'ECG_P_Onsets': np.array(P1_data), 'ECG_P_Peaks': np.array(P_data),
                           'ECG_P_Offsets': np.array(P2_data),
                           'ECG_Q_Peaks': np.array(Q_data), 'ECG_R_Peaks': np.array(R_data),
                           'ECG_S_Peaks': np.array(S_data),
                           'ECG_T_Onsets': np.array(T1_data), 'ECG_T_Peaks': np.array(T_data),
                           'ECG_T_Offsets': np.array(T2_data),
                           'ECG_R_Offsets': np.array(J_data)}

    return normalized_data


def signal_caracterization(fiducial_normalized):
    """
        Toma los puntos fiduciales normalizados de una persona y extrae información estadística de estos

        Parámetros
        ----------
        fiducial_normalized: dict
            puntos fiduciales normalizados de la persona.

        Salidas
        -------
        features: dict
            diccionario con las características estadísticas de la persona
    """
    features = dict()
    for key, value in fiducial_normalized.items():
        features['mean_{}'.format(key)] = np.mean(value)
        features['std_{}'.format(key)] = np.std(value)
        # features['kurtosis_{}'.format(key)] = kurtosis(value,bias=False)
        # features['skew_{}'.format(key)] = skew(value,bias=False)
    return features


def flatten_features(features_list):
    flatten_features_list = defaultdict(list)
    for person in features_list:
        # you can list as many input dicts as you want here
        for key, value in person.items():
            flatten_features_list[key].append(value)
    return flatten_features_list


def flatten_fiducial(fiducial_list):
    flatten_fiducial_list = defaultdict(list)

    for person in fiducial_list:
        # you can list as many input dicts as you want here
        for key, value in person.items():
            flatten_fiducial_list[key].append(value)

    for key, value in flatten_fiducial_list.items():
        flatten_fiducial_list[key] = [item for sublist in value for item in sublist]

    return flatten_fiducial_list


if __name__ == '__main__':

    path = 'D:/Humath-Curie-General/Humath-Curie-ECG/datos/ecg_arritmias_sanos_corru70.json'
    signals = pd.read_json(path)

    # Si solo quiero lo datos de arritmia
    signals = signals.drop(signals[signals['Arrhythmia'] == 0].index)
    signals = signals.reset_index(drop=True)

    # todas las señales

    fiducial_n_list = []
    features_list = []
    for person in range(len(signals)):
        index=0

        try:
            signal_person = signals.loc[person, 'ECG_II'][0]
            signal_person = np.array(signal_person)
            signal_person = np.where(signal_person == None, np.nan, signal_person)
            label = signals.loc[person, 'Arrhythmia']

            fiducial_points_R, fiducial_points_nk, signal_filtered = ecg_delineation(signal=signal_person, fs=250)
            fiducial_points_n = normalized_fiducial2(fiducial_points_R, 250)
            features_signal = signal_caracterization(fiducial_points_n)
            features_signal['Arrhythmia'] = label
            fiducial_n_list.append(fiducial_points_n)

            if len(features_signal) == 21:
                features_list.append(features_signal)
            else:
                continue
        except:
            continue

    flattened_features = flatten_features(features_list)
    flattened_dic = dict(flattened_features)
    df_flattened_features = pd.DataFrame(flattened_dic)
    df_flattened_features.to_csv('features_arritmias.txt')

    # flattened_fiducial_n = flatten_fiducial(fiducial_n_list)
    # flattened_dic = dict(flattened_fiducial_n)
    # df_flattened = pd.DataFrame(flattened_dic)
    # df_flattened.to_csv('fiducial_normalized_arritmias.txt')
