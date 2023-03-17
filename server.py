from fastapi import FastAPI, UploadFile, File
import numpy as np
import uvicorn
import pandas as pd
import sys
sys.path.append('/ecg_feature_extraction')
from ecg_feature_extraction.ecg_taxonomy import temporal_ecg_features
from ecg_feature_extraction.fiducial_point_detection import ecg_delineation, vector_fiducial





app= FastAPI()

def main(signal, fs):
    ecg_fiducial_R, ecg_fiducial_points_nk, signal_filtered = ecg_delineation(signal=signal, fs=fs)
    signal_fiducial_zeros = np.zeros(signal.shape, dtype=int)
    vector_ecg_fiducial = vector_fiducial(ecg_fiducial_R, signal_fiducial_zeros)
    features = temporal_ecg_features(ecg_fiducial_R,signal_filtered,fs)
    return vector_ecg_fiducial,features

@app.get("/")
def read_root():
    return {"Humath Curie": "Análisis de señales electrocardiográficas"}


@app.post("/features_ecg/")
async def process_image_file(csv_file: UploadFile = File(...)):
    dataframe = pd.read_csv(csv_file.file,index_col=0)
    vector, features= main(dataframe,250)
    return features

if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
