"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def extract_features(our_DB):
    audio_files = os.listdir(our_DB)
    f_min = 20
    f_max = sr // 2
    all_features = []
    all_statistics = []
    
    print(f"{len(audio_files)} arquivos encontrados.")
    
    for i, audio_file in enumerate(audio_files, start=1):
        print(f"Processando {i}/{len(audio_files)}: {audio_file}")
        file_path = os.path.join(our_DB, audio_file)
        
        y, fs = librosa.load(file_path, sr=sr, mono=True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)
        spectral_contrast = librosa.feature.spectral_contrast(y=y)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y)
        f0 = librosa.yin(y, f_min, f_max)
        f0[f0 == f_max] = 0
        rms = librosa.feature.rms(y=y)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        onset_env = librosa.onset.onset_strength(y=y)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env)

        mfcc_stats = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.min(mfcc, axis=1),
            np.median(mfcc, axis=1), np.max(mfcc, axis=1), stats.skew(mfcc, axis=1), stats.kurtosis(mfcc, axis=1)
        ]).flatten()

        print(f"{mfcc_stats.shape} nahahha")
        
        spectral_contrast_stats = np.hstack([
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1), np.min(spectral_contrast, axis=1),
            np.median(spectral_contrast, axis=1), np.max(spectral_contrast, axis=1), stats.skew(spectral_contrast, axis=1), stats.kurtosis(spectral_contrast, axis=1)
        ]).flatten()

        other_features = np.hstack([
            np.mean(spectral_centroid), np.std(spectral_centroid), np.min(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.min(spectral_bandwidth),
            np.mean(spectral_flatness), np.std(spectral_flatness), np.min(spectral_flatness),
            np.mean(spectral_rolloff), np.std(spectral_rolloff), np.min(spectral_rolloff),
            np.mean(f0), np.std(f0), np.min(f0),
            np.mean(rms), np.std(rms), np.min(rms),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate), np.min(zero_crossing_rate),
            np.mean(tempo), np.std(tempo), np.min(tempo)
        ])

        features = np.hstack([mfcc_stats, spectral_contrast_stats, other_features])
        all_features.append(features)
        
    
    all_features = np.array(all_features)
    all_statistics = np.array(all_statistics)
    
    min_vals = np.min(all_features, axis=0)
    max_vals = np.max(all_features, axis=0)
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    all_normalized_features = (all_features - min_vals) / range_vals
    
    output_file = "features_db.csv"
    save_features_to_file(all_features, min_vals, max_vals, output_file)
    
    print(f"Extração e salvamento das features para a base de dados {our_DB} concluído.")

def save_features_to_file(features, min_vals, max_vals, output_file):
    with open(output_file, 'w') as f:
        #np.savetxt(f, np.vstack([min_vals, max_vals]), delimiter=",", fmt="%.6f")
        np.savetxt(f, features, delimiter=",", fmt="%.6f")
    print(f"Arquivo {output_file} salvo com sucesso!")


if __name__ == "__main__":
    plt.close('all')
    
    #--- Load file
    fName = "Queries/MT0000414517.mp3"
    our_DB="Queries" 
    global sr   
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono = mono)
    print(y.shape)
    print(fs)
    
    #--- Play Sound
    #sd.play(y, sr, blocking=False)
    
    #--- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)
    
    #--- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
    #--- Extract features
    extract_features(our_DB)    
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')
    