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
import csv


def normalize_features(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maxs = np.max(matrix, axis=0)
    mins = np.min(matrix, axis=0)
    range_ = maxs - mins
    range_[range_ == 0] = 1.0
    
    norm_matrix = (matrix - mins) / range_
    return norm_matrix, maxs, mins

def extract_stats(feature):
    feature_mean = np.mean(feature, axis=1)
    feature_std = np.std(feature, axis=1)
    feature_skewness = stats.skew(feature, axis=1, nan_policy='omit')
    feature_kurtosis = stats.kurtosis(feature, axis=1, nan_policy='omit')
    feature_median = np.median(feature, axis=1)
    feature_max = np.max(feature, axis=1)
    feature_min = np.min(feature, axis=1)
    return np.column_stack([feature_mean, feature_std, feature_skewness, feature_kurtosis, feature_median, feature_max,feature_min])

def extract_features(our_DB):
    audio_files = sorted(os.listdir(our_DB))
    all_features = []
    print(f"{len(audio_files)} arquivos encontrados.")
    f_min = 20
    f_max = 22050 // 2

    for i, audio_file in enumerate(audio_files, start=1):
        print(f"A processar a musica {i}/{len(audio_files)}: {audio_file}")
        file_path = os.path.join(our_DB, audio_file)
        
        y, sr = librosa.load(file_path)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)
        spectral_contrast = librosa.feature.spectral_contrast(y=y)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y)
        f0 = librosa.yin(y, fmin=f_min, fmax=f_max)
        f0[f0 == f_max] = 0
        rms = librosa.feature.rms(y=y)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        tempo = np.array([librosa.beat.tempo(y=y)[0]])
        
        mfcc_stats = extract_stats(mfcc).flatten()
        spectral_centroid_stats = extract_stats(spectral_centroid)[0, :]
        spectral_bandwidth_stats = extract_stats(spectral_bandwidth)[0, :]
        spectral_contrast_stats = extract_stats(spectral_contrast).flatten()
        spectral_flatness_stats = extract_stats(spectral_flatness)[0, :]
        spectral_rolloff_stats = extract_stats(spectral_rolloff)[0, :]
        f0 = np.expand_dims(f0, axis=0)
        f0_stats = extract_stats(f0)[0,:]
        rms_stats = extract_stats(rms)[0, :]
        zero_crossing_rate_stats = extract_stats(zero_crossing_rate)[0, :]
        
        music_features = np.concatenate((
            mfcc_stats,
            spectral_centroid_stats,
            spectral_bandwidth_stats,
            spectral_contrast_stats,
            spectral_flatness_stats,
            spectral_rolloff_stats,
            f0_stats,
            rms_stats,
            zero_crossing_rate_stats,
            tempo
        ))
        
        all_features.append(music_features)
        
    all_features = np.array(all_features)
    all_normalized_features, maxs, mins = normalize_features(all_features)
    data = np.vstack([mins, maxs, all_normalized_features])
    
    output_file = "OUR_FM_ALL.csv"
    np.savetxt(output_file, data, delimiter=",", fmt="%.6f")
    print(f"Feito.")

def compare(file1, file2, tolerance=1e-4):
    differences = []
    with open(file1, newline='') as f1, open(file2, newline='') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for row_num, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            for col_num, (val1, val2) in enumerate(zip(row1, row2), start=1):
                if(col_num >175 and col_num<169): #incluir isto para excluir f0
                    val1 = val1.strip()
                    val2 = val2.strip()
                    try:
                        num1 = float(val1)
                        num2 = float(val2)
                        if abs(num1 - num2) > tolerance:
                            differences.append(
                                f"Linha {row_num}, Coluna {col_num}: {num1} != {num2}"
                            )
                    except ValueError:
                        if val1 != val2:
                            differences.append(
                                f"Linha {row_num}, Coluna {col_num}: '{val1}' != '{val2}'"
                            )
                            
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]
    output_file = f"{base1}_{base2}without_F0.txt"
    with open(output_file, "w", encoding="utf-8") as out_file:
        if differences:
            out_file.write("Diferenças encontradas:\n")
            out_file.writelines(f"{line}\n" for line in differences)
        else:
            out_file.write("Os arquivos são considerados equivalentes (com tolerância).\n")

if __name__ == "__main__":
    plt.close('all')
    
    #--- Load file
    fName = "Queries/MT0000414517.mp3"
    our_DB="allsongs"
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
    compare("./OUR_FM_ALL.csv","./validacao/FM_All.csv")    
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')
    