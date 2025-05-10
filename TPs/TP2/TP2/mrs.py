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
import scipy.fft
from scipy.spatial.distance import euclidean, cityblock, cosine

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
        
        feature_parts = [
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
        ]
        
        total_length = sum(part.size for part in feature_parts)
        music_features = np.zeros(total_length)
        
        current_pos = 0
        for part in feature_parts:
            part_size = part.size
            music_features[current_pos:current_pos + part_size] = part
            current_pos += part_size
        
        if i == 1:
            all_features = np.zeros((len(audio_files), music_features.size))
        all_features[i-1, :] = music_features
        
    all_normalized_features, maxs, mins = normalize_features(all_features)
    data = np.vstack([mins, maxs, all_normalized_features])
    
    output_file = "OUR_FM_ALL.csv"
    np.savetxt(output_file, data, delimiter=",", fmt="%.6f")
    print(f"Feito.")

def custom_spectral_centroid(y, sr=22050, w=2048, h=512):
    n_samples = len(y)
    
    pad_length = w + h * (int(np.ceil((n_samples - w) / h)))
    if pad_length > n_samples:
        y = np.pad(y, (0, pad_length - n_samples))
    
    n_samples = len(y)
    n_frames = 1 + (n_samples - w) // h
    
    sc = np.zeros(n_frames)
    
    window = np.hanning(w)
    freq_bins = np.fft.rfftfreq(w, 1/sr)
    for i in range(n_frames):
        start = i * h
        end = start + w
        
        if end <= n_samples:
            frame = y[start:end]
            if len(frame) == w:
                frame = frame * window
                spectrum = np.abs(scipy.fft.rfft(frame))
                
                if np.sum(spectrum) > 0:
                    sc[i] = np.sum(freq_bins * spectrum) / np.sum(spectrum)
                else:
                    sc[i] = 0
    
    return sc

def compare_spectral_centroids(y, sr=22050):
    sc_librosa = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
    
    sc_custom = custom_spectral_centroid(y, sr)
    
    min_length = min(len(sc_librosa) - 2, len(sc_custom))
    
    if min_length <= 0:
        return 0.0, 0.0
    
    sc_librosa_adjusted = sc_librosa[2:2+min_length]
    sc_custom_adjusted = sc_custom[:min_length]
    
    correlation = scipy.stats.pearsonr(sc_librosa_adjusted, sc_custom_adjusted)[0]
    rmse = np.sqrt(np.mean((sc_librosa_adjusted - sc_custom_adjusted) ** 2))
    
    return correlation, rmse

def save_metrics(filename, audios_folder):
    audio_files = sorted(os.listdir(audios_folder))
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"Processing file {i}/{len(audio_files)}: {audio_file}")
        file_path = os.path.join(audios_folder, audio_file)
        
        try:
            y, sr = librosa.load(file_path, sr=22050)
            correlation, rmse = compare_spectral_centroids(y, sr)
            results.append([correlation, rmse])
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results.append([0.0, 0.0])

    np.savetxt(filename, results, delimiter=',', comments='', fmt="%.6f")
    print(f"Results saved to {filename}")

def compare(file1, file2, tolerance=1e-4):
    differences = []
    with open(file1, newline='') as f1, open(file2, newline='') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for row_num, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            for col_num, (val1, val2) in enumerate(zip(row1, row2), start=1):
                if(col_num >175 and col_num<169): #descomentar esta linha para excluir f0
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


def compute_similarity_matrices(pretended_top, query_file, db_file, audio_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    full_db = np.loadtxt(db_file, delimiter=",")
    query = np.loadtxt(query_file, delimiter=",")
    db_features = full_db[2:]
    query_features = query[2]

    n_songs = db_features.shape[0]
    euclidean_distances = np.zeros(n_songs)
    manhattan_distances = np.zeros(n_songs)
    cosine_distances = np.zeros(n_songs)

    for i in range(n_songs):
        song_vector = db_features[i]
        euclidean_distances[i] = euclidean(query_features, song_vector)
        manhattan_distances[i] = cityblock(query_features, song_vector)
        cosine_distances[i] = cosine(query_features, song_vector)


    euclidean_distances = np.array(euclidean_distances)
    manhattan_distances = np.array(manhattan_distances)
    cosine_distances = np.array(cosine_distances)

    np.savetxt(os.path.join(output_folder, "similarity_euclidean.csv"), euclidean_distances[:, None], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_folder, "similarity_manhattan.csv"), manhattan_distances[:, None], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_folder, "similarity_cosine.csv"), cosine_distances[:, None], delimiter=",", fmt="%.6f")

    euclidean_top10_idx = np.argsort(euclidean_distances)[:pretended_top]
    manhattan_top10_idx = np.argsort(manhattan_distances)[:pretended_top]
    cosine_top10_idx = np.argsort(cosine_distances)[:pretended_top]

    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".mp3")])

    global euclidean_top10
    global manhattan_top10
    global cosine_top10

    euclidean_top10 = [(audio_files[i], euclidean_distances[i]) for i in euclidean_top10_idx]
    manhattan_top10 = [(audio_files[i], manhattan_distances[i]) for i in manhattan_top10_idx]
    cosine_top10 = [(audio_files[i], cosine_distances[i]) for i in cosine_top10_idx]

    with open(os.path.join(output_folder, "rankings.txt"), "w") as f:
        f.write("Ranking: Euclidean-------------\n")
        for name, dist in euclidean_top10:
            f.write(f"{name}\t{dist:.6f}\n")
        f.write("\nRanking: Manhattan-------------\n")
        for name, dist in manhattan_top10:
            f.write(f"{name}\t{dist:.6f}\n")
        f.write("\nRanking: Cosine-------------\n")
        for name, dist in cosine_top10:
            f.write(f"{name}\t{dist:.6f}\n")

def precision(meta_ranking, distance_ranking, name):
    meta_titles = set([title for title, _ in meta_ranking])
    distance_titles = set([title for title, _ in distance_ranking])
    
    intersecao = meta_titles.intersection(distance_titles)
    precisao = ((len(intersecao)-1) / (len(meta_titles)-1)) * 100 

    with open("results_ranking/rankings.txt", "a", encoding="utf-8") as f:
        f.write(f"\nprecision ({name} with Metadata): {precisao:.1f}")


def count_rows(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1

def metadata(pretended_top, query_file, db_file, audio_folder):
    query_size = count_rows(query_file)
    db_size = count_rows(db_file)

    query_metadata = [None] * query_size
    db_metadata = [None] * db_size

    with open(query_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            query_metadata[i] = row

    with open(db_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            db_metadata[i] = row

    if not query_metadata or not db_metadata:
        print("Error: Metadata files are empty or could not be read.")
        return
    
    query = query_metadata[0]
    
    query_artist = query.get('Artist', '').lower()
    print(f"Query Artist: {query_artist}")
    query_genres = [genre.strip().lower() for genre in query.get('GenresStr', '').split(';') if genre.strip()]
    print(f"Query Genres: {query_genres}")
    query_moods = [mood.strip().lower() for mood in query.get('MoodsStrSplit', '').split(';') if mood.strip()]
    print(f"Query Moods: {query_moods}")
    
    similarity_scores = np.zeros(db_size)
    song_ids = [None]*(db_size)
    i=0
    for item in db_metadata:
        score = 0
        db_artist = item.get('Artist', '').lower()
        db_genres = [genre.strip().lower() for genre in item.get('GenresStr', '').split(';') if genre.strip()]
        db_moods = [mood.strip().lower() for mood in item.get('MoodsStrSplit', '').split(';') if mood.strip()]
        
        if query_artist == db_artist:
            score += 1
        
        for genre in query_genres:
            if genre in db_genres:
                score += 1
        
        for mood in query_moods:
            if mood in db_moods:
                score += 1
        
        similarity_scores[i]=score
        song_ids[i]=item.get('SONG_ID', item.get('SongID', ''))
        i=i+1
    
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".mp3")])
    
    indices = np.argsort(similarity_scores)[::-1]
    top_10_indices = indices[:pretended_top]
    
    top_10_songs = [(audio_files[i], similarity_scores[i]) for i in top_10_indices if i < len(audio_files)]
    
    os.makedirs("results_ranking", exist_ok=True)
    with open("results_ranking/rankings.txt", "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("Ranking: Metadata Similarity-------------\n")
        for title, score in top_10_songs:
            f.write(f"{title}\t{score}\n")
    
    np.savetxt("results_ranking/similarity_metadata.csv", np.array(similarity_scores)[:, None], delimiter=",", fmt="%.1f")
    
    precision(top_10_songs, euclidean_top10, "Euclidean")
    precision(top_10_songs, manhattan_top10, "Manhattan")
    precision(top_10_songs, cosine_top10, "Cosine")




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
    #extract_features(our_DB)
    #compare("./OUR_FM_ALL.csv","./validacao/FM_All.csv")   
    #correlation, rmse = compare_spectral_centroids(y, sr)
    #print(f"Pearson Correlation: {correlation:.6f}")
    #print(f"RMSE: {rmse:.6f}") 
    #save_metrics("spectral_centroid_metrics.csv", our_DB)
    #compare("./spectral_centroid_metrics.csv", "./validacao/metricsSpectralCentroid.csv")
    compute_similarity_matrices(11,query_file="validacao/FM_Q.csv",db_file="./validacao/FM_All.csv",audio_folder="./allsongs",output_folder="results_ranking")
    metadata(11,query_file="./query_metadata.csv",db_file="./panda_dataset_taffc_metadata.csv",audio_folder="./allsongs"); 
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')
    