import numpy as np, librosa

def rms_db(wav):
    eps = 1e-10
    return float(20*np.log10(np.sqrt(np.mean(np.square(wav))+eps)+eps))

def clipping_ratio(wav, threshold=0.99):
    return float(np.mean(np.abs(wav) >= threshold))

def silence_ratio_amp(wav, frame_length=400, hop_length=160, db_threshold=-40.0):
    if len(wav) < frame_length:
        return 1.0 if rms_db(wav) < db_threshold else 0.0
    frames = librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length).T
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1)+1e-10)
    frame_db = 20*np.log10(frame_rms+1e-10)
    return float(np.mean(frame_db < db_threshold))

def compute_basic_metrics(wav, sr):
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=wav)
    zcr = librosa.feature.zero_crossing_rate(y=wav)
    return {
        'rms_db': rms_db(wav),
        'peak_abs': float(np.max(np.abs(wav))) if len(wav) else 0.0,
        'clipping_ratio': clipping_ratio(wav),
        'silence_ratio_amp': silence_ratio_amp(wav),
        'dc_offset': float(np.abs(np.mean(wav))) if len(wav) else 0.0,
        'zero_crossing_rate': float(np.mean(zcr)),
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_flatness_mean': float(np.mean(flatness)),
    }
