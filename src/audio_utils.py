import numpy as np, librosa, soundfile as sf

def load_and_standardize_audio(path, target_sr=16000, mono=True):
    try:
        wav, sr = sf.read(path, always_2d=False)
        wav = wav.astype(np.float32)
        if wav.ndim > 1 and mono:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        if not np.isfinite(wav).all():
            wav = np.nan_to_num(wav)
        return {'ok': True, 'waveform': wav.astype(np.float32), 'sample_rate': int(sr), 'duration_sec': float(len(wav)/sr) if sr > 0 else 0.0}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
