from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _compute_energy_snr(wav: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    """
    Robust no-reference SNR estimate using frame energies.

    Method:
    - split into short frames
    - lowest-energy 10% => noise estimate
    - highest-energy 30% => signal estimate
    - SNR = 10 * log10(signal / noise)

    This is not a perfect clean-reference SNR, but it is practical and stable
    for corpus filtering.
    """
    if wav is None or len(wav) < int(0.1 * sr):
        return None, "too_short"

    frame_length = int(0.02 * sr)   # 20 ms
    hop_length = int(0.01 * sr)     # 10 ms

    if len(wav) < frame_length:
        return None, "too_short"

    frames = []
    for start in range(0, len(wav) - frame_length + 1, hop_length):
        frame = wav[start:start + frame_length]
        frames.append(np.mean(frame ** 2))

    if len(frames) < 10:
        return None, "too_few_frames"

    energies = np.asarray(frames, dtype=np.float32)
    energies = np.maximum(energies, 1e-10)

    noise_cut = np.percentile(energies, 10)
    signal_cut = np.percentile(energies, 70)

    noise_frames = energies[energies <= noise_cut]
    signal_frames = energies[energies >= signal_cut]

    if len(noise_frames) == 0 or len(signal_frames) == 0:
        return None, "degenerate"

    noise_energy = float(np.mean(noise_frames))
    signal_energy = float(np.mean(signal_frames))

    if noise_energy <= 0:
        return None, "invalid_noise"

    snr_db = 10.0 * np.log10(signal_energy / noise_energy)
    snr_db = float(np.clip(snr_db, -5.0, 50.0))
    return round(snr_db, 2), "energy_percentile"

def _compute_wada_snr_if_available(wav: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    """
    Optional WADA-SNR path.
    If wada_snr is unavailable or fails, return None.
    """
    try:
        import librosa
        from wada_snr import wada_snr

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        snr_db = float(wada_snr(wav))
        snr_db = float(np.clip(snr_db, -5.0, 50.0))
        return round(snr_db, 2), "wada_snr"
    except Exception as e:
        logger.debug("WADA-SNR unavailable or failed: %s", e)
        return None, "wada_unavailable"

def compute_snr_db(wav: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    """
    Primary SNR interface.

    Priority:
    1. WADA-SNR if available
    2. robust energy-percentile fallback
    """
    snr_db, backend = _compute_wada_snr_if_available(wav, sr)
    if snr_db is not None:
        return snr_db, backend
    return _compute_energy_snr(wav, sr)