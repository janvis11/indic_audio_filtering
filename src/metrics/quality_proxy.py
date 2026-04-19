from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DNSMOS_CACHE = Path.home() / ".cache" / "dnsmos"
_DNSMOS_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
)

_DNSMOS_SESSION = None
_DNSMOS_TRIED = False
_DNSMOS_DISABLED = False


def _load_dnsmos_session():
    try:
        import onnxruntime as ort
        import requests

        _DNSMOS_CACHE.mkdir(parents=True, exist_ok=True)
        model_path = _DNSMOS_CACHE / "sig_bak_ovr.onnx"

        if not model_path.exists():
            logger.info("Downloading DNSMOS ONNX model...")
            r = requests.get(_DNSMOS_URL, timeout=60)
            r.raise_for_status()
            model_path.write_bytes(r.content)

        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.warning("DNSMOS unavailable, continuing without it: %s", e)
        return None


def _get_dnsmos_session():
    global _DNSMOS_SESSION, _DNSMOS_TRIED
    if not _DNSMOS_TRIED:
        _DNSMOS_TRIED = True
        _DNSMOS_SESSION = _load_dnsmos_session()
    return _DNSMOS_SESSION


def compute_quality_proxy(basic: dict, vad: dict) -> dict:
    """
    Lightweight no-reference proxy used as the primary demonstrated quality signal.
    """
    rms_norm = np.clip((basic["rms_db"] + 50.0) / 35.0, 0.0, 1.0)
    clip_penalty = np.clip(1.0 - (basic["clipping_ratio"] / 0.02), 0.0, 1.0)
    silence_penalty = np.clip(1.0 - basic["silence_ratio_amp"], 0.0, 1.0)
    speech_score = np.clip(vad["speech_ratio"] / 0.6, 0.0, 1.0)
    flatness_penalty = np.clip(
        1.0 - abs(basic["spectral_flatness_mean"] - 0.2) / 0.4, 0.0, 1.0
    )

    overall = float(
        0.25 * rms_norm
        + 0.20 * clip_penalty
        + 0.20 * silence_penalty
        + 0.20 * speech_score
        + 0.15 * flatness_penalty
    )

    return {
        "quality_proxy": round(overall, 4),
        "quality_proxy_mos_like": round(1.0 + 4.0 * overall, 4),
    }


def compute_dnsmos_from_wav(wav: np.ndarray, sr: int) -> Optional[dict]:
    """
    Optional DNSMOS helper.

    Returns None if:
    - DNSMOS unavailable
    - model input shape incompatible
    - inference fails

    IMPORTANT:
    If the ONNX model shape is incompatible, disable DNSMOS for the rest of the run.
    """
    global _DNSMOS_DISABLED

    if _DNSMOS_DISABLED:
        return None

    sess = _get_dnsmos_session()
    if sess is None:
        return None

    try:
        import librosa

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000

        target_len = 9 * sr
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]

        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        # Most stable fallback: feed raw waveform as 2D [1, T]
        inp = wav.astype(np.float32)[None, :]

        out = sess.run(None, {input_name: inp})[0]

        # Be defensive about output format
        out = np.asarray(out).reshape(-1)
        if len(out) >= 3:
            sig, bak, ovrl = float(out[0]), float(out[1]), float(out[2])
            return {
                "dnsmos_sig": round(sig, 3),
                "dnsmos_bak": round(bak, 3),
                "dnsmos_ovrl": round(ovrl, 3),
            }

        logger.warning("DNSMOS output shape unexpected: %s. Disabling DNSMOS.", out.shape)
        _DNSMOS_DISABLED = True
        return None

    except Exception as e:
        msg = str(e)
        logger.warning("DNSMOS inference failed, disabling it for this run: %s", msg)
        _DNSMOS_DISABLED = True
        return None