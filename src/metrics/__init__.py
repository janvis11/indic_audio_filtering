# Metrics package for Indic Audio Filtering Pipeline
#
# Available metrics:
#   - basic: RMS, clipping, silence, spectral features
#   - vad: Voice activity detection (Silero VAD)
#   - quality_proxy: Composite quality score + DNSMOS
#   - snr: WADA-SNR or energy-difference fallback
#   - asr_bonus: ASR-based confidence and language ID
#   - langid_bonus: Language match verification
from .snr import compute_snr_db

__all__ = ["compute_snr_db", "snr_decision"]
