"""
ASR-based quality check (bonus feature).

Uses faster-whisper (whisper-tiny) to:
  1. Estimate transcription confidence via avg_logprob — low confidence
     indicates unintelligible or heavily degraded audio.
  2. Detect the spoken language for cross-checking against the manifest label.
  3. Compute intelligibility proxy from multiple ASR signals.

Research grounding:
  - TITW (Jung et al., Interspeech 2024): uses Whisper transcription +
    DNSMOS filtering as the two-stage quality gate for in-the-wild TTS data.
    Whisper log-probability is their proxy for intelligibility.
  - IndicVoices-R (Sankar et al., NeurIPS 2024): applies ASR-based
    filtering to verify that transcription text matches the detected language
    and speaking rate is within 0.5-6.0 wps.

Installation (run in Colab before importing):
    !pip install faster-whisper
"""

import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def _ensure_faster_whisper() -> bool:
    """Install faster-whisper if not present. Safe to call multiple times."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        logger.info("faster-whisper not found — attempting pip install...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "faster-whisper"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import faster_whisper  # noqa: F401
            logger.info("faster-whisper installed successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not install faster-whisper: {e}")
            return False


_HAS_FW = _ensure_faster_whisper()
if _HAS_FW:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        _HAS_FW = False


def compute_intelligibility_proxy(
    asr_avg_logprob: Optional[float],
    asr_text: str,
    speech_ratio: float,
    duration_sec: float
) -> dict:
    """
    Compute an explicit intelligibility proxy score from ASR signals.

    Intelligibility measures how understandable the speech is, combining:
      1. ASR confidence (avg_logprob): Low confidence = unintelligible
      2. Transcript non-emptiness: Empty transcript = unintelligible
      3. Speech ratio: Low speech ratio = mostly silence/noise
      4. Duration sanity: Too short = likely not speech

    Returns:
        dict with:
          - asr_text_len: character count of transcription
          - chars_per_sec: speaking rate in characters/second
          - intelligibility_proxy: composite [0, 1] score
          - empty_transcript_flag: True if ASR produced no text
          - low_confidence_flag: True if avg_logprob < -1.25 (TITW threshold)
          - speech_rate_proxy: chars/sec normalized to [0, 1]
          - possible_non_speech_flag: True if speech_ratio < 0.15 or duration < 0.3s

    Scoring:
      - logprob component: maps [-2, 0] to [0, 1]
      - text component: 1.0 if non-empty, 0.0 if empty
      - speech component: speech_ratio clipped to [0, 1]
      - Final: weighted average (0.4 logprob + 0.3 text + 0.3 speech)
    """
    # ASR text length
    asr_text_len = len(asr_text.strip()) if asr_text else 0

    # Characters per second (proxy for speaking rate)
    chars_per_sec = round(asr_text_len / duration_sec, 2) if duration_sec > 0 else 0.0

    # Logprob component: map [-2, 0] to [0, 1]
    if asr_avg_logprob is not None:
        logprob_score = max(0.0, min(1.0, (asr_avg_logprob + 2.0) / 2.0))
    else:
        logprob_score = 0.5  # Neutral if ASR unavailable

    # Text component: 1.0 if non-empty, 0.0 if empty
    text_score = 1.0 if asr_text_len > 0 else 0.0

    # Speech component: use speech_ratio directly
    speech_score = min(1.0, max(0.0, speech_ratio))

    # Weighted composite
    intelligibility = (
        0.4 * logprob_score +
        0.3 * text_score +
        0.3 * speech_score
    )

    # === NEW: Explicit ASR-based flags for bonus evaluation ===

    # Flag 1: Empty transcript (unintelligible or non-speech)
    empty_transcript_flag = asr_text_len == 0

    # Flag 2: Low confidence (TITW uses -1.25 as review threshold)
    low_confidence_flag = (
        asr_avg_logprob is not None and
        asr_avg_logprob < -1.25
    )

    # Flag 3: Speech rate proxy (normal speaking rate ~10-60 chars/sec for Indic)
    # Normalize: <5 chars/sec = too slow (0.0), >80 chars/sec = too fast (0.0)
    if chars_per_sec < 5:
        speech_rate_proxy = 0.0
    elif chars_per_sec > 80:
        speech_rate_proxy = 0.0
    else:
        # Peak at ~30 chars/sec (typical speaking rate)
        speech_rate_proxy = max(0.0, min(1.0, 1.0 - abs(chars_per_sec - 30) / 50))

    # Flag 4: Possible non-speech (very low speech ratio or extremely short)
    possible_non_speech_flag = (
        speech_ratio < 0.15 or
        (duration_sec is not None and duration_sec < 0.3)
    )

    return {
        "asr_text_len": asr_text_len,
        "chars_per_sec": chars_per_sec,
        "intelligibility_proxy": round(intelligibility, 4),
        "empty_transcript_flag": empty_transcript_flag,
        "low_confidence_flag": low_confidence_flag,
        "speech_rate_proxy": round(speech_rate_proxy, 4),
        "possible_non_speech_flag": possible_non_speech_flag,
    }


class ASRBonus:
    """
    Whisper-based ASR confidence and language-ID checker.

    Applied to samples that pass hard gates, providing:
      - asr_avg_logprob: Whisper internal confidence.
        Values below -1.25 indicate unintelligible audio (TITW threshold).
      - asr_detected_language: detected BCP-47 language code.
      - asr_confidence_proxy: normalised [0, 1] confidence score.
      - asr_text_len: character count of transcription.
      - chars_per_sec: speaking rate in chars/second.
      - intelligibility_proxy: composite [0, 1] score.

    Model: whisper-tiny (39M params).
    Compute type falls back: float16 (GPU) -> int8 (CPU) -> float32.
    """

    _COMPUTE_FALLBACKS = ["float16", "int8", "float32"]

    def __init__(self, model_size: str = "tiny", device: str = "auto", compute_type: str = "float16"):
        self.ok = False
        self.model = None
        self.backend_info = "unavailable"

        if not _HAS_FW:
            logger.warning(
                "faster-whisper unavailable. ASR metrics will be null. "
                "Run: !pip install faster-whisper  then restart runtime."
            )
            return

        for ct in self._COMPUTE_FALLBACKS:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=ct)
                self.ok = True
                self.backend_info = f"faster_whisper/{model_size}/{ct}"
                logger.info(f"ASRBonus ready: {self.backend_info}")
                break
            except Exception as e:
                logger.debug(f"WhisperModel({ct}) failed: {e} — trying next fallback")

        if not self.ok:
            logger.warning("ASRBonus: all compute_type fallbacks failed. ASR disabled.")

    def transcribe(self, audio_path: str, duration_sec: float = None, speech_ratio: float = None) -> dict:
        """
        Transcribe audio and return ASR quality signals.

        Log-probability normalisation:
            logprob in [-2, 0] maps to confidence in [0, 1].
            Below -2.0 = very low confidence (unintelligible or wrong language).
            TITW paper uses -1.25 as the review threshold.

        Args:
            audio_path: Path to audio file
            duration_sec: Duration of audio (for chars_per_sec calculation)
            speech_ratio: VAD speech ratio (for intelligibility proxy)

        Returns:
            dict with ASR metrics + intelligibility proxy
        """
        if not self.ok:
            result = {
                "asr_text": "",
                "asr_avg_logprob": None,
                "asr_detected_language": None,
                "asr_confidence_proxy": None,
                "asr_backend": "unavailable",
            }
            result.update(compute_intelligibility_proxy(None, "", speech_ratio or 0.0, duration_sec or 0.0))
            return result

        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segs = list(segments)
            text = " ".join((s.text or "").strip() for s in segs).strip()
            logprobs = [s.avg_logprob for s in segs if getattr(s, "avg_logprob", None) is not None]
            avg_logprob = float(sum(logprobs) / len(logprobs)) if logprobs else None
            confidence_proxy = (
                round(max(0.0, min(1.0, (avg_logprob + 2.0) / 2.0)), 4)
                if avg_logprob is not None else None
            )
            result = {
                "asr_text": text,
                "asr_avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
                "asr_detected_language": getattr(info, "language", None),
                "asr_confidence_proxy": confidence_proxy,
                "asr_backend": self.backend_info,
            }
            # Add intelligibility proxy metrics
            result.update(compute_intelligibility_proxy(
                avg_logprob, text, speech_ratio or 0.0, duration_sec or 0.0
            ))
            return result
        except Exception as e:
            logger.warning(f"ASR failed for {audio_path}: {e}")
            result = {
                "asr_text": "",
                "asr_avg_logprob": None,
                "asr_detected_language": None,
                "asr_confidence_proxy": None,
                "asr_backend": f"error:{type(e).__name__}",
            }
            result.update(compute_intelligibility_proxy(None, "", speech_ratio or 0.0, duration_sec or 0.0))
            return result
