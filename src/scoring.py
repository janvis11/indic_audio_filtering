from __future__ import annotations


def _normalize(value, lo, hi):
    if value is None:
        return None
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def apply_hard_rules(record: dict, cfg: dict) -> list[str]:
    """
    Hard reject should only catch clearly unusable audio.
    Everything else should flow into review.
    """
    t = cfg["thresholds"]["reject"]
    reasons = []

    if record.get("io_error"):
        reasons.append("io_error")
    if record.get("duration_sec", 0) < t["min_duration_sec"]:
        reasons.append("too_short")
    if record.get("duration_sec", 0) > t["max_duration_sec"]:
        reasons.append("too_long")
    if record.get("rms_db", 0) < t["min_rms_db"]:
        reasons.append("low_energy")
    if record.get("clipping_ratio", 0) > t["max_clipping_ratio"]:
        reasons.append("high_clipping")

    # Keep these extremely permissive in hard reject
    if record.get("silence_ratio_amp", 0) > t["max_silence_ratio"]:
        reasons.append("extreme_silence")
    if record.get("speech_ratio", 0) < t["min_speech_ratio"]:
        reasons.append("almost_no_speech")
    if record.get("quality_proxy", 1.0) < t["min_quality_proxy"]:
        reasons.append("extremely_low_quality")
    if record.get("snr_db") is not None and record["snr_db"] < t.get("min_snr_db", -5.0):
        reasons.append("extremely_low_snr")

    return reasons


def compute_scores(record: dict, cfg: dict) -> dict:
    signal_parts = [
        _normalize(record.get("rms_db"), -50, -10),
        1.0 - _normalize(record.get("clipping_ratio"), 0.0, 0.03),
        1.0 - _normalize(record.get("silence_ratio_amp"), 0.0, 0.90),
    ]
    signal_parts = [x for x in signal_parts if x is not None]
    signal_score = sum(signal_parts) / max(1, len(signal_parts))

    vad_parts = [
        _normalize(record.get("speech_ratio"), 0.05, 0.75),
        1.0 - _normalize(record.get("max_silence_sec"), 0.5, 15.0),
    ]
    vad_parts = [x for x in vad_parts if x is not None]
    vad_score = sum(vad_parts) / max(1, len(vad_parts))

    quality_score = record.get("quality_proxy", 0.0)

    snr_score = _normalize(record.get("snr_db"), 0, 30)
    if snr_score is None:
        snr_score = 0.5

    asr_score = record.get("intelligibility_proxy")
    if asr_score is None:
        asr_score = record.get("asr_confidence_proxy")
    if asr_score is None:
        asr_score = 0.5

    # Soft penalties only
    if record.get("empty_transcript_flag", False):
        asr_score = min(asr_score, 0.35)
    if record.get("low_confidence_flag", False):
        asr_score = min(asr_score, 0.45)
    if record.get("possible_non_speech_flag", False):
        asr_score = min(asr_score, 0.30)

    language_score = record.get("language_score")
    if language_score is None:
        language_score = 0.5
    elif language_score == 0.0:
        language_score = 0.40  # only soft penalty

    weights = cfg["scoring"]["weights"]

    final = 100.0 * (
        weights["signal"] * signal_score
        + weights["vad"] * vad_score
        + weights["quality"] * quality_score
        + weights["snr"] * snr_score
        + weights["asr"] * asr_score
        + weights["language"] * language_score
    )

    return {
        "signal_score": round(signal_score * 100, 2),
        "vad_score": round(vad_score * 100, 2),
        "quality_score": round(quality_score * 100, 2),
        "snr_score": round(snr_score * 100, 2),
        "asr_score": round(asr_score * 100, 2),
        "language_score_pct": round(language_score * 100, 2),
        "final_score": round(final, 2),
    }


def final_decision(record: dict, hard_reasons: list[str], cfg: dict) -> tuple[str, list[str]]:
    reasons = list(hard_reasons)

    if hard_reasons:
        return "reject", reasons

    rt = cfg["thresholds"]["review"]

    if record.get("clipping_ratio", 0) > rt["max_clipping_ratio"]:
        reasons.append("review_clipping")
    if record.get("silence_ratio_amp", 0) > rt["max_silence_ratio"]:
        reasons.append("review_silence")
    if record.get("speech_ratio", 1.0) < rt["min_speech_ratio"]:
        reasons.append("review_low_speech")
    if record.get("quality_proxy", 1.0) < rt["min_quality_proxy"]:
        reasons.append("review_low_quality")

    if record.get("asr_avg_logprob") is not None and record["asr_avg_logprob"] < rt["min_asr_confidence"]:
        reasons.append("review_low_asr_confidence")

    if record.get("snr_db") is not None and record["snr_db"] < rt.get("min_snr_db", 8.0):
        reasons.append("review_low_snr")

    # Language mismatch stays soft and conditional
    if record.get("language_match") is False:
        weak_asr = (
            record.get("asr_confidence_proxy") is not None
            and record["asr_confidence_proxy"] < 0.40
        )
        weak_quality = record.get("quality_proxy", 1.0) < 0.40
        weak_speech = record.get("speech_ratio", 1.0) < 0.20
        if weak_asr or weak_quality or weak_speech:
            reasons.append("review_language_mismatch")

    if record.get("empty_transcript_flag", False):
        reasons.append("review_empty_transcript")
    if record.get("possible_non_speech_flag", False):
        reasons.append("review_possible_non_speech")

    # Deduplicate while preserving order
    reasons = list(dict.fromkeys(reasons))

    score = record.get("final_score", 0.0)

    if score >= cfg["scoring"]["keep_min"] and not reasons:
        return "keep", reasons

    if score < cfg["scoring"]["review_min"]:
        if "low_final_score" not in reasons:
            reasons.append("low_final_score")
        return "reject", reasons

    return "review", reasons