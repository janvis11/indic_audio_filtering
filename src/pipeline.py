from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .audio_utils import load_and_standardize_audio
from .metrics.basic import compute_basic_metrics
from .metrics.vad import VADWrapper
from .metrics.quality_proxy import compute_dnsmos_from_wav, compute_quality_proxy
from .metrics.snr import compute_snr_db
from .metrics.asr_bonus import ASRBonus
from .metrics.langid_bonus import compute_language_match
from .scoring import apply_hard_rules, compute_scores, final_decision


class FilteringPipeline:
    def __init__(self, cfg: dict, enable_asr: bool = True):
        self.cfg = cfg
        self.vad = VADWrapper(sr=cfg["audio"]["target_sr"])
        self.asr = (
            ASRBonus(model_size="tiny", device="auto", compute_type="float16")
            if enable_asr and cfg["modules"].get("enable_asr_bonus", True)
            else None
        )

    def process_one(self, row: dict) -> dict:
        sample_id = row.get("sample_id") or Path(
            row.get("audio_filepath") or row.get("audio_path")
        ).stem
        audio_path = row.get("audio_filepath") or row.get("audio_path")
        language = row.get("language") or row.get("lang")

        record = {
            "sample_id": sample_id,
            "audio_path": audio_path,
            "language": language,
        }

        loaded = load_and_standardize_audio(
            audio_path,
            target_sr=self.cfg["audio"]["target_sr"],
            mono=self.cfg["audio"]["mono"],
        )
        if not loaded["ok"]:
            record.update(
                {
                    "io_error": loaded["error"],
                    "decision": "reject",
                    "reason_codes": ["io_error"],
                }
            )
            return record

        wav = loaded["waveform"]
        sr = loaded["sample_rate"]
        record["duration_sec"] = round(loaded["duration_sec"], 4)

        basic = compute_basic_metrics(wav, sr)
        vad = self.vad.compute(wav)
        record.update(basic)
        record.update(vad)

        record.update(compute_quality_proxy(basic, vad))

        snr_db, snr_backend = compute_snr_db(wav, sr)
        record["snr_db"] = snr_db
        record["snr_backend"] = snr_backend

        dnsmos = compute_dnsmos_from_wav(wav, sr)
        if dnsmos is not None:
            record.update(dnsmos)

        hard = apply_hard_rules(record, self.cfg)

        if self.asr is not None and not hard:
            asr_result = self.asr.transcribe(
                audio_path,
                duration_sec=record.get("duration_sec"),
                speech_ratio=record.get("speech_ratio"),
            )
            record.update(asr_result)
            record.update(compute_language_match(language, record.get("asr_detected_language")))
        else:
            record.update(
                {
                    "asr_text": "",
                    "asr_avg_logprob": None,
                    "asr_detected_language": None,
                    "asr_confidence_proxy": None,
                    "asr_backend": "skipped",
                }
            )
            record.update(compute_language_match(language, None))
            record.update(
                {
                    "asr_text_len": 0,
                    "chars_per_sec": 0.0,
                    "intelligibility_proxy": 0.0,
                    "empty_transcript_flag": True,
                    "low_confidence_flag": False,
                    "speech_rate_proxy": 0.0,
                    "possible_non_speech_flag": True,
                }
            )

        record.update(compute_scores(record, self.cfg))
        decision, reasons = final_decision(record, hard, self.cfg)
        record["decision"] = decision
        record["reason_codes"] = reasons

        return record

    def run(self, rows: list[dict]) -> list[dict]:
        out = []
        for row in tqdm(rows, desc="Filtering audio"):
            out.append(self.process_one(row))
        return out