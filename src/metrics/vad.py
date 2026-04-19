from __future__ import annotations

import numpy as np

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    _HAS_SILERO = True
except Exception:
    _HAS_SILERO = False


class VADWrapper:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.model = load_silero_vad() if _HAS_SILERO else None

    def compute(self, wav: np.ndarray) -> dict:
        if len(wav) == 0:
            return {
                "speech_duration_sec": 0.0,
                "speech_ratio": 0.0,
                "num_speech_segments": 0,
                "max_silence_sec": 0.0,
                "mean_speech_segment_sec": 0.0,
                "vad_backend": "none",
            }

        if self.model is not None:
            ts = get_speech_timestamps(wav, self.model, sampling_rate=self.sr)
            segs = [(d["start"] / self.sr, d["end"] / self.sr) for d in ts]
            backend = "silero"
        else:
            # More forgiving energy fallback for real-world noisy speech
            frame = int(0.03 * self.sr)
            hop = int(0.01 * self.sr)

            segs = []
            backend = "energy_fallback"
            in_speech = False
            start = 0.0

            global_rms = float(np.sqrt(np.mean(wav ** 2) + 1e-10))
            # Lower threshold than before to avoid killing quiet speech
            thr = max(0.006, 0.28 * global_rms)

            for i in range(0, max(1, len(wav) - frame), hop):
                rms = float(np.sqrt(np.mean(wav[i:i + frame] ** 2) + 1e-10))
                sp = rms > thr
                t = i / self.sr

                if sp and not in_speech:
                    start = t
                    in_speech = True
                elif not sp and in_speech:
                    if (t - start) >= 0.08:  # ignore tiny bursts
                        segs.append((start, t))
                    in_speech = False

            if in_speech:
                end_t = len(wav) / self.sr
                if (end_t - start) >= 0.08:
                    segs.append((start, end_t))

            # Merge nearby segments to avoid fragmentation
            merged = []
            for s, e in segs:
                if not merged:
                    merged.append([s, e])
                else:
                    prev_s, prev_e = merged[-1]
                    if s - prev_e <= 0.15:
                        merged[-1][1] = e
                    else:
                        merged.append([s, e])
            segs = [(s, e) for s, e in merged]

        total = len(wav) / self.sr
        speech_dur = sum(max(0.0, e - s) for s, e in segs)

        silences = []
        prev = 0.0
        for s, e in segs:
            silences.append(max(0.0, s - prev))
            prev = e
        silences.append(max(0.0, total - prev))

        seg_lens = [max(0.0, e - s) for s, e in segs]

        return {
            "speech_duration_sec": float(speech_dur),
            "speech_ratio": float(speech_dur / total) if total > 0 else 0.0,
            "num_speech_segments": int(len(segs)),
            "max_silence_sec": float(max(silences) if silences else 0.0),
            "mean_speech_segment_sec": float(np.mean(seg_lens) if seg_lens else 0.0),
            "vad_backend": backend,
        }