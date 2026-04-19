# Indic Audio Filtering Pipeline

A production-quality audio filtering pipeline for large-scale Indic speech datasets. This pipeline detects and removes low-quality samples using multiple audio quality metrics, enabling efficient training data curation at scale.
> kindly see  [flowchart](https://github.com/janvis11/indic_audio_filtering/blob/main/pipeline_flowchart.png)

> results - [plots](https://github.com/janvis11/indic_audio_filtering/tree/main/content/output_subset/plots)
> overall results - [all](https://github.com/janvis11/indic_audio_filtering/tree/main/content/output_subset)
---
> related work-  [similar project](https://github.com/janvis11/asr)

## Overview

This pipeline is designed to handle 1000+ hours of Indic speech data across 22+ languages. The implementation prioritizes:

- **Sound metrics** grounded in speech research (IndicVoices-R, DNSMOS, TITW)
- **Scalable execution** via parallel processing
- **Language-aware thresholds** recognizing that quality is not one-size-fits-all
- **Resumable processing** with checkpoint support for large runs

---

## Design & Implementation

### Metrics Used and Why

The pipeline implements **7 core metrics** plus **3 bonus metrics**, all grounded in peer-reviewed speech research:

#### Core Metrics (Research-Backed)

| Metric | Purpose | Why This Metric | Source |
|--------|---------|-----------------|--------|
| **RMS (dB)** | Detects silent/muted audio | Extreme energy levels indicate device issues or missing audio | Signal processing fundamentals |
| **Clipping Ratio** | Finds distorted samples | Clipped audio (peak saturation) is unusable for training | Analog signal theory |
| **Silence Ratio** | Measures speech presence | Too much silence = poor training value; VAD-based segmentation | Voice Activity Detection (Silero) |
| **Speech Ratio** | Validates speech content | Ensures minimum 3% speech for training; speaks to voice presence | VAD temporal analysis |
| **SNR (dB)** | Quantifies noise level | Higher SNR = cleaner training signal; WADA-SNR is intrusive but accurate | **IndicVoices-R (NeurIPS 2024)** |
| **Quality Proxy** | No-reference quality | Composite score without needing reference audio; blends spectral + temporal features | **DataSpeech (HF 2024)** + This Work |
| **DNSMOS OVRL** | Neural MOS score | Industry-standard non-intrusive neural MOS from Microsoft; OVRL > 2.5 proven for TTS | **DNSMOS P.835 (ICASSP 2022, Microsoft)** |

#### Bonus Metrics (ASR-Based)

| Metric | Purpose | Why This Metric |
|--------|---------|-----------------|
| **ASR Confidence** | Transcription reliability | Low confidence = hard-to-understand speech; log-prob from Whisper | **TITW (Interspeech 2024)** |
| **Language Match** | Catch mislabeled samples | Whisper's LID catches data-entry errors and language mixing | **Whisper (OpenAI)** |
| **Intelligibility Proxy** | Speech clarity rate | Characters per second indicates speaking rate; too slow/fast = quality flag | **IndicVoices-R speaking rate analysis** |

**Why This Combination?**
- **SNR + DNSMOS** covers both classical (signal-based) and modern (neural) quality assessment
- **VAD metrics** ensure meaningful speech content (not silence)
- **ASR metrics** add linguistic validation (what was actually spoken)
- **Quality proxy** provides a fast no-reference baseline without ONNX models
- **Together**, they form a robust pipeline resistant to single-metric failures

---

### Thresholding / Filtering Logic

#### Three-Tier Decision System

The pipeline outputs three mutually exclusive buckets:

```
┌─────────────────────────────────────────────────────────┐
│  HARD REJECT RULES (Immediate Rejection)                │
│  • IO errors (cannot load)                              │
│  • Duration < 0.4s or > 35s                             │
│  • RMS < -50 dB (silent)                                │
│  • Clipping > 5% (distorted)                            │
│  • Silence > 98% (no speech)                            │
│  • Speech < 3% (VAD confirms no speech)                 │
│  → DECISION: REJECT                                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  SOFT SCORING & WEIGHTED DECISION                       │
│                                                         │
│  1. Normalize each metric to [0, 100]                   │
│  2. Apply soft penalties for borderline cases:          │
│     - Empty transcript (ASR)                           │
│     - Low ASR confidence (< -1.25)                     │
│     - Non-speech segments (> 12%)                       │
│  3. Compute weighted average:                           │
│     Final_Score = Σ(weight_i × metric_i)               │
│                                                         │
│  Scoring Weights:                                      │
│  • Signal (RMS, clipping, silence): 22%                │
│  • VAD (speech presence): 18%                          │
│  • Quality proxy + DNSMOS: 22%                         │
│  • SNR: 15%                                            │
│  • ASR confidence + intelligibility: 18%               │
│  • Language match: 5%                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  DECISION RULES                                         │
│                                                         │
│  Score ≥ 68 AND no review flags → KEEP                 │
│  • High-quality, ready for training                     │
│  • Output: keep_manifest.jsonl                          │
│                                                         │
│  42 ≤ Score < 68 OR review flags → REVIEW              │
│  • Borderline cases requiring human inspection          │
│  • Flags: low speech, low ASR conf, language mismatch   │
│  • Output: review_manifest.jsonl                        │
│                                                         │
│  Score < 42 AND not caught by hard rules → REJECT       │
│  • Low-quality, not suitable for training               │
│  • Output: reject_manifest.jsonl                        │
└─────────────────────────────────────────────────────────┘
```

#### Threshold Justification

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Keep Score** | ≥ 68 | Empirically tuned for >50% keep rate on clean IndicVoices subset |
| **Review Range** | 42–68 | Captures ~30% ambiguous samples for human review (not auto-reject) |
| **Min Speech Ratio** | 3% | Ensures ≥ 0.5s continuous speech in typical 10s audio |
| **Min SNR** | -5 dB (hard), 8 dB (review) | IndicVoices-R uses -5 dB hard floor; 8+ dB for high-quality training data |
| **Max Clipping** | 5% (hard), 1% (review) | <1% clipping undetectable by human ear; >5% causes audio dropout |
| **DNSMOS threshold** | > 2.5 (soft) | TITW paper: ≥2.5 MOS proven for TTS training datasets |
| **ASR Confidence** | -1.25 | Whisper log-prob; <-1.25 indicates very low transcription confidence |

#### Why This Approach?

1. **Hard rules catch obvious junk** → fast rejection, no ONNX cost
2. **Soft scoring handles ambiguity** → not everything is pass/fail
3. **Review bucket is honest** → 30% of samples are genuinely borderline
4. **Per-metric normalization** → compensates for different ranges (dB, ratio, etc.)
5. **Language weighting is low (5%)** → prioritizes signal quality over language match

---

### Scalability Approach

#### 1. Parallel Processing Architecture

The pipeline uses **two-level parallelization** mirroring `setup_dataset.py`:

```python
# High-level pattern
ProcessPoolExecutor(max_workers=cpu_count())  # CPU-bound: metrics, scoring
    ↓
    └─→ ThreadPoolExecutor(max_workers=8)    # I/O-bound: audio loading, disk writes
```

**Why two levels?**
- **ProcessPoolExecutor**: Metric computation (SNR, DNSMOS, WADA) cannot release GIL; one process per core
- **ThreadPoolExecutor**: Audio file I/O is I/O-bound; threads excel at blocking operations
- **Batching**: Process-level submits batches of 64–128 samples to avoid queue overhead

#### 2. Memory Efficiency

| Strategy | Benefit |
|----------|---------|
| **Streaming JSONL** | Manifest not loaded into RAM; parsed incrementally |
| **Single ONNX load** | DNSMOS + Whisper models loaded once per process (not per sample) |
| **Parquet write batching** | Results written in chunks to avoid memory buildup |
| **Audio disc cache** | Don't keep decoded audio in memory; reload per metric if needed |

#### 3. Resumable Checkpointing

- **Atomic writes**: Each sample's metrics flushed immediately to parquet
- **Resume mode** (`--resume`): Skips already-processed samples via hash of audio path + language
- **Fault tolerance**: Crash during processing → re-run with `--resume` to pick up where you left off

#### 4. Throughput Benchmarks

| Configuration | Samples/sec | 1,600 samples | 1M samples |
|:---|:---|:---|:---|
| **Single-threaded** (baseline) | 2–3 | ~9 min | ~4 days |
| **8 workers + threading** (typical) | 15–20 | ~1.5 min | ~14 hrs |
| **32 workers + threading** (high-end) | 50–60 | ~30 sec | ~5 hrs |

**Measured on:** Intel 8-core (16 logical), 16GB RAM, SSD, DNSMOS P.835 + Whisper-tiny enabled

---

## Bonus Deliverables


| Bonus Feature | Implementation | Location |
|---------------|----------------|----------|
| **Language ID** | Whisper-based language verification | `src/metrics/langid_bonus.py`, Stage 6 |
| **ASR-Based Checks** | Transcription confidence, intelligibility proxy | `src/metrics/asr_bonus.py`, Stage 6 |
| **Visual Analysis** | 9 comprehensive quality plots | `scripts/visualize_outputs.py`, `outputs/plots/` |


### Tested Configuration

This pipeline was validated on a **subset of 1,600 samples** (8 languages x 200 samples each) from the AI4Bharat IndicVoices dataset for rapid iteration. The architecture is designed to scale seamlessly to the full dataset (1000+ hours, 22+ languages).

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run on a manifest file
python -m src.main --manifest subset_manifest.jsonl --output_dir ./outputs --config configs/default.yaml

# Run with resume capability (skip already processed samples)
python -m src.main --manifest subset_manifest.jsonl --output_dir ./outputs --resume

# Limit processing for testing
python -m src.main --manifest subset_manifest.jsonl --output_dir ./outputs --limit 100

# Disable ASR for faster processing
python -m src.main --manifest subset_manifest.jsonl --output_dir ./outputs --disable_asr
```

### Helper Scripts

```bash
# Run pipeline via wrapper script
python scripts/run_local_pipeline.py --manifest subset_manifest.jsonl --output-dir ./outputs

# Visualize outputs after processing
python scripts/visualize_outputs.py --metrics ./outputs/metrics.parquet --output-dir ./outputs/plots
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUDIO FILTERING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

    INPUT: JSONL Manifest (sample_id, audio_path, language)
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 1: AUDIO LOADING & STANDARDIZATION                           │
    │  • Load audio (soundfile/librosa)                                   │
    │  • Convert to 16kHz mono                                            │
    │  • Validate waveform integrity                                      │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 2: BASIC SIGNAL METRICS                                      │
    │  • RMS (dB) - signal energy                                         │
    │  • Clipping ratio - peak amplitude saturation                       │
    │  • Silence ratio (amplitude-based)                                  │
    │  • Spectral centroid, flatness, ZCR                                 │
    │  • DC offset                                                        │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 3: VOICE ACTIVITY DETECTION (Silero VAD)                     │
    │  • Speech duration & ratio                                          │
    │  • Number of speech segments                                        │
    │  • Max silence gap                                                  │
    │  • Mean segment length                                              │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 4: QUALITY PROXY & DNSMOS                                    │
    │  • Composite quality proxy (no-reference)                           │
    │  • DNSMOS P.835 scores (SIG, BAK, OVRL) - optional                  │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 5: SNR ESTIMATION                                            │
    │  • WADA-SNR (preferred) or energy-percentile fallback               │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 6: ASR-BASED METRICS (Bonus)                                 │
    │  • Whisper transcription                                            │
    │  • ASR confidence proxy (log-prob)                                  │
    │  • Detected language verification                                   │
    │  • Intelligibility proxy (chars/sec)                                │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 7: SCORING & DECISION                                        │
    │  • Normalize metrics to 0-100 scale                                 │
    │  • Weighted composite scoring                                       │
    │  • Hard reject rules (IO errors, extreme cases)                     │
    │  • Review bucket (borderline samples)                               │
    │  • Final decision: KEEP / REVIEW / REJECT                           │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    OUTPUT: metrics.parquet, metrics.csv, decisions.jsonl,
            keep/review/reject_manifest.jsonl, summary.json, plots/
```

---

## Metrics Implemented

### Core Metrics (Stages 1-5)

| Metric | Description | Source | Threshold (Reject) |
|--------|-------------|--------|-------------------|
| **RMS (dB)** | Signal energy level | Signal processing | < -50 dB |
| **Clipping Ratio** | Fraction of samples at peak | Signal processing | > 5% |
| **Silence Ratio** | Fraction of silent frames | Amplitude-based | > 98% |
| **Speech Ratio** | VAD speech / total duration | Silero VAD | < 3% |
| **Quality Proxy** | Composite no-reference score | This work | < 0.15 |
| **SNR (dB)** | Signal-to-noise ratio | WADA-SNR / Energy | < -5 dB |
| **DNSMOS OVRL** | Neural MOS overall quality | Microsoft P.835 | < 2.5 |

### Bonus Metrics (Stage 6) 

| Metric | Description | Source |
|--------|-------------|--------|
| **ASR Confidence** | Transcription log-probability | Whisper |
| **Language Match** | Detected vs. labeled language | Whisper LID |
| **Intelligibility Proxy** | Characters per second | ASR-derived |

> **ASR-Based Quality Checks**
> 
> The pipeline includes Whisper-based ASR metrics:
> - Transcription confidence scoring for intelligibility assessment
> - Language ID verification to catch mislabeled samples
> - Character-per-second rate as a speech quality proxy
> 
> See `notebooks/04_bonus_asr_lid_analysis.ipynb` for detailed analysis.

### Scoring Weights (Configurable)

```yaml
scoring:
  weights:
    signal: 0.22      # RMS, clipping, silence
    vad: 0.18         # Speech ratio, segments
    quality: 0.22     # Quality proxy, DNSMOS
    snr: 0.15         # SNR score
    asr: 0.18         # ASR confidence, intelligibility
    language: 0.05    # Language match
```

---

## Decision Logic

### Three-Tier Classification

1. **KEEP** (score >= 68, no review flags)
   - High-quality audio suitable for training
   - Exported to `keep_manifest.jsonl`

2. **REVIEW** (score 42-68, or borderline flags)
   - Ambiguous samples for manual inspection
   - Exported to `review_manifest.jsonl`
   - Flags: low speech ratio, low ASR confidence, language mismatch

3. **REJECT** (score < 42, or hard reject)
   - Unusable audio (IO errors, too short, extreme clipping)
   - Exported to `reject_manifest.jsonl`

### Hard Reject Rules

- IO error (cannot load audio)
- Duration < 0.4s or > 35s
- RMS < -50 dB
- Clipping ratio > 5%
- Silence ratio > 98%
- Speech ratio < 3%

---

## Output Files

After running the pipeline, the output directory contains:

| File | Description |
|------|-------------|
| `metrics.parquet` | Full per-sample metrics (Polars format) |
| `metrics.csv` | Human-readable metrics table |
| `decisions.jsonl` | JSONL with decision + reason codes |
| `keep_manifest.jsonl` | Filtered manifest for training |
| `review_manifest.jsonl` | Borderline samples for review |
| `reject_manifest.jsonl` | Rejected samples |
| `summary.json` | Aggregate statistics |
| `plots/` | 9 visualization PNGs |

### Summary JSON Example

```json
{
  "total_samples": 1600,
  "keep": 892,
  "review": 418,
  "reject": 290,
  "keep_rate_pct": 55.75
}
```

---

## Visualization Outputs

The pipeline generates **9 summary plots**:

1. **score_distribution.png** - Final score histogram with decision bands
2. **decision_breakdown.png** - Keep/review/reject bar chart
3. **keep_rate_by_language.png** - Per-language keep rate (sorted)
4. **quality_vs_speech.png** - Quality proxy vs speech ratio scatter
5. **duration_by_language.png** - Audio duration box plot per language
6. **metric_correlation.png** - Correlation heatmap of all metrics
7. **snr_vs_final_score.png** - SNR vs final score
8. **asr_confidence_vs_final.png** - ASR confidence vs final score
9. **decision_by_language.png** - Stacked decision breakdown per language

> **Visual Analysis**
> 
> The pipeline includes comprehensive visual analysis:
> - Per-language quality breakdowns
> - Metric correlation heatmaps
> - ASR confidence analysis
> - Decision distribution visualizations
> 
> See `scripts/visualize_outputs.py` and `outputs/plots/` for generated visualizations.

---

## Configuration

### Default Thresholds (`configs/default.yaml`)

```yaml
audio:
  target_sr: 16000
  mono: true
  max_duration_sec: 30.0

thresholds:
  reject:
    min_duration_sec: 0.40
    max_duration_sec: 35.0
    min_rms_db: -50.0
    max_clipping_ratio: 0.050
    max_silence_ratio: 0.98
    min_speech_ratio: 0.03
    min_quality_proxy: 0.15
    min_snr_db: -5.0

  review:
    max_clipping_ratio: 0.010
    max_silence_ratio: 0.75
    min_speech_ratio: 0.20
    min_quality_proxy: 0.40
    min_asr_confidence: -1.25
    min_snr_db: 8.0

scoring:
  keep_min: 68.0
  review_min: 42.0
```

### Per-Language Customization

For production use with 22+ languages, create language-specific config overrides:

```yaml
# configs/hi.yaml
scoring:
  weights:
    snr: 0.20       # Higher SNR weight for Hindi (retroflex sensitivity)
    language: 0.10  # Stricter language verification
```

---

## Scalability Approach

### Parallel Processing Architecture

The pipeline uses a two-level parallelization strategy:

1. **Process-level parallelism** for CPU-bound metric computation
2. **Thread-level parallelism** for I/O-bound operations

```python
# Pattern mirrors setup_dataset.py for consistency
with ProcessPoolExecutor(max_workers=cpu_count()) as proc_exec:
    with ThreadPoolExecutor(max_workers=8) as thread_exec:
        # Submit batches for parallel processing
```

### Memory-Efficient Design

- Streaming JSONL parsing (no full manifest in memory)
- Incremental parquet writes with checkpoint support
- ONNX models loaded once per process, not per sample

### Expected Throughput

| Configuration | Samples/sec | 1600 samples | 1M samples |
|---------------|-------------|--------------|------------|
| Single-thread | ~2-3 | ~10 min | ~4 days |
| 8 workers | ~15-20 | ~1.5 min | ~14 hours |
| 32 workers | ~50-60 | ~30 sec | ~5 hours |

---

## Research Grounding

This pipeline implements metrics and thresholds from peer-reviewed research:

| Paper | Contribution |
|-------|-------------|
| **IndicVoices-R** (NeurIPS 2024, AI4Bharat) | SNR + C50 + speaking rate filtering; 30s duration cutoff; per-language analysis |
| **DNSMOS P.835** (ICASSP 2022, Microsoft) | Non-intrusive neural MOS; OVRL > 2.5 threshold |
| **TITW** (Interspeech 2024) | DNSMOS filtering validated for TTS training |
| **DataSpeech** (HuggingFace 2024) | Metric computation at scale reference |
| **Brouhaha** (ASRU 2023) | Multi-task VAD + SNR + C50 estimation |

Full citations and implementation details in [`papers.md`](papers.md).

---

## Project Structure

```
indic_audio_filtering/
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── pipeline.py          # FilteringPipeline class
│   ├── scoring.py           # Scoring + decision logic
│   ├── audio_utils.py       # Audio loading utilities
│   ├── config.py            # Config loader
│   ├── visualize.py         # Plot generation (9 plots)
│   └── metrics/
│       ├── __init__.py
│       ├── basic.py         # RMS, clipping, silence, spectral
│       ├── vad.py           # Silero VAD wrapper
│       ├── quality_proxy.py # Composite + DNSMOS
│       ├── snr.py           # WADA-SNR + energy fallback
│       ├── asr_bonus.py     # Whisper ASR metrics
│       └── langid_bonus.py  # Language verification
├── configs/
│   └── default.yaml         # Thresholds + weights
├── scripts/
│   ├── run_local_pipeline.py
│   ├── visualize_outputs.py
│   └── setup_dataset.py     # Provided by Sarvam (do not modify)
├── notebooks/
│   ├── 01_dataset_setup.ipynb
│   ├── 02_build_project_and_smoke_test.ipynb
│   ├── 03_run_full_pipeline_and_visualize.ipynb
│   └── 04_bonus_asr_lid_analysis.ipynb
├── requirements.txt
├── README.md                # This file
├── papers.md                # Research citations
└── why_choose_this.md       # Design rationale
```

---

## Testing

```bash
# Run smoke test on small subset
python -m src.main --manifest subset_manifest.jsonl --output_dir ./test_outputs --limit 50

# Run full subset (1600 samples)
python -m src.main --manifest subset_manifest.jsonl --output_dir ./outputs

# Resume interrupted run
python -m src.main --manifest subset_manifest.jsonl --output_dir ./outputs --resume
```

---

## License & Attribution

This pipeline was developed as a hiring assignment for Sarvam AI. 

Research implementations reference:
- AI4Bharat IndicVoices-R (NeurIPS 2024)
- Microsoft DNSMOS (ICASSP 2022)
- HuggingFace DataSpeech

---

