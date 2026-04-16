## Papers used and exactly what was taken
### IndicVoices-R - Sankar et al., NeurIPS 2024 (AI4Bharat)
The direct predecessor to this exact task. They filtered IndicVoices using SNR, C50, speaking rate, and pitch. The 30-second duration cutoff, the C50 mean of 53.45 dB as the quality benchmark, and the speaking rate bounds of 0.5-6.0 words/second all come directly from this paper. The per-language keep rates and the comparison against LJSpeech/LibriTTS as quality benchmarks also come from here.
### DNSMOS P.835 - Reddy, Gopal, Cutler, ICASSP 2022 (Microsoft)
The DNSMOS ONNX models, the three output scores (SIG, BAK, OVRL), and the threshold of 2.5 OVRL as the quality floor all come from this paper. The core justification - that conventional metrics require a clean reference signal while DNSMOS doesn't - is stated verbatim in the paper abstract.
### TITW (Text-to-Speech In The Wild) - Jung et al., Interspeech 2024
The validation that DNSMOS > 2.5 is sufficient to train a working TTS model came from this paper. They built TITW-Easy by applying DNSMOS filtering to VoxCeleb1 and showed that TTS models trained on it achieve UTMOS > 3.0. The pipeline structure of transcription -> segmentation -> DNSMOS filtering is theirs.
### DataSpeech - HuggingFace, 2024
The specific combination of SNR + C50 + speaking rate as a metric set, and the ProcessPoolExecutor + GPU-parallel computation pattern. IndicVoices-R explicitly cites DataSpeech as their metric computation tool.
### NISQA crowdsourced TTS pipeline - arXiv 2410.13357, 2024
The finding that quality-gated filtering produces a 0.4 UTMOS improvement over unfiltered data. Used to justify that filtering actually matters for downstream model quality, not just as a data hygiene exercise.
### Brouhaha - Lavechin et al., ASRU 2023
The C50 computation method. Brouhaha is a multi-task model that estimates VAD + SNR + C50 jointly. IndicVoices-R uses it specifically for C50. The heuristic fallback (early/late energy ratio) is an approximation of what Brouhaha computes properly.
### WADA-SNR - Kim & Stern, Interspeech 2008
The blind SNR estimation method that works without a clean reference signal. The key property taken: it estimates SNR from the shape of the amplitude distribution, so it works on single-channel community recordings with no paired clean audio.