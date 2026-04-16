Why I Chose the Audio Filtering Pipeline
Speech models are bounded by data quality. You can fine-tune architectures indefinitely, but if the training distribution is noisy, mislabeled, or clipped, model performance has a hard ceiling. The filtering pipeline sits upstream of everything - fix it once, and every downstream system benefits.
That's why I chose this task over the codec evaluation.

It's a production problem, not a benchmark exercise
The filtering pipeline asks: how do you systematically ensure data quality at scale, across 22 Indic languages, in a way that actually ships? 
That's an engineering problem with a reusable artifact at the end.
The provided setup_dataset.py already shows how the team thinks about data infrastructure - parallel parquet processing, manifest generation, structured logging. A well-designed filtering pipeline slots directly into that architecture.

Quality in speech data is multi-dimensional and language-sensitive
A naive pipeline checks SNR and calls it done. But:

High SNR doesn't guarantee intelligibility
Silence ratio can't catch semantic errors or wrong-language audio
Clipping thresholds that work for Hindi may reject valid retroflex-heavy Tamil speech
IndicVoices is community-contributed - mislabeling and language bleed are real

This requires a layered quality framework: fast heuristics first (clipping, silence), then signal-level metrics (SNR, spectral flatness), then semantic checks (ASR confidence, language ID). Each layer filters what the previous one misses. That kind of thinking is more interesting to design and more useful in production than running three codecs through the same eval harness.

Scalability isn't a bonus - it's the problem
1000+ hours of multilingual audio means millions of samples. 

At that scale:
Sequential processing is a non-starter
I/O becomes the bottleneck before compute does
Memory-aware design matters more than algorithmic elegance

The interesting engineering lives here: how do you parallelize metric computation without blowing memory? How do you structure per-language output manifests so they're usable in NeMo or ESPnet without post-processing? These are the questions this task forces you to answer.

The output is immediately usable
The end product is a filtered manifest with per-sample quality scores and a keep flag - something a training pipeline can consume directly. It's not a report or a notebook. It's infrastructure.
That's what I set out to build.
