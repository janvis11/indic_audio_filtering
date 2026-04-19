"""
Visualization module — generates all summary plots for the pipeline output.

Produces 9 plots saved as PNG to output_dir/plots/:
  1. score_distribution.png       — final score histogram + decision bands
  2. decision_breakdown.png       — keep/review/reject bar chart
  3. keep_rate_by_language.png    — horizontal bar, sorted by keep rate
  4. quality_vs_speech.png        — quality_proxy vs speech_ratio scatter
  5. duration_by_language.png     — box plot per language
  6. metric_correlation.png       — correlation heatmap of all numeric metrics
  7. snr_vs_final_score.png       — SNR (dB) vs final score scatter (bonus)
  8. asr_confidence_vs_final.png  — ASR confidence vs final score (bonus)
  9. decision_by_language.png     — stacked bar of keep/review/reject per language (bonus)

All plots use a consistent color scheme:
  keep = #2ecc71 (green), review = #f39c12 (amber), reject = #e74c3c (red)
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in Colab and headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

DECISION_COLORS = {
    "keep":   "#2ecc71",
    "review": "#f39c12",
    "reject": "#e74c3c",
}
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
}


def _load(metrics_path: str):
    """Load metrics parquet into pandas DataFrame."""
    return pl.read_parquet(metrics_path).to_pandas()


def plot_score_distribution(pdf, out: Path):
    """Final score histogram with keep/review/reject decision bands."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(pdf["final_score"], bins=50, color="#4a90d9", alpha=0.8, edgecolor="none")
        ax.axvline(70, color=DECISION_COLORS["keep"],   linestyle="--", linewidth=1.5, label="keep threshold (70)")
        ax.axvline(45, color=DECISION_COLORS["reject"],  linestyle=":",  linewidth=1.5, label="reject threshold (45)")
        ax.axvspan(70, 100, alpha=0.07, color=DECISION_COLORS["keep"])
        ax.axvspan(45,  70, alpha=0.07, color=DECISION_COLORS["review"])
        ax.axvspan(0,   45, alpha=0.07, color=DECISION_COLORS["reject"])
        ax.set_xlabel("Final quality score (0–100)", fontsize=11)
        ax.set_ylabel("Sample count", fontsize=11)
        ax.set_title("Quality score distribution with decision bands", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 100)
        fig.tight_layout()
        fig.savefig(out / "score_distribution.png", dpi=150)
        plt.close(fig)
    logger.info("Saved score_distribution.png")


def plot_decision_breakdown(pdf, out: Path):
    """Keep / review / reject counts with percentages."""
    with plt.rc_context(STYLE):
        counts = pdf["decision"].value_counts().reindex(["keep", "review", "reject"]).fillna(0)
        total = counts.sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[DECISION_COLORS.get(d, "#888") for d in counts.index],
            edgecolor="none",
            width=0.5,
        )
        for bar, val in zip(bars, counts.values):
            pct = 100 * val / total if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{int(val)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_ylabel("Sample count", fontsize=11)
        ax.set_title("Filtering decision breakdown", fontsize=12, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.25)
        fig.tight_layout()
        fig.savefig(out / "decision_breakdown.png", dpi=150)
        plt.close(fig)
    logger.info("Saved decision_breakdown.png")


def plot_keep_rate_by_language(pdf, out: Path):
    """Horizontal bar — keep rate % per language, sorted descending."""
    if "language" not in pdf.columns or pdf["language"].isna().all():
        logger.warning("No language column — skipping keep_rate_by_language plot")
        return
    with plt.rc_context(STYLE):
        lang_keep = (
            pdf.groupby("language")["decision"]
            .apply(lambda s: round(100 * (s == "keep").mean(), 1))
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(9, max(4, len(lang_keep) * 0.45)))
        colors = [
            DECISION_COLORS["keep"] if v >= 70
            else DECISION_COLORS["review"] if v >= 45
            else DECISION_COLORS["reject"]
            for v in lang_keep.values
        ]
        bars = ax.barh(lang_keep.index, lang_keep.values, color=colors, edgecolor="none", height=0.6)
        for bar, val in zip(bars, lang_keep.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val}%", va="center", fontsize=9)
        ax.axvline(70, color=DECISION_COLORS["keep"],  linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(45, color=DECISION_COLORS["reject"], linestyle=":",  linewidth=1, alpha=0.7)
        ax.set_xlabel("Keep rate (%)", fontsize=11)
        ax.set_title("Keep rate by language\n(per-language thresholds applied)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 110)
        fig.tight_layout()
        fig.savefig(out / "keep_rate_by_language.png", dpi=150)
        plt.close(fig)
    logger.info("Saved keep_rate_by_language.png")


def plot_quality_vs_speech(pdf, out: Path):
    """Scatter: quality_proxy vs speech_ratio, colored by decision."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for decision, color in DECISION_COLORS.items():
            sub = pdf[pdf["decision"] == decision]
            ax.scatter(
                sub["speech_ratio"],
                sub["quality_proxy"],
                c=color, alpha=0.4, s=18, label=decision, edgecolors="none",
            )
        ax.set_xlabel("Speech ratio (VAD speech / total duration)", fontsize=11)
        ax.set_ylabel("Quality proxy score [0–1]", fontsize=11)
        ax.set_title("Quality proxy vs speech ratio by decision", fontsize=12, fontweight="bold")
        handles = [mpatches.Patch(color=c, label=d) for d, c in DECISION_COLORS.items()]
        ax.legend(handles=handles, fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out / "quality_vs_speech.png", dpi=150)
        plt.close(fig)
    logger.info("Saved quality_vs_speech.png")


def plot_duration_by_language(pdf, out: Path):
    """Box plot of audio duration per language."""
    if "language" not in pdf.columns or pdf["language"].isna().all():
        logger.warning("No language column — skipping duration_by_language plot")
        return
    with plt.rc_context(STYLE):
        langs = sorted(pdf["language"].dropna().unique())
        data = [pdf[pdf["language"] == lang]["duration_sec"].dropna().values for lang in langs]
        fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.2), 5))
        bp = ax.boxplot(data, labels=langs, patch_artist=True, notch=False, widths=0.5)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4a90d9")
            patch.set_alpha(0.6)
        ax.axhline(0.5,  color=DECISION_COLORS["reject"], linestyle="--", linewidth=1, label="min (0.5s)")
        ax.axhline(30.0, color=DECISION_COLORS["review"],  linestyle=":",  linewidth=1, label="max (30s)")
        ax.set_ylabel("Duration (seconds)", fontsize=11)
        ax.set_xlabel("Language", fontsize=11)
        ax.set_title("Audio duration distribution by language", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "duration_by_language.png", dpi=150)
        plt.close(fig)
    logger.info("Saved duration_by_language.png")


def plot_metric_correlation(pdf, out: Path):
    """Heatmap of correlations between numeric quality metrics."""
    numeric_cols = [
        c for c in [
            "final_score", "signal_score", "vad_score", "quality_score",
            "rms_db", "clipping_ratio", "silence_ratio_amp", "speech_ratio",
            "spectral_flatness_mean", "quality_proxy", "duration_sec",
        ]
        if c in pdf.columns
    ]
    if len(numeric_cols) < 3:
        logger.warning("Not enough numeric columns for correlation heatmap")
        return

    with plt.rc_context(STYLE):
        corr = pdf[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(len(numeric_cols) * 0.9, len(numeric_cols) * 0.75))
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels([c.replace("_", "\n") for c in numeric_cols], fontsize=7, rotation=0)
        ax.set_yticklabels([c.replace("_", " ") for c in numeric_cols], fontsize=7)
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 0.7 else "white")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        ax.set_title("Metric correlation matrix", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "metric_correlation.png", dpi=150)
        plt.close(fig)
    logger.info("Saved metric_correlation.png")


def plot_snr_vs_final_score(pdf, out: Path):
    """
    Scatter plot: SNR (dB) vs final score, colored by decision.

    This plot shows whether SNR is a good predictor of overall quality.
    Low SNR should correlate with low final scores if the pipeline is working correctly.
    """
    if "snr_db" not in pdf.columns or pdf["snr_db"].isna().all():
        logger.warning("No SNR data available — skipping snr_vs_final_score plot")
        return

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for decision, color in DECISION_COLORS.items():
            sub = pdf[pdf["decision"] == decision]
            valid = sub["snr_db"].notna()
            ax.scatter(
                sub.loc[valid, "snr_db"],
                sub.loc[valid, "final_score"],
                c=color, alpha=0.5, s=25, label=decision, edgecolors="none",
            )
        ax.set_xlabel("SNR (dB)", fontsize=11)
        ax.set_ylabel("Final quality score (0–100)", fontsize=11)
        ax.set_title("SNR vs Final Score\n(higher SNR should correlate with higher quality)", fontsize=12, fontweight="bold")
        handles = [mpatches.Patch(color=c, label=d) for d, c in DECISION_COLORS.items()]
        ax.legend(handles=handles, fontsize=9)
        ax.set_xlim(-10, 50)  # Typical SNR range
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(out / "snr_vs_final_score.png", dpi=150)
        plt.close(fig)
    logger.info("Saved snr_vs_final_score.png")


def plot_asr_confidence_vs_final(pdf, out: Path):
    """
    Scatter plot: ASR confidence proxy vs final score, colored by decision.

    This plot shows whether ASR confidence (intelligibility) correlates with
    overall quality. Low ASR confidence should indicate borderline or reject samples.
    """
    if "asr_confidence_proxy" not in pdf.columns or pdf["asr_confidence_proxy"].isna().all():
        logger.warning("No ASR confidence data — skipping asr_confidence_vs_final plot")
        return

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for decision, color in DECISION_COLORS.items():
            sub = pdf[pdf["decision"] == decision]
            valid = sub["asr_confidence_proxy"].notna()
            ax.scatter(
                sub.loc[valid, "asr_confidence_proxy"],
                sub.loc[valid, "final_score"],
                c=color, alpha=0.5, s=25, label=decision, edgecolors="none",
            )
        ax.set_xlabel("ASR confidence proxy [0–1]", fontsize=11)
        ax.set_ylabel("Final quality score (0–100)", fontsize=11)
        ax.set_title("ASR Confidence vs Final Score\n(higher confidence = more intelligible speech)", fontsize=12, fontweight="bold")
        handles = [mpatches.Patch(color=c, label=d) for d, c in DECISION_COLORS.items()]
        ax.legend(handles=handles, fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(out / "asr_confidence_vs_final.png", dpi=150)
        plt.close(fig)
    logger.info("Saved asr_confidence_vs_final.png")


def plot_decision_by_language(pdf, out: Path):
    """
    Stacked bar chart: keep/review/reject counts per language.

    This shows the decision breakdown by language, which is important for
    evaluating whether the pipeline is fair across different Indic languages.
    """
    if "language" not in pdf.columns or pdf["language"].isna().all():
        logger.warning("No language column — skipping decision_by_language plot")
        return

    with plt.rc_context(STYLE):
        # Compute counts per language per decision
        grouped = pdf.groupby("language")["decision"].value_counts().unstack(fill_value=0)
        # Reorder columns
        for col in ["keep", "review", "reject"]:
            if col not in grouped.columns:
                grouped[col] = 0
        grouped = grouped[["keep", "review", "reject"]]

        langs = grouped.index.tolist()
        fig, ax = plt.subplots(figsize=(max(10, len(langs) * 1.3), 5))

        x = np.arange(len(langs))
        width = 0.6

        # Stacked bars
        bottoms = np.zeros(len(langs))
        for decision in ["keep", "review", "reject"]:
            values = grouped[decision].values
            ax.bar(
                x, values, width, bottom=bottoms,
                color=DECISION_COLORS[decision],
                label=decision, edgecolor="none"
            )
            # Add count labels on bars
            for i, (bx, val) in enumerate(zip(bottoms, values)):
                if val > 0:
                    ax.text(
                        x[i], bx + val / 2, str(int(val)),
                        ha="center", va="center", fontsize=7, color="white"
                    )
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=0, fontsize=9)
        ax.set_ylabel("Sample count", fontsize=11)
        ax.set_xlabel("Language", fontsize=11)
        ax.set_title("Decision breakdown by language", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "decision_by_language.png", dpi=150)
        plt.close(fig)
    logger.info("Saved decision_by_language.png")


def create_summary_plots(metrics_path: str, output_dir: str) -> None:
    """
    Generate all 9 summary plots from the metrics parquet file.

    Args:
        metrics_path: Path to metrics.parquet
        output_dir:   Directory to write PNG files
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        pdf = _load(metrics_path)
    except Exception as e:
        logger.error(f"Could not load metrics from {metrics_path}: {e}")
        return

    logger.info(f"Generating plots for {len(pdf)} samples...")

    plot_score_distribution(pdf, out)
    plot_decision_breakdown(pdf, out)
    plot_keep_rate_by_language(pdf, out)
    plot_quality_vs_speech(pdf, out)
    plot_duration_by_language(pdf, out)
    plot_metric_correlation(pdf, out)
    # Bonus plots (Priority 3 visual analysis)
    plot_snr_vs_final_score(pdf, out)
    plot_asr_confidence_vs_final(pdf, out)
    plot_decision_by_language(pdf, out)

    logger.info(f"All plots saved to {out}/")
