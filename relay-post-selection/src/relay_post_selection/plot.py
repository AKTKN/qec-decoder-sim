"""Step 4: Plot post-selection performance metrics from decoded results.

Reads parquet files produced by decode.py and generates:
  1. Rejection rate vs. threshold (if multiple thresholds available)
  2. Post-LER vs. threshold
  3. Improvement ratio (post_ler / original_ler) vs. threshold
  4. LER curves: original vs. post-selected (if multiple noise params / distances)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def _load_results(results_dir: Path) -> pl.DataFrame:
    """Load all parquet result files from a directory into a single DataFrame."""
    parquet_files = sorted(results_dir.glob("results_batch=*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No result parquet files found in {results_dir}")

    frames = [pl.read_parquet(str(p)) for p in parquet_files]
    return pl.concat(frames)


def compute_metrics(df: pl.DataFrame) -> dict:
    """Compute post-selection metrics from a results DataFrame."""
    n_total = len(df)
    n_errors = df["is_logical_error"].sum()
    n_rejected = df["reject"].sum()
    n_accepted = n_total - n_rejected
    n_errors_accepted = df.filter(~pl.col("reject"))["is_logical_error"].sum()

    original_ler = n_errors / n_total if n_total > 0 else float("nan")
    rejection_rate = n_rejected / n_total if n_total > 0 else float("nan")
    post_ler = n_errors_accepted / n_accepted if n_accepted > 0 else float("nan")
    improvement_ratio = post_ler / original_ler if original_ler > 0 else float("nan")

    return {
        "n_total": n_total,
        "n_errors": int(n_errors),
        "n_rejected": int(n_rejected),
        "n_accepted": int(n_accepted),
        "n_errors_accepted": int(n_errors_accepted),
        "original_ler": original_ler,
        "rejection_rate": rejection_rate,
        "post_ler": post_ler,
        "improvement_ratio": improvement_ratio,
    }


def plot_single(
    results_dir: str | Path,
    out_dir: str | Path,
    label: Optional[str] = None,
) -> dict:
    """Plot metrics for a single results directory.

    Returns the computed metrics dict.
    """
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_results(results_dir)
    metrics = compute_metrics(df)

    config_path = results_dir / "decode_config.json"
    threshold = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        threshold = cfg.get("reject_threshold")

    title_suffix = f" (threshold={threshold})" if threshold is not None else ""
    label_str = label or results_dir.name

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Post-Selection Metrics{title_suffix}")

    metric_names = ["original_ler", "rejection_rate", "post_ler", "improvement_ratio"]
    metric_labels = [
        "Original LER",
        "Rejection Rate",
        "Post-Selection LER",
        "Improvement Ratio",
    ]
    bar_values = [
        metrics["original_ler"],
        metrics["rejection_rate"],
        metrics["post_ler"],
        metrics["improvement_ratio"],
    ]

    axes[0].bar([label_str], [metrics["original_ler"]], color="steelblue")
    axes[0].set_ylabel("Logical Error Rate")
    axes[0].set_title("Original LER")

    axes[1].bar([label_str], [metrics["rejection_rate"]], color="coral")
    axes[1].set_ylabel("Rejection Rate")
    axes[1].set_title("Rejection Rate")

    axes[2].bar([label_str], [metrics["post_ler"]], color="mediumseagreen", label="Post-LER")
    axes[2].axhline(metrics["original_ler"], color="steelblue", linestyle="--", label="Original LER")
    axes[2].set_ylabel("Logical Error Rate")
    axes[2].set_title(f"Post-LER (improvement: {metrics['improvement_ratio']:.3f}x)")
    axes[2].legend()

    plt.tight_layout()
    plot_path = out_dir / "summary_metrics.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary plot to %s", plot_path)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    return metrics


def plot_threshold_sweep(
    result_dirs: list[Path],
    thresholds: list[float],
    out_dir: str | Path,
    label: str = "",
) -> None:
    """Plot metrics vs. threshold for a sweep of result directories."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for rdir in result_dirs:
        df = _load_results(rdir)
        all_metrics.append(compute_metrics(df))

    original_lers = [m["original_ler"] for m in all_metrics]
    rejection_rates = [m["rejection_rate"] for m in all_metrics]
    post_lers = [m["post_ler"] for m in all_metrics]
    improvement_ratios = [m["improvement_ratio"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Post-Selection vs. Threshold {label}")

    axes[0, 0].plot(thresholds, rejection_rates, "o-", color="coral")
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Rejection Rate")
    axes[0, 0].set_title("Rejection Rate vs. Threshold")

    axes[0, 1].plot(thresholds, original_lers, "s--", color="steelblue", label="Original LER")
    axes[0, 1].plot(thresholds, post_lers, "o-", color="mediumseagreen", label="Post-LER")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Logical Error Rate")
    axes[0, 1].set_title("LER vs. Threshold")
    axes[0, 1].legend()

    axes[1, 0].plot(thresholds, improvement_ratios, "o-", color="purple")
    axes[1, 0].axhline(1.0, color="gray", linestyle="--", label="No improvement")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Improvement Ratio (post/original)")
    axes[1, 0].set_title("Improvement Ratio vs. Threshold")
    axes[1, 0].legend()

    axes[1, 1].plot(rejection_rates, post_lers, "o-", color="darkorange")
    axes[1, 1].set_xlabel("Rejection Rate")
    axes[1, 1].set_ylabel("Post-LER")
    axes[1, 1].set_title("Post-LER vs. Rejection Rate")

    plt.tight_layout()
    plot_path = out_dir / "threshold_sweep.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold sweep plot to %s", plot_path)

    summary = {
        "thresholds": thresholds,
        "rejection_rates": rejection_rates,
        "original_lers": original_lers,
        "post_lers": post_lers,
        "improvement_ratios": improvement_ratios,
    }
    with open(out_dir / "threshold_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
