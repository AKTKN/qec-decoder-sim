"""Dispatch CLI for relay-post-selection.

Sub-commands:
    step3   Decode sampled data and write parquet results.
    step4   Plot post-selection performance metrics.

Examples:
    python -m relay_post_selection step3 \\
        --data_dir sampled_data/surface_d5/ \\
        --out_dir results/surface_d5_thresh-2/ \\
        --threshold -2.0 \\
        --processes 8

    python -m relay_post_selection step4 \\
        --results_dir results/surface_d5_thresh-2/ \\
        --out_dir plots/surface_d5/
"""
from __future__ import annotations

import argparse
import logging
import sys


def _add_decoder_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Decoder options")
    g.add_argument("--decoder_type", default="relay", choices=["relay", "membp"])
    g.add_argument("--alpha", type=float, default=None)
    g.add_argument("--gamma0", type=float, default=0.1,
                   help="Mem-BP memory strength (default: 0.1)")
    g.add_argument("--gamma_dist_interval", type=float, nargs=2,
                   default=[-0.24, 0.66], metavar=("LO", "HI"),
                   help="Relay gamma distribution interval [lo, hi] (default: -0.24 0.66)")
    g.add_argument("--llr_clip_min", type=float, default=None,
                   help="Minimum LLR clip value (e.g. -4.0)")
    g.add_argument("--llr_clip_max", type=float, default=None,
                   help="Maximum LLR clip value (e.g. 8.0)")
    g.add_argument("--pre_iter", type=int, default=60)
    g.add_argument("--num_sets", type=int, default=60)
    g.add_argument("--set_max_iter", type=int, default=60)
    g.add_argument("--stop_nconv", type=int, default=5)
    g.add_argument("--stopping_criterion", default="nconv",
                   choices=["nconv", "pre_iter", "all"])
    g.add_argument("--max_iter", type=int, default=200,
                   help="Max iterations (membp only)")


def _cmd_step3(args: argparse.Namespace) -> None:
    from .config import DecoderConfig, PostSelectionConfig
    from .decode import run_decode

    decoder_cfg = DecoderConfig(
        alpha=args.alpha,
        gamma0=args.gamma0,
        gamma_dist_interval=tuple(args.gamma_dist_interval),
        llr_clip_min=args.llr_clip_min,
        llr_clip_max=args.llr_clip_max,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        stopping_criterion=args.stopping_criterion,
        decoder_type=args.decoder_type,
        max_iter=args.max_iter,
    )
    cfg = PostSelectionConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        reject_threshold=args.threshold,
        processes=args.processes,
        decoder=decoder_cfg,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    out_dir = run_decode(cfg)
    print(f"Decode complete. Results written to: {out_dir}")


def _cmd_step4(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .plot import plot_single, plot_threshold_sweep

    out_dir = Path(args.out_dir)

    if args.sweep_dirs:
        # Multiple result dirs for threshold sweep
        sweep_dirs = [Path(d) for d in args.sweep_dirs]
        if not args.sweep_thresholds:
            print("Error: --sweep_thresholds is required with --sweep_dirs.", file=sys.stderr)
            sys.exit(1)
        thresholds = [float(t) for t in args.sweep_thresholds]
        if len(sweep_dirs) != len(thresholds):
            print(
                f"Error: --sweep_dirs ({len(sweep_dirs)} entries) and "
                f"--sweep_thresholds ({len(thresholds)} entries) must have the same count.",
                file=sys.stderr,
            )
            sys.exit(1)
        plot_threshold_sweep(sweep_dirs, thresholds, out_dir, label=args.label)
    else:
        if args.results_dir is None:
            print("Error: --results_dir is required when --sweep_dirs is not given.", file=sys.stderr)
            sys.exit(1)
        results_dir = Path(args.results_dir)
        metrics = plot_single(results_dir, out_dir, label=args.label)
        print("Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
    print(f"Plots written to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="relay_post_selection",
        description="relay-bp post-selection simulation pipeline",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # step3
    p3 = subparsers.add_parser("step3", help="Decode sampled data and compute reject flags.")
    p3.add_argument("--data_dir", required=True,
                    help="Sampled data directory from step2 (contains dem.dem, sampled_data/).")
    p3.add_argument("--out_dir", required=True, help="Output directory for parquet results.")
    p3.add_argument("--threshold", type=float, required=True,
                    help="LLR threshold for reject metric (e.g. -2.0).")
    p3.add_argument("--processes", type=int, default=4,
                    help="Number of parallel worker processes.")
    p3.add_argument("--overwrite", action="store_true", help="Overwrite existing results.")
    p3.add_argument("--verbose", action="store_true",
                    help="Print real-time progress with per-shot timing and ETA.")
    _add_decoder_args(p3)

    # step4
    p4 = subparsers.add_parser("step4", help="Plot post-selection metrics.")
    p4.add_argument("--results_dir", default=None,
                    help="Single results directory (from step3).")
    p4.add_argument("--out_dir", required=True, help="Output directory for plots.")
    p4.add_argument("--label", default="", help="Label for plot titles.")
    p4.add_argument(
        "--sweep_dirs",
        nargs="*",
        help="Multiple result directories for threshold sweep (in threshold order).",
    )
    p4.add_argument(
        "--sweep_thresholds",
        nargs="*",
        help="Threshold values corresponding to --sweep_dirs.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.command == "step3":
        _cmd_step3(args)
    elif args.command == "step4":
        _cmd_step4(args)


if __name__ == "__main__":
    main()
