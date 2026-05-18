"""Step 2: Sample syndrome and observable flip data from a stim circuit.

Usage:
    python -m qec_decoder_sim.sampling \\
        --circuit path/to/circuit.stim \\
        --out_dir output/sampled/ \\
        --shots 100000 \\
        --seed 42 \\
        --batch_size 10000
"""
from __future__ import annotations

import argparse
import logging

from .sampler import SyndromeDataSampler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample syndrome + observable flip data from a stim circuit."
    )
    parser.add_argument(
        "--circuit",
        required=True,
        help="Path to the .stim circuit file.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory. Will contain dem.dem, metadata.json, and sampled_data/.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
        help="Total number of shots to sample (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
        help="Number of shots per batch file (default: 10000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing batch files.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    sampler = SyndromeDataSampler.from_file(args.circuit)
    out_dir = sampler.sample(
        out_dir=args.out_dir,
        num_shots=args.shots,
        seed=args.seed,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(f"Sampling complete. Output written to: {out_dir}")


if __name__ == "__main__":
    main()
