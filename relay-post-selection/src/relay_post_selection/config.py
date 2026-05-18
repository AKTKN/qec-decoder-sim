from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DecoderConfig:
    """relay-bp decoder hyperparameters."""

    # MinSumBP / RelayBP shared
    alpha: Optional[float] = None
    gamma0: float = 0.1
    llr_clip_min: Optional[float] = None
    llr_clip_max: Optional[float] = None

    # Relay-specific
    pre_iter: int = 60
    num_sets: int = 60
    set_max_iter: int = 60
    gamma_dist_interval: tuple[float, float] = (-0.24, 0.66)
    stop_nconv: int = 5
    stopping_criterion: str = "nconv"

    # Decoder selection
    decoder_type: str = "relay"  # "relay" or "membp"

    # membp-specific
    max_iter: int = 200


@dataclass
class PostSelectionConfig:
    """Full configuration for the post-selection decode step."""

    data_dir: str
    out_dir: str
    reject_threshold: float
    processes: int = 4
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    overwrite: bool = False
    verbose: bool = False
