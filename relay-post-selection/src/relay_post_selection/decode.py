"""Step 3: Decode sampled data with relay-bp and compute per-shot reject flags.

Each worker process handles one .b8 batch file independently.
Results are written as parquet files: columns [shot_id, is_logical_error, reject].
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import stim

from .config import DecoderConfig, PostSelectionConfig

logger = logging.getLogger(__name__)

# Chunk size for progress reporting within each worker (shots per report).
_PROGRESS_CHUNK = 500


def _build_observable_decoder(
    check_matrix: Any,
    error_priors: Any,
    observables_matrix: Any,
    decoder_cfg: DecoderConfig,
    reject_threshold: float,
) -> Any:
    """Build an ObservableDecoderRunner from relay-bp."""
    import relay_bp

    if decoder_cfg.decoder_type == "relay":
        inner = relay_bp.RelayDecoderF64(
            check_matrix,
            error_priors=error_priors,
            alpha=decoder_cfg.alpha,
            gamma0=decoder_cfg.gamma0,
            pre_iter=decoder_cfg.pre_iter,
            num_sets=decoder_cfg.num_sets,
            set_max_iter=decoder_cfg.set_max_iter,
            gamma_dist_interval=decoder_cfg.gamma_dist_interval,
            stop_nconv=decoder_cfg.stop_nconv,
            stopping_criterion=decoder_cfg.stopping_criterion,
            llr_clip_min=decoder_cfg.llr_clip_min,
            llr_clip_max=decoder_cfg.llr_clip_max,
        )
    elif decoder_cfg.decoder_type == "membp":
        inner = relay_bp.MinSumBPDecoderF64(
            check_matrix,
            error_priors=error_priors,
            max_iter=decoder_cfg.max_iter,
            alpha=decoder_cfg.alpha,
            gamma0=decoder_cfg.gamma0,
            llr_clip_min=decoder_cfg.llr_clip_min,
            llr_clip_max=decoder_cfg.llr_clip_max,
        )
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_cfg.decoder_type!r}")

    runner = relay_bp.ObservableDecoderRunner(
        inner,
        observables_matrix,
        include_decode_result=False,
        reject_threshold=reject_threshold,
    )
    return runner


def _load_check_matrices(dem_path: Path) -> Any:
    """Load DEM and return a CheckMatrices object."""
    from relay_bp.stim.sinter.check_matrices import CheckMatrices

    dem = stim.DetectorErrorModel.from_file(str(dem_path))
    return CheckMatrices.from_dem(dem, decomposed_hyperedges=None, prune_decided_errors=True)


def _decode_batch_worker(args: tuple) -> None:
    """Worker function: decode one .b8 batch and write parquet output.

    If args contains a progress_queue (last element, not None), the worker
    sends (shots_done: int) updates every _PROGRESS_CHUNK shots.
    """
    (
        batch_path,
        out_path,
        dem_path,
        num_detectors,
        num_observables,
        batch_index,
        shot_offset,
        decoder_cfg_dict,
        reject_threshold,
        progress_queue,  # Optional[multiprocessing.Queue]; None when verbose=False
    ) = args

    decoder_cfg = DecoderConfig(**decoder_cfg_dict)

    cm = _load_check_matrices(dem_path)
    runner = _build_observable_decoder(
        cm.check_matrix, cm.error_priors, cm.observables_matrix, decoder_cfg, reject_threshold
    )

    raw = stim.read_shot_data_file(
        path=str(batch_path),
        format="b8",
        num_detectors=num_detectors + num_observables,
        bit_packed=False,
    )
    detectors = raw[:, :num_detectors].astype(np.uint8)
    actual_observables = raw[:, num_detectors:].astype(np.uint8)
    num_shots = detectors.shape[0]

    if cm.syndrome_bias is not None:
        detectors = (detectors + cm.syndrome_bias) % 2

    is_logical_error_list: list[bool] = []
    reject_list: list[bool] = []

    chunk = _PROGRESS_CHUNK
    for start in range(0, num_shots, chunk):
        end = min(start + chunk, num_shots)
        sub_det = detectors[start:end]
        sub_obs = actual_observables[start:end]

        results = runner.decode_observables_detailed_batch(
            sub_det,
            parallel=False,
            progress_bar=False,
        )

        for i, result in enumerate(results):
            predicted_obs = np.array(result.observables, dtype=np.uint8)
            if cm.observables_bias is not None:
                predicted_obs = (predicted_obs + cm.observables_bias) % 2
            is_logical_error_list.append(bool(np.any(predicted_obs != sub_obs[i])))
            reject_list.append(bool(result.reject) if result.reject is not None else False)

        if progress_queue is not None:
            progress_queue.put(end - start)

    shot_ids = np.arange(shot_offset, shot_offset + num_shots, dtype=np.int64)
    table = pa.table(
        {
            "shot_id": shot_ids,
            "is_logical_error": np.array(is_logical_error_list, dtype=bool),
            "reject": np.array(reject_list, dtype=bool),
        }
    )
    pq.write_table(table, str(out_path))
    logger.info("Batch %d: wrote %d shots to %s", batch_index, num_shots, out_path)


def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def _progress_printer(
    queue: Any,
    total_shots: int,
    start_time: float,
) -> None:
    """Daemon thread: reads shot counts from queue and prints a live progress line."""
    processed = 0
    while processed < total_shots:
        try:
            n = queue.get(timeout=0.5)
        except Exception:
            # Timeout — redraw with current numbers so the ETA stays fresh.
            n = 0

        if n is None:  # sentinel: all workers finished
            break

        processed += n
        elapsed = time.time() - start_time
        pct = processed / total_shots * 100
        per_shot_ms = elapsed / processed * 1000 if processed > 0 else 0.0
        eta_sec = (total_shots - processed) / processed * elapsed if processed > 0 else 0.0

        line = (
            f"\r[step3] {processed:>{len(str(total_shots))}}/{total_shots} shots"
            f"  {pct:5.1f}%"
            f"  elapsed {_fmt_time(elapsed)}"
            f"  {per_shot_ms:.2f} ms/shot"
            f"  ETA {_fmt_time(eta_sec)}"
            f"   "  # trailing spaces to overwrite longer previous lines
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    # Final line
    elapsed = time.time() - start_time
    sys.stdout.write(
        f"\r[step3] {total_shots}/{total_shots} shots  100.0%"
        f"  elapsed {_fmt_time(elapsed)}"
        f"  done.           \n"
    )
    sys.stdout.flush()


def run_decode(cfg: PostSelectionConfig) -> Path:
    """Run the decode step and write per-batch parquet results.

    Returns the output directory path.
    """
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    dem_path = data_dir / "dem.dem"
    num_detectors: int = metadata["num_detectors"]
    num_observables: int = metadata["num_observables"]
    num_batches: int = metadata["num_batches"]
    batch_size: int = metadata["batch_size"]

    # Build worker args list (progress_queue filled in later).
    pending_batches = []
    skipped_shots = 0
    for batch_idx in range(1, num_batches + 1):
        batch_path = data_dir / "sampled_data" / f"det_batch={batch_idx}.b8"
        if not batch_path.exists():
            logger.warning("Batch file not found, skipping: %s", batch_path)
            continue

        out_path = out_dir / f"results_batch={batch_idx}.parquet"
        if not cfg.overwrite and out_path.exists():
            logger.info("Skipping existing result: %s", out_path)
            # Estimate shots skipped (last batch may be smaller).
            skipped_shots += batch_size
            continue

        shot_offset = (batch_idx - 1) * batch_size
        pending_batches.append((
            batch_path,
            out_path,
            dem_path,
            num_detectors,
            num_observables,
            batch_idx,
            shot_offset,
            asdict(cfg.decoder),
            cfg.reject_threshold,
        ))

    if not pending_batches:
        logger.info("All batches already decoded.")
        return out_dir

    # Estimate total shots to decode (last batch may be smaller than batch_size).
    total_shots = metadata["num_shots"] - skipped_shots

    # Set up progress queue and printer thread when verbose.
    progress_queue: Optional[Any] = None
    printer_thread: Optional[threading.Thread] = None
    start_time = time.time()

    if cfg.verbose:
        manager = mp.Manager()
        progress_queue = manager.Queue()
        printer_thread = threading.Thread(
            target=_progress_printer,
            args=(progress_queue, total_shots, start_time),
            daemon=True,
        )
        printer_thread.start()

    # Attach queue (or None) as the last element of each worker's args.
    worker_args = [batch + (progress_queue,) for batch in pending_batches]

    try:
        if cfg.processes <= 1:
            for args in worker_args:
                _decode_batch_worker(args)
        else:
            with mp.Pool(processes=cfg.processes) as pool:
                pool.map(_decode_batch_worker, worker_args)
    finally:
        if progress_queue is not None:
            progress_queue.put(None)  # sentinel to stop printer
        if printer_thread is not None:
            printer_thread.join(timeout=5.0)

    config_path = out_dir / "decode_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "reject_threshold": cfg.reject_threshold,
                "decoder": asdict(cfg.decoder),
            },
            f,
            indent=2,
        )

    return out_dir
