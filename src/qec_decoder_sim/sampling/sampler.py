from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import stim

logger = logging.getLogger(__name__)


@dataclass
class SamplingMetadata:
    circuit_path: Optional[str]
    num_shots: int
    batch_size: int
    num_batches: int
    seed: int
    num_detectors: int
    num_observables: int


class SyndromeDataSampler:
    """Samples detector syndromes and observable flips from a stim circuit.

    Output layout (per batch file, b8 format with append_observables=True):
        Each row has ceil((num_detectors + num_observables) / 8) bytes.
        The last num_observables bits of each row are the observable flips.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        circuit_path: Optional[str] = None,
    ) -> None:
        self.circuit = circuit
        self.circuit_path = circuit_path
        self._dem: Optional[stim.DetectorErrorModel] = None

    @classmethod
    def from_file(cls, circuit_path: str | Path) -> "SyndromeDataSampler":
        path = Path(circuit_path)
        circuit = stim.Circuit.from_file(str(path))
        return cls(circuit, circuit_path=str(path))

    @property
    def dem(self) -> stim.DetectorErrorModel:
        if self._dem is None:
            self._dem = self.circuit.detector_error_model(
                decompose_errors=False,
                ignore_decomposition_failures=True,
            )
        return self._dem

    def sample(
        self,
        out_dir: str | Path,
        num_shots: int,
        seed: int = 0,
        batch_size: int = 10_000,
        overwrite: bool = False,
    ) -> Path:
        """Sample num_shots shots and write to out_dir.

        Returns the output directory path.
        """
        out_dir = Path(out_dir)
        sampled_data_dir = out_dir / "sampled_data"
        sampled_data_dir.mkdir(parents=True, exist_ok=True)

        dem_path = out_dir / "dem.dem"
        if overwrite or not dem_path.exists():
            self.dem.to_file(str(dem_path))
            logger.info("Wrote DEM to %s", dem_path)

        num_batches = (num_shots + batch_size - 1) // batch_size
        remaining = num_shots

        for batch_idx in range(1, num_batches + 1):
            shots_this_batch = min(batch_size, remaining)
            remaining -= shots_this_batch

            out_path = sampled_data_dir / f"det_batch={batch_idx}.b8"
            if not overwrite and out_path.exists():
                logger.info("Skipping existing batch: %s", out_path)
                continue

            batch_seed = seed + (batch_idx - 1)
            sampler = self.circuit.compile_detector_sampler(seed=batch_seed)
            sampler.sample_write(
                shots=shots_this_batch,
                filepath=out_path,
                format="b8",
                append_observables=True,
            )
            logger.info(
                "Wrote batch %d/%d (%d shots) to %s",
                batch_idx,
                num_batches,
                shots_this_batch,
                out_path,
            )

        metadata = SamplingMetadata(
            circuit_path=self.circuit_path,
            num_shots=num_shots,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
            num_detectors=self.dem.num_detectors,
            num_observables=self.dem.num_observables,
        )
        metadata_path = out_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        logger.info("Wrote metadata to %s", metadata_path)

        return out_dir
