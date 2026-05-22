from __future__ import annotations

import argparse
import itertools
import pathlib
from dataclasses import dataclass

import stim

from qec_decoder_sim.circuits import BivariateBicycleCode
from qec_decoder_sim.circuits._topological_memory_circuits import make_topological_memory_circuit
from qec_decoder_sim.codes.code_factory import BBCodeSpec

_SUPPORTED_CODES = {
    "surface_code_Z",
    "superdense_color_code_Z",
    "midout_color_code_Z",
    "bivariate_bicycle_code_Z",
}

_NOISE_MODEL_ALIASES = {
    "uniform": "uniform-cln",
    "uniform-cln": "uniform-cln",
    "si1000": "si1000",
    "bitflip": "bitflip",
    "phenomenological": "phenomenological",
}


@dataclass(frozen=True)
class CircuitJob:
    code: str
    distance: int
    rounds_spec: str
    noise_model_raw: str
    noise_strength: float
    use_both: bool = False
    ibm_reproduce: bool = False


def _parse_int_list(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid integer list; use comma-separated ints") from exc
    return values


def _parse_rounds(rounds_spec: str, *, distance: int) -> int:
    spec = rounds_spec.strip().lower().replace(" ", "")
    if not spec:
        raise ValueError("rounds must be a positive integer, 'd', or 'k*d'")
    if spec == "d":
        return _require_positive(distance, name="distance")
    if spec.isdigit():
        return _require_positive(int(spec), name="rounds")
    if spec.endswith("*d"):
        factor = _require_positive(int(spec[:-2]), name="rounds multiplier")
        return factor * _require_positive(distance, name="distance")
    if spec.startswith("d*"):
        factor = _require_positive(int(spec[2:]), name="rounds multiplier")
        return factor * _require_positive(distance, name="distance")
    raise ValueError("rounds must be a positive integer, 'd', or 'k*d'")


def _require_positive(value: int, *, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _resolve_noise_model(raw: str) -> str:
    key = raw.strip().lower()
    if key not in _NOISE_MODEL_ALIASES:
        raise ValueError(
            "noise_model must be one of "
            f"{sorted(_NOISE_MODEL_ALIASES.keys())}; got {raw!r}"
        )
    return _NOISE_MODEL_ALIASES[key]


def _basis_from_code(code: str) -> str:
    if code.endswith("_Z"):
        return "z"
    if code.endswith("_X"):
        return "x"
    raise ValueError("code must end with '_Z' or '_X'")


def _build_bb_circuit(
    *,
    job: CircuitJob,
    bb_spec: BBCodeSpec,
) -> stim.Circuit:
    rounds = _parse_rounds(job.rounds_spec, distance=job.distance)
    noise_model = _resolve_noise_model(job.noise_model_raw)
    builder = BivariateBicycleCode(
        bbcode_spec=bb_spec,
        noise_model=noise_model,
        p=job.noise_strength,
        num_rounds=rounds,
        basis=_basis_from_code(job.code),
        use_both=job.use_both,
        ibm_reproduce=job.ibm_reproduce,
    )
    return builder.get_circuit()


def _build_topological_circuit(*, job: CircuitJob) -> stim.Circuit:
    rounds = _parse_rounds(job.rounds_spec, distance=job.distance)
    return make_topological_memory_circuit(
        style=job.code,
        rounds=rounds,
        diameter=job.distance,
        noise_model=job.noise_model_raw,
        noise_strength=job.noise_strength,
    )


def _write_circuit(
    *,
    circuit: stim.Circuit,
    out_dir: pathlib.Path,
    job: CircuitJob,
) -> pathlib.Path:
    noise_tag = job.noise_model_raw.strip().lower()
    rounds = 1 if noise_tag == "bitflip" else _parse_rounds(job.rounds_spec, distance=job.distance)

    # Under the phenomenological noise model, superdense and midout color codes
    # produce identical circuits, so we save them under the unified name.
    code_name = job.code
    if noise_tag == "phenomenological" and code_name in {"superdense_color_code_Z", "midout_color_code_Z"}:
        code_name = "color_code_Z"

    stem = (
        f"code={code_name},d={job.distance},rounds={rounds},"
        f"noisemodel={noise_tag},p={job.noise_strength}"
    )
    if job.code == "bivariate_bicycle_code_Z":
        stem += f",use_both={job.use_both}"
    path = out_dir / f"{stem}.stim"
    with open(path, "w", encoding="utf-8") as handle:
        print(circuit, file=handle)
    return path


def _build_jobs(args: argparse.Namespace) -> list[CircuitJob]:
    jobs: list[CircuitJob] = []
    for (code, distance, rounds_spec, noise_model_raw, noise_strength) in itertools.product(
        args.code,
        args.distance,
        args.rounds,
        args.noise_model,
        args.noise_strength,
    ):
        jobs.append(
            CircuitJob(
                code=code,
                distance=int(distance),
                rounds_spec=rounds_spec,
                noise_model_raw=noise_model_raw,
                noise_strength=float(noise_strength),
                use_both=args.use_both,
                ibm_reproduce=args.ibm_reproduce,
            )
        )
    return jobs


def _validate_args(args: argparse.Namespace) -> None:
    unknown_codes = sorted(set(args.code) - _SUPPORTED_CODES)
    if unknown_codes:
        raise ValueError(f"Unknown codes: {unknown_codes}")

    if "bivariate_bicycle_code_Z" in args.code:
        missing = [
            name
            for name in ("l", "m", "a_x_pows", "a_y_pows", "b_x_pows", "b_y_pows")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "bivariate_bicycle_code_Z requires " + ", ".join(f"--{name}" for name in missing)
            )

    if args.ibm_reproduce:
        non_uniform = [nm for nm in args.noise_model if nm not in {"uniform", "uniform-cln"}]
        if non_uniform:
            raise ValueError(
                f"--ibm_reproduce requires --noise_model uniform; got {non_uniform}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stim circuits for QEC simulation.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--code", nargs="+", required=True, choices=sorted(_SUPPORTED_CODES))
    parser.add_argument("--distance", nargs="+", required=True, type=int)
    parser.add_argument("--rounds", nargs="+", required=True, type=str)
    parser.add_argument("--noise_model", nargs="+", required=True, choices=sorted(_NOISE_MODEL_ALIASES.keys()))
    parser.add_argument("--noise_strength", nargs="+", required=True, type=float)
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--l", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--a_x_pows", type=_parse_int_list)
    parser.add_argument("--a_y_pows", type=_parse_int_list)
    parser.add_argument("--b_x_pows", type=_parse_int_list)
    parser.add_argument("--b_y_pows", type=_parse_int_list)
    parser.add_argument("--use_both", action="store_true", default=False)
    parser.add_argument("--ibm_reproduce", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    jobs = _build_jobs(args)
    if args.stdout and len(jobs) != 1:
        raise ValueError("--stdout requires exactly one circuit job")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bb_spec = None
    if "bivariate_bicycle_code_Z" in args.code:
        bb_spec = BBCodeSpec(
            l=int(args.l),
            m=int(args.m),
            a_x_pows=tuple(args.a_x_pows),
            a_y_pows=tuple(args.a_y_pows),
            b_x_pows=tuple(args.b_x_pows),
            b_y_pows=tuple(args.b_y_pows),
        )

    for job in jobs:
        if job.code == "bivariate_bicycle_code_Z":
            assert bb_spec is not None
            circuit = _build_bb_circuit(job=job, bb_spec=bb_spec)
        elif job.code in {"surface_code_Z", "superdense_color_code_Z", "midout_color_code_Z"}:
            circuit = _build_topological_circuit(job=job)
        else:
            raise NotImplementedError(f"Circuit generation is not implemented for {job.code}")

        if args.stdout:
            print(circuit)
            continue

        path = _write_circuit(circuit=circuit, out_dir=out_dir, job=job)
        print(f"wrote file://{path.absolute()}")


if __name__ == "__main__":
    main()
