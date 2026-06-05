from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qec_decoder_sim.decoder.sinter_registry import SUPPORTED_DECODERS

# All topological (non-BB) circuit styles supported by the memory experiment runner.
TOPOLOGICAL_CIRCUIT_TYPES: frozenset[str] = frozenset(
    {"surface_code_Z", "superdense_color_code_Z", "midout_color_code_Z"}
)

BB_CIRCUIT_TYPE = "bivariate_bicycle_code_Z"
HGP_CIRCUIT_TYPE = "hgp_code_Z"


@dataclass(frozen=True)
class ExperimentConfig:
    # --- shots / sweep ---
    shots_by_p: tuple[int, ...]
    p_values: tuple[float, ...]

    # --- circuit identity ---
    circuit_type: str
    """Which code/circuit style to simulate.  One of TOPOLOGICAL_CIRCUIT_TYPES or BB_CIRCUIT_TYPE."""

    basis: str

    # --- topological code parameters (required when circuit_type != BB_CIRCUIT_TYPE) ---
    distances: tuple[int, ...] | None
    """Diameter / distance sweep for topological codes.  None for BB codes.
    A single-element tuple runs one distance; a multi-element tuple sweeps all values."""

    # --- BB code parameters (required when circuit_type == BB_CIRCUIT_TYPE) ---
    l: int | None
    m: int | None
    a_x_pows: tuple[int, ...] | None
    a_y_pows: tuple[int, ...] | None
    b_x_pows: tuple[int, ...] | None
    b_y_pows: tuple[int, ...] | None
    ibm_reproduce: bool
    use_both: bool

    # --- HGP code parameters (required when circuit_type == HGP_CIRCUIT_TYPE) ---
    hgp_pcm_path: str | None
    """Path to the classical parity-check matrix file for HGP codes."""
    hgp_build_graph_seed: int | None
    """Seed for cardinal edge orientation; None means use default (22)."""
    hgp_computed_distance: int | None
    """Quantum code distance computed from hgp_pcm_path at config load time."""

    # --- rounds spec ---
    num_rounds_spec: str
    """Number of syndrome-extraction rounds per shot.
    Accepts a positive integer (e.g. ``"12"``), ``"d"`` (= distance), or
    ``"k*d"`` / ``"d*k"`` (= k × distance).  For BB codes only integer strings
    are valid (distance is not a single well-defined scalar there)."""

    # --- noise ---
    noise_mode: str
    noise_model: str
    num_circuit_samples: int
    noise_sigma_rel: float
    circuit_seed_base: int
    circuit_seeds: tuple[int, ...] | None

    # --- decoding ---
    xyz_decoding: bool
    """True (default): decode X and Z errors jointly.
    False: filter the circuit to a single Pauli basis before building the check matrix.
    Applies only to BP-type decoders (bposd, bplsd, relay-bp)."""

    # --- execution ---
    processes: int
    detector_seed_base: int
    print_progress: bool
    output_root: Path
    decoders: tuple[str, ...]


@dataclass(frozen=True)
class DecoderVariant:
    alias: str
    decoder_type: str
    params: dict[str, Any]


def default_config_path(*, repo_root: Path) -> Path:
    return repo_root / "scripts_shell" / "config_example.yaml"


def load_config(
    config_path: Path, *, repo_root: Path
) -> tuple[ExperimentConfig, dict[str, DecoderVariant], bool]:
    payload = _load_yaml(config_path)

    runtime = _require_section(payload, "runtime")
    output = _require_section(payload, "output")
    experiment = _require_section(payload, "experiment")

    decoder_params = payload.get("decoder_params", {})
    if decoder_params is None:
        decoder_params = {}
    if not isinstance(decoder_params, dict):
        raise ValueError("decoder_params must be a mapping")

    variants = _parse_decoder_variants(payload)

    output_root_raw = str(output.get("output_root", "results/memory_runs"))
    output_root = Path(output_root_raw)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    noise_mode, noise_model = _parse_noise_mode_and_model(experiment)
    p_values = _parse_p_values(experiment.get("p_values", [0.002]))
    shots_by_p = _parse_shots(experiment.get("shots", 10000), p_values=p_values)

    num_circuit_samples = int(experiment.get("num_circuit_samples", 1))
    noise_sigma_rel = float(
        experiment.get(
            "noise_sigma_rel",
            experiment.get("sigma_rel", experiment.get("sigma", 0.0)),
        )
    )
    circuit_seed_base = int(
        experiment.get("circuit_seed_base", experiment.get("detector_seed_base", 1000))
    )
    circuit_seeds = _parse_optional_int_list(experiment.get("circuit_seeds"))

    if variants:
        aliases = _parse_decoder_aliases(experiment.get("decoders"), fallback=variants.keys())
        unknown = sorted(set(aliases) - set(variants.keys()))
        if unknown:
            raise ValueError(
                "Unknown decoder alias(es) in experiment.decoders: " + ", ".join(unknown)
            )
    else:
        aliases = _parse_decoder_aliases(experiment.get("decoders"), fallback=decoder_params.keys())
        variants = _build_default_variants(aliases, decoder_params)

    # Circuit type and per-type parameters
    circuit_type = str(experiment.get("circuit", BB_CIRCUIT_TYPE)).strip()

    distances = _parse_distances(experiment.get("distance"))

    num_rounds_spec = _parse_num_rounds_spec(experiment.get("num_rounds", 1))

    ibm_reproduce = bool(experiment.get("ibm_reproduce", False))
    use_both = bool(experiment.get("use_both", False))
    xyz_decoding = bool(experiment.get("xyz_decoding", True))

    # BB-specific fields (optional — only required for BB codes)
    l_raw = experiment.get("l")
    m_raw = experiment.get("m")
    l = int(l_raw) if l_raw is not None else None
    m = int(m_raw) if m_raw is not None else None
    a_x_pows = _parse_pow_tuple_optional(experiment.get("a_x_pows"))
    a_y_pows = _parse_pow_tuple_optional(experiment.get("a_y_pows"))
    b_x_pows = _parse_pow_tuple_optional(experiment.get("b_x_pows"))
    b_y_pows = _parse_pow_tuple_optional(experiment.get("b_y_pows"))

    # HGP-specific fields (optional — only required for HGP codes)
    hgp_pcm_path_raw = experiment.get("pcm_path")
    hgp_pcm_path: str | None = None
    if hgp_pcm_path_raw is not None:
        raw_path = Path(str(hgp_pcm_path_raw))
        hgp_pcm_path = str(raw_path if raw_path.is_absolute() else repo_root / raw_path)
    hgp_build_graph_seed_raw = experiment.get("build_graph_seed")
    hgp_build_graph_seed: int | None = (
        int(hgp_build_graph_seed_raw) if hgp_build_graph_seed_raw is not None else None
    )
    # Distance computation for HGP codes (done early so it can be stored in config)
    hgp_computed_distance: int | None = None
    if circuit_type == HGP_CIRCUIT_TYPE:
        if hgp_pcm_path is None:
            raise ValueError("experiment.pcm_path is required for hgp_code_Z")
        from qec_decoder_sim.circuits.hgpcode import compute_hgp_distance, load_pcm
        _h = load_pcm(hgp_pcm_path)
        hgp_computed_distance = compute_hgp_distance(_h)
        # Validate against reference distance if provided
        _ref_distances = _parse_distances(experiment.get("distance"))
        if _ref_distances is not None:
            if len(_ref_distances) != 1:
                raise ValueError(
                    "experiment.distance for hgp_code_Z must be a single reference value"
                )
            ref_d = _ref_distances[0]
            if ref_d != hgp_computed_distance:
                raise ValueError(
                    f"experiment.distance={ref_d} does not match computed HGP code "
                    f"distance {hgp_computed_distance} for PCM {hgp_pcm_path!r}."
                )

    cfg = ExperimentConfig(
        shots_by_p=shots_by_p,
        p_values=p_values,
        circuit_type=circuit_type,
        basis=str(experiment.get("basis", "z")).lower(),
        distances=distances,
        num_rounds_spec=num_rounds_spec,
        l=l,
        m=m,
        a_x_pows=a_x_pows,
        a_y_pows=a_y_pows,
        b_x_pows=b_x_pows,
        b_y_pows=b_y_pows,
        ibm_reproduce=ibm_reproduce,
        use_both=use_both,
        hgp_pcm_path=hgp_pcm_path,
        hgp_build_graph_seed=hgp_build_graph_seed,
        hgp_computed_distance=hgp_computed_distance,
        noise_mode=str(noise_mode).lower(),
        noise_model=str(noise_model).lower(),
        num_circuit_samples=num_circuit_samples,
        noise_sigma_rel=noise_sigma_rel,
        circuit_seed_base=circuit_seed_base,
        circuit_seeds=circuit_seeds,
        xyz_decoding=xyz_decoding,
        processes=max(1, int(experiment.get("processes", 4))),
        detector_seed_base=int(experiment.get("detector_seed_base", 1000)),
        print_progress=bool(experiment.get("print_progress", True)),
        output_root=output_root,
        decoders=aliases,
    )

    _validate_experiment_config(cfg)

    write_plots = bool(runtime.get("write_plots", False))
    return cfg, variants, write_plots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import error path
        raise ImportError("pyyaml is required to load YAML experiment configs.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML top-level must be a mapping")
    return payload


def _require_section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    section = payload.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid section: {key}")
    return section


def _parse_p_values(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, list):
        vals = [float(v) for v in raw]
    elif isinstance(raw, str):
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        raise ValueError("experiment.p_values must be a list[float] or csv string")

    if not vals:
        raise ValueError("experiment.p_values must be non-empty")
    return tuple(vals)


def _parse_distances(raw: Any) -> tuple[int, ...] | None:
    """Parse ``distance`` YAML value into a tuple of ints, or None if absent."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, list):
        if not raw:
            raise ValueError("experiment.distance list must be non-empty")
        return tuple(int(v) for v in raw)
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        if not parts:
            raise ValueError("experiment.distance csv must be non-empty")
        return tuple(int(v) for v in parts)
    raise ValueError("experiment.distance must be an int, list[int], or csv string")


def _parse_num_rounds_spec(raw: Any) -> str:
    """Normalise the ``num_rounds`` YAML value to a string spec.

    Accepts:
    - positive integer  → ``"12"``
    - string ``"d"``    → ``"d"``
    - string ``"k*d"``  → ``"k*d"``
    """
    if isinstance(raw, int):
        return str(raw)
    s = str(raw).strip()
    if not s:
        raise ValueError("experiment.num_rounds must not be empty")
    return s


def _parse_pow_tuple(raw: Any) -> tuple[int, ...]:
    if raw is None:
        raise ValueError("bb code power lists must be provided")
    return _parse_pow_tuple_optional(raw)  # type: ignore[return-value]


def _parse_pow_tuple_optional(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw]
    elif isinstance(raw, str):
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        raise ValueError("bb code powers must be list[int], tuple[int], or csv string")

    if not values:
        raise ValueError("bb code powers must be non-empty")
    return tuple(values)


def _parse_shots(raw: Any, *, p_values: tuple[float, ...]) -> tuple[int, ...]:
    if isinstance(raw, list):
        shots = [int(v) for v in raw]
        if len(shots) != len(p_values):
            raise ValueError("experiment.shots list length must match experiment.p_values length")
    elif isinstance(raw, str):
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        if len(parts) > 1:
            shots = [int(x) for x in parts]
            if len(shots) != len(p_values):
                raise ValueError("experiment.shots csv length must match experiment.p_values length")
        else:
            if len(parts) != 1:
                raise ValueError("experiment.shots must be non-empty")
            shots = [int(parts[0])] * len(p_values)
    else:
        shots = [int(raw)] * len(p_values)

    if not shots:
        raise ValueError("experiment.shots must be non-empty")
    if any(v < 1 for v in shots):
        raise ValueError("experiment.shots values must be >= 1")
    return tuple(shots)


def _parse_optional_int_list(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return tuple(int(v) for v in raw)
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        return tuple(int(v) for v in parts)
    raise ValueError("circuit_seeds must be list[int] or csv string")


def _parse_decoder_aliases(raw: Any, *, fallback: Any) -> tuple[str, ...]:
    if raw is None:
        aliases = list(fallback) if fallback is not None else []
    elif isinstance(raw, str):
        aliases = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        aliases = [str(x).strip() for x in raw if str(x).strip()]
    else:
        raise ValueError("experiment.decoders must be a list[str] or csv string")

    if not aliases:
        raise ValueError("experiment.decoders is empty")
    return tuple(aliases)


def _parse_decoder_variants(payload: dict[str, Any]) -> dict[str, DecoderVariant]:
    raw = payload.get("decoder_variants")
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("decoder_variants must be a non-empty mapping")

    variants: dict[str, DecoderVariant] = {}
    for alias, cfg in raw.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("decoder_variants keys must be non-empty strings")
        if not isinstance(cfg, dict):
            raise ValueError(f"decoder_variants.{alias} must be a mapping")

        decoder_type = str(cfg.get("type", "")).strip()
        if decoder_type not in SUPPORTED_DECODERS:
            raise ValueError(
                f"decoder_variants.{alias}.type must be one of {sorted(SUPPORTED_DECODERS)}"
            )

        params = cfg.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"decoder_variants.{alias}.params must be a mapping")

        variants[alias] = DecoderVariant(alias=alias, decoder_type=decoder_type, params=dict(params))

    return variants


def _build_default_variants(
    aliases: tuple[str, ...],
    decoder_params: dict[str, Any],
) -> dict[str, DecoderVariant]:
    variants: dict[str, DecoderVariant] = {}
    for alias in aliases:
        if alias not in SUPPORTED_DECODERS:
            raise ValueError(f"Unsupported decoder: {alias!r}")
        params = decoder_params.get(alias, {})
        if not isinstance(params, dict):
            raise ValueError(f"decoder_params.{alias} must be a mapping")
        variants[alias] = DecoderVariant(alias=alias, decoder_type=alias, params=dict(params))
    return variants


def _parse_noise_mode_and_model(experiment: dict[str, Any]) -> tuple[str, str]:
    mode = str(experiment.get("noise_mode", "theoretical")).strip().lower()
    model = str(experiment.get("noise_model", "si1000")).strip().lower()
    return mode, model


def _validate_experiment_config(cfg: ExperimentConfig) -> None:
    known_types = TOPOLOGICAL_CIRCUIT_TYPES | {BB_CIRCUIT_TYPE, HGP_CIRCUIT_TYPE}
    if cfg.circuit_type not in known_types:
        raise ValueError(
            f"experiment.circuit must be one of {sorted(known_types)}, "
            f"got {cfg.circuit_type!r}"
        )

    if cfg.basis not in {"x", "z"}:
        raise ValueError("experiment.basis must be 'x' or 'z'")

    if cfg.circuit_type == HGP_CIRCUIT_TYPE:
        if not cfg.hgp_pcm_path:
            raise ValueError("experiment.pcm_path is required for hgp_code_Z")
        noise_key = cfg.noise_model.strip().lower()
        if noise_key not in {"uniform", "uniform-cln"}:
            raise ValueError(
                "hgp_code_Z only supports noise_model='uniform'; "
                f"got {cfg.noise_model!r}"
            )
        if cfg.noise_mode != "theoretical":
            raise ValueError(
                "hgp_code_Z only supports noise_mode='theoretical'"
            )
        _validate_num_rounds_spec(cfg.num_rounds_spec)
        return

    if cfg.circuit_type == BB_CIRCUIT_TYPE:
        # BB-specific validation
        missing = [
            name
            for name, val in [
                ("l", cfg.l), ("m", cfg.m),
                ("a_x_pows", cfg.a_x_pows), ("a_y_pows", cfg.a_y_pows),
                ("b_x_pows", cfg.b_x_pows), ("b_y_pows", cfg.b_y_pows),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                "experiment.circuit=bivariate_bicycle_code_Z requires: "
                + ", ".join(f"experiment.{n}" for n in missing)
            )
        if cfg.l < 1 or cfg.m < 1:  # type: ignore[operator]
            raise ValueError("experiment.l and experiment.m must be >= 1")
        # For BB codes num_rounds_spec must be a plain integer (distance is not available)
        try:
            nr = int(cfg.num_rounds_spec)
            if nr < 1:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                "experiment.num_rounds for bivariate_bicycle_code_Z must be a positive "
                f"integer; got {cfg.num_rounds_spec!r}"
            )
    else:
        # Topological code validation
        if not cfg.distances:
            raise ValueError(
                f"experiment.circuit={cfg.circuit_type!r} requires experiment.distance "
                "(int or list of ints)"
            )
        for d in cfg.distances:
            if d < 1:
                raise ValueError(
                    f"experiment.distance values must be >= 1; got {d}"
                )
        # Validate num_rounds_spec is parseable (distance-free check; full check at build time)
        _validate_num_rounds_spec(cfg.num_rounds_spec)
        # Warn if BB-specific options are set for topological codes
        if cfg.ibm_reproduce:
            warnings.warn(
                "experiment.ibm_reproduce is ignored for topological circuit types.",
                stacklevel=3,
            )
        if cfg.use_both:
            warnings.warn(
                "experiment.use_both is ignored for topological circuit types.",
                stacklevel=3,
            )

    if cfg.noise_mode not in {"theoretical", "realistic"}:
        raise ValueError("experiment.noise_mode must be 'theoretical' or 'realistic'")
    if cfg.noise_mode == "realistic":
        if cfg.circuit_type != BB_CIRCUIT_TYPE:
            raise ValueError(
                "noise_mode='realistic' is only supported for bivariate_bicycle_code_Z"
            )
        if cfg.num_circuit_samples < 1:
            raise ValueError("num_circuit_samples must be >= 1")
        if cfg.noise_sigma_rel < 0:
            raise ValueError("noise_sigma_rel must be >= 0")
        if cfg.circuit_seed_base < 0:
            raise ValueError("circuit_seed_base must be >= 0")
        if cfg.circuit_seeds is not None and len(cfg.circuit_seeds) != cfg.num_circuit_samples:
            raise ValueError("circuit_seeds length must match num_circuit_samples")


def _validate_num_rounds_spec(spec: str) -> None:
    """Raise ValueError if spec is not a valid num_rounds expression."""
    s = spec.strip().lower().replace(" ", "")
    if not s:
        raise ValueError("experiment.num_rounds must not be empty")
    if s.isdigit():
        if int(s) < 1:
            raise ValueError(f"experiment.num_rounds must be >= 1; got {spec!r}")
        return
    if s == "d":
        return
    if s.endswith("*d") and s[:-2].isdigit() and int(s[:-2]) >= 1:
        return
    if s.startswith("d*") and s[2:].isdigit() and int(s[2:]) >= 1:
        return
    raise ValueError(
        f"experiment.num_rounds must be a positive integer, 'd', or 'k*d'; got {spec!r}"
    )
