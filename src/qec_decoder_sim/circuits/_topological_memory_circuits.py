from __future__ import annotations

import pathlib
import sys
import warnings
from typing import Callable, Literal

import stim


_CHROMOBIUS_ROOT = pathlib.Path(__file__).resolve().parent / "from_chromobius"


def _ensure_chromobius_on_path() -> None:
    if not _CHROMOBIUS_ROOT.exists():
        raise FileNotFoundError(f"Chromobius sources not found at {_CHROMOBIUS_ROOT}")
    path_str = str(_CHROMOBIUS_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_chromobius_on_path()

import gen  # type: ignore
from clorco._make_circuit import make_circuit as _chromobius_make_circuit  # type: ignore

NoiseModelName = Literal["uniform", "si1000", "bitflip", "phenomenological"]
TopologicalStyle = Literal["surface_code_Z", "superdense_color_code_Z", "midout_color_code_Z"]

_STYLE_MAP: dict[str, str] = {
    "surface_code_Z": "surface_code_Z",
    "superdense_color_code_Z": "superdense_color_code_Z",
    "midout_color_code_Z": "midout_color_code_Z",
}

_NOISE_MODEL_BUILDERS: dict[str, Callable[[float], "gen.NoiseModel"]] = {
    "uniform": gen.NoiseModel.uniform_depolarizing,
    "si1000": gen.NoiseModel.si1000,
    "bitflip": gen.NoiseModel.bitflip,
}


def _make_bitflip_code_capacity_circuit(
    *,
    style_key: str,
    rounds: int,
    diameter: int,
    noise_strength: float,
) -> stim.Circuit:
    """Generate a code capacity circuit with X_ERROR noise and perfect syndrome extraction.

    Uses make_code_capacity_circuit (single syndrome extraction round with MPP measurements).
    X_ERROR is applied to data qubits; measurements are perfect (no flip noise).
    rounds must be 1; a warning is issued if a different value is supplied.
    """
    if rounds != 1:
        warnings.warn(
            f"bitflip noise model uses make_code_capacity_circuit which is a single syndrome "
            f"extraction round. rounds={rounds} is ignored; the circuit is always rounds=1.",
            stacklevel=4,
        )

    noise_rule = gen.NoiseRule(after={"X_ERROR": float(noise_strength)})

    if style_key == "surface_code_Z":
        from clorco.surface_code._surface_code_layouts import make_surface_code_layout  # type: ignore
        from clorco.surface_code._keyed_constructions import f2c  # type: ignore
        code = make_surface_code_layout(width=diameter, height=diameter).z_basis_subset()
        return code.make_code_capacity_circuit(noise=noise_rule, extra_coords_func=f2c)

    if style_key == "superdense_color_code_Z":
        if diameter <= 2 or diameter % 2 == 0:
            raise ValueError(
                f"superdense_color_code_Z with bitflip requires an odd diameter > 2, got {diameter}"
            )
        from clorco.color_code._superdense_planar_color_code_circuits import (  # type: ignore
            make_color_code_layout_for_superdense,
        )
        from clorco.color_code._keyed_constructions import f2c as color_f2c  # type: ignore
        code = make_color_code_layout_for_superdense(base_data_width=diameter).z_basis_subset()
        return code.make_code_capacity_circuit(noise=noise_rule, extra_coords_func=color_f2c)

    if style_key == "midout_color_code_Z":
        raise ValueError(
            "midout_color_code_Z does not support the 'bitflip' noise model. "
            "Use 'uniform' or 'si1000' instead."
        )

    raise ValueError(f"Unsupported style for bitflip: {style_key!r}")


def _make_phenom_topological_circuit(
    *,
    style_key: str,
    rounds: int,
    diameter: int,
    noise_strength: float,
) -> stim.Circuit:
    """Generate a phenomenological-noise circuit for surface or color codes.

    Uses make_phenom_circuit from the StabilizerCode class, which applies
    DEPOLARIZE1(p) to all data qubits and flip_result=p to all measurements
    in each syndrome extraction round.

    Both superdense_color_code_Z and midout_color_code_Z produce the same
    circuit under this noise model (the smooth color code layout).
    """
    noise = float(noise_strength)

    if style_key == "surface_code_Z":
        from clorco.surface_code._surface_code_layouts import make_surface_code_layout  # type: ignore
        from clorco.surface_code._keyed_constructions import f2c  # type: ignore
        code = make_surface_code_layout(width=diameter, height=diameter)
        return code.make_phenom_circuit(noise=noise, rounds=rounds, extra_coords_func=f2c)

    if style_key in {"superdense_color_code_Z", "midout_color_code_Z"}:
        from clorco.color_code._color_code_layouts import make_color_code_layout  # type: ignore
        from clorco.color_code._keyed_constructions import f2c as color_f2c  # type: ignore
        code = make_color_code_layout(
            base_width=diameter,
            coord_style="rect",
            single_rgb_layer_instead_of_actual_code=False,
            spurs="smooth",
        )
        return code.make_phenom_circuit(noise=noise, rounds=rounds, extra_coords_func=color_f2c)

    raise ValueError(f"Unsupported style for phenomenological noise model: {style_key!r}")


def make_topological_memory_circuit(
    *,
    style: TopologicalStyle | str,
    rounds: int,
    diameter: int,
    noise_model: NoiseModelName | str,
    noise_strength: float,
) -> stim.Circuit:
    style_key = style.strip()
    if style_key not in _STYLE_MAP:
        raise ValueError(
            "Unsupported topological style. "
            f"Expected one of {sorted(_STYLE_MAP.keys())}, got {style!r}"
        )

    noise_key = noise_model.strip().lower()
    if noise_key not in _NOISE_MODEL_BUILDERS and noise_key != "phenomenological":
        raise ValueError(
            "Topological circuits only support noise_model 'uniform', 'si1000', 'bitflip', "
            f"or 'phenomenological'. Got {noise_model!r}"
        )

    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if diameter < 1:
        raise ValueError("diameter must be >= 1")
    if noise_strength < 0:
        raise ValueError("noise_strength must be >= 0")

    if noise_key == "phenomenological":
        return _make_phenom_topological_circuit(
            style_key=style_key,
            rounds=int(rounds),
            diameter=int(diameter),
            noise_strength=float(noise_strength),
        )

    if noise_key == "bitflip":
        return _make_bitflip_code_capacity_circuit(
            style_key=style_key,
            rounds=int(rounds),
            diameter=int(diameter),
            noise_strength=float(noise_strength),
        )

    noise = _NOISE_MODEL_BUILDERS[noise_key](float(noise_strength))
    convert_to_cz = noise_key == "si1000"
    return _chromobius_make_circuit(
        style=_STYLE_MAP[style_key],
        noise_model=noise,
        noise_strength=float(noise_strength),
        rounds=int(rounds),
        diameter=int(diameter),
        debug_out_dir=None,
        convert_to_cz=convert_to_cz,
        editable_extras={},
    )
