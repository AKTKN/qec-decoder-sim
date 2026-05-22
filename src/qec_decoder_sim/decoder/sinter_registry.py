from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sinter

SUPPORTED_DECODERS = {
    "bposd",
    "bplsd",
    "relay-bp",
    "chromobius",
    "mwpm",
}

# Decoder types whose DEM must be built with decompose_errors=True.
_DECOMPOSE_ERRORS_DECODERS: frozenset[str] = frozenset({"mwpm"})

# Decoder types that are sinter built-ins (no custom decoder object needed).
# Maps our decoder type name → the sinter built-in decoder name.
_BUILTIN_DECODER_MAP: dict[str, str] = {
    "mwpm": "pymatching",
}

# Decoder types that apply BP-style decoding (relevant for xyz_decoding filtering).
BP_DECODER_TYPES: frozenset[str] = frozenset({"bposd", "bplsd", "relay-bp"})


@dataclass(frozen=True)
class ResolvedDecoder:
    """Result of resolving a DecoderVariant into a concrete sinter decoder configuration."""

    sinter_name: str
    """The decoder name to use in sinter.Task(decoder=...).  May be a built-in name."""

    decoder: "sinter.Decoder | None"
    """Custom sinter.Decoder object, or None if the decoder is a sinter built-in."""

    needs_decompose_errors: bool
    """Whether the DEM for this decoder must be built with decompose_errors=True."""


def resolve_decoder(variant: "DecoderVariant") -> ResolvedDecoder:  # type: ignore[name-defined]  # noqa: F821
    """Resolve a DecoderVariant into a concrete sinter decoder configuration.

    Returns a ResolvedDecoder describing:
    - which name to pass to sinter.Task(decoder=...)
    - the custom sinter.Decoder object (or None for built-ins)
    - whether the DEM must be built with decompose_errors=True
    """
    dtype = variant.decoder_type

    if dtype == "mwpm":
        correlated = bool(variant.params.get("correlated_decoding", False))
        if correlated:
            decoder = _build_beliefmatching_decoder(variant.params)
            return ResolvedDecoder(
                sinter_name=variant.alias,
                decoder=decoder,
                needs_decompose_errors=True,
            )
        else:
            return ResolvedDecoder(
                sinter_name=_BUILTIN_DECODER_MAP["mwpm"],
                decoder=None,
                needs_decompose_errors=True,
            )

    if dtype == "chromobius":
        decoder = _import_chromobius_decoder()
        return ResolvedDecoder(
            sinter_name=variant.alias,
            decoder=decoder,
            needs_decompose_errors=False,
        )

    # BP decoder types: bposd, bplsd, relay-bp
    bp_decoders = build_custom_sinter_decoders(
        selected_decoders=(dtype,),
        decoder_params={dtype: variant.params},
    )
    return ResolvedDecoder(
        sinter_name=variant.alias,
        decoder=bp_decoders[dtype],
        needs_decompose_errors=False,
    )


def _import_bposd_decoder_cls() -> type:
    try:
        from ldpc.sinter_decoders import SinterBpOsdDecoder  # type: ignore

        return SinterBpOsdDecoder
    except Exception as exc:
        raise ImportError(
            "Decoder 'bposd' requires ldpc with sinter decoders installed. "
            "Install with `pip install ldpc` (or an environment providing ldpc.sinter_decoders)."
        ) from exc


def _import_bplsd_decoder_cls() -> type:
    try:
        from ldpc.sinter_decoders import SinterLsdDecoder  # type: ignore

        return SinterLsdDecoder
    except Exception as exc:
        raise ImportError(
            "Decoder 'bplsd' requires ldpc with sinter decoders installed. "
            "Install with `pip install ldpc` (or an environment providing ldpc.sinter_decoders)."
        ) from exc


def _import_relay_bp_decoder_cls() -> type:
    try:
        from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP  # type: ignore[import-not-found]

        return SinterDecoder_RelayBP
    except Exception as exc:
        raise ImportError(
            "Decoder 'relay-bp' requires relay-bp with sinter decoders available. "
            "Install relay-bp or ensure it is on PYTHONPATH."
        ) from exc


def _import_chromobius_decoder() -> "sinter.Decoder":
    try:
        import chromobius  # type: ignore

        decoders = chromobius.sinter_decoders()
        return decoders["chromobius"]
    except ImportError as exc:
        raise ImportError(
            "Decoder 'chromobius' requires the chromobius package to be installed. "
            "Build and install chromobius from DonotSync/chromobius, or install via pip."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise chromobius decoder: {exc}"
        ) from exc


def _build_beliefmatching_decoder(params: dict[str, Any]) -> "sinter.Decoder":
    filtered = {k: v for k, v in params.items() if k != "correlated_decoding"}
    try:
        from beliefmatching import BeliefMatchingDecoder  # type: ignore

        return BeliefMatchingDecoder(**filtered)
    except ImportError as exc:
        raise ImportError(
            "Decoder 'mwpm' with correlated_decoding=True requires the beliefmatching "
            "package. Install with `pip install beliefmatching`."
        ) from exc


def build_custom_sinter_decoders(
    *,
    selected_decoders: tuple[str, ...],
    decoder_params: dict[str, Any],
) -> dict[str, "sinter.Decoder"]:
    import sinter

    decoders: dict[str, sinter.Decoder] = {}

    if "relay-bp" in selected_decoders:
        RelayBP = _import_relay_bp_decoder_cls()
        decoders["relay-bp"] = RelayBP(**dict(decoder_params.get("relay-bp", {})))

    if "bposd" in selected_decoders:
        BpOsd = _import_bposd_decoder_cls()
        decoders["bposd"] = BpOsd(**dict(decoder_params.get("bposd", {})))

    if "bplsd" in selected_decoders:
        BpLsd = _import_bplsd_decoder_cls()
        decoders["bplsd"] = BpLsd(**dict(decoder_params.get("bplsd", {})))

    missing = sorted(set(selected_decoders) - set(decoders.keys()))
    if missing:
        raise ValueError(f"Failed to initialize decoder(s): {', '.join(missing)}")

    return decoders
