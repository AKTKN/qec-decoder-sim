from __future__ import annotations

from typing import Any

SUPPORTED_DECODERS = {
    "bposd",
    "bplsd",
    "relay-bp",
}


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
