"""End-to-end tests for extended memory experiment support.

Covers all requirements added in the surface/color code extension:
  1. midout_color_code_Z circuit generation
  2. filter_circuit_by_basis()
  3. Config loading for topological codes and BB codes with new fields
  4. Task building: MWPM (decompose_errors, built-in pymatching), xyz_decoding, chromobius
  5. Backward compatibility with existing BB YAML configs
"""
from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path

import pytest
import stim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_surface_code_circuit(d: int = 3, rounds: int = 2) -> stim.Circuit:
    from qec_decoder_sim.circuits._topological_memory_circuits import make_topological_memory_circuit
    return make_topological_memory_circuit(
        style="surface_code_Z",
        rounds=rounds,
        diameter=d,
        noise_model="uniform",
        noise_strength=0.001,
    )


def _make_midout_circuit(d: int = 3, rounds: int = 2) -> stim.Circuit:
    from qec_decoder_sim.circuits._topological_memory_circuits import make_topological_memory_circuit
    return make_topological_memory_circuit(
        style="midout_color_code_Z",
        rounds=rounds,
        diameter=d,
        noise_model="uniform",
        noise_strength=0.001,
    )


def _load_cfg_from_yaml(yaml_str: str, repo_root: Path):
    import yaml
    from qec_decoder_sim.runner.config import load_config

    tmp = repo_root / "_test_config.yaml"
    tmp.write_text(textwrap.dedent(yaml_str), encoding="utf-8")
    try:
        return load_config(tmp, repo_root=repo_root)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. midout_color_code_Z circuit generation
# ---------------------------------------------------------------------------

class TestMidoutCircuitGeneration:
    def test_returns_valid_circuit_with_detectors(self):
        circuit = _make_midout_circuit(d=3, rounds=2)
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0

    def test_circuit_has_observables(self):
        circuit = _make_midout_circuit(d=3, rounds=2)
        assert circuit.num_observables > 0

    def test_bitflip_noise_raises(self):
        from qec_decoder_sim.circuits._topological_memory_circuits import make_topological_memory_circuit
        with pytest.raises(ValueError, match="bitflip"):
            make_topological_memory_circuit(
                style="midout_color_code_Z",
                rounds=2,
                diameter=3,
                noise_model="bitflip",
                noise_strength=0.01,
            )

    def test_midout_in_supported_codes(self):
        from qec_decoder_sim.circuits.generate_circuits import _SUPPORTED_CODES
        assert "midout_color_code_Z" in _SUPPORTED_CODES


class TestGenerateCircuitsCLIMidout:
    def test_main_writes_stim_file(self, tmp_path):
        from qec_decoder_sim.circuits.generate_circuits import main as gen_main
        sys.argv = [
            "generate_circuits",
            "--code", "midout_color_code_Z",
            "--distance", "3",
            "--rounds", "2",
            "--noise_model", "uniform",
            "--noise_strength", "0.001",
            "--out_dir", str(tmp_path),
        ]
        gen_main()
        stim_files = list(tmp_path.glob("*.stim"))
        assert len(stim_files) == 1
        circuit = stim.Circuit(stim_files[0].read_text())
        assert circuit.num_detectors > 0


# ---------------------------------------------------------------------------
# 2. filter_circuit_by_basis
# ---------------------------------------------------------------------------

class TestFilterCircuitByBasis:
    def test_removes_x_detectors_from_surface_code(self):
        from qec_decoder_sim.utils.dem_utils import CHROMOBIUS_COORD_BASIS_MAP, filter_circuit_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        filtered = filter_circuit_by_basis(circuit, remove_basis="X")

        coords = filtered.get_detector_coordinates()
        for det_coords in coords.values():
            if len(det_coords) >= 4:
                basis = CHROMOBIUS_COORD_BASIS_MAP.get(int(det_coords[3]))
                assert basis != "X", f"Found X-basis detector after filtering: coords={det_coords}"

    def test_removes_z_detectors_from_surface_code(self):
        from qec_decoder_sim.utils.dem_utils import CHROMOBIUS_COORD_BASIS_MAP, filter_circuit_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        filtered = filter_circuit_by_basis(circuit, remove_basis="Z")

        coords = filtered.get_detector_coordinates()
        for det_coords in coords.values():
            if len(det_coords) >= 4:
                basis = CHROMOBIUS_COORD_BASIS_MAP.get(int(det_coords[3]))
                assert basis != "Z"

    def test_filtered_detector_count_is_less_than_original(self):
        from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        filtered = filter_circuit_by_basis(circuit, remove_basis="X")
        assert filtered.num_detectors < circuit.num_detectors

    def test_filter_x_plus_z_removes_all_basis_detectors(self):
        from qec_decoder_sim.utils.dem_utils import CHROMOBIUS_COORD_BASIS_MAP, filter_circuit_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        filtered_x = filter_circuit_by_basis(circuit, remove_basis="X")
        filtered_both = filter_circuit_by_basis(filtered_x, remove_basis="Z")
        coords = filtered_both.get_detector_coordinates()
        for det_coords in coords.values():
            if len(det_coords) >= 4:
                basis = CHROMOBIUS_COORD_BASIS_MAP.get(int(det_coords[3]))
                assert basis is None, "All basis-labelled detectors should be gone"

    def test_roundtrip_consistency_with_filter_dem_by_basis(self):
        """filter_circuit_by_basis + DEM ≡ filter_dem_by_basis applied to the full DEM.

        Both approaches should produce a DEM with the same number of detectors.
        """
        from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis, filter_dem_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        full_dem = circuit.detector_error_model(
            decompose_errors=False, ignore_decomposition_failures=True
        )

        # Path A: filter DEM directly
        filtered_dem = filter_dem_by_basis(full_dem, remove_basis="X")

        # Path B: filter circuit, then build DEM
        filtered_circuit = filter_circuit_by_basis(circuit, remove_basis="X")
        circuit_dem = filtered_circuit.detector_error_model(
            decompose_errors=False, ignore_decomposition_failures=True
        )

        assert filtered_dem.num_detectors == circuit_dem.num_detectors

    def test_invalid_basis_raises(self):
        from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        with pytest.raises(AssertionError):
            filter_circuit_by_basis(circuit, remove_basis="Y")

    def test_midout_color_code_filtering(self):
        from qec_decoder_sim.utils.dem_utils import CHROMOBIUS_COORD_BASIS_MAP, filter_circuit_by_basis

        circuit = _make_midout_circuit(d=3, rounds=2)
        filtered = filter_circuit_by_basis(circuit, remove_basis="X")
        coords = filtered.get_detector_coordinates()
        for det_coords in coords.values():
            if len(det_coords) >= 4:
                basis = CHROMOBIUS_COORD_BASIS_MAP.get(int(det_coords[3]))
                assert basis != "X"


# ---------------------------------------------------------------------------
# 3. Config loading
# ---------------------------------------------------------------------------

_BB_YAML = """\
runtime:
  write_plots: false
output:
  output_root: results/test
experiment:
  shots: 100
  p_values: [0.001]
  basis: z
  num_rounds: 3
  l: 6
  m: 6
  a_x_pows: [3]
  a_y_pows: [1, 2]
  b_x_pows: [1, 2]
  b_y_pows: [3]
  noise_mode: theoretical
  noise_model: uniform
  processes: 1
  decoders: [bplsd]
decoder_params:
  bplsd:
    max_iter: 5
"""

_SURFACE_YAML = """\
runtime:
  write_plots: false
output:
  output_root: results/test
experiment:
  circuit: surface_code_Z
  distance: 5
  shots: 100
  p_values: [0.001]
  basis: z
  num_rounds: 5
  noise_mode: theoretical
  noise_model: uniform
  processes: 1
  decoders: [mwpm]
"""

_MIDOUT_YAML = """\
runtime:
  write_plots: false
output:
  output_root: results/test
experiment:
  circuit: midout_color_code_Z
  distance: 3
  shots: 100
  p_values: [0.001]
  basis: z
  num_rounds: 3
  noise_mode: theoretical
  noise_model: uniform
  xyz_decoding: false
  processes: 1
  decoders: [bplsd]
decoder_params:
  bplsd:
    max_iter: 5
"""


class TestConfigLoading:
    def test_bb_yaml_no_circuit_field_defaults_to_bb(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE
        assert cfg.circuit_type == BB_CIRCUIT_TYPE

    def test_bb_yaml_fields_parsed(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        assert cfg.l == 6
        assert cfg.m == 6
        assert cfg.a_x_pows == (3,)
        assert cfg.a_y_pows == (1, 2)
        assert cfg.b_x_pows == (1, 2)
        assert cfg.b_y_pows == (3,)
        assert cfg.xyz_decoding is True  # default

    def test_surface_code_yaml_parsed(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        assert cfg.circuit_type == "surface_code_Z"
        assert cfg.distance == 5
        assert cfg.l is None
        assert cfg.m is None

    def test_midout_yaml_xyz_decoding_false(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_MIDOUT_YAML, tmp_path)
        assert cfg.circuit_type == "midout_color_code_Z"
        assert cfg.distance == 3
        assert cfg.xyz_decoding is False

    def test_missing_distance_for_topological_raises(self, tmp_path):
        yaml_str = _SURFACE_YAML.replace("  distance: 5\n", "")
        with pytest.raises(ValueError, match="distance"):
            _load_cfg_from_yaml(yaml_str, tmp_path)

    def test_missing_l_for_bb_code_raises(self, tmp_path):
        yaml_str = _BB_YAML.replace("  l: 6\n", "")
        with pytest.raises(ValueError, match="l"):
            _load_cfg_from_yaml(yaml_str, tmp_path)

    def test_ibm_reproduce_and_use_both_parsed(self, tmp_path):
        yaml_str = _BB_YAML.replace(
            "  decoders: [bplsd]",
            "  ibm_reproduce: false\n  use_both: true\n  decoders: [bplsd]",
        )
        cfg, _, _ = _load_cfg_from_yaml(yaml_str, tmp_path)
        assert cfg.ibm_reproduce is False
        assert cfg.use_both is True

    def test_unknown_circuit_type_raises(self, tmp_path):
        yaml_str = _SURFACE_YAML.replace(
            "  circuit: surface_code_Z", "  circuit: magic_code_Z"
        )
        with pytest.raises(ValueError, match="circuit"):
            _load_cfg_from_yaml(yaml_str, tmp_path)

    def test_mwpm_accepted_as_decoder_alias(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        assert "mwpm" in cfg.decoders
        assert variants["mwpm"].decoder_type == "mwpm"


# ---------------------------------------------------------------------------
# 4. Decoder resolution
# ---------------------------------------------------------------------------

class TestDecoderResolution:
    def test_mwpm_no_correlated_uses_pymatching_builtin(self):
        from qec_decoder_sim.decoder.sinter_registry import ResolvedDecoder, resolve_decoder
        from qec_decoder_sim.runner.config import DecoderVariant

        variant = DecoderVariant(alias="mwpm", decoder_type="mwpm", params={})
        res = resolve_decoder(variant)
        assert res.sinter_name == "pymatching"
        assert res.decoder is None
        assert res.needs_decompose_errors is True

    def test_mwpm_needs_decompose_errors(self):
        from qec_decoder_sim.decoder.sinter_registry import resolve_decoder
        from qec_decoder_sim.runner.config import DecoderVariant

        variant = DecoderVariant(alias="my_mwpm", decoder_type="mwpm", params={})
        res = resolve_decoder(variant)
        assert res.needs_decompose_errors is True

    def test_bplsd_no_decompose_errors(self):
        from qec_decoder_sim.decoder.sinter_registry import resolve_decoder
        from qec_decoder_sim.runner.config import DecoderVariant

        variant = DecoderVariant(alias="bplsd", decoder_type="bplsd", params={"max_iter": 5})
        res = resolve_decoder(variant)
        assert res.needs_decompose_errors is False
        assert res.decoder is not None

    def test_chromobius_import_error_has_helpful_message(self):
        from qec_decoder_sim.decoder.sinter_registry import resolve_decoder
        from qec_decoder_sim.runner.config import DecoderVariant

        # Skip if chromobius is actually installed
        try:
            import chromobius  # type: ignore  # noqa: F401
            pytest.skip("chromobius is installed — import-error path not reachable")
        except ImportError:
            pass

        variant = DecoderVariant(alias="chromobius", decoder_type="chromobius", params={})
        with pytest.raises(ImportError, match="chromobius"):
            resolve_decoder(variant)


# ---------------------------------------------------------------------------
# 5. Task building — surface code + MWPM
# ---------------------------------------------------------------------------

class TestTaskBuildingSurfaceMwpm:
    def test_tasks_built_with_decompose_errors(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)

        assert len(tasks) == 1
        task = tasks[0]
        # MWPM task must use sinter's built-in "pymatching"
        assert task.decoder == "pymatching"
        # DEM must have been generated with decompose_errors=True — each component of every
        # error instruction (split by ^ separators) must have at most 2 detector targets.
        dem = task.detector_error_model
        assert dem is not None
        for inst in dem.flattened():
            if inst.type == "error":
                components: list[list] = [[]]
                for t in inst.targets_copy():
                    if t.is_separator():
                        components.append([])
                    elif t.is_relative_detector_id():
                        components[-1].append(t)
                for comp in components:
                    assert len(comp) <= 2, (
                        f"DEM component has {len(comp)} detectors after decomposition"
                    )

    def test_no_custom_decoder_registered_for_builtin_mwpm(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        _, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        # sinter's built-in "pymatching" must NOT appear in custom_decoders
        assert "pymatching" not in custom_decoders
        assert "mwpm" not in custom_decoders


# ---------------------------------------------------------------------------
# 6. Task building — xyz_decoding
# ---------------------------------------------------------------------------

class TestTaskBuildingXyzDecoding:
    def test_xyz_decoding_false_removes_x_detectors_from_task(self, tmp_path):
        """When xyz_decoding=False and decoder is BP, task circuit has no X detectors."""
        from qec_decoder_sim.utils.dem_utils import CHROMOBIUS_COORD_BASIS_MAP

        cfg, variants, _ = _load_cfg_from_yaml(_MIDOUT_YAML, tmp_path)
        assert cfg.xyz_decoding is False

        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders
        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)

        assert len(tasks) == 1
        task = tasks[0]
        coords = task.circuit.get_detector_coordinates()
        for det_coords in coords.values():
            if len(det_coords) >= 4:
                basis = CHROMOBIUS_COORD_BASIS_MAP.get(int(det_coords[3]))
                assert basis != "X", "X-basis detector found in xyz_decoding=False task circuit"

    def test_xyz_decoding_true_keeps_all_detectors(self, tmp_path):
        yaml_str = _MIDOUT_YAML.replace("  xyz_decoding: false\n", "  xyz_decoding: true\n")
        cfg, variants, _ = _load_cfg_from_yaml(yaml_str, tmp_path)

        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders
        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)

        # Circuit in task should equal the full unfiltered circuit
        from qec_decoder_sim.runner.circuit_builder import build_circuit
        base_circuit = build_circuit(cfg=cfg, p=cfg.p_values[0], circuit_seed=None)
        assert tasks[0].circuit.num_detectors == base_circuit.num_detectors

    def test_mwpm_not_affected_by_xyz_decoding_false(self, tmp_path):
        """MWPM is not a BP decoder — xyz_decoding=False should not filter its circuit."""
        yaml_str = _SURFACE_YAML.replace(
            "  decoders: [mwpm]", "  decoders: [mwpm]"
        )
        yaml_str = yaml_str.rstrip() + "\n  xyz_decoding: false\n"
        cfg, variants, _ = _load_cfg_from_yaml(yaml_str, tmp_path)

        from qec_decoder_sim.runner.circuit_builder import build_circuit
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        base_circuit = build_circuit(cfg=cfg, p=cfg.p_values[0], circuit_seed=None)

        # MWPM circuit should NOT be filtered
        assert tasks[0].circuit.num_detectors == base_circuit.num_detectors


# ---------------------------------------------------------------------------
# 7. BB code backward compatibility
# ---------------------------------------------------------------------------

class TestBBBackwardCompatibility:
    def test_existing_bb_yaml_loads_without_error(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE
        assert cfg.circuit_type == BB_CIRCUIT_TYPE

    def test_bb_circuit_builds_correctly(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        from qec_decoder_sim.runner.circuit_builder import build_circuit
        circuit = build_circuit(cfg=cfg, p=0.001, circuit_seed=None)
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0

    def test_bb_ibm_reproduce_forwarded(self, tmp_path):
        yaml_str = _BB_YAML.replace(
            "  noise_model: uniform\n",
            "  noise_model: uniform\n  ibm_reproduce: true\n",
        )
        cfg, _, _ = _load_cfg_from_yaml(yaml_str, tmp_path)
        assert cfg.ibm_reproduce is True

        from qec_decoder_sim.runner.circuit_builder import build_circuit
        # Should not raise
        circuit = build_circuit(cfg=cfg, p=0.001, circuit_seed=None)
        assert circuit.num_detectors > 0

    def test_bb_use_both_forwarded(self, tmp_path):
        yaml_str = _BB_YAML.replace(
            "  decoders: [bplsd]",
            "  use_both: true\n  decoders: [bplsd]",
        )
        cfg, _, _ = _load_cfg_from_yaml(yaml_str, tmp_path)
        assert cfg.use_both is True

        from qec_decoder_sim.runner.circuit_builder import build_circuit
        circuit = build_circuit(cfg=cfg, p=0.001, circuit_seed=None)
        assert circuit.num_detectors > 0

    def test_bb_metadata_contains_expected_keys(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders
        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        meta = tasks[0].json_metadata
        assert "l" in meta
        assert "m" in meta
        assert "a_x_pows" in meta
        assert "circuit_type" in meta

    def test_topological_metadata_contains_distance(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders
        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        meta = tasks[0].json_metadata
        assert "distance" in meta
        assert "l" not in meta


# ---------------------------------------------------------------------------
# 8. Sinter integration smoke test
# ---------------------------------------------------------------------------

class TestSinterSmoke:
    """Light end-to-end sinter.collect() runs with tiny shot counts."""

    def test_surface_code_mwpm_collects(self, tmp_path):
        """sinter.collect() completes for surface_code_Z + pymatching."""
        import sinter

        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        # Replace shot count with tiny value for speed
        tasks = [
            sinter.Task(
                circuit=t.circuit,
                detector_error_model=t.detector_error_model,
                decoder=t.decoder,
                json_metadata=t.json_metadata,
                collection_options=sinter.CollectionOptions(max_shots=10),
            )
            for t in tasks
        ]
        results = sinter.collect(
            tasks=tasks,
            custom_decoders=custom_decoders,
            num_workers=1,
            print_progress=False,
        )
        assert len(results) == len(tasks)

    def test_midout_bplsd_xyz_decoding_false_collects(self, tmp_path):
        """sinter.collect() completes for midout_color_code_Z + bplsd + xyz_decoding=False."""
        import sinter

        cfg, variants, _ = _load_cfg_from_yaml(_MIDOUT_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        tasks = [
            sinter.Task(
                circuit=t.circuit,
                detector_error_model=t.detector_error_model,
                decoder=t.decoder,
                json_metadata=t.json_metadata,
                collection_options=sinter.CollectionOptions(max_shots=10),
            )
            for t in tasks
        ]
        results = sinter.collect(
            tasks=tasks,
            custom_decoders=custom_decoders,
            num_workers=1,
            print_progress=False,
        )
        assert len(results) == len(tasks)
