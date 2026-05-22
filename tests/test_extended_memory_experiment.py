"""End-to-end tests for extended memory experiment support.

Covers all requirements added in the surface/color code extension:
  1. midout_color_code_Z circuit generation
  2. filter_circuit_by_basis()
  3. Config loading for topological codes and BB codes (distances list, num_rounds_spec)
  4. Task building: MWPM, xyz_decoding, distance sweep, metadata correctness
  5. Backward compatibility with existing BB YAML configs
"""
from __future__ import annotations

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


def _build_circuit_from_cfg(cfg, p=0.001):
    """Helper to call build_circuit with the right resolved distance/num_rounds."""
    from qec_decoder_sim.runner.circuit_builder import build_circuit, resolve_num_rounds
    from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE

    if cfg.circuit_type == BB_CIRCUIT_TYPE:
        distance = None
        num_rounds = int(cfg.num_rounds_spec)
    else:
        distance = cfg.distances[0]
        num_rounds = resolve_num_rounds(cfg.num_rounds_spec, distance=distance)
    return build_circuit(cfg=cfg, p=p, circuit_seed=None, distance=distance, num_rounds=num_rounds)


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
        """filter_circuit_by_basis + DEM ≡ filter_dem_by_basis applied to the full DEM."""
        from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis, filter_dem_by_basis

        circuit = _make_surface_code_circuit(d=3, rounds=2)
        full_dem = circuit.detector_error_model(
            decompose_errors=False, ignore_decomposition_failures=True
        )

        filtered_dem = filter_dem_by_basis(full_dem, remove_basis="X")
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
  num_rounds: 5
  shots: 100
  p_values: [0.001]
  basis: z
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
  num_rounds: 3
  shots: 100
  p_values: [0.001]
  basis: z
  noise_mode: theoretical
  noise_model: uniform
  xyz_decoding: false
  processes: 1
  decoders: [bplsd]
decoder_params:
  bplsd:
    max_iter: 5
"""

_SURFACE_MULTI_D_YAML = """\
runtime:
  write_plots: false
output:
  output_root: results/test
experiment:
  circuit: surface_code_Z
  distance: [3, 5, 7]
  num_rounds: "d"
  shots: 50
  p_values: [0.001, 0.005]
  basis: z
  noise_mode: theoretical
  noise_model: uniform
  processes: 1
  decoders: [mwpm]
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
        assert cfg.num_rounds_spec == "3"
        assert cfg.xyz_decoding is True  # default

    def test_surface_code_yaml_parsed(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        assert cfg.circuit_type == "surface_code_Z"
        assert cfg.distances == (5,)
        assert cfg.num_rounds_spec == "5"
        assert cfg.l is None
        assert cfg.m is None

    def test_midout_yaml_xyz_decoding_false(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_MIDOUT_YAML, tmp_path)
        assert cfg.circuit_type == "midout_color_code_Z"
        assert cfg.distances == (3,)
        assert cfg.xyz_decoding is False

    def test_distance_list_parsed_as_tuple(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        assert cfg.distances == (3, 5, 7)

    def test_num_rounds_d_spec_stored_as_string(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        assert cfg.num_rounds_spec == "d"

    def test_num_rounds_kd_spec(self, tmp_path):
        yaml_str = _SURFACE_YAML.replace("  num_rounds: 5", "  num_rounds: '3*d'")
        cfg, _, _ = _load_cfg_from_yaml(yaml_str, tmp_path)
        assert cfg.num_rounds_spec == "3*d"

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
# 4. resolve_num_rounds
# ---------------------------------------------------------------------------

class TestResolveNumRounds:
    def test_integer_spec(self):
        from qec_decoder_sim.runner.circuit_builder import resolve_num_rounds
        assert resolve_num_rounds("12", distance=5) == 12

    def test_d_spec(self):
        from qec_decoder_sim.runner.circuit_builder import resolve_num_rounds
        assert resolve_num_rounds("d", distance=7) == 7

    def test_kd_spec(self):
        from qec_decoder_sim.runner.circuit_builder import resolve_num_rounds
        assert resolve_num_rounds("3*d", distance=5) == 15

    def test_dk_spec(self):
        from qec_decoder_sim.runner.circuit_builder import resolve_num_rounds
        assert resolve_num_rounds("d*4", distance=3) == 12

    def test_invalid_spec_raises(self):
        from qec_decoder_sim.runner.circuit_builder import resolve_num_rounds
        with pytest.raises(ValueError):
            resolve_num_rounds("foo", distance=5)


# ---------------------------------------------------------------------------
# 5. Decoder resolution
# ---------------------------------------------------------------------------

class TestDecoderResolution:
    def test_mwpm_no_correlated_uses_pymatching_builtin(self):
        from qec_decoder_sim.decoder.sinter_registry import resolve_decoder
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

        try:
            import chromobius  # type: ignore  # noqa: F401
            pytest.skip("chromobius is installed — import-error path not reachable")
        except ImportError:
            pass

        variant = DecoderVariant(alias="chromobius", decoder_type="chromobius", params={})
        with pytest.raises(ImportError, match="chromobius"):
            resolve_decoder(variant)


# ---------------------------------------------------------------------------
# 6. Task building — surface code + MWPM
# ---------------------------------------------------------------------------

class TestTaskBuildingSurfaceMwpm:
    def test_tasks_built_with_decompose_errors(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)

        assert len(tasks) == 1
        task = tasks[0]
        assert task.decoder == "pymatching"
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
                    assert len(comp) <= 2

    def test_no_custom_decoder_registered_for_builtin_mwpm(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        _, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        assert "pymatching" not in custom_decoders
        assert "mwpm" not in custom_decoders


# ---------------------------------------------------------------------------
# 7. Distance sweep
# ---------------------------------------------------------------------------

class TestDistanceSweep:
    def test_multi_distance_creates_correct_task_count(self, tmp_path):
        """3 distances × 2 p_values × 1 decoder = 6 tasks."""
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        assert len(tasks) == 3 * 2 * 1  # distances × p_values × decoders

    def test_each_task_has_correct_distance_in_metadata(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        distances_in_meta = {t.json_metadata["distance"] for t in tasks}
        assert distances_in_meta == {3, 5, 7}

    def test_each_task_has_num_rounds_equal_to_distance(self, tmp_path):
        """num_rounds_spec='d' → num_rounds in metadata equals distance for each task."""
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        for task in tasks:
            meta = task.json_metadata
            assert meta["num_rounds"] == meta["distance"]

    def test_circuits_differ_across_distances(self, tmp_path):
        """Circuits at different distances must differ in detector count."""
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        # Group by distance and verify detector counts differ
        by_d: dict[int, int] = {}
        for task in tasks:
            d = task.json_metadata["distance"]
            p = task.json_metadata["p"]
            if p == cfg.p_values[0]:  # same p for fair comparison
                by_d[d] = task.circuit.num_detectors
        assert len(set(by_d.values())) == 3  # all three distinct

    def test_metadata_contains_num_rounds_spec(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        for task in tasks:
            assert "num_rounds_spec" in task.json_metadata
            assert task.json_metadata["num_rounds_spec"] == "d"

    def test_single_distance_backward_compat(self, tmp_path):
        """distance: 5 (single int) → distances=(5,) → 1 task."""
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        assert len(tasks) == 1
        assert tasks[0].json_metadata["distance"] == 5


# ---------------------------------------------------------------------------
# 8. Task building — xyz_decoding
# ---------------------------------------------------------------------------

class TestTaskBuildingXyzDecoding:
    def test_xyz_decoding_false_removes_x_detectors_from_task(self, tmp_path):
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

        base_circuit = _build_circuit_from_cfg(cfg)
        assert tasks[0].circuit.num_detectors == base_circuit.num_detectors

    def test_mwpm_not_affected_by_xyz_decoding_false(self, tmp_path):
        yaml_str = _SURFACE_YAML.rstrip() + "\n  xyz_decoding: false\n"
        cfg, variants, _ = _load_cfg_from_yaml(yaml_str, tmp_path)

        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders
        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        base_circuit = _build_circuit_from_cfg(cfg)

        assert tasks[0].circuit.num_detectors == base_circuit.num_detectors


# ---------------------------------------------------------------------------
# 9. BB code backward compatibility
# ---------------------------------------------------------------------------

class TestBBBackwardCompatibility:
    def test_existing_bb_yaml_loads_without_error(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE
        assert cfg.circuit_type == BB_CIRCUIT_TYPE

    def test_bb_circuit_builds_correctly(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        circuit = _build_circuit_from_cfg(cfg)
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0

    def test_bb_ibm_reproduce_forwarded(self, tmp_path):
        yaml_str = _BB_YAML.replace(
            "  noise_model: uniform\n",
            "  noise_model: uniform\n  ibm_reproduce: true\n",
        )
        cfg, _, _ = _load_cfg_from_yaml(yaml_str, tmp_path)
        assert cfg.ibm_reproduce is True
        circuit = _build_circuit_from_cfg(cfg)
        assert circuit.num_detectors > 0

    def test_bb_use_both_forwarded(self, tmp_path):
        yaml_str = _BB_YAML.replace(
            "  decoders: [bplsd]",
            "  use_both: true\n  decoders: [bplsd]",
        )
        cfg, _, _ = _load_cfg_from_yaml(yaml_str, tmp_path)
        assert cfg.use_both is True
        circuit = _build_circuit_from_cfg(cfg)
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
        assert "num_rounds" in meta
        assert "num_rounds_spec" in meta

    def test_topological_metadata_contains_distance(self, tmp_path):
        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders
        tasks, _ = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        meta = tasks[0].json_metadata
        assert "distance" in meta
        assert "l" not in meta

    def test_bb_distances_is_none(self, tmp_path):
        cfg, _, _ = _load_cfg_from_yaml(_BB_YAML, tmp_path)
        assert cfg.distances is None


# ---------------------------------------------------------------------------
# 10. Sinter integration smoke tests
# ---------------------------------------------------------------------------

class TestSinterSmoke:
    def test_surface_code_mwpm_collects(self, tmp_path):
        import sinter

        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_YAML, tmp_path)
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

    def test_midout_bplsd_xyz_decoding_false_collects(self, tmp_path):
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

    def test_multi_distance_surface_code_mwpm_collects(self, tmp_path):
        """Distance sweep with MWPM: 3 distances × 2 p_values = 6 tasks, all collected."""
        import sinter

        cfg, variants, _ = _load_cfg_from_yaml(_SURFACE_MULTI_D_YAML, tmp_path)
        from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

        tasks, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=variants)
        assert len(tasks) == 6

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
        assert len(results) == 6
