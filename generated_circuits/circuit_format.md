Before run simulation, use need to generate stim circuits. Simulator load the circuit when execute sampling.
How to indicate the circuit information for simulation?

**Supported codes**  
    - `surface_code_Z`  
    - `superdense_color_code_Z`  
    - `bivariate_bicycle_code_Z`  

(Z means the basis of observable used in memory experiment)

Then, you need to specify detail parameters as folows:  
    - distance :code distance for code.  
    - rounds : the number of rounds of syndrome extraction.  
    - noise_model : noise model you want to insert.  
    - noise_strength : error rate of noise model.  

If you specify `bivariate_bicycle_code_Z`, you must specify additional parameters related to BB code like:  
    - l: 12  
    - m: 6  
    - a_x_pows: [3]  
    - a_y_pows: [1, 2]  
    - b_x_pows: [1, 2]  
    - b_y_pows: [3]  
(This is an example to generate $[[144, 12, 12]]$ BB code.)

You can generate specified stim circuit by running the following shell script(example): 
``` bash
./scripts/generate_circuits \
    -- code surface_code_Z \
    -- out_dir circuits/surface \
    -- noise_model uniform \
    -- rounds "d" \ 
    -- distance {1} \
    -- noise_strength {2} \
    ::: 3 5 7   # you can generate collectively
    ::: 0.001 0.002 0.005
```
- rounds are specified int or "{int}*d" where "d" is distance. For example, "4*d" means 4 * (distance) rounds. If you want to specify (distance) rounds, set "d" rather "1*d". 

**File format**
code={code},d={distance},rounds={rounds},noisemodel={noise_model},p={noise_strength}.stim


First, we implement wrapper, then build circuit generator module in src.


## Circuit Generation Methods

There are two distinct generation paths depending on the code type.

### Surface Code and Superdense Color Code (via `from_chromobius`)

Both codes are generated through a local copy of the [chromobius](https://github.com/quantumlib/Chromobius)
library located at `src/qec_decoder_sim/circuits/from_chromobius/`.

**Entry point:** `_topological_memory_circuits.py` → `make_topological_memory_circuit()`
→ `_chromobius_make_circuit()` in `from_chromobius/clorco/_make_circuit.py`

The generation pipeline uses a *chunk* and *flow* abstraction:

1. **Chunk**: a single syndrome extraction round represented as a `stim.Circuit` plus a
   list of `Flow` objects that describe how stabilizers propagate in/out of the round.
2. **Flow**: one stabilizer measurement, carrying `flags` such as `basis=X` / `basis=Z`
   and `color=r` / `color=g` / `color=b`.
3. `compile_chunks_into_circuit()` converts the flows into `DETECTOR` instructions,
   attaching extra coordinates via the `f2c` function.

**Surface code** (`surface_code_Z`):
- `make_xz_memory_experiment_chunks(basis="Z", ...)` builds [init, body×(r-2), final] chunks.
- Every round measures *both* X-type and Z-type stabilizer tiles; flows for both are emitted.
- The `basis="Z"` argument only selects which logical observable is tracked — **detectors
  for both X and Z stabilizers are always included**.
  Source: `from_chromobius/clorco/surface_code/_surface_code_chunks.py`

**Superdense color code** (`superdense_color_code_Z`):
- `make_superdense_color_code_circuit_round_chunk()` measures both X and Z ancillae in
  every round (each plaquette has one X and one Z measurement qubit).
- Again, flows for both bases are emitted regardless of the `basis` argument.
  Source: `from_chromobius/clorco/color_code/_superdense_planar_color_code_circuits.py`

### Bivariate Bicycle Code (custom builder)

**Entry point:** `generate_circuits.py` → `BivariateBicycleCode.build()` →
`_bbcode_circuit_builder.py`

Detector insertion is controlled explicitly:
- `_append_detectors_for_round()` inserts Z-basis detectors.
- `_append_detectors_for_round_x()` inserts X-basis detectors.
- The `use_both` parameter (default `False`) controls whether both bases appear in
  repeated rounds. With `use_both=False`, only the single matching basis is inserted.


## Detector Coordinate Convention

Every `DETECTOR` instruction in the generated STIM circuits carries four coordinates:

```
DETECTOR(x, y, time, basis_color) rec[-k] ...
```

| Index | Meaning |
|-------|---------|
| 0 | Physical x-coordinate of the stabilizer center |
| 1 | Physical y-coordinate of the stabilizer center |
| 2 | Syndrome extraction round (time) |
| **3** | **Basis + color encoding (see table below)** |

The 4th coordinate (index 3) is computed by the `f2c` function
(`from_chromobius/clorco/surface_code/_keyed_constructions.py` and the equivalent for
color code) as:

```python
coord = color_offset + basis_offset
# color_offset: r=0, g=1, b=2
# basis_offset: X=0, Z=3
```

### Basis–Color Mapping Table

| 4th coord | Color | Basis |
|-----------|-------|-------|
| 0 | red   | X |
| 1 | green | X |
| 2 | blue  | X |
| 3 | red   | Z |
| 4 | green | Z |
| 5 | blue  | Z |

**Surface code** uses only two colors (red and green), so only coordinates **0, 1, 3, 4**
appear. Coordinates 2 and 5 are absent.

**Superdense color code** uses all three colors, so coordinates **0–5** all appear.

### Filtering Detectors by Basis

To obtain a DEM restricted to a single basis (e.g., Z-basis detectors only for decoding
X errors), filter on the 4th coordinate:

```python
# Keep Z-basis detectors (4th coord >= 3)
z_det_ids = {i for i, c in circuit.get_detector_coordinates().items() if len(c) >= 4 and c[3] >= 3}
```

A utility function `filter_dem_by_basis` in
`src/qec_decoder_sim/utils/dem_utils.py` automates this at the DEM level.
