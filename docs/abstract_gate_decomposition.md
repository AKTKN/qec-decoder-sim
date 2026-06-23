# Abstract One-Qubit Gate Decomposition Before Noise

`qec_decoder_sim` decomposes selected abstract one-qubit Clifford gates
immediately before circuit-level noise insertion. This happens after circuit
generation and optimization, so optimizers can still use compact Clifford
representatives internally. The decomposition ensures that noise is applied to
the intended physical one-qubit basis gates instead of to an abstract composite
gate.

The pass is applied in both noise insertion implementations used by the
repository:

| Implementation | Entry point |
| --- | --- |
| qec-decoder-sim noise inserter | `qec_decoder_sim.noise.noise_model.insert_arbitrary_noise` and `insert_realistic_noise` |
| Chromobius-derived generator noise inserter | `gen.NoiseModel.noisy_circuit` |

`H` is intentionally kept as `H`.

## Physical Basis

The current physical one-qubit basis for this preprocessing step is:

| Gate | Meaning |
| --- | --- |
| `H` | Hadamard basis change. Kept as a physical gate. |
| `S`, `S_DAG` | Quarter-turn phase gates. |
| `SQRT_X`, `SQRT_X_DAG` | Half-turn X-axis gates. |

## Decomposition Table

The following replacements are applied recursively through `REPEAT` blocks
before inserting noise. When an abstract gate is expanded into multiple
physical gates inside one original moment, the moment is split into physical
sub-moments with `TICK`s between them. Non-expanded operations from the original
moment remain in the first physical sub-moment.

| Abstract gate | Physical basis decomposition |
| --- | --- |
| `C_XYZ` | `SQRT_X`, then `S` |
| `C_ZYX` | `S_DAG`, then `SQRT_X_DAG` |
| `H_XY` | `S_DAG`, then `SQRT_X`, then `SQRT_X` |
| `H_YZ` | `S`, then `S`, then `SQRT_X_DAG` |
| `SQRT_Y` | `S`, then `SQRT_X_DAG`, then `S_DAG` |
| `SQRT_Y_DAG` | `S`, then `SQRT_X`, then `S_DAG` |

The products above are verified against Stim tableaus in
`tests/test_noise_gate_decomposition.py`.

## Noise Semantics

After decomposition, each physical gate receives its own one-qubit gate noise.
For example, under `si1000` with physical error rate `p`:

```text
C_XYZ q
```

is converted before noise insertion into:

```text
SQRT_X q
TICK
S q
```

and then each of `SQRT_X` and `S` receives `DEPOLARIZE1(p / 10)`.

This differs intentionally from treating `C_XYZ` as one physical gate, which
would add only one one-qubit noise event. The decomposition therefore matters
for detector error model priors and for comparisons against circuits expressed
in an explicit physical basis.
