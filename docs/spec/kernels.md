# kernels

> Owns `trainer_rsi.kernels`: the device-to-target mapping, the tile geometry
> derived from a device document, and the emitted tile-level source.

## 1. Where this sits

The analysis layer decides **what** to change; this layer decides **how the
resulting operator is shaped on this device**, and emits it in a tile-level
language rather than in vendor intrinsics. That is what lets one description run
on a second vendor: the target changes, the tile arithmetic does not.

- Emitting MUST stop at a runnable, checkable program. A skeleton with a hole in
  it is not an emitted kernel.
- The emitted program MUST carry its own correctness gate and its own timing, so
  `trsi drive` can measure it with no extra machinery.
- Nothing here is applied to the user's code. An emitted kernel is a candidate
  that the harness then has to accept.

## 2. Target mapping

`kernels.target.ARCHITECTURES` maps an installed document's `architecture` to the
compiler's own spelling, e.g. `nvidia.sm90` to `{"kind": "cuda", "arch": "sm_90"}`.

- The mapping MUST be a table, not a branch inside the emitter: adding a vendor is
  a new row.
- An architecture with no row MUST be refused, listing the rows that exist.
- The emitter MUST read the target, never the device. A per-device `if` in emitted
  source is how portability is lost.

## 3. Geometry

`matmul_geometry` chooses `block_m`, `block_n`, `block_k`, `stages` and `threads`
from the document alone.

| Constraint | Source |
| --- | --- |
| staged buffers fit one block | `memory.shared.per_cta` |
| resident threads fit one block | `compute.max_threads_per_cta` |
| accumulator stays in registers | `ACCUMULATORS_PER_THREAD` |
| blocks resident per SM | `memory.shared.per_sm` |
| residency rounds the problem needs | `compute.sm_count` |

- A tile that violates any constraint MUST be dropped, never clamped.
- A problem shorter than the shallowest stage depth MUST run in one stage: the
  depth is capped by the problem's own `k` rather than refused.
- The choice MUST be stated as a heuristic in the emitted file. The geometry is a
  prediction; `trsi drive`'s measurement is the only thing that settles it.

## 4. Emitted source

One file, Python, importing the tile language only.

- The header MUST name the device id, the document digest, the geometry, the
  residency rounds and the roofline, so the file is readable without this project.
- The correctness gate MUST require the caller to state a bound (`--rtol`) and
  refuse to run without one. A generated default is the same mistake as a default
  in [`verify`](./verify.md), with the same consequence.
- Timing MUST use device events around one iteration per sample and print one
  `us=<float>` line per sample, which is the contract
  [`harness`](./harness.md) measures against.

## 5. Scope

One operator, matmul, for the dtypes the compiler spells. Every other op kind is
refused rather than emitted approximately.

Not yet: realising a `Plan` into a kernel request automatically, per-target kernel
specialisation, and autotuning the geometry against measurements. Until then the
geometry is a starting point and the harness is what refutes it.
