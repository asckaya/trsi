---
type: FEAT
component: measure
---

# [FEAT][measure] Close the loop: measure on the device, then say what it supports

## Description

`plan` prices a replacement and `verify` says what a measurement supports, but
nothing produces a measurement. The loop is open at exactly the point where a
prediction becomes a result.

This milestone adds `measure`: warmup policy, event timing, per-step samples, and
the files `compare` already reads. It adds no new claim logic.

### Current state

- `src/trainer_rsi/plan/candidates.py` prices four rewrite kinds; nothing applies
  one and nothing times one.
- `src/trainer_rsi/verify/stats.py` compares two timing sets; the only producer of
  a timing set today is a hand-written JSON file.
- `src/trainer_rsi/capture/record.py` measures a *bundle*: profiler events for a
  window of steps. It does not produce a distribution, has no warmup policy, and
  its step time is the span of the profiler window, which includes launch gaps at
  the edges.
- Nothing times a step. The only duration a report holds is the profiler window's
  span, which includes launch gaps at its edges and is not a distribution.

### Decisions

- D0 This milestone is partly delivered ahead of its plan by `harness/measure.py`,
  which times the measure command as a process and reads `us=` lines from it.
  In-process step timing with CUDA events remains open, and the harness contract
  does not depend on which of the two produces the file.
- D1 Timing uses CUDA events around each step rather than a wall clock over many
  steps, because a wall clock cannot separate step cost from host jitter, and
  because `compare` needs a distribution rather than a total.
- D2 Warmup is a stated count rather than a duration, because a duration hides
  what it warmed: the first iteration of a graph-captured step is not the same
  work as the tenth.
- D3 A timing set is a file with a name and a list, not an entry in the bundle,
  because `compare` must be able to run on two files from two machines without a
  bundle either side.

## Milestones

### Milestone M0: measure a step on the device

#### Depends
- None

#### Target State Design

##### Delivered

```diff
# src/trainer_rsi/measure/record.py
+@contextmanager
+def record(*, name: str, out: Path, warmup: int = 5, reps: int = 20) -> Generator[None]:
+    """Time the block it wraps, once per iteration, and write the samples."""
```

```text
$ trsi measure --name eager --out eager.json -- python bench_eager.py
wrote eager.json: 20 samples, median 812.4 us, p10 806.1, p90 819.0
```

##### Accepted by

A script that opens the window, runs twenty iterations after five warmup
iterations, and writes a file `compare` accepts. A run with fewer than
`MIN_SAMPLES` iterations is refused with the same words `compare` uses.

#### Related Files
- `docs/spec/verify.md`

#### Rejected

- Reusing `capture.record` and deriving samples from profiler events: the window
  span includes edge gaps, and slicing a chrome trace into steps needs step
  annotations the capture does not write.
- Timing with `torch.utils.benchmark`: it owns the loop it measures, and the loop
  belongs to the model.
