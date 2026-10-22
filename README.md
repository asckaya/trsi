# trainer-rsi

A full-pipeline optimizer for training programs.

It answers one question about a real training step: **where did the time go, and
what is the most it could be worth to fix it.** It does that by pricing the
captured program against the hardware's own published rates, comparing the
result with a measured device timeline, and reporting the difference ranked by
what each cause could return.

It is not a kernel library, not an autotuner, and it embeds no agent framework.
A report states what was observed, what it could be worth at most, and what must
be verified before any of it is claimed — never a promised speedup.

## What runs where

Everything runs on the machine that holds the model. The weights and the data are
already there, so nothing is shipped anywhere.

| | Needs | Does |
| --- | --- | --- |
| Facts | nothing | reads the installed device documents |
| Capture | that machine's own `torch` | profiles the step, exports the graph, writes one bundle |
| Analysis | no GPU | prices the bundle, ranks the gaps, prices the rewrites, compares two timings |
| Kernels | no GPU to emit, a GPU to run | derives a tile from the device document and writes the kernel |
| Harness | a coding agent runtime | hands the step over, then measures what came back |

A bundle and a timing set are files, so a report is produced in the same session
as the capture, and can be read again later without the model present.

## Install

```sh
uv sync                 # the tool
uv sync --extra agent   # the tool plus the agent runtime `trsi drive` drives
```

`torch` is not a dependency of this package and never will be. `capture` imports
the torch already installed beside the model, so the stack being measured is the
stack the model runs on rather than a second one that happens to be importable.
Run it inside that environment; a venv without torch can still do everything
except a capture.

Without `uv`, the same two lines are `pip install -e .` and `pip install -e '.[agent]'`.

## Five minutes without a GPU

```sh
.venv/bin/trsi facts
.venv/bin/trsi facts nvidia.h800_sxm throughput.bf16
.venv/bin/python examples/synthetic_bundle.py
.venv/bin/trsi analyze examples/synthetic.bundle.json.gz
```

`examples/synthetic_bundle.py` builds a bundle by hand — a two-projection block
on an A100 plus a hand-written device timeline — so the report can be read
before any accelerator is involved.

## Capture your own step

`trsi capture` prints the snippet for a named device:

```sh
.venv/bin/trsi capture --device nvidia.h800_sxm --steps 3
```

It is four lines in your own loop:

```python
import trainer_rsi.capture as capture

with capture.record(
    device_id="nvidia.h800_sxm", out="step.bundle.json.gz", steps=3, module=model, example=(batch,)
):
    for _ in range(3):
        train_step()
```

The `with` block **is** the measured window: open it after warmup, close it after
the steps whose cost the report should describe. `module=` and `example=` are
optional; without them the bundle carries a timeline and no graph, and the report
says so instead of guessing at one.

## What the report says

| Field | Meaning |
| --- | --- |
| measured | device time the trace covered, per step |
| per-op floor | sum of each op's own roofline: the floor for this program, unfused |
| absolute floor | the larger of all arithmetic and all traffic: the floor for any schedule |
| headroom | measured ÷ per-op floor; 1.0 means every kernel ran at its own roofline |
| priced coverage | share of modelled traffic carried by ops that have a cost rule |
| device busy | time at least one queue was running work, out of the trace span |

Both floors exist because one number cannot tell structure from physics: the
per-op floor says what this program owes, and the absolute floor says what the
arithmetic owes anyone.

Findings are grouped by the waste they expose, and ordered by `ceiling_us`:

| Waste | Question |
| --- | --- |
| structure | is an intermediate written and read back because two ops did not become one? |
| schedule | did the device queue idle, or is the step split into too many short kernels? |
| parallel | is communication running where no computation covers it? |
| precision | is compute-bound arithmetic issued below the widest rate the device has? |
| io | do copies sit on the critical path? |
| memory | do live activations alone constrain the step? |

`ceiling_us` is a model upper bound, not a promise. Every finding also states
what must be verified before the saving is claimed — a numeric bound per output,
a gradient comparison, a bit-identical parameter update — because a measurement
of the rewritten program is the only thing that turns a ceiling into a result.

## Prescriptions: `trsi plan`

`analyze` says what is wrong. `plan` says what to replace, and prices the
replacement with the same cost rules before anything runs:

```
2. fuse · fuse aten.rms_norm.default into aten.gelu.default
   · 4.1 us -> 2.1 us at the roofline
   · drop n (2.00 MiB) as a kernel boundary
   → save 2.1 us  (0.3% of the step)
   requires per-output comparison against the current program at a declared bound
   requires the fused kernel's own measurement on this device
```

Four kinds are priced: `fuse` (two adjacent kernels become one, keeping the
producer's inputs and the consumer's output and dropping the intermediate),
`precision` (arithmetic re-issued at the fastest rate the device states, repriced
as memory bound when that is what it becomes), `capture` (idle time between
kernels, from the trace), and `overlap` (exposed collective time, from the trace).

The summary takes a **disjoint** set — two rewrites of the same op are not
additive — and refuses its own arithmetic when the total would predict a step
below the absolute floor:

```
sum | 3 disjoint plans predict 112.1 us per step out of 600.0 us measured (19%)
```

Nothing is applied. A plan states what it touches, what it would owe, and what
must be verified first; the rewrite and the measurement are the next layer.

## The operator side: `trsi kernel`

The analysis side stops at "these two ops become one kernel". This is the other
half: the tile that kernel runs, derived from the device document rather than
from a template.

```sh
trsi kernel --device nvidia.h800_sxm --m 8192 --n 8192 --k 8192 --out mm.py
```

```
    tile  256x128x32, 3 stages
 threads  256
  shared  72 KiB, 3 block(s) per SM
   waves  5.2
roofline  1111.2 us
```

`mm.py` is a complete tile-level matmul: staged copies, a pipelined tensor-core
`gemm`, a correctness gate that **refuses to run without a bound you state**, and
event timing that prints one `us=` line per sample. That last part is what makes
it measurable by the same harness:

```sh
trsi drive --repo . --device nvidia.h800_sxm \
  --entry "python mm.py --rtol 2e-2" --measure "python mm.py --rtol 2e-2"
```

Geometry comes from the document: staged buffers must fit
`memory.shared.per_cta`, the accumulator must fit
`ACCUMULATORS_PER_THREAD` registers, residency follows `memory.shared.per_sm`
and `compute.sm_count`. A tile that violates any of those is dropped, never
clamped — and the choice is printed in the file, because it is a heuristic that
the measurement is there to refute.

The target mapping is a table (`nvidia.sm90` → `{"kind": "cuda", "arch": "sm_90"}`),
so a second vendor is a new row rather than a branch in the emitter.

## Claiming a win: `trsi compare`

Two timing sets in, one claim out:

```sh
trsi compare before.json after.json     # {"name": ..., "us": [...]}
```

The claim is read off a bootstrap interval of the median ratio, not off the point
estimate, and there are three answers: `faster`, `slower`, or
`indistinguishable`. Fewer than five samples cannot produce a claim at all.
Correctness has the same discipline: `verify.judge` refuses an unjudged output,
refuses a default bound, refuses an aggregate on a discrete output, and refuses a
relative measure against a reference whose norm is near zero.

## Handing one step to an agent: `trsi drive`

The agent edits; the harness measures; the audit judges. Those are three
different parties on purpose.

```sh
trsi drive --repo ~/train-repo --device nvidia.h800_sxm \
  --entry "python train.py --max-steps 3" \
  --measure "python bench_step.py"      # prints one us=<float> per iteration
  --gate "python check_step.py"          # must exit 0 to be accepted
```

What happens: a copy of the repository is made without its history; the step is
measured `warmup + reps` times; a Codex thread is given the copy, the device
facts, and the priced plans; the step is measured again; the gate runs; the audit
compares the agent's claims with the samples it took itself.

```
3 turned out to be 2 turns, artifacts in runs/20261022-193000
812.4 us -> 641.0 us  x1.267 [1.244, 1.291]  faster
gate: exit 0  python check_step.py
agent: fused the rmsnorm into the up projection's epilogue
  claim supported: the fused epilogue removes a pass — x1.267 [1.244, 1.291]
accepted
```

A run is accepted when the samples separate the two programs and the gate passed.
It is rejected when a claim does not match the measurement, even if the timing
improved — a report that contradicts its own evidence is not evidence. Failed
runs feed their findings into the next prompt verbatim.

`--print-prompt` shows what the agent is told without running anything, which is
also how the prompt is reviewed. The Codex adapter lives in
`src/trainer_rsi/harness/runner.py`; `openai-codex` is an optional extra
(`pip install -e '.[agent]'`), and the package imports without it.

## Hardware facts carry provenance

Every figure a bound divides by comes from an installed document, and every
figure names where it came from:

```toml
[facts.throughput.bf16]
value = 989500000000000
unit = "flop/s"
origin = "derived"
source = "https://serverflow.ru/.../NVIDIA H800.pdf"
conditions = "dense; the datasheet states 1979 TFLOPS BF16 with the footnote that dense figures are half."
```

`trsi facts nvidia.h800_sxm` prints all of them. Absent and unavailable are
different answers: a dtype the product has no mode for is recorded as
`status = "unavailable"`, so the analysis reports it as unrated rather than
pricing it at something it is not. Installed: `nvidia.a100_sxm_80gb`,
`nvidia.h800_sxm`, `nvidia.rtx_a6000`, `nvidia.rtx_pro_6000_blackwell`.

Adding a device is one file under `src/trainer_rsi/facts/hardware/` with schema
`trainer_rsi.hardware/v1`; `trsi facts` picks it up with its digest.

## Architecture

```
capture/record.py ──► Bundle { Graph, Trace, Environment }  ──► analyze/report.py
     (that machine)          (one gzipped JSON)                  (no GPU)
```

| Stage | File | Owns |
| --- | --- | --- |
| capture | `capture/record.py`, `capture/graph.py`, `capture/kineto.py` | profiler window, `torch.export` graph, chrome-trace parsing |
| data model | `ir/graph.py`, `ir/trace.py`, `ir/bundle.py` | tensors, ops, mesh; events; the bundle and its digest |
| work | `analyze/work.py` | flops and bytes per op kind |
| roofline | `analyze/roofline.py` | per-op bounds, the two floors, coverage |
| timeline | `analyze/timeline.py` | busy time, overlap, gaps, exposed collectives and copies |
| findings | `analyze/gaps.py` | the six waste classes, each with a ceiling and a required check |
| plans | `plan/candidates.py` | rewrites priced against the same device, taken as a disjoint set |
| kernels | `kernels/` | tile geometry from the device document, emitted as tile-level source |
| verification | `verify/predicates.py`, `verify/stats.py` | no-default bounds, and a claim read off an interval |
| facts | `facts/` | installed documents, digests, the rates a bound divides by |

The invariant: **the graph states work, the trace states duration, and the
analysis is their disagreement.** Nothing stores a derived timing next to the
event it came from.

Cost rules are one function, `analyze/work.py:price`, dispatching on op kind:
matmul `2·M·N·K`, attention `4·B·H·L·S·E`, norm `5·elements`, collective
`bytes × ring factor`, and so on. Elementwise arithmetic is counted at one flop
per output element per input on purpose: those ops are priced to be memory bound,
and inflating their flop count would move a bound the memory side already sets.
An op with no rule keeps its traffic and is reported as unpriced, so coverage
falls instead of the report quietly narrowing its own scope.

Adding a detector is a function in `analyze/gaps.py` returning `Finding | None`,
wired into `detect()`, with a case in `tests/test_analysis.py`.

## Where the contracts live

| Spec | Owns |
| --- | --- |
| [architecture](docs/spec/architecture.md) | the pipeline map, the one invariant, the ownership table |
| [bundle](docs/spec/bundle.md) | `Graph`, `Trace`, `Bundle`, the on-disk format, the digest |
| [facts](docs/spec/facts.md) | the document schema, provenance, absent versus unavailable |
| [analysis](docs/spec/analysis.md) | cost rules, the two floors, coverage, the finding contract |
| [plan](docs/spec/plan.md) | rewrite kinds, their pricing, disjointness, self-refusal |
| [verify](docs/spec/verify.md) | predicates, refusals, the claim and its interval |
| [kernels](docs/spec/kernels.md) | target mapping, tile geometry, the emitted source contract |
| [harness](docs/spec/harness.md) | the agent/harness split, the measure contract, acceptance |
| [0001 measure](docs/plans/0001-measure-and-verify.md) | the open milestone, partly delivered by the harness |

## Not built yet

- **No rewrite is applied.** `plan` prices a replacement; nothing edits your code
  and nothing measures the rewritten program end to end.
- **Only forward graphs are captured.** Backward is measured, not modelled, so
  recompute and optimizer-state plans have no time model.
- **Collectives are priced per op** from payload and link rate. There is no
  cluster fabric document, so a group wider than the NVLink domain is unpriced.
- **One operator emitted.** `trsi kernel` writes a matmul; every other op kind is
  refused rather than approximated. Realising a plan into a kernel request, and
  tuning the geometry against measurements, are not built.
- **No scheduler.** One command runs one thing on the machine it is given; there
  is no queue, no fleet, and no second machine involved.

## Develop

`docs/spec/` owns the contracts — the map is
[architecture](docs/spec/architecture.md) and its ownership table says which
document owns what. `docs/plans/` owns the work: three passes, delivered shapes
before prose. `docs/policies/project-policy.json` states the acceptance criteria
that apply by path.

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

Conventions worth knowing before changing anything: every data structure is a
pydantic model on `trainer_rsi.base.Value` (frozen, strict, closed), so it is
also a wire format; public names carry no leading underscore; a module file
starts with its imports; and nothing guesses — an unknown device, dtype, extent
or op is refused with what is installed, not filled in with a default.
