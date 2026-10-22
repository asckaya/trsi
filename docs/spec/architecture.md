# trainer-rsi — architecture

> The map. Every other document owns one contract; this one owns the picture and
> the table that says which document owns what.
>
> A change to a contract updates its owner first and this table second. A
> structural change here obliges the owner to follow.

## 1. Pipeline

```text
   capture            bundle              analysis                 plan
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ profiler    │   │ Graph        │   │ work   → roofline│   │ candidates   │
│ torch.export│──►│ Trace        │──►│ timeline         │──►│ combination  │
│ chrome trace│   │ Environment  │   │ gaps             │   │              │
└─────────────┘   └──────────────┘   └──────────────────┘   └──────┬───────┘
   that machine     one file,          no GPU needed               │ priced,
                    gzipped JSON                                   │ not applied
                                                                   ▼
                                                    verify: bounds in, claim out
```

Two halves meet at the bundle. Everything left of it needs the model's own
environment; everything right of it needs nothing but the file.

## 2. The one invariant

**The graph states work; the trace states duration; the analysis is their
disagreement.**

No derived timing is ever stored beside the event it came from. A figure that can
be recomputed is recomputed, so a report cannot hold a stale number that once
agreed with a different capture.

## 3. Units of exchange

| Unit | Shape | Produced by | Consumed by |
| --- | --- | --- | --- |
| hardware document | one `.toml`, schema `trainer_rsi.hardware/v1` | a person, with a source for every figure | `facts` |
| bundle | `*.bundle.json.gz`, schema `trainer_rsi.bundle/v1` | `capture.record` | `analyze`, `plan` |
| report | JSON on stdout or to a path | `analyze` | a reader, or an agent |
| plan set | JSON on stdout or to a path | `plan` | a reader, or an agent |
| timing set | `{"name": str, "us": [float]}` | `measure` (not built) or hand | `compare` |

## 4. Layers

| Layer | Package | Owns |
| --- | --- | --- |
| data | `ir/` | `Graph` / `Tensor` / `Op` / `Mesh`, `Trace` / `Event`, `Bundle` |
| facts | `facts/` | installed documents, provenance, digests, the rates a bound divides by |
| capture | `capture/` | the measured window, the exported graph, chrome-trace parsing |
| analysis | `analyze/` | work rules, the two floors, the timeline, the finding contract |
| plan | `plan/` | rewrites priced against the same device, taken as a disjoint set |
| verification | `verify/` | bounds with no default, and a claim read off an interval |
| harness | `harness/` | the task, the workspace copy, the measure and gate runs, the audit |

`ir`, `facts`, `analyze`, `plan` and `verify` import no accelerator library.
`capture` imports `torch` and is the only package that does.

## 5. Specification ownership

| Spec | Owns |
| --- | --- |
| [architecture](./architecture.md) (this) | the pipeline map, the invariant, the ownership table |
| [bundle](./bundle.md) | `Graph`, `Trace`, `Bundle`, the on-disk format, the digest |
| [facts](./facts.md) | the document schema, provenance, absent versus unavailable, registry lookup |
| [analysis](./analysis.md) | op-kind cost rules, the two floors, coverage, the finding contract |
| [plan](./plan.md) | rewrite kinds, how each is priced, disjointness, self-refusal |
| [harness](./harness.md) | the agent/harness split, the measure contract, acceptance |
| [verify](./verify.md) | predicates, refusals, the claim and its interval |

## 6. Where an agent fits

The harness is not a compiler pass. It copies the repository, measures the step,
hands the copy to a coding agent, measures again, and judges the difference
against its own samples ([harness](./harness.md)). The agent never measures, and
its report is checked rather than believed.

## 7. What this project is not

It is not a kernel library, an autotuner, or an agent framework. Nothing here
edits a training program, and nothing here claims a speedup: `plan` prices a
replacement and `verify` states what a measurement supports. The edit and the
measurement are the reader's, or a later layer's.
