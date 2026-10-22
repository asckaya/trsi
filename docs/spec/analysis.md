# analysis

> Owns `trainer_rsi.analyze`: the cost rules, the two floors, coverage, and the
> finding contract. Reads a [bundle](./bundle.md) and a [facts](./facts.md)
> projection, and produces a report.

## 1. Cost rules

One function, `analyze/work.py:price`, dispatches on `Op.kind`. `flops` and
`bytes` are the work an op owes, independent of any machine.

| kind | flops | bytes |
| --- | --- | --- |
| `matmul` | `2 · prod(out.shape[:-1]) · out.shape[-1] · in[0].shape[-1]` | inputs + output |
| `attention` | `4 · B · H · L · S · E` | q + k + v + out |
| `conv` | `out.numel · 2 · prod(weight.shape[1:])` | inputs + output |
| `norm` | `5 · out.numel` | inputs + output |
| `softmax` | `4 · out.numel` | inputs + output |
| `elementwise` | `(inputs + 1) · out.numel` | inputs + output |
| `reduce` | `in[0].numel` | inputs + output |
| `movement` | `0` | inputs + output |
| `gather` | `0` | inputs + output + out (rows touched) |
| `collective` | `0` | payload × ring factor |
| `optimizer` | `10 · out.numel` | 4 reads + 3 writes of the parameter |
| `constant`, `view` | `0` | `0` |
| `other` | `0` | inputs + output, and the op is **unpriced** |

- Elementwise arithmetic MUST be counted at one flop per output element per input.
  These ops are priced to be memory bound, and a flop count inflated toward the
  real transcendental cost would move a bound the memory side already sets.
- A collective's ring factors MUST be `all_reduce` 2, `reduce_scatter` 1,
  `all_gather` 1, `all_to_all` 1, `broadcast` 1, `send_recv` 1.
- `kind = other` MUST keep its traffic and MUST NOT be priced. Feeding its bytes
  into the floor would claim knowledge of how it consumes them.

## 2. Two floors

`roofline()` states both, because one number cannot tell structure from physics.

- `ideal_us` MUST be the sum of each op's own bound: `max(flops/peak,
  bytes/bandwidth)`. It is the floor for this program, unfused.
- `floor_us` MUST be `max(total flops / fastest stated rate, total bytes / HBM
  bandwidth)`. It is the floor for *any* schedule of this arithmetic, and it is
  unreachable in general.
- An op whose dtype has no stated rate MUST be bounded by memory alone and marked
  `unrated`. Its bound is a floor with no promise behind it.
- `coverage` MUST be the share of modelled traffic carried by priced ops. A report
  MUST show it, because a floor computed over 60% of the work is a floor over a
  program nobody described.

## 3. Timeline

`analyze/timeline.py` accounts for device time, per step, from the trace alone.

- `busy_us` MUST be the union of all device intervals; `overlap_us` MUST count
  time two or more queues ran, once per extra queue.
- A gap MUST be an interval on the busiest queue longer than `GAP_US`. It is launch
  distance, not idle hardware, and the report MUST separate it from
  `idle_share`, which is time no queue ran at all.
- `exposed_collective_us` MUST be collective time minus its overlap with
  non-collective device work. The same rule applies to copies.

## 4. Findings

A `Finding` states one cause of a gap and what it could be worth.

- Fields MUST include `waste`, `title`, `evidence`, `ceiling_us`,
  `ceiling_share`, `action`, `requires`.
- `ceiling_us` MUST be an upper bound read off the model. It MUST NOT be described
  as a saving anywhere in the report.
- `requires` MUST be non-empty for every timing finding: a claim nobody can check
  is a claim nobody should act on.
- Findings MUST be ordered by `ceiling_us`, with non-timing findings last.
- A detector MUST return `None` when its threshold is not met. A finding that fires
  on every capture trains its reader to ignore the list.

## 5. Adding a detector

Write a function in `analyze/gaps.py` returning `Finding | None`, wire it into
`detect()`, and cover it in `tests/test_analysis.py` with a case that fires and a
case that stays quiet. Evidence lines MUST carry the numbers the finding was read
from, not a restatement of the title.
