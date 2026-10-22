# plan

> Owns `trainer_rsi.plan`. Prices a replacement for a named set of ops using the
> same rules as [analysis](./analysis.md), and never applies one.

## 1. What a plan is

A `Plan` names what it replaces (`touches`), what those ops owe now (`now_us`),
what the replacement would owe (`then_us`), and what must be verified first
(`requires`).

- `delta_us` MUST be `now_us - then_us`, and a plan whose delta is not positive
  MUST NOT be offered. A rewrite that the model says is not faster is noise.
- `requires` MUST be non-empty for every kind. Nothing here has been run.
- `touches` MUST list graph op ids, so two plans over the same op can be told
  apart from two that compose.

## 2. Kinds

| kind | replaces | priced as |
| --- | --- | --- |
| `fuse` | a producer, its single-consumer intermediate, and the consumer | one kernel: producer's inputs + consumer's output, summed flops |
| `precision` | one compute-bound matmul-like op | same flops at the fastest stated rate, re-bounded by memory |
| `capture` | the step's launch structure | the idle gap the trace shows |
| `overlap` | the collective's schedule | the exposed collective time the trace shows |

- `fuse` MUST require the intermediate to have exactly one consumer, to not be a
  program output, and to be at least `CHAIN_FLOOR_BYTES`, because every step has
  hundreds of tiny intermediates and naming them buries the ones that matter.
- `fuse` MUST NOT span a collective, a view, or a constant.
- `fuse` MUST price its arithmetic at the rate of the member with the most flops.
  One kernel runs one instruction stream, and charging a matmul with the rate of
  the elementwise op beside it prices a kernel nobody can write.
- `precision` MUST re-bound the op by memory after the change. An op that becomes
  memory bound does not gain the whole rate ratio.

## 3. Combination

- `combination()` MUST take a greedy disjoint set over `touches`, largest first.
  Two rewrites of the same op are not additive, and summing both counts the same
  microseconds twice.
- `capture` and `overlap` MUST compose with any arithmetic rewrite, because they
  change launch and schedule, not arithmetic.
- The report MUST refuse its own sum when the predicted step falls below
  `floor_us`: the plans cannot all hold, and saying so is worth more than a
  headline number.

## 4. Out of scope

Applying a plan. There is no source rewrite, no patch format, and no code
generation here; a later layer owns those, and it will be judged by
[verify](./verify.md).
