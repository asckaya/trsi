# verify

> Owns `trainer_rsi.verify`: what counts as agreement, and what a timing
> difference supports.

## 1. No default bound

`judge(produced, bounds, reference)` MUST refuse in every one of these cases:

- nothing was produced, because measuring nothing is not agreement;
- no bounds were stated, because a number nobody stated is a number nobody can
  fail;
- an output was left unjudged, because a skipped comparison reports the same PASS
  as a performed one;
- a bound names an output that was not produced;
- a two-sided bound has no reference.

There MUST be no default tolerance anywhere. A single `f32 → bf16` rounding
measures `rel_l2 ≈ 1.7e-3`; a default of `1e-3` would teach its reader that FAIL is
the normal state of a correct program.

## 2. Predicates

| predicate | measures | refuses |
| --- | --- | --- |
| `AllClose(atol, rtol)` | worst per-element violation | discrete outputs |
| `RelL2(bound)` | relative L2 error | discrete outputs, references near zero |
| `Exact()` | count of differing elements | — (needs a reference) |

- `AllClose` MUST report the worst element, never an aggregate: one wrong element
  is a total failure and an average hides it.
- A relative measure MUST refuse a reference whose norm is below `NEAR_ZERO`,
  because near zero it divides by nothing and reports a number with no scale.
- Every verdict MUST carry the reference's own norm, so the reader can see what
  the error was relative to.

## 3. Claims

`compare(before, after)` states one of `faster`, `slower`, `indistinguishable`.

- Fewer than `MIN_SAMPLES` timings MUST be refused. A distribution is not a
  handful of numbers.
- The claim MUST be read off a bootstrap interval of the median ratio at
  `CLAIM_LEVEL`, not off the point estimate. Two percent is inside the noise of
  most runs, and only an interval can say so.
- The resampling MUST be seeded, so a comparison is reproducible from its inputs.
- `indistinguishable` MUST NOT be rendered as a weak win.

## 4. Out of scope

Producing the timings. `measure` — warmup policy, event timing, per-step
sampling — is a later layer; this document owns only what may be concluded from
its output.
