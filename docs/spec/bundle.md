# bundle

> Owns `trainer_rsi.ir.graph`, `trainer_rsi.ir.trace`, `trainer_rsi.ir.bundle`.
> Read by `capture/`, `analyze/`, `plan/`.

## 1. Graph

A `Graph` is a captured step as dataflow. One `Op` per call, one `Tensor` per
materialised value.

- `Tensor` MUST carry `id`, `shape`, `dtype`, `role`. `role` is one of
  `parameter`, `gradient`, `optimizer_state`, `activation`, `input`, `output`,
  `temporary`; it decides whether a tensor outlives the step.
- `Op` MUST carry `id`, `target`, `kind`, `phase`, `inputs`, `outputs`. `kind` is
  the cost rule applied to it (see [analysis](./analysis.md)); `target` is the
  captured spelling and is for the reader, not for dispatch.
- `Op.attrs["count"]` MAY state that capture merged repeated calls into one node.
  Every price is multiplied by it.
- A tensor MUST be written by exactly one op. `Graph.producers()` raises rather
  than choosing.
- An op whose kind is `other` MUST be kept, not dropped. Dropping it would raise
  coverage to 100% by removing the work that could not be priced.

## 2. Trace

A `Trace` is what ran, as reported, with nothing derived.

- `Event.start_us` MUST be relative to the first device event; `Trace.step_us`
  MUST be the span those events cover.
- `Trace.steps` MUST state how many iterations the trace covers. Every per-step
  figure is divided by it, so a trace of ten iterations reports one.
- `category` MUST be one of `kernel`, `memset`, `h2d`, `d2h`, `d2d`, `cpu_op`,
  `annotation`. Only the first five occupy a device queue, and only those enter a
  timeline.
- `stream` MUST be the device queue, and MAY be absent for host work.
- Aggregates (busy, overlap, gaps, exposed work) MUST NOT be stored on a `Trace`.
  They are computed in [analysis](./analysis.md) from the events.

## 3. Bundle

A `Bundle` is one capture: `environment`, `capture`, and at least one of `graph`
and `trace`.

- `environment.device_id` MUST name an installed hardware document. A bundle that
  names none is refused at analysis time, not defaulted.
- `schema_version` MUST be `trainer_rsi.bundle/v1`. A file stating another version
  MUST be refused with the words `re-capture it`.
- A bundle carrying no trace MUST say so in `notes`; a bundle carrying no graph
  MUST leave `capture.graph_source` at `none`.

## 4. On disk

Gzipped JSON, one file, written by `write_bundle`. The digest returned is SHA-256
over the uncompressed payload, so two captures of the same content agree
regardless of compression.

Reading MUST validate against the schema in strict mode: no coercion, no extra
keys. A field added to a model without a schema bump is a bug, because an older
build will refuse the file rather than mis-read it.

## 5. Out of scope

Timing of individual ops. A trace gives kernels, not the graph nodes they
realised; correlating the two needs per-kernel annotations that this capture does
not yet write.
