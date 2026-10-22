from typing import Literal

from trainer_rsi.analyze.roofline import Roofline
from trainer_rsi.analyze.timeline import Timeline
from trainer_rsi.base import Value
from trainer_rsi.facts.envelope import Limits
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph, Op, Tensor

type Waste = Literal["structure", "schedule", "parallel", "precision", "io", "memory"]

FUSIBLE = frozenset({"elementwise", "norm", "softmax", "reduce"})
"""Kinds that fuse into a neighbour without needing a different algorithm."""

CHAIN_FLOOR_BYTES = 1 << 20
"""An intermediate smaller than this is not worth naming; every step has hundreds."""

MATMUL_LIKE = frozenset({"matmul", "attention", "conv"})


class Finding(Value):
    """One cause of a gap between what a step costs and what it could cost.

    A finding states what was observed, what it could be worth at most, and what
    must be verified before any of it is claimed. `ceiling_us` is an upper bound
    read off a model, never a promised saving: the saving is what a measurement
    of the rewritten program says afterwards.
    """

    waste: Waste
    title: str
    evidence: tuple[str, ...]
    ceiling_us: float | None
    """Most this could save, per step. `None` when the finding is not about time."""
    ceiling_share: float | None
    """That ceiling as a share of the measured step, when a measurement exists."""
    action: str
    requires: tuple[str, ...] = ()


def share(ceiling_us: float | None, measured_us: float | None) -> float | None:
    """Which fraction of a measured step a ceiling stands for."""
    if ceiling_us is None or measured_us in (None, 0):
        return None
    return ceiling_us / measured_us


def bytes_to_us(count: int, limits: Limits) -> float:
    """Time *count* bytes take at the device's memory bandwidth."""
    return count / limits.hbm_bandwidth * 1e6


def fusion_finding(graph: Graph, limits: Limits, measured_us: float | None) -> Finding | None:
    """Intermediates written and read back because two ops did not become one.

    A tensor produced by one op and consumed by exactly one other costs its own
    size twice when the two run as separate kernels: once to write it and once
    to read it. Fusing them removes both crossings, and nothing else about the
    arithmetic changes.
    """
    producers = graph.producers()
    consumers = graph.consumers()
    outputs = {tensor.id for tensor in graph.tensors if tensor.role == "output"}
    chains: list[tuple[Tensor, Op]] = []
    for tensor in graph.tensors:
        producer = producers.get(tensor.id)
        readers = consumers.get(tensor.id, ())
        if producer is None or len(readers) != 1 or tensor.id in outputs:
            continue
        consumer = readers[0]
        if producer.kind in ("collective", "view", "constant") or consumer.kind in (
            "collective",
            "view",
            "constant",
        ):
            continue
        if tensor.nbytes < CHAIN_FLOOR_BYTES:
            continue
        if producer.kind not in FUSIBLE and consumer.kind not in FUSIBLE:
            continue
        chains.append((tensor, consumer))
    if not chains:
        return None
    ceiling = bytes_to_us(2 * sum(tensor.nbytes for tensor, _ in chains), limits)
    largest = sorted(chains, key=lambda item: item[0].nbytes, reverse=True)[:3]
    return Finding(
        waste="structure",
        title=f"{len(chains)} intermediates are written and read back for a single consumer",
        evidence=(
            *(
                f"{tensor.id} {list(tensor.shape)} {tensor.nbytes / 2**20:.1f} MiB -> {consumer.target}"
                for tensor, consumer in largest
            ),
            f"traffic removable: {2 * sum(tensor.nbytes for tensor, _ in chains) / 2**20:.1f} MiB",
        ),
        ceiling_us=ceiling,
        ceiling_share=share(ceiling, measured_us),
        action=" fuse each producer into its single consumer, keeping the consumer's launch",
        requires=("per-output comparison against the current program at a declared bound",),
    )


def precision_finding(
    priced: Roofline, limits: Limits, measured_us: float | None
) -> Finding | None:
    """Compute-bound arithmetic issued at a rate the device can beat.

    Only ops whose bound is set by arithmetic are named: a memory-bound op gains
    nothing from a wider tensor core path, and naming it would pad the ceiling
    with a saving that cannot exist.
    """
    fastest_dtype, fastest_rate = limits.fastest()
    candidates: list[tuple[float, Op, DType]] = []
    for item in priced.ops:
        if item.bound_by != "compute" or item.op.kind not in MATMUL_LIKE:
            continue
        rate = limits.peak(item.work.dtype)
        if rate is None or rate >= fastest_rate:
            continue
        candidates.append((item.compute_us * (1.0 - rate / fastest_rate), item.op, item.work.dtype))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    ceiling = sum(gain for gain, _, _ in candidates)
    if ceiling <= 0:
        return None
    return Finding(
        waste="precision",
        title=f"{len(candidates)} compute-bound ops run below {fastest_dtype.value}",
        evidence=tuple(
            f"{op.target} {op.module_path or '-'} in {dtype.value}: {gain:.1f} us at best"
            for gain, op, dtype in candidates[:3]
        ),
        ceiling_us=ceiling,
        ceiling_share=share(ceiling, measured_us),
        action=f"issue the arithmetic in {fastest_dtype.value} and re-measure both the time and the error",
        requires=(
            "a declared numeric bound per output, measured over the same inputs",
            "a gradient comparison over a random projection, for training steps",
        ),
    )


def schedule_finding(step: Timeline) -> Finding | None:
    """Device time the step could have spent working and did not.

    Two shapes of the same waste: the queue went idle between kernels, or it was
    kept busy by so many tiny kernels that launch distance, not work, set the
    pace.
    """
    if step.span_us == 0 or step.steps == 0:
        return None
    fragmented = step.kernels > 300 and step.mean_kernel_us < 10.0
    idle = step.gap_us > 0.01 * step.span_us
    if not fragmented and not idle:
        return None
    ceiling = step.gap_us
    return Finding(
        waste="schedule",
        title="the device queue idles between kernels"
        if idle
        else "the step is split into many short kernels",
        evidence=(
            f"{step.kernels} kernels per step, mean {step.mean_kernel_us:.1f} us, median {step.median_kernel_us:.1f} us",
            f"idle inside the step: {step.gap_us:.1f} us of {step.span_us:.1f} us",
            f"device-only idle (no queue running): {step.idle_share:.1%} of the span",
        ),
        ceiling_us=ceiling,
        ceiling_share=share(ceiling, step.span_us),
        action="capture the step into one graph, or launch the tail from a side stream",
        requires=(
            "a re-measurement of the same step, since graph capture changes allocator behaviour",
        ),
    )


def collective_finding(step: Timeline) -> Finding | None:
    """Communication that no computation covered.

    A collective on its own queue can overlap with compute. When it does not, its
    whole duration is added to the step instead of hidden inside it.
    """
    if step.span_us == 0 or step.collective_us == 0:
        return None
    if step.exposed_collective_us <= 0.01 * step.span_us:
        return None
    ceiling = step.exposed_collective_us
    return Finding(
        waste="parallel",
        title="communication is not covered by computation",
        evidence=(
            f"{step.collective_us:.1f} us of collective time per step",
            f"{step.exposed_collective_us:.1f} us of it ({step.exposed_collective_us / step.collective_us:.0%}) has no kernel running beside it",
        ),
        ceiling_us=ceiling,
        ceiling_share=share(ceiling, step.span_us),
        action="split the collective into chunks and issue each on its own queue ahead of its consumer",
        requires=("a bit-identical parameter update across the reordered collective",),
    )


def io_finding(step: Timeline) -> Finding | None:
    """Host-to-device traffic on the critical path."""
    if step.span_us == 0 or step.exposed_copy_us <= 0.01 * step.span_us:
        return None
    ceiling = step.exposed_copy_us
    return Finding(
        waste="io",
        title="copies run on the critical path",
        evidence=(
            f"{step.copy_us:.1f} us of copy time per step",
            f"{step.exposed_copy_us:.1f} us of it has no kernel running beside it",
        ),
        ceiling_us=ceiling,
        ceiling_share=share(ceiling, step.span_us),
        action="prefetch the next batch on a side stream and pin the host buffer",
        requires=("identical batch contents across the reordered pipeline",),
    )


def live_peak(graph: Graph) -> tuple[int, tuple[Tensor, ...]]:
    """Peak bytes of activation alive at once, and the tensors resident there.

    Parameters are excluded: they are live for the whole step, so including them
    would make every peak look like the model size and hide what recompute can
    actually reclaim.
    """
    order = {op.id: index for index, op in enumerate(graph.ops)}
    producers = graph.producers()
    consumers = graph.consumers()
    last = len(graph.ops) - 1
    births: list[tuple[int, int, Tensor]] = []
    for tensor in graph.tensors:
        if tensor.role not in ("activation", "temporary", "input"):
            continue
        producer = producers.get(tensor.id)
        birth = 0 if producer is None else order[producer.id]
        readers = consumers.get(tensor.id, ())
        death = last if not readers else max(order[reader.id] for reader in readers)
        births.append((birth, death, tensor))
    peak = 0
    peak_at = 0
    for index in range(last + 1):
        live = sum(tensor.nbytes for birth, death, tensor in births if birth <= index <= death)
        if live > peak:
            peak, peak_at = live, index
    resident = tuple(
        sorted(
            (tensor for birth, death, tensor in births if birth <= peak_at <= death),
            key=lambda t: t.nbytes,
            reverse=True,
        )
    )
    return peak, resident


def memory_finding(graph: Graph, limits: Limits) -> Finding | None:
    """Activations that occupy enough of the device to constrain the step."""
    peak, resident = live_peak(graph)
    if peak == 0 or peak < 0.25 * limits.hbm_capacity:
        return None
    top = resident[:3]
    return Finding(
        waste="memory",
        title=f"activations alone hold {peak / limits.hbm_capacity:.0%} of the device",
        evidence=(
            f"{peak / 2**30:.2f} GiB of {limits.hbm_capacity / 2**30:.0f} GiB live at the peak",
            *(
                f"{tensor.id} {list(tensor.shape)} {tensor.nbytes / 2**20:.1f} MiB"
                for tensor in top
            ),
        ),
        ceiling_us=None,
        ceiling_share=None,
        action="recompute the largest live activations in the backward pass instead of keeping them",
        requires=("a gradient comparison after the recompute change",),
    )


def detect(
    *,
    graph: Graph | None,
    limits: Limits,
    priced: Roofline,
    step: Timeline | None,
) -> tuple[Finding, ...]:
    """Every finding the captured material supports, largest ceiling first."""
    measured = priced.measured_us
    candidates: list[Finding | None] = []
    if graph is not None:
        candidates.append(fusion_finding(graph, limits, measured))
        candidates.append(precision_finding(priced, limits, measured))
        candidates.append(memory_finding(graph, limits))
    if step is not None:
        candidates.extend((schedule_finding(step), collective_finding(step), io_finding(step)))
    found = [finding for finding in candidates if finding is not None]
    return tuple(
        sorted(
            found, key=lambda finding: (finding.ceiling_us is None, -(finding.ceiling_us or 0.0))
        )
    )
