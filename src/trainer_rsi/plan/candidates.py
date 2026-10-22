from typing import Literal

from trainer_rsi.analyze.roofline import PricedOp, Roofline, microseconds
from trainer_rsi.analyze.timeline import Timeline
from trainer_rsi.base import Value
from trainer_rsi.facts.envelope import Limits
from trainer_rsi.ir.graph import Graph

type PlanKind = Literal["fuse", "precision", "capture", "overlap"]

FUSIBLE = frozenset({"elementwise", "norm", "softmax", "reduce", "matmul"})
"""Kinds that can share a kernel with a neighbour.

A matmul belongs here because its epilogue is where a norm or an activation
usually lands; a collective does not, because it cannot share an instruction
stream with arithmetic.
"""

SKIPPED = frozenset({"collective", "view", "constant"})
MATMUL_LIKE = frozenset({"matmul", "attention", "conv"})


class Plan(Value):
    """One rewrite, priced against the same device as the program it rewrites.

    A plan is not advice: it is a replacement for a named set of ops, and
    `delta_us` is what the cost model says the replacement would owe instead of
    what the ops owe now. Nothing here has been run, which is why `requires` is
    part of the record rather than a note beside it.
    """

    id: str
    kind: PlanKind
    title: str
    touches: tuple[str, ...]
    """Graph op ids this replaces or changes, which is also what makes two plans conflict."""
    now_us: float | None
    """What the touched ops owe at their own rooflines today."""
    then_us: float | None
    """What the replacement would owe, by the same rules."""
    delta_us: float | None
    delta_share: float | None
    change: str
    requires: tuple[str, ...]

    @property
    def gain(self) -> float:
        """Predicted saving, zero when there is none to state."""
        return 0.0 if self.delta_us is None else self.delta_us


def plan_id(kind: PlanKind, touches: tuple[str, ...]) -> str:
    """A stable name for one rewrite, derived from what it touches."""
    return f"{kind}:{'+'.join(touches)}"


def percent(delta_us: float | None, measured_us: float | None) -> float | None:
    """A saving as a share of the measured step."""
    if delta_us is None or measured_us in (None, 0):
        return None
    return delta_us / measured_us


def bound_us(limits: Limits, flops: int, bytes_total: int, dtype_rate: float) -> float:
    """The roofline time of one arithmetic amount at one rate and one bandwidth."""
    compute = 0.0 if dtype_rate == 0 or flops == 0 else microseconds(flops, dtype_rate)
    return max(compute, microseconds(bytes_total, limits.hbm_bandwidth))


def fusion_plans(graph: Graph, limits: Limits, priced: Roofline) -> list[Plan]:
    """Every adjacent pair that could become one kernel, priced as one kernel.

    The fused bound keeps the producer's inputs and the consumer's output, which
    is the traffic that remains when the intermediate stops being a kernel
    boundary. Pairs that would not be faster are not reported.
    """
    by_id = {item.op.id: item for item in priced.ops}
    producers = graph.producers()
    consumers = graph.consumers()
    outputs = {tensor.id for tensor in graph.tensors if tensor.role == "output"}
    plans: list[Plan] = []
    for tensor in graph.tensors:
        producer = producers.get(tensor.id)
        readers = consumers.get(tensor.id, ())
        if producer is None or len(readers) != 1 or tensor.id in outputs:
            continue
        consumer = readers[0]
        if producer.kind in SKIPPED or consumer.kind in SKIPPED:
            continue
        if producer.kind not in FUSIBLE and consumer.kind not in FUSIBLE:
            continue
        before, after = by_id.get(producer.id), by_id.get(consumer.id)
        if before is None or after is None:
            continue
        now = before.bound_us + after.bound_us
        then = bound_us(
            limits,
            before.work.flops + after.work.flops,
            before.work.bytes_in + after.work.bytes_out,
            arithmetic_rate(limits, before, after),
        )
        if now <= then:
            continue
        plans.append(
            Plan(
                id=plan_id("fuse", (producer.id, consumer.id)),
                kind="fuse",
                title=f"fuse {producer.target} into {consumer.target}",
                touches=(producer.id, consumer.id),
                now_us=now,
                then_us=then,
                delta_us=max(0.0, now - then),
                delta_share=percent(max(0.0, now - then), priced.measured_us),
                change=f"drop {tensor.id} ({tensor.nbytes / 2**20:.2f} MiB) as a kernel boundary",
                requires=(
                    "per-output comparison against the current program at a declared bound",
                    "the fused kernel's own measurement on this device",
                ),
            )
        )
    return plans


def arithmetic_rate(limits: Limits, first: PricedOp, second: PricedOp) -> float:
    """The rate a fused kernel issues at, taken from the member doing the arithmetic.

    One kernel runs one instruction stream, so the member with the most flops
    picks the precision; taking the slower of the two would charge a matmul with
    the rate of the elementwise op beside it, which is the wrong direction.

    Raises:
        KeyError: Neither dtype has a stated rate, so the pair cannot be priced.
    """
    heavier, lighter = (first, second) if first.work.flops >= second.work.flops else (second, first)
    rate = limits.peak(heavier.work.dtype) or limits.peak(lighter.work.dtype)
    if rate is None:
        raise KeyError(
            f"neither {first.work.dtype} nor {second.work.dtype} has a rate on {limits.device_id}"
        )
    return rate


def precision_plans(limits: Limits, priced: Roofline) -> list[Plan]:
    """Compute-bound arithmetic re-issued at the fastest rate the device states.

    A re-issued op can become memory bound, and when it does the plan says so by
    pricing the new time as the larger of the two sides rather than assuming the
    rate applies to the whole duration.
    """
    fastest_dtype, fastest_rate = limits.fastest()
    plans: list[Plan] = []
    for item in priced.ops:
        if item.bound_by != "compute" or item.op.kind not in MATMUL_LIKE:
            continue
        rate = limits.peak(item.work.dtype)
        if rate is None or rate >= fastest_rate:
            continue
        then = bound_us(limits, item.work.flops, item.work.bytes_total, fastest_rate)
        if item.bound_us <= then:
            continue
        plans.append(
            Plan(
                id=plan_id("precision", (item.op.id,)),
                kind="precision",
                title=f"issue {item.op.target} in {fastest_dtype.value} instead of {item.work.dtype.value}",
                touches=(item.op.id,),
                now_us=item.bound_us,
                then_us=then,
                delta_us=item.bound_us - then,
                delta_share=percent(item.bound_us - then, priced.measured_us),
                change=f"{item.op.module_path or item.op.id} becomes "
                f"{'memory' if then > microseconds(item.work.flops, fastest_rate) else 'compute'} bound",
                requires=(
                    f"a declared numeric bound per output, measured at {fastest_dtype.value}",
                    "a gradient comparison over a random projection, for a training step",
                ),
            )
        )
    return plans


def capture_plan(step: Timeline, measured_us: float | None) -> Plan | None:
    """Wrap the step in one graph capture, priced at the idle time it removes.

    A capture changes how the step is launched, not what it computes, so it
    touches no op and composes with every arithmetic rewrite.
    """
    if step.gap_us <= 0:
        return None
    return Plan(
        id=plan_id("capture", ("step",)),
        kind="capture",
        title="capture the step into one graph",
        touches=(),
        now_us=step.gap_us,
        then_us=0.0,
        delta_us=step.gap_us,
        delta_share=percent(step.gap_us, measured_us),
        change=f"remove {len(step.gaps)} launch gaps totalling {step.gap_us:.1f} us per step",
        requires=(
            "a re-measurement of the same step, since capture changes allocator behaviour",
            "the same outputs from the captured step as from the eager one",
        ),
    )


def overlap_plan(step: Timeline, measured_us: float | None) -> Plan | None:
    """Chunk the exposed collective and issue each piece ahead of its consumer."""
    if step.exposed_collective_us <= 0:
        return None
    return Plan(
        id=plan_id("overlap", ("collective",)),
        kind="overlap",
        title="split the collective so computation covers it",
        touches=(),
        now_us=step.exposed_collective_us,
        then_us=0.0,
        delta_us=step.exposed_collective_us,
        delta_share=percent(step.exposed_collective_us, measured_us),
        change=f"cover {step.exposed_collective_us:.1f} us of the {step.collective_us:.1f} us spent communicating",
        requires=(
            "a bit-identical parameter update across the reordered collective",
            "a re-measurement showing the collective still finishes before its consumer",
        ),
    )


def candidate_plans(
    *, graph: Graph | None, limits: Limits, priced: Roofline, step: Timeline | None
) -> tuple[Plan, ...]:
    """Every rewrite this capture supports, largest predicted saving first."""
    plans = list(precision_plans(limits, priced))
    if graph is not None:
        plans.extend(fusion_plans(graph, limits, priced))
    if step is not None:
        plans.extend(
            plan
            for plan in (
                capture_plan(step, priced.measured_us),
                overlap_plan(step, priced.measured_us),
            )
            if plan
        )
    return tuple(sorted(plans, key=lambda plan: plan.delta_us or 0.0, reverse=True))


def combination(plans: tuple[Plan, ...]) -> tuple[Plan, ...]:
    """A set of plans that do not touch the same op, taken greedily by saving.

    Two rewrites of the same op are not additive: pricing both and adding them
    counts the same microseconds twice. Taking a disjoint set cannot.
    """
    taken: list[Plan] = []
    used: set[str] = set()
    for plan in plans:
        if plan.kind in ("capture", "overlap"):
            taken.append(plan)
            continue
        if used.intersection(plan.touches):
            continue
        used.update(plan.touches)
        taken.append(plan)
    return tuple(taken)
