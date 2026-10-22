from typing import Literal

from trainer_rsi.analyze.work import Work, price
from trainer_rsi.base import Value
from trainer_rsi.facts.envelope import Limits
from trainer_rsi.ir.graph import Op, Phase, Tensor
from trainer_rsi.ir.trace import Trace

type Bound = Literal["compute", "memory", "balanced", "unrated"]


class PricedOp(Value):
    """One op against the machine's rates: what it owes, and what set the amount."""

    op: Op
    work: Work
    compute_us: float
    memory_us: float
    bound_us: float
    bound_by: Bound


class Roofline(Value):
    """The modelled floor of a captured step, and what it left out.

    Two aggregates, because one number would hide the difference:

    - `ideal_us` sums each op's own bound: the floor for this program with every
      kernel at its own roofline and no fusion.
    - `floor_us` is the larger of total arithmetic and total traffic over the
      whole step: the floor for *any* schedule of this arithmetic. It is
      unreachable in general, because not everything fuses.

    A report quoting only the first cannot say whether structure is the problem;
    one quoting only the second blames the program for physics.
    """

    device_id: str
    ops: tuple[PricedOp, ...]
    unpriced: tuple[tuple[Op, Work], ...]
    flops_total: int
    bytes_total: int
    ideal_us: float
    floor_us: float
    measured_us: float | None
    """Measured time for the same scope, when a trace covers it."""
    coverage: float
    """Share of modelled traffic carried by ops that were priced."""

    def by_phase(self) -> dict[Phase, float]:
        """Ideal microseconds per phase, so a report can say where the floor sits."""
        totals: dict[Phase, float] = {}
        for priced in self.ops:
            totals[priced.op.phase] = totals.get(priced.op.phase, 0.0) + priced.bound_us
        return totals


def microseconds(amount: int, rate: float) -> float:
    """Time *amount* units take at *rate* per second, in microseconds."""
    return amount / rate * 1e6


def roofline(
    ops: tuple[Op, ...], tensors: dict[str, Tensor], limits: Limits, trace: Trace | None
) -> Roofline:
    """Bound every op against one device and aggregate the result.

    An op whose arithmetic the device publishes no rate for is bounded by memory
    alone and marked `unrated`. Its bound is a floor, not a target: calling it
    anything else would credit the device with a rate it never stated.
    """
    priced: list[PricedOp] = []
    unpriced: list[tuple[Op, Work]] = []
    flops_total = 0
    bytes_total = 0
    priced_bytes = 0
    for op in ops:
        work = price(op, tensors)
        if not work.priced:
            unpriced.append((op, work))
            continue
        rate = limits.peak(work.dtype)
        compute_us = 0.0 if rate is None or work.flops == 0 else microseconds(work.flops, rate)
        memory_us = microseconds(work.bytes_total, limits.hbm_bandwidth)
        bound = max(compute_us, memory_us)
        if work.flops and rate is None:
            kind: Bound = "unrated"
        elif compute_us > memory_us:
            kind = "compute"
        elif memory_us > compute_us:
            kind = "memory"
        else:
            kind = "balanced"
        priced.append(
            PricedOp(
                op=op,
                work=work,
                compute_us=compute_us * op.scale,
                memory_us=memory_us * op.scale,
                bound_us=bound * op.scale,
                bound_by=kind,
            )
        )
        flops_total += work.flops * op.scale
        bytes_total += work.bytes_total * op.scale
        priced_bytes += work.bytes_total * op.scale
    arithmetic_floor = (
        microseconds(flops_total, max(limits.peak_flops.values())) if flops_total else 0.0
    )
    coverage_total = bytes_total + sum(work.bytes_total * op.scale for op, work in unpriced)
    return Roofline(
        device_id=limits.device_id,
        ops=tuple(priced),
        unpriced=tuple(unpriced),
        flops_total=flops_total,
        bytes_total=bytes_total,
        ideal_us=sum(item.bound_us for item in priced),
        floor_us=max(arithmetic_floor, microseconds(bytes_total, limits.hbm_bandwidth)),
        measured_us=None if trace is None else trace.step_us,
        coverage=1.0 if coverage_total == 0 else priced_bytes / coverage_total,
    )
