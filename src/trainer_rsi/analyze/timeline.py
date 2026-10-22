from itertools import pairwise
from statistics import median

from trainer_rsi.base import Value
from trainer_rsi.ir.trace import Event, Trace

type Interval = tuple[float, float]

GAP_US = 2.0
"""Idle time on a device queue below this is scheduling jitter, not a stall."""

COLLECTIVE_MARKERS = (
    "nccl",
    "allreduce",
    "allgather",
    "reducescatter",
    "alltoall",
    "broadcast",
    "sendrecv",
)


class Gap(Value):
    """An interval where the main device queue had nothing to run."""

    start_us: float
    dur_us: float
    stream: int


class Timeline(Value):
    """What device time the trace accounts for, per step, and how it overlapped.

    Every figure is per step: a trace of ten iterations reports one iteration's
    worth, so it can be compared with a bound computed for one iteration.
    """

    steps: int
    span_us: float
    busy_us: float
    """Time at least one queue was running work."""
    overlap_us: float
    """Time two or more queues were running at once, counted once per extra queue."""
    kernels: int
    mean_kernel_us: float
    median_kernel_us: float
    gaps: tuple[Gap, ...]
    gap_us: float
    collective_us: float
    exposed_collective_us: float
    copy_us: float
    exposed_copy_us: float
    syncs: int

    @property
    def idle_share(self) -> float:
        """Share of the span with no device work at all."""
        return 0.0 if self.span_us == 0 else 1.0 - self.busy_us / self.span_us


def merged(intervals: tuple[Interval, ...]) -> tuple[Interval, ...]:
    """Union of intervals, sorted and with touching ones joined."""
    if not intervals:
        return ()
    ordered = sorted(intervals)
    union: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = union[-1]
        if start <= last_end:
            union[-1] = (last_start, max(last_end, end))
        else:
            union.append((start, end))
    return tuple(union)


def covered(intervals: tuple[Interval, ...]) -> float:
    """Total length of the union of *intervals*."""
    return sum(end - start for start, end in merged(intervals))


def intersection(left: tuple[Interval, ...], right: tuple[Interval, ...]) -> float:
    """Total length two unions share."""
    first, second = merged(left), merged(right)
    total = 0.0
    i = j = 0
    while i < len(first) and j < len(second):
        a_start, a_end = first[i]
        b_start, b_end = second[j]
        total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return total


def is_collective(event: Event) -> bool:
    """Whether a device event is communication rather than computation."""
    lowered = event.name.lower()
    return any(marker in lowered for marker in COLLECTIVE_MARKERS)


def is_copy(event: Event) -> bool:
    """Whether a device event moves data across a device boundary."""
    return event.category in ("h2d", "d2h", "d2d")


def by_stream(events: tuple[Event, ...]) -> dict[int, tuple[Interval, ...]]:
    """Merge each device queue's work into intervals."""
    grouped: dict[int, list[Interval]] = {}
    for event in events:
        if event.stream is None:
            continue
        grouped.setdefault(event.stream, []).append((event.start_us, event.end_us))
    return {stream: merged(tuple(spans)) for stream, spans in grouped.items()}


def timeline(trace: Trace) -> Timeline:
    """Account for the trace's device time, per step."""
    events = trace.device_events
    streams = by_stream(events)
    spans = tuple((event.start_us, event.end_us) for event in events)
    busy = covered(spans)
    overlap = sum(
        intersection(streams[first], streams[second])
        for index, first in enumerate(streams)
        for second in list(streams)[index + 1 :]
    )
    main = max(streams, key=lambda stream: covered(streams[stream])) if streams else 0
    gaps: list[Gap] = []
    main_spans = streams.get(main, ())
    for (_, end), (next_start, _) in pairwise(main_spans):
        idle = next_start - end
        if idle > GAP_US:
            gaps.append(Gap(start_us=end, dur_us=idle, stream=main))
    collectives = tuple((event.start_us, event.end_us) for event in events if is_collective(event))
    compute = tuple(
        (event.start_us, event.end_us)
        for event in events
        if not is_collective(event) and not is_copy(event)
    )
    copies = tuple((event.start_us, event.end_us) for event in events if is_copy(event))
    kernels = [event.dur_us for event in events if event.category == "kernel"]
    steps = trace.steps
    return Timeline(
        steps=steps,
        span_us=trace.step_us / steps,
        busy_us=busy / steps,
        overlap_us=overlap / steps,
        kernels=len(kernels) // steps,
        mean_kernel_us=0.0 if not kernels else sum(kernels) / len(kernels),
        median_kernel_us=0.0 if not kernels else median(kernels),
        gaps=tuple(
            Gap(start_us=g.start_us, dur_us=g.dur_us / steps, stream=g.stream) for g in gaps
        ),
        gap_us=sum(gap.dur_us for gap in gaps) / steps,
        collective_us=covered(collectives) / steps,
        exposed_collective_us=(covered(collectives) - intersection(collectives, compute)) / steps,
        copy_us=covered(copies) / steps,
        exposed_copy_us=(covered(copies) - intersection(copies, compute)) / steps,
        syncs=sum(1 for event in trace.events if "synchronize" in event.name.lower()) // steps,
    )
