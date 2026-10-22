from typing import Literal

from trainer_rsi.base import Value

type Category = Literal["kernel", "memset", "h2d", "d2h", "d2d", "cpu_op", "annotation"]

DEVICE_CATEGORIES: frozenset[str] = frozenset({"kernel", "memset", "h2d", "d2h", "d2d"})


class Event(Value):
    """One profiler event, in profiler units (microseconds)."""

    name: str
    category: Category
    start_us: float
    dur_us: float
    stream: int | None = None
    """The device queue the work ran on; `None` for host-side events."""
    device: int = 0
    nbytes: int | None = None
    """Bytes moved, for a copy or memset; `None` where the profiler states none."""

    @property
    def end_us(self) -> float:
        """Timestamp the event finished."""
        return self.start_us + self.dur_us

    @property
    def on_device(self) -> bool:
        """Whether this event occupies a device queue."""
        return self.category in DEVICE_CATEGORIES


class Trace(Value):
    """The device-side history of one or more steps, as reported and nothing derived."""

    events: tuple[Event, ...]
    device_id: str
    """The hardware document this trace ran on, e.g. ``nvidia.h200_sxm``."""
    step_us: float
    """Wall time covered by the trace, so uncovered time can be read as a share."""
    source_digest: str = ""
    steps: int = 1
    """Iterations covered, so per-step figures divide honestly."""

    @property
    def device_events(self) -> tuple[Event, ...]:
        """Every event that occupies a device queue."""
        return tuple(event for event in self.events if event.on_device)

    @property
    def streams(self) -> tuple[int, ...]:
        """Distinct device queues the trace used, in ascending order."""
        return tuple(
            sorted({event.stream for event in self.device_events if event.stream is not None})
        )
