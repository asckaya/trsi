import json
from pathlib import Path

from trainer_rsi.ir.trace import Category, Event, Trace

CATEGORIES: dict[str, Category] = {
    "kernel": "kernel",
    "gpu_memset": "memset",
    "Memcpy HtoD": "h2d",
    "Memcpy DtoH": "d2h",
    "Memcpy DtoD": "d2d",
    "Memcpy Device to Device": "d2d",
    "Memcpy Device to Host": "d2h",
    "Memcpy Host to Device": "h2d",
    "cpu_op": "cpu_op",
    "gpu_user_annotation": "annotation",
    "user_annotation": "annotation",
}


def number(raw: dict[str, object], key: str, default: float) -> float:
    """One numeric field of a trace entry.

    Raises:
        TypeError: The field is present and is not a number.
    """
    value = raw.get(key, default)
    if not isinstance(value, int | float):
        raise TypeError(f"trace field {key!r} is {type(value).__name__}, not a number")
    return float(value)


def integer(raw: dict[str, object], key: str, default: int) -> int:
    """One integer field of a trace entry, refusing a non-integer."""
    return int(number(raw, key, default))


def event_of(raw: dict[str, object]) -> Event | None:
    """One chrome-trace entry as an event, or `None` for categories this ignores.

    Chrome traces carry flow arrows, counters and metadata frames beside the
    complete events; only complete events describe work that took time.
    """
    if raw.get("ph") != "X":
        return None
    category = CATEGORIES.get(str(raw.get("cat")))
    if category is None:
        return None
    args = raw.get("args") or {}
    assert isinstance(args, dict)
    return Event(
        name=str(raw.get("name", "")),
        category=category,
        start_us=number(raw, "ts", 0.0),
        dur_us=number(raw, "dur", 0.0),
        stream=None if args.get("stream") is None else integer(args, "stream", 0),
        device=integer(args, "device", 0),
        nbytes=None if args.get("bytes") is None else integer(args, "bytes", 0),
    )


def trace_of(payload: dict[str, object], *, device_id: str, steps: int) -> Trace | None:
    """Build a trace from a chrome-trace payload, rebased to start at zero.

    Args:
        payload: The `export_chrome_trace` output.
        device_id: The hardware document the devices are.
        steps: How many iterations the payload covers.

    Returns:
        The trace, with every timestamp relative to the first device event, or
        `None` when the payload holds no device event. A capture of host work
        alone has no device timeline, and a timeline of nothing would still
        claim a device.
    """
    raw_events = payload.get("traceEvents", [])
    if not isinstance(raw_events, list):
        raise ValueError("traceEvents is not a list")
    events = tuple(
        event
        for event in (event_of(raw) for raw in raw_events if isinstance(raw, dict))
        if event is not None
    )
    device_events = tuple(event for event in events if event.on_device)
    if not device_events:
        return None
    origin = min(event.start_us for event in device_events)
    span = max(event.end_us for event in device_events) - origin
    return Trace(
        events=tuple(
            event.model_copy(update={"start_us": event.start_us - origin})
            if event.on_device
            else event
            for event in events
        ),
        device_id=device_id,
        step_us=span,
        steps=steps,
    )


def read_chrome(path: Path, *, device_id: str, steps: int) -> Trace | None:
    """Read a chrome trace from disk."""
    return trace_of(json.loads(path.read_text(encoding="utf-8")), device_id=device_id, steps=steps)
