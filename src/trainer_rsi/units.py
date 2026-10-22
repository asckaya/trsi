SYMBOLS: dict[str, str] = {
    "byte": "B",
    "byte/s": "B/s",
    "flop/s": "FLOP/s",
    "op/s": "op/s",
    "count": "",
    "s": "s",
    "Hz": "Hz",
}

PREFIXES: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
)


def quantity(value: float, unit: str) -> str:
    """A figure with the prefix that keeps it readable, e.g. ``989.5 TFLOP/s``.

    Decimal prefixes throughout, because that is how every installed document
    quotes its figures: a report that answers ``74.5 GiB`` to a datasheet saying
    ``80 GB`` makes the reader do arithmetic to check the tool.
    """
    symbol = SYMBOLS[unit]
    for scale, prefix in PREFIXES:
        if abs(value) >= scale:
            return f"{value / scale:g} {prefix}{symbol}".strip()
    return f"{value:g} {symbol}".strip()
