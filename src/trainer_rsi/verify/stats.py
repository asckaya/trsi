from math import ceil
from random import Random
from statistics import median, stdev
from typing import Literal

from trainer_rsi.base import Value

MIN_SAMPLES = 5
"""Fewer than this is not a distribution, and a claim from it is noise."""

RESAMPLES = 2000
CLAIM_LEVEL = 0.95


class Samples(Value):
    """Timings of one program, in microseconds, in the order they were taken."""

    name: str
    us: tuple[float, ...]

    @property
    def count(self) -> int:
        """How many timings were taken."""
        return len(self.us)

    @property
    def median_us(self) -> float:
        """The middle timing, which is what a step costs most of the time."""
        return median(self.us)

    @property
    def spread_us(self) -> float:
        """Standard deviation, which says whether the median is worth quoting."""
        return 0.0 if len(self.us) < 2 else stdev(self.us)

    def percentile(self, fraction: float) -> float:
        """One percentile by nearest rank, so a p10 is a measurement and not an interpolation."""
        if not self.us:
            raise ValueError(f"{self.name} has no samples")
        ordered = sorted(self.us)
        rank = max(0, min(len(ordered) - 1, ceil(fraction * len(ordered)) - 1))
        return ordered[rank]


class Comparison(Value):
    """Two distributions and the only claim their ratio supports."""

    before: Samples
    after: Samples
    speedup: float
    """Ratio of medians: above one means the change made the step faster."""
    low: float
    high: float
    resamples: int
    claim: Literal["faster", "slower", "indistinguishable"]


def bootstrap_speedup(
    before: Samples, after: Samples, resamples: int, seed: int
) -> tuple[float, float]:
    """The interval of the median-ratio under resampling, at the claim level.

    Resampling is what keeps a two percent difference from being reported as a
    win: the interval widens with the noise the run actually showed, and the
    claim is read off the interval rather than off the point estimate.
    """
    random = Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        drawn_before = [before.us[random.randrange(before.count)] for _ in range(before.count)]
        drawn_after = [after.us[random.randrange(after.count)] for _ in range(after.count)]
        ratios.append(median(drawn_before) / median(drawn_after))
    ratios.sort()
    tail = (1.0 - CLAIM_LEVEL) / 2.0
    return ratios[round(tail * resamples)], ratios[round((1.0 - tail) * resamples) - 1]


def compare(
    before: Samples, after: Samples, *, resamples: int = RESAMPLES, seed: int = 0
) -> Comparison:
    """Compare two timing sets and state what their difference supports.

    Raises:
        ValueError: Either side has fewer than `MIN_SAMPLES` timings.
    """
    short = [side.name for side in (before, after) if side.count < MIN_SAMPLES]
    if short:
        raise ValueError(
            f"{', '.join(short)}: fewer than {MIN_SAMPLES} samples cannot support a claim"
        )
    speedup = before.median_us / after.median_us
    low, high = bootstrap_speedup(before, after, resamples, seed)
    if low > 1.0:
        claim: Literal["faster", "slower", "indistinguishable"] = "faster"
    elif high < 1.0:
        claim = "slower"
    else:
        claim = "indistinguishable"
    return Comparison(
        before=before,
        after=after,
        speedup=speedup,
        low=low,
        high=high,
        resamples=resamples,
        claim=claim,
    )
