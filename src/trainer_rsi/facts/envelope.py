from dataclasses import dataclass
from typing import Literal

from trainer_rsi.facts.document import HardwareDoc
from trainer_rsi.ir.dtype import DType

PREFERENCE: tuple[DType, ...] = (
    DType.F4E2M1,
    DType.FP8E4M3,
    DType.FP8E5M2,
    DType.BF16,
    DType.F16,
    DType.TF32,
    DType.F32,
    DType.F64,
)
"""Which of the equally fast dtypes to name first, narrowest useful precision."""

PCIE_DOMAIN = 8
"""Widest single-host PCIe group this project prices, not a hardware claim.

A wider group needs a scale-out fabric, and no document here describes one, so
a collective over it is reported unpriced rather than priced at a link it does
not use.
"""


@dataclass(frozen=True, slots=True)
class Link:
    """One interconnect a collective can cross, at the rate a ring is limited by."""

    kind: Literal["nvlink", "pcie"]
    bytes_per_second: float
    """Per-direction rate: what a ring collective pays, not the bidirectional sum."""
    domain: int
    """Largest group this link joins, in devices."""


@dataclass(frozen=True, slots=True)
class Limits:
    """The rates a bound divides by, projected once from one device document."""

    device_id: str
    label: str
    digest: str
    peak_flops: dict[DType, float]
    hbm_bandwidth: float
    hbm_capacity: int
    nvlink: Link | None
    pcie: Link

    def peak(self, dtype: DType) -> float | None:
        """Rate for one arithmetic dtype, or `None` when the document states none."""
        return self.peak_flops.get(dtype)

    def fastest(self) -> tuple[DType, float]:
        """The dtype with the highest stated rate, and that rate."""
        return max(self.peak_flops.items(), key=lambda item: (item[1], -PREFERENCE.index(item[0])))

    def link(self, group: int) -> Link | None:
        """The link a collective over *group* devices crosses, or `None` if unpriced."""
        if self.nvlink is not None and group <= self.nvlink.domain:
            return self.nvlink
        if group <= self.pcie.domain:
            return self.pcie
        return None


def limits(doc: HardwareDoc) -> Limits:
    """Project one hardware document into the rates the analysis divides by.

    Args:
        doc: An installed document.

    Returns:
        The rates, with absent dtypes and links left out rather than filled in.

    Raises:
        KeyError: The document states neither an HBM bandwidth nor a capacity.
    """
    peaks: dict[DType, float] = {}
    for dtype in DType:
        path = f"throughput.{dtype.value}"
        if path in doc.facts:
            rate = doc.maybe(path)
            if rate is not None:
                peaks[dtype] = rate
    nvlink = doc.maybe("interconnect.nvlink.bandwidth")
    return Limits(
        device_id=doc.id,
        label=doc.label,
        digest=doc.digest,
        peak_flops=peaks,
        hbm_bandwidth=doc.amount("memory.hbm.bandwidth"),
        hbm_capacity=doc.size("memory.hbm.capacity"),
        nvlink=None
        if nvlink is None
        else Link(
            kind="nvlink",
            bytes_per_second=nvlink / 2,
            domain=doc.size("interconnect.nvlink.domain"),
        ),
        pcie=Link(
            kind="pcie",
            bytes_per_second=doc.amount("interconnect.pcie.bandwidth") / 2,
            domain=PCIE_DOMAIN,
        ),
    )
