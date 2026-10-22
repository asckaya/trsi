from typing import Literal

from trainer_rsi.facts.document import HardwareDoc

type TileTarget = dict[str, str]
"""A target in the form the kernel compiler takes it: a kind and the machine's own name."""

ARCHITECTURES: dict[str, TileTarget] = {
    "nvidia.sm80": {"kind": "cuda", "arch": "sm_80"},
    "nvidia.sm86": {"kind": "cuda", "arch": "sm_86"},
    "nvidia.sm90": {"kind": "cuda", "arch": "sm_90"},
    "nvidia.sm120": {"kind": "cuda", "arch": "sm_120"},
}
"""Architectures this build can emit for, mapped to the compiler's own spelling.

This table is the portability boundary, and it is deliberately a table rather
than a rule: adding a vendor means naming its architecture strings, not editing
a branch inside the emitter.
"""


class UnsupportedArchitectureError(KeyError):
    """The installed document names an architecture no emitter here covers."""


def tile_target(doc: HardwareDoc) -> TileTarget:
    """The compiler target for one installed device.

    Raises:
        UnsupportedArchitectureError: The document's architecture has no mapping,
            listing the ones that do.
    """
    try:
        return ARCHITECTURES[doc.architecture]
    except KeyError:
        raise UnsupportedArchitectureError(
            f"{doc.id} is {doc.architecture}, which has no target mapping; known: {sorted(ARCHITECTURES)}"
        ) from None


type Vendor = Literal["cuda"]
"""The only compiler backend this build emits for. A second one is a new mapping,
not a new branch: the emitter reads the target, never the device."""


def vendor_of(target: TileTarget) -> Vendor:
    """Which backend emits for a target.

    Raises:
        UnsupportedArchitectureError: The target's kind has no emitter here.
    """
    kind = target["kind"]
    if kind != "cuda":
        raise UnsupportedArchitectureError(f"no emitter for target kind {kind!r}")
    return "cuda"
