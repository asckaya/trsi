from math import ceil

from trainer_rsi.base import Value
from trainer_rsi.facts.document import HardwareDoc
from trainer_rsi.facts.envelope import Limits
from trainer_rsi.ir.dtype import DType

BLOCK_SIZES: tuple[int, ...] = (256, 128, 64)
"""Side lengths offered to a matmul tile, largest first."""

BLOCK_DEPTHS: tuple[int, ...] = (128, 64, 32)
"""Reduction depths offered per stage, deepest first."""

STAGES: tuple[int, ...] = (4, 3)
"""Pipeline depths offered, deepest first. Two stages expose the copy rather than hide it."""

MAX_BLOCKS_PER_SM = 32

ACCUMULATORS_PER_THREAD = 128
"""Accumulator elements one thread may hold. An FP32 accumulator costs a register
each, a thread has 255 in total, and the rest go to addresses and the pipeline; a
tile above this does not fail to launch, it fails to keep its accumulator in
registers, which is worse because it looks like it worked."""


class Geometry(Value):
    """A tile shape and the arithmetic that chose it.

    Every field is a decision the kernel layer owns, derived from the installed
    device document rather than from a template: what fits in shared memory, how
    many threads can be resident, and how many blocks the problem has to fill.
    """

    block_m: int
    block_n: int
    block_k: int
    stages: int
    threads: int
    smem_bytes: int
    blocks_per_sm: int
    waves: float
    """Residency rounds the problem needs: below one, the device is not filled."""
    roofline_us: float
    """Time this problem owes at the device's own rates, whatever the tile."""


def smem_of(block_m: int, block_n: int, block_k: int, stages: int, width: int) -> int:
    """Shared memory a staged tile holds: two buffers, one per operand, times the stages."""
    return (block_m * block_k + block_k * block_n) * width * stages


def tile_candidates(
    dtype: DType, doc: HardwareDoc, *, k: int | None = None
) -> tuple[Geometry, ...]:
    """Every tile that fits this device's shared memory with a real pipeline.

    A candidate is rejected when the staged buffers exceed the shared memory one
    block may hold, when its threads exceed what a block may hold, or when the
    accumulator would not fit in registers. Nothing is clamped to fit: a tile
    that does not fit is not a tile.

    Args:
        dtype: Precision the operands are stored in.
        doc: The device the tile has to fit.
        k: Reduction depth of the problem, which caps the per-stage depth. A
            problem shorter than the shallowest offered stage runs in one stage
            rather than being refused.
    """
    per_cta = doc.size("memory.shared.per_cta")
    per_sm = doc.size("memory.shared.per_sm")
    max_threads = doc.size("compute.max_threads_per_cta")
    width = dtype.nbytes(1)
    depths = (
        BLOCK_DEPTHS
        if k is None
        else tuple(sorted({min(depth, k) for depth in BLOCK_DEPTHS}, reverse=True))
    )
    found: list[Geometry] = []
    for block_m in BLOCK_SIZES:
        for block_n in BLOCK_SIZES:
            for block_k in depths:
                threads = 128 if block_m * block_n <= 128 * 128 else 256
                if threads > max_threads or block_m * block_n // threads > ACCUMULATORS_PER_THREAD:
                    continue
                for stages in STAGES:
                    smem = smem_of(block_m, block_n, block_k, stages, width)
                    if smem > per_cta:
                        continue
                    found.append(
                        Geometry(
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            stages=stages,
                            threads=threads,
                            smem_bytes=smem,
                            blocks_per_sm=max(1, min(per_sm // smem, MAX_BLOCKS_PER_SM)),
                            waves=0.0,
                            roofline_us=0.0,
                        )
                    )
    return tuple(found)


def matmul_geometry(
    *, m: int, n: int, k: int, dtype: DType, limits: Limits, doc: HardwareDoc
) -> Geometry:
    """The tile this device should run for an MxNxK matmul, and why.

    The choice maximises resident blocks times tile area, which is data reuse per
    launch, then prefers the square tile, then the deeper reduction. That is a
    heuristic, not a measurement: it is the geometry a `trsi drive` run then has
    to beat.

    Raises:
        ValueError: No tile fits the device's shared memory at this depth, so
            nothing is emitted rather than a kernel that cannot launch.
    """
    candidates = tile_candidates(dtype, doc, k=k)
    if not candidates:
        raise ValueError(
            f"no {dtype.value} tile fits {doc.id}: {doc.size('memory.shared.per_cta')} bytes of shared "
            f"memory per block, and the shallowest offered stage is {min(BLOCK_DEPTHS)} deep"
        )
    sm_count = doc.size("compute.sm_count")
    rate = limits.peak(dtype)
    compute_us = 0.0 if rate is None else 2 * m * n * k / rate * 1e6
    moved = (m * k + k * n + m * n) * dtype.nbytes(1)
    roofline_us = max(compute_us, moved / limits.hbm_bandwidth * 1e6)
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.blocks_per_sm * item.block_m * item.block_n,
            -abs(item.block_m - item.block_n),
            item.block_k,
            item.stages,
        ),
        reverse=True,
    )
    chosen = ranked[0]
    blocks = ceil(m / chosen.block_m) * ceil(n / chosen.block_n)
    return chosen.model_copy(
        update={
            "waves": blocks / (sm_count * chosen.blocks_per_sm),
            "roofline_us": roofline_us,
        }
    )
