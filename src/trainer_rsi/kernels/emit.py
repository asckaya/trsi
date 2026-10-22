from pathlib import Path

from trainer_rsi.base import Value
from trainer_rsi.facts.document import HardwareDoc
from trainer_rsi.facts.envelope import Limits
from trainer_rsi.ir.dtype import DType
from trainer_rsi.kernels.geometry import Geometry, matmul_geometry
from trainer_rsi.kernels.target import TileTarget, tile_target, vendor_of

DTYPE_SPELLINGS: dict[DType, str] = {
    DType.BF16: "bfloat16",
    DType.F16: "float16",
    DType.F32: "float32",
    DType.FP8E4M3: "float8_e4m3fn",
    DType.FP8E5M2: "float8_e5m2",
}
"""The dtype names the kernel compiler takes, for the types worth emitting today."""

PROGRAM = '''"""A tile-level matmul for {device}, emitted by trsi.

Geometry chosen from the installed device document ({digest12}), not from a
template: {block_m}x{block_n} tiles, {block_k}-deep stages, {stages} stages in
flight, {threads} threads, {smem} bytes of staged shared memory, {blocks} block(s)
resident per SM.

The problem is {m}x{n}x{k} in {dtype}: {waves:.1f} residency rounds, and a
roofline of {roofline:.1f} us on this device.

This file is a starting point, not a result. Run it, read `--rtol`, and let
`trsi drive` decide whether the geometry was right.
"""

import argparse
import sys

import tilelang
import tilelang.language as T

DEVICE = {device!r}
TARGET = {target!r}
M, N, K = {m}, {n}, {k}
DTYPE = {dtype!r}
ACCUM = "float"
BLOCK_M, BLOCK_N, BLOCK_K = {block_m}, {block_n}, {block_k}
STAGES, THREADS = {stages}, {threads}


@tilelang.jit(target=TARGET)
def matmul(A, B, block_M: int, block_N: int, block_K: int, dtype: str, accum_dtype: str, num_stages: int):
    """One tiled matmul: staged copies, a tensor-core gemm, and the accumulator written out."""
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=THREADS) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def main(argv: list[str]) -> int:
    """Check against the reference, then time the kernel, printing one `us=` per iteration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtol", type=float, required=True, help="relative bound you require of this kernel")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=getattr(torch, DTYPE))
    b = torch.randn(K, N, device="cuda", dtype=getattr(torch, DTYPE))
    kernel = matmul(a, b, BLOCK_M, BLOCK_N, BLOCK_K, DTYPE, ACCUM, STAGES)
    kernel(a, b)

    produced = kernel(a, b).float()
    expected = (a.float() @ b.float())
    scale = expected.norm()
    if float(scale) < 1e-12:
        print("reference norm is near zero", file=sys.stderr)
        return 1
    error = float((produced - expected).norm() / scale)
    print(f"rel_l2={{error:.3e}} vs required {{args.rtol:.1e}}")
    if error > args.rtol:
        print("FAIL: the kernel does not meet the bound you stated", file=sys.stderr)
        return 1

    for _ in range(args.warmup):
        kernel(a, b)
    torch.cuda.synchronize()
    for _ in range(args.reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel(a, b)
        end.record()
        torch.cuda.synchronize()
        print(f"us={{start.elapsed_time(end) * 1000.0:.2f}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


class Kernel(Value):
    """One emitted kernel: its source, and the geometry it was emitted for."""

    name: str
    device_id: str
    target: TileTarget
    dtype: DType
    m: int
    n: int
    k: int
    geometry: Geometry
    source: str


def matmul_kernel(
    *, m: int, n: int, k: int, dtype: DType, limits: Limits, doc: HardwareDoc
) -> Kernel:
    """Emit the tile-level matmul this device should run for this problem.

    Args:
        m: Rows of the output.
        n: Columns of the output.
        k: Reduction depth.
        dtype: Precision the inputs are stored in.
        limits: The device's rates, which price the roofline printed in the file.
        doc: The installed document the geometry is derived from.

    Returns:
        The kernel, with its source and the geometry behind it.

    Raises:
        KeyError: No spelling exists for this dtype, so no source is emitted.
        UnsupportedArchitectureError: The device has no target mapping.
        ValueError: No tile fits the device's shared memory.
    """
    vendor_of(tile_target(doc))
    geometry = matmul_geometry(m=m, n=n, k=k, dtype=dtype, limits=limits, doc=doc)
    source = PROGRAM.format(
        device=doc.id,
        digest12=doc.digest[:12],
        target=tile_target(doc),
        m=m,
        n=n,
        k=k,
        dtype=DTYPE_SPELLINGS[dtype],
        block_m=geometry.block_m,
        block_n=geometry.block_n,
        block_k=geometry.block_k,
        stages=geometry.stages,
        threads=geometry.threads,
        smem=geometry.smem_bytes,
        blocks=geometry.blocks_per_sm,
        waves=geometry.waves,
        roofline=geometry.roofline_us,
    )
    return Kernel(
        name=f"matmul_{doc.id.replace('.', '_')}_{m}x{n}x{k}_{dtype.value}",
        device_id=doc.id,
        target=tile_target(doc),
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        geometry=geometry,
        source=source,
    )


def write(kernel: Kernel, path: Path) -> Path:
    """Write the kernel source to *path*, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kernel.source, encoding="utf-8")
    return path
