import ast
import importlib.util
from pathlib import Path

import pytest

from trainer_rsi.cli import main
from trainer_rsi.facts import limits, load
from trainer_rsi.facts.document import Fact, HardwareDoc
from trainer_rsi.ir.dtype import DType
from trainer_rsi.kernels import (
    matmul_geometry,
    matmul_kernel,
    tile_candidates,
    tile_target,
    vendor_of,
    write,
)
from trainer_rsi.kernels.emit import DTYPE_SPELLINGS
from trainer_rsi.kernels.geometry import ACCUMULATORS_PER_THREAD, smem_of


def fact(value: int, unit: str) -> Fact:
    """One installed figure, for a device document this test builds by hand."""
    return Fact(value=value, unit=unit, origin="vendor", source="test", conditions="test")


def cramped() -> HardwareDoc:
    """A device whose shared memory fits no offered tile."""
    return HardwareDoc(
        id="test.cramped",
        label="a device with no room",
        architecture="nvidia.sm90",
        facts={
            "compute.sm_count": fact(1, "count"),
            "compute.max_threads_per_cta": fact(1024, "count"),
            "memory.shared.per_cta": fact(1024, "byte"),
            "memory.shared.per_sm": fact(1024, "byte"),
        },
        digest="0" * 64,
        path="test.toml",
    )


@pytest.mark.parametrize(
    "device",
    [
        "nvidia.a100_sxm_80gb",
        "nvidia.h800_sxm",
        "nvidia.rtx_a6000",
        "nvidia.rtx_pro_6000_blackwell",
    ],
)
def test_every_installed_device_has_a_compiler_target(device: str):
    target = tile_target(load(device))
    assert target["kind"] == "cuda"
    assert target["arch"].startswith("sm_")


def test_an_unmapped_architecture_is_refused_with_what_is_known():
    unknown = cramped().model_copy(update={"architecture": "amd.gfx942"})
    with pytest.raises(KeyError, match="no target mapping"):
        tile_target(unknown)


def test_a_target_no_emitter_covers_is_refused():
    with pytest.raises(KeyError, match="no emitter for target kind 'hip'"):
        vendor_of({"kind": "hip", "mcpu": "gfx942"})


def test_the_chosen_tile_fits_the_shared_memory_it_was_measured_against():
    doc = load("nvidia.h800_sxm")
    geometry = matmul_geometry(
        m=8192, n=8192, k=8192, dtype=DType.BF16, limits=limits(doc), doc=doc
    )
    assert geometry.smem_bytes == smem_of(
        geometry.block_m, geometry.block_n, geometry.block_k, geometry.stages, DType.BF16.nbytes(1)
    )
    assert geometry.smem_bytes <= doc.size("memory.shared.per_cta")
    assert geometry.stages >= 3
    assert geometry.block_m * geometry.block_n // geometry.threads <= ACCUMULATORS_PER_THREAD
    assert geometry.waves > 0
    assert geometry.roofline_us == pytest.approx(2 * 8192**3 / 989.5e12 * 1e6, rel=1e-9)


def test_the_roofline_is_the_memory_side_when_the_problem_is_thin():
    doc = load("nvidia.h800_sxm")
    geometry = matmul_geometry(m=32, n=32, k=8, dtype=DType.BF16, limits=limits(doc), doc=doc)
    moved = (32 * 8 + 8 * 32 + 32 * 32) * 2
    assert geometry.roofline_us == pytest.approx(moved / 3.35e12 * 1e6, rel=1e-9)


def test_a_reduction_depth_deeper_than_the_problem_is_not_offered():
    doc = load("nvidia.h800_sxm")
    geometry = matmul_geometry(m=128, n=128, k=16, dtype=DType.BF16, limits=limits(doc), doc=doc)
    assert geometry.block_k <= 16


def test_a_device_with_no_room_refuses_rather_than_emitting_a_tile():
    with pytest.raises(ValueError, match=r"no bf16 tile fits test\.cramped"):
        matmul_geometry(
            m=4096,
            n=4096,
            k=4096,
            dtype=DType.BF16,
            limits=limits(load("nvidia.h800_sxm")),
            doc=cramped(),
        )


def test_candidates_never_exceed_memory_or_registers():
    doc = load("nvidia.a100_sxm_80gb")
    candidates = tile_candidates(DType.BF16, doc)
    assert candidates
    assert all(item.smem_bytes <= doc.size("memory.shared.per_cta") for item in candidates)
    assert all(
        item.block_m * item.block_n // item.threads <= ACCUMULATORS_PER_THREAD
        for item in candidates
    )


def test_the_emitted_source_is_valid_python_that_names_its_own_geometry():
    doc = load("nvidia.h800_sxm")
    kernel = matmul_kernel(m=4096, n=2048, k=1024, dtype=DType.BF16, limits=limits(doc), doc=doc)
    ast.parse(kernel.source)
    compile(kernel.source, kernel.name, "exec")
    geometry = kernel.geometry
    assert (
        f"BLOCK_M, BLOCK_N, BLOCK_K = {geometry.block_m}, {geometry.block_n}, {geometry.block_k}"
        in kernel.source
    )
    assert f"STAGES, THREADS = {geometry.stages}, {geometry.threads}" in kernel.source
    assert "'arch': 'sm_90'" in kernel.source
    assert "T.Pipelined" in kernel.source and "T.gemm" in kernel.source


def test_the_emitted_harness_refuses_to_invent_a_bound():
    doc = load("nvidia.rtx_a6000")
    kernel = matmul_kernel(m=512, n=512, k=512, dtype=DType.F16, limits=limits(doc), doc=doc)
    assert "required=True" in kernel.source
    assert "rel_l2" in kernel.source
    assert 'print(f"us=' in kernel.source


def test_a_dtype_with_no_spelling_is_refused_before_any_source_exists():
    doc = load("nvidia.h800_sxm")
    assert DType.I32 not in DTYPE_SPELLINGS
    with pytest.raises(KeyError):
        matmul_kernel(m=64, n=64, k=64, dtype=DType.I32, limits=limits(doc), doc=doc)


def test_the_emitted_kernel_is_accepted_by_the_real_compiler(tmp_path: Path):
    pytest.importorskip(
        "tilelang", reason="checking emitted source against tilelang: uv sync --group kernel"
    )

    doc = load("nvidia.h800_sxm")
    kernel = matmul_kernel(m=1024, n=1024, k=1024, dtype=DType.BF16, limits=limits(doc), doc=doc)
    path = write(kernel, tmp_path / "mm.py")
    spec = importlib.util.spec_from_file_location("emitted", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert type(module.matmul).__name__ == "JITImpl", (
        "the decorator returned something other than a kernel"
    )
    assert module.TARGET == {"kind": "cuda", "arch": "sm_90"}


def test_the_kernel_command_writes_the_source(tmp_path: Path, capsys):
    out = tmp_path / "mm.py"
    assert (
        main(
            [
                "kernel",
                "--device",
                "nvidia.h800_sxm",
                "--m",
                "1024",
                "--n",
                "1024",
                "--k",
                "1024",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert out.exists()
    assert "sm_90" in out.read_text()
    assert "wrote" in capsys.readouterr().out
