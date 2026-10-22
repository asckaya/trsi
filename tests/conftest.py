from pathlib import Path

import pytest

from trainer_rsi.ir.bundle import Bundle, Capture, Environment, new_bundle, write_bundle
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph, Mesh, Op, Role, Tensor
from trainer_rsi.ir.trace import Event, Trace

DEVICE = "nvidia.a100_sxm_80gb"


def tensor(name: str, shape: tuple[int, ...], dtype: DType, role: Role) -> Tensor:
    """One graph tensor with the fields the analysis reads."""
    return Tensor(id=name, shape=shape, dtype=dtype, role=role)


def tiny_graph() -> Graph:
    """A two-projection block whose intermediates each have a single consumer."""
    tensors = (
        tensor("x", (256, 1024), DType.BF16, "input"),
        tensor("w1", (4096, 1024), DType.BF16, "parameter"),
        tensor("h", (256, 4096), DType.BF16, "activation"),
        tensor("n", (256, 4096), DType.BF16, "activation"),
        tensor("g", (256, 4096), DType.BF16, "activation"),
        tensor("w2", (1024, 4096), DType.BF16, "parameter"),
        tensor("y", (256, 1024), DType.BF16, "activation"),
        tensor("loss", (1,), DType.F32, "output"),
    )
    ops = (
        Op(
            id="mm1",
            target="aten.mm.default",
            kind="matmul",
            phase="forward",
            inputs=("x", "w1"),
            outputs=("h",),
            module_path="block.up",
        ),
        Op(
            id="norm",
            target="aten.rms_norm.default",
            kind="norm",
            phase="forward",
            inputs=("h",),
            outputs=("n",),
            module_path="block.norm",
        ),
        Op(
            id="act",
            target="aten.gelu.default",
            kind="elementwise",
            phase="forward",
            inputs=("n",),
            outputs=("g",),
            module_path="block.act",
        ),
        Op(
            id="mm2",
            target="aten.mm.default",
            kind="matmul",
            phase="forward",
            inputs=("g", "w2"),
            outputs=("y",),
            module_path="block.down",
        ),
        Op(
            id="loss",
            target="aten.sum.default",
            kind="reduce",
            phase="forward",
            inputs=("y",),
            outputs=("loss",),
            module_path="",
        ),
    )
    return Graph(tensors=tensors, ops=ops, mesh=Mesh(dp=8), step="train_step")


def tiny_trace() -> Trace:
    """Two compute stretches with an idle gap, an exposed collective, and an exposed copy."""
    events = (
        Event(name="ampere_sgemm_128x128", category="kernel", start_us=0.0, dur_us=100.0, stream=0),
        Event(
            name="ampere_sgemm_128x128", category="kernel", start_us=110.0, dur_us=90.0, stream=0
        ),
        Event(name="elementwise_kernel", category="kernel", start_us=200.0, dur_us=100.0, stream=0),
        Event(name="Memcpy HtoD", category="h2d", start_us=0.0, dur_us=50.0, stream=3, nbytes=4096),
        Event(
            name="Memcpy HtoD", category="h2d", start_us=430.0, dur_us=50.0, stream=3, nbytes=4096
        ),
        Event(
            name="ncclDevKernel_AllReduce", category="kernel", start_us=120.0, dur_us=70.0, stream=7
        ),
        Event(
            name="ncclDevKernel_AllReduce",
            category="kernel",
            start_us=320.0,
            dur_us=100.0,
            stream=7,
        ),
        Event(name="cudaDeviceSynchronize", category="cpu_op", start_us=420.0, dur_us=2.0),
    )
    return Trace(events=events, device_id=DEVICE, step_us=500.0)


@pytest.fixture
def bundle() -> Bundle:
    """A bundle carrying both halves of a capture."""
    return new_bundle(
        environment=Environment(
            device_id=DEVICE,
            python="3.14.8",
            torch="2.14.1",
            device_name="NVIDIA A100-SXM4-80GB",
            git_commit="deadbeef",
        ),
        capture=Capture(
            entry="train.py --steps 2",
            iters=2,
            warmup=1,
            activities=("cpu", "cuda"),
            graph_source="torch.export",
        ),
        graph=tiny_graph(),
        trace=tiny_trace(),
    )


@pytest.fixture
def bundle_path(tmp_path: Path, bundle: Bundle) -> Path:
    """The same bundle on disk, in the format capture writes."""
    path = tmp_path / "step.bundle.json.gz"
    write_bundle(path, bundle)
    return path
