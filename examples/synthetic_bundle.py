"""Build a capture bundle without a GPU, so the report can be read anywhere.

Run it from the repo root:

    .venv/bin/python examples/synthetic_bundle.py
    .venv/bin/trsi analyze examples/synthetic.bundle.json.gz

The graph is a two-projection block on an A100: the numbers are the ones the
cost rules produce from those shapes, and the timeline is a hand-written device
history with an idle gap, an all-reduce that nothing covers, and a copy that
lands on the critical path. It exists to show what a bundle is made of.
"""

from pathlib import Path

from trainer_rsi.ir.bundle import Bundle, Capture, Environment, new_bundle, write_bundle
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph, Mesh, Op, Role, Tensor
from trainer_rsi.ir.trace import Event, Trace

DEVICE = "nvidia.a100_sxm_80gb"
OUT = Path(__file__).parent / "synthetic.bundle.json.gz"


def graph() -> Graph:
    """A block whose every intermediate has exactly one consumer."""
    shapes: dict[str, tuple[tuple[int, ...], DType, Role]] = {
        "x": ((256, 1024), DType.BF16, "input"),
        "w1": ((4096, 1024), DType.BF16, "parameter"),
        "h": ((256, 4096), DType.BF16, "activation"),
        "n": ((256, 4096), DType.BF16, "activation"),
        "g": ((256, 4096), DType.BF16, "activation"),
        "w2": ((1024, 4096), DType.BF16, "parameter"),
        "y": ((256, 1024), DType.BF16, "activation"),
        "loss": ((1,), DType.F32, "output"),
    }
    tensors = tuple(
        Tensor(id=name, shape=shape, dtype=dtype, role=role)
        for name, (shape, dtype, role) in shapes.items()
    )
    ops = (
        Op(
            id="up",
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
            id="down",
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
        ),
    )
    return Graph(tensors=tensors, ops=ops, mesh=Mesh(dp=8), step="train_step")


def trace() -> Trace:
    """One step of device history: 300 us of compute, 200 us that is not covered."""
    events = (
        Event(name="ampere_sgemm_128x128", category="kernel", start_us=0.0, dur_us=120.0, stream=0),
        Event(
            name="vectorized_elementwise_kernel",
            category="kernel",
            start_us=130.0,
            dur_us=90.0,
            stream=0,
        ),
        Event(
            name="ampere_sgemm_128x128", category="kernel", start_us=230.0, dur_us=140.0, stream=0
        ),
        Event(
            name="ncclDevKernel_AllReduce_LL",
            category="kernel",
            start_us=150.0,
            dur_us=60.0,
            stream=7,
        ),
        Event(
            name="ncclDevKernel_AllReduce_LL",
            category="kernel",
            start_us=420.0,
            dur_us=90.0,
            stream=7,
        ),
        Event(
            name="Memcpy HtoD", category="h2d", start_us=0.0, dur_us=40.0, stream=3, nbytes=1 << 20
        ),
        Event(
            name="Memcpy HtoD",
            category="h2d",
            start_us=520.0,
            dur_us=60.0,
            stream=3,
            nbytes=1 << 20,
        ),
        Event(name="cudaDeviceSynchronize", category="cpu_op", start_us=580.0, dur_us=3.0),
    )
    return Trace(events=events, device_id=DEVICE, step_us=600.0)


def main() -> None:
    """Write the bundle beside this file."""
    bundle: Bundle = new_bundle(
        environment=Environment(
            device_id=DEVICE,
            python="3.14.8",
            torch="2.14.1",
            device_name="NVIDIA A100-SXM4-80GB",
            torch_cuda="12.8",
            driver="570.00",
            git_commit="synthetic",
        ),
        capture=Capture(
            entry="examples/synthetic_bundle.py",
            iters=1,
            warmup=0,
            activities=("cpu", "cuda"),
            graph_source="manual",
        ),
        graph=graph(),
        trace=trace(),
        notes=("hand-written: this bundle is an example, not a measurement",),
    )
    digest = write_bundle(OUT, bundle)
    print(f"wrote {OUT}  ({digest[:16]})")
    print(f"read it with: trsi analyze {OUT}")


if __name__ == "__main__":
    main()
