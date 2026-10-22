from pathlib import Path

import pytest
import torch
from torch import nn

from trainer_rsi import capture
from trainer_rsi.capture.aten import kind_of
from trainer_rsi.capture.graph import graph_of
from trainer_rsi.capture.kineto import trace_of
from trainer_rsi.cli import main
from trainer_rsi.ir.bundle import read_bundle
from trainer_rsi.ir.graph import Mesh


class Block(nn.Module):
    """A layer whose export contains an elementwise chain, a tuple and a join."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)
        self.down = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-mutable-override]
        hidden = self.up(x)
        hidden, mean, _rstd = torch.native_layer_norm(
            hidden, [128], self.norm.weight, self.norm.bias, 1e-5
        )
        hidden = hidden * torch.sigmoid(hidden)
        return torch.cat([self.down(hidden), self.down(hidden)], dim=0).sum(dim=0), mean


@pytest.fixture
def block() -> tuple[Block, torch.Tensor]:
    """A module on CPU with one example input."""
    torch.manual_seed(0)
    return Block(), torch.randn(8, 64)


def chrome(device: bool) -> dict[str, object]:
    """A chrome trace shaped like kineto's, with or without device events."""
    events: list[dict[str, object]] = [
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 100.0, "dur": 5.0},
        {"ph": "M", "name": "process_name", "args": {"name": "x"}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0.0, "dur": 1.0},
    ]
    if device:
        events += [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm",
                "ts": 1000.0,
                "dur": 30.0,
                "args": {"stream": 7, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm",
                "ts": 1040.0,
                "dur": 20.0,
                "args": {"stream": 7, "device": 0},
            },
            {
                "ph": "X",
                "cat": "Memcpy HtoD",
                "name": "Memcpy HtoD",
                "ts": 1030.0,
                "dur": 10.0,
                "args": {"stream": 3, "device": 0, "bytes": 4096},
            },
            {
                "ph": "X",
                "cat": "gpu_memset",
                "name": "memset",
                "ts": 1070.0,
                "dur": 5.0,
                "args": {"stream": 7, "device": 0},
            },
        ]
    return {"traceEvents": events}


def test_kind_table_covers_the_shapes_a_transformer_trains_with():
    assert kind_of("aten.linear.default") == "matmul"
    assert kind_of("aten.scaled_dot_product_attention.default") == "attention"
    assert kind_of("aten.native_layer_norm.default") == "norm"
    assert kind_of("aten.add.Tensor") == "elementwise"
    assert kind_of("aten.copy_.default") == "movement"
    assert kind_of("aten.slice.Tensor") == "view"
    assert kind_of("custom::mystery") == "other"


def test_trace_keeps_device_events_and_rebases_them_to_zero():
    trace = trace_of(chrome(device=True), device_id="nvidia.h800_sxm", steps=2)
    assert trace is not None
    assert trace.device_id == "nvidia.h800_sxm"
    starts = [event.start_us for event in trace.device_events]
    assert min(starts) == 0.0
    assert trace.step_us == 75.0
    assert trace.streams == (3, 7)
    assert {event.category for event in trace.events if not event.on_device} == {"cpu_op"}


def test_trace_of_host_work_alone_is_none():
    assert trace_of(chrome(device=False), device_id="nvidia.h800_sxm", steps=1) is None


def test_graph_prices_an_exported_block(block):
    module, example = block
    program = torch.export.export(module, (example,))
    graph = graph_of(program, mesh=Mesh(dp=2), step="train_step")
    kinds = {op.target: op.kind for op in graph.ops}
    assert kinds["aten.linear.default"] == "matmul"
    assert kinds["aten.native_layer_norm.default"] == "norm"
    assert kinds["operator.getitem"] == "view"
    assert graph.mesh.dp == 2
    assert graph.by_id["p_up_weight"].role == "parameter"
    assert graph.by_id["x"].role == "input"
    assert graph.by_id["x"].shape == (8, 64)
    assert graph.producers()
    assert all(op.inputs for op in graph.ops if op.kind not in ("constant",))


def test_tuple_outputs_and_getitem_resolve_to_distinct_tensors(block):
    module, example = block
    graph = graph_of(torch.export.export(module, (example,)), mesh=Mesh(), step="s")
    norm = next(op for op in graph.ops if op.kind == "norm")
    assert len(norm.outputs) == 3
    getitems = [op for op in graph.ops if op.target == "operator.getitem"]
    assert getitems
    assert {op.inputs[0] for op in getitems} <= set(norm.outputs)
    assert all(op.kind == "view" for op in getitems)


def test_record_on_host_work_writes_a_graph_only_bundle(tmp_path: Path, block):
    module, example = block
    out = tmp_path / "graph.bundle.json.gz"
    with capture.record(
        device_id="nvidia.h800_sxm",
        out=out,
        step="train_step",
        steps=2,
        module=module,
        example=(example,),
        activities=("cpu",),
    ):
        for _ in range(2):
            module(example)
    bundle = read_bundle(out)
    assert bundle.graph is not None
    assert bundle.trace is None
    assert bundle.capture.graph_source == "torch.export"
    assert any("no device event" in note for note in bundle.notes)
    assert bundle.environment.device_name == "cpu"


def test_record_refuses_an_export_without_examples(tmp_path: Path, block):
    module, _ = block
    with (
        pytest.raises(ValueError, match="both a module and its example inputs"),
        capture.record(
            device_id="nvidia.h800_sxm",
            out=tmp_path / "x.json.gz",
            module=module,
            activities=("cpu",),
        ),
    ):
        pass


def test_record_refuses_cuda_activities_without_a_device(tmp_path: Path):
    if torch.cuda.is_available():
        pytest.skip("this machine has an accelerator")
    with (
        pytest.raises(RuntimeError, match="no accelerator is visible"),
        capture.record(device_id="nvidia.h800_sxm", out=tmp_path / "x.json.gz"),
    ):
        pass


def test_report_renders_a_bundle_whose_trace_is_missing(tmp_path: Path, block, capsys):
    module, example = block
    out = tmp_path / "graph.bundle.json.gz"
    with capture.record(
        device_id="nvidia.h800_sxm",
        out=out,
        module=module,
        example=(example,),
        activities=("cpu",),
    ):
        module(example)
    assert main(["analyze", str(out)]) == 0
    assert "no trace" in capsys.readouterr().out
