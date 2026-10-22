import gzip
import json
from pathlib import Path

import pytest

from trainer_rsi.ir.bundle import read_bundle, write_bundle
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph, Op


def test_dtype_resolves_both_profiler_spellings():
    assert DType.of_torch("torch.bfloat16") is DType.BF16
    assert DType.of_torch("c10::BFloat16") is DType.BF16
    assert DType.of_torch("float") is DType.F32
    assert DType.of_torch("torch.float8_e4m3fn") is DType.FP8E4M3


def test_dtype_refuses_an_unknown_name():
    with pytest.raises(KeyError, match="unknown dtype"):
        DType.of_torch("torch.complex128")


def test_sub_byte_width_packs_into_whole_bytes():
    assert DType.F4E2M1.nbytes(3) == 2
    assert DType.BF16.nbytes(3) == 6


def test_bundle_round_trips_through_disk(tmp_path: Path, bundle):
    path = tmp_path / "x.bundle.json.gz"
    digest = write_bundle(path, bundle)
    loaded = read_bundle(path)
    assert loaded.digest() == bundle.digest()
    assert loaded.graph == bundle.graph
    assert loaded.trace == bundle.trace
    assert len(digest) == 64


def test_bundle_refuses_a_foreign_schema(tmp_path: Path, bundle):
    path = tmp_path / "x.bundle.json.gz"
    tampered = bundle.model_dump()
    tampered["schema_version"] = "trainer_rsi.bundle/v0"
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(tampered).encode())
    with pytest.raises(ValueError, match="re-capture"):
        read_bundle(path)


def test_graph_refuses_two_producers_for_one_tensor(bundle):
    first, second = bundle.graph.ops[0], bundle.graph.ops[1]
    duplicated = Op(
        id="clone",
        target="aten.clone.default",
        kind="movement",
        phase="forward",
        inputs=("h",),
        outputs=("h",),
    )
    graph = Graph(tensors=bundle.graph.tensors, ops=(first, second, duplicated))
    with pytest.raises(ValueError, match=f"written by {first.id!r}"):
        graph.producers()
