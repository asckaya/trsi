from pathlib import Path

import pytest

from trainer_rsi.facts import installed, limits, load
from trainer_rsi.facts.document import Fact, flatten_facts, load_document
from trainer_rsi.ir.dtype import DType


def test_every_installed_document_loads_and_matches_its_filename():
    docs = installed()
    assert set(docs) == {
        "nvidia.a100_sxm_80gb",
        "nvidia.h800_sxm",
        "nvidia.rtx_a6000",
        "nvidia.rtx_pro_6000_blackwell",
    }
    for doc in docs.values():
        assert Path(doc.path).exists()
        assert len(doc.digest) == 64


def test_a100_limits_project_the_rates_the_bound_divides_by():
    a100 = limits(load("nvidia.a100_sxm_80gb"))
    assert a100.peak(DType.BF16) == 312e12
    assert a100.peak(DType.FP8E4M3) is None
    assert a100.hbm_bandwidth == 2039e9
    assert a100.nvlink is not None
    assert a100.nvlink.bytes_per_second == 300e9
    assert a100.fastest() == (DType.BF16, 312e12)


def test_link_choice_follows_the_group_size():
    h800 = limits(load("nvidia.h800_sxm"))
    near = h800.link(8)
    assert near is not None
    assert near.kind == "nvlink"
    assert h800.link(9) is None


def test_a_card_without_nvlink_links_through_pcie():
    pro = limits(load("nvidia.rtx_pro_6000_blackwell"))
    assert pro.nvlink is None
    pcie = pro.link(2)
    assert pcie is not None
    assert pcie.kind == "pcie"
    assert pro.peak(DType.BF16) == 500e12


def test_unavailable_is_not_the_same_as_absent():
    a6000 = load("nvidia.rtx_a6000")
    assert a6000.fact("interconnect.nvlink.bandwidth").status == "unavailable"
    with pytest.raises(KeyError, match="unavailable"):
        a6000.amount("interconnect.nvlink.bandwidth")
    with pytest.raises(KeyError, match="has no fact"):
        a6000.fact("throughput.f64")


def test_fact_refuses_an_inconsistent_state():
    with pytest.raises(ValueError, match="status=unavailable"):
        Fact(
            value=1, unit="byte", origin="vendor", source="s", conditions="c", status="unavailable"
        )


def test_flatten_refuses_a_leaf_that_is_not_a_fact():
    with pytest.raises(TypeError, match="not a fact table"):
        flatten_facts({"compute": {"sm_count": 132}}, "")


def test_documents_carry_provenance_for_every_figure():
    for doc in installed().values():
        for path, fact in doc.facts.items():
            assert fact.source.startswith("http"), f"{doc.id}.{path} cites no source"
            assert fact.conditions, f"{doc.id}.{path} states no conditions"


def test_load_refuses_an_unknown_device():
    with pytest.raises(KeyError, match="no hardware document"):
        load("nvidia.rtx_5090")


def test_load_refuses_a_document_with_the_wrong_schema(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text('[spec]\nschema = "nope"\nkind = "device"\nid = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_document(path)
