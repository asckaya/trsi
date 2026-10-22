import pytest

from trainer_rsi.analyze import build
from trainer_rsi.analyze.gaps import memory_finding
from trainer_rsi.analyze.roofline import roofline
from trainer_rsi.analyze.timeline import timeline
from trainer_rsi.analyze.work import price
from trainer_rsi.facts import limits, load
from trainer_rsi.ir.bundle import new_bundle
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph, Mesh, Op, Tensor


def test_matmul_work_is_two_flops_per_multiply_add(bundle):
    graph = bundle.graph
    work = price(graph.ops[0], graph.by_id)
    assert work.flops == 2 * 256 * 4096 * 1024
    assert work.bytes_in == graph.by_id["x"].nbytes + graph.by_id["w1"].nbytes
    assert work.bytes_out == graph.by_id["h"].nbytes
    assert work.dtype is DType.BF16
    assert work.priced


def test_unpriced_ops_are_kept_and_lower_coverage(bundle):
    graph = bundle.graph
    exotic = Op(
        id="exotic",
        target="custom::fused_thing",
        kind="other",
        phase="forward",
        inputs=("y",),
        outputs=("loss",),
    )
    with_exotic = Graph(tensors=graph.tensors, ops=(*graph.ops[:-1], exotic), mesh=graph.mesh)
    priced = roofline(
        with_exotic.ops, with_exotic.by_id, limits(load("nvidia.a100_sxm_80gb")), None
    )
    assert priced.coverage < 1.0
    assert [op.target for op, _ in priced.unpriced] == ["custom::fused_thing"]
    assert priced.unpriced[0][1].priced is False


def test_roofline_names_the_binding_side(bundle):
    graph = bundle.graph
    priced = roofline(graph.ops, graph.by_id, limits(load("nvidia.a100_sxm_80gb")), None)
    bounds = {item.op.id: item.bound_by for item in priced.ops}
    assert bounds["mm1"] == "compute"
    assert bounds["norm"] == "memory"
    assert priced.ideal_us == pytest.approx(18.1, abs=0.5)
    assert priced.floor_us < priced.ideal_us
    assert priced.measured_us is None


def test_timeline_accounts_for_idle_overlap_and_exposed_work(bundle):
    step = timeline(bundle.trace)
    assert step.busy_us == 440.0
    assert step.overlap_us == 120.0
    assert step.gap_us == 10.0
    assert step.collective_us == 170.0
    assert step.exposed_collective_us == 100.0
    assert step.copy_us == 100.0
    assert step.exposed_copy_us == 50.0
    assert step.kernels == 5
    assert step.syncs == 1


def test_findings_rank_by_what_they_could_be_worth(bundle):
    report = build(bundle)
    order = [finding.waste for finding in report.findings]
    assert order == ["parallel", "io", "schedule", "structure"]
    ceilings = [finding.ceiling_us or 0.0 for finding in report.findings]
    assert ceilings == sorted(ceilings, reverse=True)
    structure = report.findings[-1]
    assert structure.ceiling_us == pytest.approx(2 * 3 * 2 * 2**20 / 2039e9 * 1e6, rel=1e-6)
    assert "12.0 MiB" in structure.evidence[-1]


def test_every_finding_states_what_must_be_verified_first(bundle):
    report = build(bundle)
    assert all(finding.requires for finding in report.findings if finding.waste != "memory")


def test_memory_finding_stays_quiet_when_activations_fit(bundle):
    a100 = limits(load("nvidia.a100_sxm_80gb"))
    assert memory_finding(bundle.graph, a100) is None


def test_memory_finding_speaks_when_activations_dominate(bundle):
    giant = Tensor(id="giant", shape=(1, 20_000_000_000), dtype=DType.BF16, role="activation")
    graph = Graph(
        tensors=(*bundle.graph.tensors, giant),
        ops=(
            *bundle.graph.ops,
            Op(
                id="blow",
                target="aten.mul.default",
                kind="elementwise",
                phase="forward",
                inputs=("y",),
                outputs=("giant",),
            ),
        ),
        mesh=Mesh(),
    )
    finding = memory_finding(graph, limits(load("nvidia.a100_sxm_80gb")))
    assert finding is not None
    assert finding.ceiling_us is None
    assert "giant" in finding.evidence[1]


def test_detect_without_a_trace_reports_only_what_the_graph_supports(bundle):
    graph_only = new_bundle(
        environment=bundle.environment, capture=bundle.capture, graph=bundle.graph
    )
    report = build(graph_only)
    assert report.timeline is None
    assert [finding.waste for finding in report.findings] == ["structure"]
