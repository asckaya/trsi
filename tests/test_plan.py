import json

import pytest

from trainer_rsi.analyze import build
from trainer_rsi.analyze.timeline import timeline
from trainer_rsi.cli import main
from trainer_rsi.facts import limits, load
from trainer_rsi.ir.bundle import new_bundle, write_bundle
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph
from trainer_rsi.plan import Plan, candidate_plans, combination
from trainer_rsi.plan.report import summary


def priced_of(bundle):
    report = build(bundle)
    return report, report.roofline


def plans_of(bundle):
    report, priced = priced_of(bundle)
    assert priced is not None
    step = timeline(bundle.trace) if bundle.trace is not None else None
    return candidate_plans(
        graph=bundle.graph,
        limits=limits(load(report.device_id)),
        priced=priced,
        step=step,
    )


def test_fusion_plan_keeps_endpoint_traffic_and_drops_the_intermediate(bundle):
    plans = plans_of(bundle)
    fusion = next(plan for plan in plans if plan.id == "fuse:norm+act")
    traffic = bundle.graph.by_id["h"].nbytes + bundle.graph.by_id["g"].nbytes
    assert fusion.now_us == pytest.approx(4.1, abs=0.1)
    assert fusion.then_us == pytest.approx(traffic / 2039e9 * 1e6, rel=1e-6)
    assert fusion.delta_us == pytest.approx(fusion.now_us - fusion.then_us)
    assert "n (2.00 MiB)" in fusion.change


def test_a_fusion_that_would_be_slower_is_not_offered(bundle):
    plans = plans_of(bundle)
    assert all(
        plan.now_us is None or plan.then_us is None or plan.now_us > plan.then_us for plan in plans
    )


def test_plans_are_ranked_by_what_they_save(bundle):
    plans = plans_of(bundle)
    deltas = [plan.delta_us or 0.0 for plan in plans]
    assert deltas == sorted(deltas, reverse=True)
    assert plans[0].kind == "overlap"


def test_precision_plan_appears_for_wide_arithmetic(bundle):
    graph = bundle.graph
    wide = tuple(
        tensor.model_copy(update={"dtype": DType.F32}) if tensor.role != "parameter" else tensor
        for tensor in graph.tensors
    )
    fp32 = Graph(tensors=wide, ops=graph.ops, mesh=graph.mesh, step=graph.step)
    report = build(new_bundle(environment=bundle.environment, capture=bundle.capture, graph=fp32))
    assert report.roofline is not None
    plans = candidate_plans(
        graph=fp32, limits=limits(load(report.device_id)), priced=report.roofline, step=None
    )
    precision = [plan for plan in plans if plan.kind == "precision"]
    assert precision
    assert "in bf16 instead of f32" in precision[0].title
    assert (precision[0].then_us or 0.0) <= (precision[0].now_us or 0.0)


def test_a_combination_never_touches_the_same_op_twice(bundle):
    plans = plans_of(bundle)
    taken = combination(plans)
    arithmetic = [plan for plan in taken if plan.kind not in ("capture", "overlap")]
    touched = [op for plan in arithmetic for op in plan.touches]
    assert len(touched) == len(set(touched))
    total = sum(plan.delta_us or 0.0 for plan in taken)
    assert total <= sum(plan.delta_us or 0.0 for plan in plans)


def test_summary_refuses_plans_that_predict_under_the_floor():
    plan = Plan(
        id="overlap:collective",
        kind="overlap",
        title="split the collective",
        touches=(),
        now_us=90.0,
        then_us=0.0,
        delta_us=90.0,
        delta_share=0.9,
        change="cover it",
        requires=("a bit-identical parameter update",),
    )
    text = summary((plan,), measured_us=100.0, floor_us=50.0).plain
    assert "not all of them can hold" in text
    assert "below the 50.0 us floor" in text


def test_summary_keeps_quiet_when_the_prediction_stays_above_the_floor(bundle):
    plans = plans_of(bundle)
    report = build(bundle)
    assert report.roofline is not None
    text = summary(plans, report.measured_us, report.roofline.floor_us).plain
    assert "cannot hold" not in text
    assert "disjoint plans" in text


def test_plan_without_a_graph_is_refused(bundle, tmp_path, capsys):
    trace_only = new_bundle(
        environment=bundle.environment, capture=bundle.capture, trace=bundle.trace
    )
    path = tmp_path / "trace.bundle.json.gz"
    write_bundle(path, trace_only)
    assert main(["plan", str(path)]) == 1
    assert "no graph" in capsys.readouterr().err


def test_plan_command_writes_json(bundle, tmp_path):
    bundle_path = tmp_path / "step.bundle.json.gz"
    write_bundle(bundle_path, bundle)
    out = tmp_path / "plans.json"
    assert main(["plan", str(bundle_path), "--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["plans"][0]["kind"] == "overlap"
    assert payload["predicted_saving_us"] > 0
    assert payload["combination"][0] == "overlap:collective"
