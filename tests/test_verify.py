import json

import numpy as np
import pytest

from trainer_rsi.cli import main
from trainer_rsi.verify import AllClose, Exact, RelL2, Samples, compare, judge


def test_a_clear_difference_is_claimed_as_faster():
    before = Samples(name="before", us=tuple([100.0, 101.0, 99.0, 100.5, 100.2, 99.8]))
    after = Samples(name="after", us=tuple([80.0, 81.0, 79.5, 80.4, 80.1, 79.9]))
    comparison = compare(before, after)
    assert comparison.claim == "faster"
    assert comparison.speedup == pytest.approx(1.25, rel=0.01)
    assert comparison.low > 1.0


def test_a_change_that_does_not_move_the_number_is_not_a_win():
    before = Samples(name="before", us=tuple([100.0, 101.0, 99.0, 100.5, 100.2, 99.8]))
    after = Samples(name="after", us=tuple([100.4, 99.6, 100.1, 100.3, 99.9, 100.2]))
    comparison = compare(before, after)
    assert comparison.claim == "indistinguishable"
    assert comparison.low < 1.0 < comparison.high


def test_a_regression_is_reported_as_one():
    before = Samples(name="before", us=(80.0, 81.0, 79.5, 80.4, 80.1))
    after = Samples(name="after", us=(100.0, 101.0, 99.0, 100.5, 100.2))
    assert compare(before, after).claim == "slower"


def test_a_claim_needs_a_distribution():
    before = Samples(name="before", us=(100.0, 101.0))
    after = Samples(name="after", us=(80.0, 81.0))
    with pytest.raises(ValueError, match="fewer than 5 samples"):
        compare(before, after)


def test_percentiles_are_measurements_not_interpolations():
    samples = Samples(name="s", us=tuple(float(value) for value in range(1, 11)))
    assert samples.percentile(0.0) == 1.0
    assert samples.percentile(1.0) == 10.0
    assert samples.percentile(0.5) in (5.0, 6.0)


def test_compare_command_reads_two_timing_files(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({"name": "eager", "us": [100.0, 101.0, 99.5, 100.2, 100.4]}))
    after.write_text(json.dumps({"name": "cudagraph", "us": [80.0, 80.5, 79.8, 80.2, 80.1]}))
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out
    assert "faster" in out
    assert "over" in out


def test_every_output_needs_a_bound():
    produced = {"output": np.zeros(3)}
    with pytest.raises(ValueError, match="no bounds were stated"):
        judge(produced, {})


def test_an_unjudged_output_cannot_pass():
    produced = {"output": np.zeros(3), "output[1]": np.zeros(3)}
    with pytest.raises(ValueError, match=r"outputs left unjudged: \['output\[1\]'\]"):
        judge(produced, {"output": RelL2(bound=1e-3)}, {"output": np.zeros(3)})


def test_a_bound_naming_an_absent_output_is_refused():
    with pytest.raises(ValueError, match="were not produced"):
        judge(
            {"output": np.zeros(3)},
            {"output": Exact(), "output[1]": Exact()},
            {"output": np.zeros(3)},
        )


def test_a_two_sided_bound_needs_a_reference():
    with pytest.raises(ValueError, match="need one"):
        judge({"output": np.ones(3)}, {"output": AllClose(atol=0.0, rtol=0.0)})


def test_allclose_reports_the_worst_element_not_the_average():
    reference = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    candidate = np.array([1.0, 1.0, 1.5], dtype=np.float32)
    verdict = judge(
        {"output": candidate}, {"output": AllClose(atol=0.0, rtol=0.0)}, {"output": reference}
    )
    assert not verdict[0].passed
    assert verdict[0].metric == pytest.approx(0.5)
    assert verdict[0].reference_norm == pytest.approx(np.sqrt(3.0))


def test_relative_error_against_a_stated_bound():
    reference = np.array([1.0, 2.0, 3.0])
    candidate = reference + 1e-6
    verdict = judge({"output": candidate}, {"output": RelL2(bound=1e-3)}, {"output": reference})[0]
    assert verdict.passed
    assert verdict.metric < 1e-3


def test_a_reference_near_zero_refuses_a_relative_measure():
    with pytest.raises(ValueError, match="too near zero"):
        judge({"output": np.zeros(3)}, {"output": RelL2(bound=1e-3)}, {"output": np.zeros(3)})


def test_an_aggregate_on_a_discrete_output_is_refused():
    indices = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError, match="discrete output"):
        judge({"index": indices}, {"index": RelL2(bound=1.0)}, {"index": indices})


def test_exact_comparison_needs_no_reference_but_does_compare():
    indices = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError, match="needs a reference"):
        judge({"index": indices}, {"index": Exact()})
    verdict = judge({"index": indices}, {"index": Exact()}, {"index": indices.copy()})[0]
    assert verdict.passed
    wrong = judge({"index": indices}, {"index": Exact()}, {"index": indices + 1})[0]
    assert not wrong.passed
    assert wrong.metric == 4.0


def test_nothing_produced_is_not_a_pass():
    with pytest.raises(ValueError, match="nothing was produced"):
        judge({}, {"output": Exact()}, {})
