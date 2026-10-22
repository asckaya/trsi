import json
import sys
from pathlib import Path

import pytest

from trainer_rsi.harness import (
    Claim,
    CodexRunner,
    JsonObject,
    Report,
    Task,
    Turn,
    audit,
    build,
    drive,
    environment,
    measure,
    overrides,
    parse_samples,
)
from trainer_rsi.harness.loop import workspace_copy
from trainer_rsi.harness.measure import gate
from trainer_rsi.harness.runner import Runner
from trainer_rsi.verify import Samples, compare

PYTHON = sys.executable


def repo(tmp_path: Path, microseconds: float = 100.0) -> Path:
    """A repository whose step costs whatever its bench script prints."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "bench.py").write_text(f'for _ in range(5):\n    print("us={microseconds}")\n')
    (root / "step.py").write_text("def step():\n    return 1\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return root


def task(root: Path, *, gate_command: str = "", attempts: int = 1) -> Task:
    """One task against a repository, measured by its bench script."""
    return Task(
        repo=str(root),
        device="nvidia.h800_sxm",
        entry="python step.py",
        measure=f"{PYTHON} bench.py",
        gate=gate_command,
        warmup=0,
        reps=1,
        attempts=attempts,
    )


class FakeRunner(Runner):
    """An agent that makes one edit and answers with one report."""

    def __init__(self, report: Report | None, edit: float | None = None) -> None:
        self.report = report
        self.edit = edit
        self.prompts: list[str] = []

    def run(self, prompt: str, *, cwd: Path, output_schema: JsonObject | None = None) -> Turn:
        """Rewrite the bench script to the edited cost, then answer."""
        assert output_schema is None or "properties" in output_schema
        self.prompts.append(prompt)
        if self.edit is not None:
            (cwd / "bench.py").write_text(f'for _ in range(5):\n    print("us={self.edit}")\n')
        if self.report is None:
            return Turn(response="I looked at it and it seems fine.")
        payload = self.report.model_dump(mode="json")
        return Turn(response=json.dumps(payload), structured=payload)


def speed_report(
    speedup: float | None, statement: str = "the fused epilogue removes a pass"
) -> Report:
    """A report claiming one speed win."""
    return Report(
        summary="fused two ops",
        claims=(Claim(kind="speed", statement=statement, speedup=speedup),),
    )


def test_samples_are_read_from_us_lines():
    assert parse_samples("noise\nus=12.5\nus=13\n") == (12.5, 13.0)


def test_a_measure_command_that_prints_no_sample_is_refused():
    with pytest.raises(ValueError, match="no `us="):
        parse_samples("took a while\n")


def test_measure_runs_warmup_then_keeps_the_samples(tmp_path: Path):
    root = repo(tmp_path)
    samples = measure(task(root), root, tmp_path / "before.json")
    assert samples.count == 5
    assert samples.median_us == 100.0
    assert json.loads((tmp_path / "before.json").read_text())["us"] == [100.0] * 5


def test_a_failing_measure_command_is_an_error_not_a_sample(tmp_path: Path):
    root = repo(tmp_path)
    (root / "bench.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(RuntimeError, match="exited 3"):
        measure(task(root), root, tmp_path / "x.json")


def test_the_gate_reports_its_exit_code(tmp_path: Path):
    root = repo(tmp_path)
    assert gate(task(root), root) is None
    failed = gate(task(root, gate_command=f"{PYTHON} -c 'raise SystemExit(2)'"), root)
    assert failed is not None
    assert failed.exit_code == 2


def test_a_claim_that_matches_the_measurement_is_supported(tmp_path: Path):
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(80.0,) * 5)
    result = audit(
        speed_report(1.25), workspace=tmp_path, comparison=compare(before, after), gate=None
    )
    assert result.accepted
    assert result.verified[0].verdict == "supported"
    assert result.findings == ()


def test_a_claim_the_samples_do_not_share_is_contradicted(tmp_path: Path):
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(80.0,) * 5)
    result = audit(
        speed_report(1.9), workspace=tmp_path, comparison=compare(before, after), gate=None
    )
    assert not result.accepted
    assert result.verified[0].verdict == "contradicted"
    assert "claimed x1.900, measured x1.250" in result.findings[0]


def test_a_speed_claim_with_no_ratio_is_unbacked(tmp_path: Path):
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(80.0,) * 5)
    result = audit(
        speed_report(None), workspace=tmp_path, comparison=compare(before, after), gate=None
    )
    assert result.verified[0].verdict == "unbacked"
    assert not result.accepted


def test_a_change_the_samples_cannot_separate_is_not_accepted(tmp_path: Path):
    before = Samples(name="before", us=(100.0, 100.1, 99.9, 100.05, 99.95))
    after = Samples(name="after", us=(100.02, 99.98, 100.01, 100.0, 99.99))
    result = audit(
        speed_report(1.0), workspace=tmp_path, comparison=compare(before, after), gate=None
    )
    assert not result.accepted
    assert result.verified[0].verdict == "indistinguishable"


def test_a_failing_gate_rejects_a_faster_step(tmp_path: Path):
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(50.0,) * 5)
    failed = gate(task(tmp_path, gate_command=f"{PYTHON} -c 'raise SystemExit(1)'"), tmp_path)
    result = audit(
        speed_report(2.0), workspace=tmp_path, comparison=compare(before, after), gate=failed
    )
    assert not result.accepted
    assert "the gate exited 1" in result.findings[0]


def test_no_report_means_nothing_can_be_checked(tmp_path: Path):
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(80.0,) * 5)
    result = audit(None, workspace=tmp_path, comparison=compare(before, after), gate=None)
    assert not result.accepted
    assert "no report was produced" in result.findings[0]


def test_a_correctness_claim_is_checked_for_its_artifact_not_re_run(tmp_path: Path):
    (tmp_path / "verify.txt").write_text("PASS\n")
    report = Report(
        summary="fused",
        claims=(
            Claim(kind="correctness", statement="outputs match", evidence=("verify.txt",)),
            Claim(kind="correctness", statement="grads match", evidence=("absent.txt",)),
        ),
    )
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(80.0,) * 5)
    result = audit(report, workspace=tmp_path, comparison=compare(before, after), gate=None)
    assert [item.verdict for item in result.verified] == ["checked", "unbacked"]
    assert "did not re-run the verifier" in result.verified[0].detail


def test_a_changed_file_that_does_not_exist_is_a_finding(tmp_path: Path):
    before = Samples(name="before", us=(100.0,) * 5)
    after = Samples(name="after", us=(80.0,) * 5)
    report = Report(summary="x", changed=("ghost.py",))
    result = audit(report, workspace=tmp_path, comparison=compare(before, after), gate=None)
    assert "does not exist: ghost.py" in result.findings[0]


def test_drive_accepts_a_change_the_measurement_supports(tmp_path: Path):
    root = repo(tmp_path)
    result = drive(task(root), FakeRunner(speed_report(1.25), edit=80.0), root=tmp_path / "runs")
    assert result.accepted
    assert result.comparison is not None
    assert result.comparison.speedup == pytest.approx(1.25)
    assert Path(result.root).joinpath("drive.json").exists()
    assert Path(result.root).joinpath("workspace", "step.py").exists()


def test_drive_feeds_the_failure_back_to_the_next_turn(tmp_path: Path):
    root = repo(tmp_path)
    runner = FakeRunner(speed_report(1.9), edit=80.0)
    result = drive(task(root, attempts=2), runner, root=tmp_path / "runs")
    assert not result.accepted
    assert len(result.turns) == 2
    assert "What the last attempt failed to prove" in runner.prompts[1]
    assert "contradicted" in runner.prompts[1]


def test_the_workspace_copy_leaves_history_and_past_runs_behind(tmp_path: Path):
    root = repo(tmp_path)
    (root / "runs").mkdir()
    destination = workspace_copy(root, tmp_path / "copy")
    assert (destination / "bench.py").exists()
    assert not (destination / ".git").exists()
    assert not (destination / "runs").exists()


def test_the_prompt_states_the_device_and_that_the_harness_measures(tmp_path: Path):
    text = build(task(repo(tmp_path)), tmp_path, ("a contradicted claim",))
    assert "nvidia.h800_sxm" in text
    assert "already done, by the harness" in text
    assert "a contradicted claim" in text


def test_runtime_overrides_are_validated_before_they_reach_the_cli():
    assert overrides(("a=1", " b = 2 ")) == ("a=1", "b=2")
    with pytest.raises(ValueError, match="not key=value"):
        overrides(("model_provider",))


def test_runtime_environment_keeps_values_that_contain_an_equals():
    assert environment(("MY_KEY=abc=def", " PAD = keep ")) == {"MY_KEY": "abc=def", "PAD": " keep "}


def test_the_runner_carries_the_model_provider_and_its_overrides():
    runner = CodexRunner(
        model="my-model",
        provider="my-provider",
        config=("model_providers.my-provider.base_url=https://api.example.com/v1",),
        env={"MY_KEY": "secret"},
    )
    assert runner.model == "my-model"
    assert runner.provider == "my-provider"
    assert runner.config == ("model_providers.my-provider.base_url=https://api.example.com/v1",)
    assert runner.env == {"MY_KEY": "secret"}
