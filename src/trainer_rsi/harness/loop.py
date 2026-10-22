import json
import shutil
import time
from pathlib import Path

from trainer_rsi.harness.audit import audit
from trainer_rsi.harness.contract import Drive, Report, Task, Turn
from trainer_rsi.harness.measure import gate, measure
from trainer_rsi.harness.prompt import build
from trainer_rsi.harness.runner import Runner
from trainer_rsi.verify.stats import compare

SKIP = (".git", "runs", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache", "*.bundle.json.gz")
"""What a workspace copy leaves behind: history, previous runs, and caches."""


def report_of(turn: Turn, workspace: Path) -> Report | None:
    """The report a turn produced, from its structured output or from `report.json`.

    Returns `None` when the turn produced neither, which is a finding rather than
    an error: an agent that answers in prose did not answer the question.
    """
    candidate = turn.structured
    if candidate is None:
        written = workspace / "report.json"
        if not written.exists():
            return None
        candidate = json.loads(written.read_text(encoding="utf-8"))
    return Report.model_validate_json(json.dumps(candidate))


def workspace_copy(repo: Path, destination: Path) -> Path:
    """Copy *repo* to *destination*, leaving history and caches behind."""
    shutil.copytree(repo, destination, ignore=shutil.ignore_patterns(*SKIP), symlinks=True)
    return destination


def drive(task: Task, runner: Runner, *, root: Path | None = None) -> Drive:
    """Measure, hand the step to the agent, measure again, and judge the difference.

    The baseline is the repository as it was before the first turn, not the
    previous attempt: the question is whether the ending program is faster than
    the one the task started from.

    Raises:
        ValueError: The measure command states too few iterations to support a claim.
        RuntimeError: The measure command failed.
    """
    repo = Path(task.repo).expanduser().resolve()
    base = (root or repo / "runs") / time.strftime("%Y%m%d-%H%M%S")
    workspace = workspace_copy(repo, base / "workspace")
    before = measure(task, repo, base / "before.json")
    turns: list[Turn] = []
    feedback: tuple[str, ...] = ()
    report: Report | None = None
    after = None
    comparison = None
    verdict = None
    checked = None
    for attempt in range(1, task.attempts + 1):
        turn = runner.run(
            build(task, workspace, feedback),
            cwd=workspace,
            output_schema=Report.model_json_schema(),
        )
        turns.append(turn)
        report = report_of(turn, workspace)
        after = measure(task, workspace, base / f"after-{attempt}.json")
        comparison = compare(before, after)
        checked = gate(task, workspace)
        verdict = audit(report, workspace=workspace, comparison=comparison, gate=checked)
        if verdict.accepted:
            break
        feedback = verdict.findings
    result = Drive(
        task=task,
        root=str(base),
        turns=tuple(turns),
        before=before,
        after=after,
        comparison=comparison,
        gate=checked,
        report=report,
        audit=verdict,
        accepted=bool(verdict and verdict.accepted),
    )
    (base / "drive.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result
