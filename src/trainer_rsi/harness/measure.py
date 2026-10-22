import re
import shlex
import subprocess
from pathlib import Path

from trainer_rsi.harness.contract import Gate, Task
from trainer_rsi.verify.stats import Samples

SAMPLE_LINE = re.compile(r"^us=(\d+(?:\.\d+)?)$")
OUTPUT_TAIL = 400


def parse_samples(output: str) -> tuple[float, ...]:
    """Every `us=` line in a measure command's output, in order.

    Raises:
        ValueError: The output holds no sample, so the command measured nothing.
    """
    found = tuple(
        float(match.group(1))
        for line in output.splitlines()
        if (match := SAMPLE_LINE.match(line.strip()))
    )
    if not found:
        raise ValueError("the measure command printed no `us=<float>` line")
    return found


def run_once(task: Task, cwd: Path) -> tuple[float, ...]:
    """Run the measure command once and read the samples it printed.

    Raises:
        RuntimeError: The command failed, naming its exit code and its last lines.
    """
    completed = subprocess.run(
        shlex.split(task.measure), cwd=cwd, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"measure exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[-OUTPUT_TAIL:]}"
        )
    return parse_samples(completed.stdout)


def measure(task: Task, cwd: Path, out: Path) -> Samples:
    """Run the measure command `warmup` times, then `reps` more, and keep the last samples.

    Warmup output is discarded rather than trimmed from the kept run: a first
    iteration that compiles a kernel is a different program from the one being
    compared, and averaging the two describes neither.
    """
    for _ in range(task.warmup):
        run_once(task, cwd)
    samples: list[float] = []
    for _ in range(task.reps):
        samples.extend(run_once(task, cwd))
    result = Samples(name=f"{task.measure} in {cwd.name}", us=tuple(samples))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def gate(task: Task, cwd: Path) -> Gate | None:
    """Run the correctness command, or return `None` when the task states none."""
    if not task.gate:
        return None
    completed = subprocess.run(
        shlex.split(task.gate), cwd=cwd, capture_output=True, text=True, check=False
    )
    return Gate(
        command=task.gate,
        exit_code=completed.returncode,
        output=(completed.stdout + completed.stderr).strip()[-OUTPUT_TAIL:],
    )
