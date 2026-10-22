from pathlib import Path

from trainer_rsi.harness.contract import Task

SYSTEM = """You are working inside a copy of one repository, on one training step.

Rules that hold for every turn:

- Everything this project knows is asked of the `trsi` command. Run `trsi --help`
  first. Do not ask a person, and do not derive a hardware figure yourself:
  `trsi facts <device>` prints the installed ones with their sources.
- You do not measure. The harness measures the step before and after your edit and
  decides whether the change is real. Your numbers are checked against its own.
- You do not change what the step computes. If you cannot verify the numerics
  yourself, say so in `could_not_check` instead of claiming it.
- A change that the samples cannot separate from noise is not a result. That is a
  normal outcome; report it rather than dressing it up.
- Prefer the change the plan already prices. If you do something else, say why.
"""


def build(task: Task, workspace: Path, feedback: tuple[str, ...]) -> str:
    """The prompt for one turn, including what the last attempt failed to prove."""
    lines = [
        f"Goal: {task.goal}.",
        "",
        f"Repository: {workspace}",
        f"Accelerator: {task.device}",
        f"How the step runs: {task.entry}",
        f"How it is measured (already done, by the harness): {task.measure}",
    ]
    if task.gate:
        lines.append(f"How correctness is checked, and required to pass: {task.gate}")
    lines += [
        "",
        "Do this:",
        "1. `trsi analyze <bundle>` if a bundle is present, to see where the step loses time.",
        "2. `trsi plan <bundle>` for the rewrites this project has priced on this hardware.",
        "3. Make the smallest change that the evidence supports.",
        "4. Write `report.json` in the repository root, matching the schema you were given.",
        "",
        "A speed claim is supported only if you state the ratio you expect and the",
        "files that show it; a correctness claim needs the verifier output it rests on.",
    ]
    if feedback:
        lines += ["", "What the last attempt failed to prove:"]
        lines += [f"- {item}" for item in feedback]
        lines += ["", "Fix the reason, not the wording of the claim."]
    return "\n".join(lines)
