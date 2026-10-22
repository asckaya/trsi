from typing import Literal

from trainer_rsi.base import Value
from trainer_rsi.verify.stats import Comparison, Samples

type SandboxLevel = Literal["read-only", "workspace-write", "full-access"]

type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
type JsonObject = dict[str, JsonValue]
"""JSON as JSON: what a schema and a structured reply are, before any model reads them."""


class Task(Value):
    """One thing to make faster, and how it will be judged.

    The measure command is the harness's, not the agent's. An agent that reports
    its own number is an agent whose number nobody checked, and the whole point
    of this layer is that the edit and the measurement come from different
    parties.
    """

    repo: str
    device: str
    """Installed hardware document id the machine running the step is."""
    entry: str
    """How the step runs, in one line, for the agent to read."""
    measure: str
    """Command run in the repository that prints one `us=<float>` per timed iteration."""
    gate: str = ""
    """Command that must exit zero for a change to be accepted; the correctness gate."""
    goal: str = "make one step faster without changing what it computes"
    attempts: int = 2
    sandbox: SandboxLevel = "workspace-write"
    warmup: int = 2
    reps: int = 3


class Claim(Value):
    """One thing the agent says its change achieved."""

    kind: Literal["speed", "correctness"]
    statement: str
    speedup: float | None = None
    """What the agent says the ratio is; the audit recomputes it and compares."""
    evidence: tuple[str, ...] = ()
    """Files the claim rests on, relative to the workspace."""


class Report(Value):
    """What the agent must hand back, as the structured output of its turn."""

    summary: str
    changed: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()
    could_not_check: tuple[str, ...] = ()
    """What the agent knows it did not verify. An empty list is a claim of its own."""


class Turn(Value):
    """One agent turn, as the harness saw it."""

    response: str
    structured: JsonObject | None = None
    seconds: float = 0.0
    tokens: int | None = None


class Gate(Value):
    """The correctness command's verdict."""

    command: str
    exit_code: int
    output: str


class Verified(Value):
    """One claim, checked against what the harness measured."""

    claim: Claim
    verdict: Literal["supported", "contradicted", "unbacked", "indistinguishable", "checked"]
    speedup: float | None = None
    low: float | None = None
    high: float | None = None
    detail: str = ""


class Audit(Value):
    """Whether the agent's account matches the measurement."""

    accepted: bool
    verified: tuple[Verified, ...]
    findings: tuple[str, ...]


class Drive(Value):
    """One end-to-end run: what was measured, what the agent did, what it proved."""

    task: Task
    root: str
    turns: tuple[Turn, ...]
    before: Samples
    after: Samples | None = None
    comparison: Comparison | None = None
    gate: Gate | None = None
    report: Report | None = None
    audit: Audit | None = None
    accepted: bool = False
