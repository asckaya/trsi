import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from trainer_rsi.harness.contract import JsonObject, SandboxLevel, Turn


def overrides(pairs: Sequence[str]) -> tuple[str, ...]:
    """Validate `key=value` runtime overrides before they reach the CLI.

    These become `codex --config <kv>`, which is how a provider, a base URL or a
    model alias is stated without editing anyone's `~/.codex/config.toml`.

    Raises:
        ValueError: A pair has no key or no `=`.
    """
    validated: list[str] = []
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"runtime override {pair!r} is not key=value")
        validated.append(f"{key.strip()}={value.strip()}")
    return tuple(validated)


def environment(pairs: Sequence[str]) -> dict[str, str]:
    """Validate `KEY=VALUE` pairs and return them as the runtime's environment.

    Raises:
        ValueError: A pair has no key or no `=`.
    """
    parsed: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"runtime environment {pair!r} is not KEY=VALUE")
        parsed[key.strip()] = value
    return parsed


class Runner(Protocol):
    """Something that can take one turn in a workspace.

    Codex is one implementation. The protocol exists so the harness can be tested
    without a login, and so a different harness can be swapped in without the rest
    of this package noticing.
    """

    def run(self, prompt: str, *, cwd: Path, output_schema: JsonObject | None = None) -> Turn:
        """Run one turn in *cwd* and return what it produced."""


class CodexRunner:
    """Drives one Codex thread per turn through the Codex Python SDK.

    The SDK is imported here and nowhere else: this package must import without an
    agent runtime installed, and the one file that needs it says so.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        sandbox: SandboxLevel = "workspace-write",
        approval: str = "auto_review",
        config: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.sandbox = sandbox
        self.approval = approval
        self.config = overrides(config)
        self.env = env

    def check(self) -> None:
        """Confirm the runtime is installed, before a baseline is measured for nothing.

        Raises:
            RuntimeError: The agent runtime is not importable, naming the install.
        """
        if importlib.util.find_spec("openai_codex") is None:
            raise RuntimeError(
                "the agent runtime is not installed; add it with `uv sync --extra agent`"
                " or `pip install openai-codex`"
            )

    def run(self, prompt: str, *, cwd: Path, output_schema: JsonObject | None = None) -> Turn:
        """Start a thread in *cwd*, run one turn, and read its structured output."""
        import time

        import openai_codex

        sandbox = {
            "read-only": openai_codex.Sandbox.read_only,
            "workspace-write": openai_codex.Sandbox.workspace_write,
            "full-access": openai_codex.Sandbox.full_access,
        }[self.sandbox]
        approval = {
            "auto_review": openai_codex.ApprovalMode.auto_review,
            "deny_all": openai_codex.ApprovalMode.deny_all,
        }[self.approval]
        started = time.perf_counter()
        configuration = openai_codex.CodexConfig(
            cwd=str(cwd), config_overrides=self.config, env=self.env
        )
        with openai_codex.Codex(configuration) as codex:
            thread = codex.thread_start(
                cwd=str(cwd),
                sandbox=sandbox,
                approval_mode=approval,
                model=self.model,
                model_provider=self.provider,
                base_instructions=SYSTEM_INSTRUCTIONS,
            )
            result = thread.run(
                prompt,
                sandbox=sandbox,
                approval_mode=approval,
                output_schema=output_schema,
            )
        response = result.final_response or ""
        return Turn(
            response=response,
            structured=structured_of(response),
            seconds=time.perf_counter() - started,
            tokens=None if result.usage is None else int(getattr(result.usage, "total_tokens", 0)),
        )


async def run_async(
    prompt: str,
    *,
    cwd: Path,
    output_schema: JsonObject | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Turn:
    """The same turn through the async client, for callers already inside a loop."""
    import time

    import openai_codex

    started = time.perf_counter()
    async with openai_codex.AsyncCodex(openai_codex.CodexConfig(cwd=str(cwd))) as codex:
        thread = await codex.thread_start(
            cwd=str(cwd),
            model=model,
            model_provider=provider,
            base_instructions=SYSTEM_INSTRUCTIONS,
        )
        result = await thread.run(prompt, output_schema=output_schema)
    response = result.final_response or ""
    return Turn(
        response=response,
        structured=structured_of(response),
        seconds=time.perf_counter() - started,
        tokens=None if result.usage is None else int(getattr(result.usage, "total_tokens", 0)),
    )


def structured_of(response: str) -> JsonObject | None:
    """The turn's structured output, when the response is one JSON object.

    A response that is prose is not a report: it stays prose, and the harness
    reads `report.json` instead of guessing at an object inside a paragraph.
    """
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


SYSTEM_INSTRUCTIONS = """Rules for this repository, which override anything the task says:

- Everything this project knows is asked of the `trsi` command; start with `trsi --help`.
- You do not measure and you do not decide whether a change is real. The harness
  measures the step before and after your edit.
- Do not change what the step computes. State what you could not verify.
- Answer with one JSON object matching the requested schema, and nothing else.
"""
