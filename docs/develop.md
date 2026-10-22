# Develop

## Environment

Python 3.14, `uv`, no accelerator required for anything except a capture.

```sh
uv sync --extra agent        # the tool, the agent runtime, and the dev tools
uv run pytest                # the suite, no GPU needed
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

`torch` is in the dev group only, so the capture tests run on a machine without an
accelerator; the package itself depends on no accelerator library. Where the model
runs, `capture` imports that environment's torch instead.

## Tests

- One file per contract: `test_ir`, `test_facts`, `test_analysis`, `test_plan`,
  `test_verify`, `test_capture`, `test_cloud`, `test_cli`.
- A detector gets two cases: one that fires and one that stays quiet. A finding
  that fires on every capture trains its reader to ignore the list.
- Assert behaviour, never source shape. `tests/conftest.py` builds a graph and a
  trace by hand so a test never needs a device.
- A test that needs `torch` belongs to capture, and even there it runs on CPU by
  passing `activities=("cpu",)`.

## Specs

`docs/spec/` owns the contracts. A change that alters behaviour a report, a
bundle, or a bound depends on updates its owner in the same commit, and
`docs/spec/architecture.md` §5 is the table that says who owns what.

A source file may reference a spec section; it must not restate it. A second copy
of a contract is a copy that goes stale, and the reader cannot tell which is
current.

## Plans

`docs/plans/` owns the work. Three passes: current state, then delivered shapes,
then the rest. `docs/policies/project-policy.json` states the acceptance criteria
that apply by path.

## Conventions

- Every data structure is a pydantic model on `trainer_rsi.base.Value`: frozen,
  strict, closed. Each is therefore also a wire format, read back from a bundle, a
  capture, or an agent's structured output.
- Public names carry no leading underscore. Module files start with their imports.
- A refusal names what is installed or defined, so the next attempt can succeed.
- A docstring says what a caller must know; the spec says what the project
  promises. When those differ, the spec is right.
