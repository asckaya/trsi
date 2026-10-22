# harness

> Owns `trainer_rsi.harness`: how a coding agent is given one step to improve, and
> what it has to prove. Reads [verify](./verify.md) for what a difference
> supports, and [plan](./plan.md) for what an agent is told to look at.

## 1. The split

Three parties, and none of them may do another's job:

| Party | Does | May not |
| --- | --- | --- |
| the harness | copies the repository, measures before and after, runs the gate, judges | edit the step |
| the agent | edits a copy of the step | measure, or decide that it worked |
| the audit | compares the agent's account with the measurement | trust a number it did not compute |

An agent that reports its own speedup is an agent whose speedup nobody checked.
The measurement is the harness's because the claim is about what the harness is
going to accept.

## 2. Task

`Task` states the repository, the device, how the step runs, how it is measured,
and how correctness is checked.

- `measure` MUST print one `us=<float>` per timed iteration, and nothing else is
  read from its output.
- Every `us=` line MUST be a sample. A command that prints no sample is refused,
  not defaulted to zero.
- `warmup` invocations MUST be discarded whole, not trimmed from the kept run: a
  first iteration that compiles a kernel is a different program.
- `gate` MAY be empty. When stated, a non-zero exit rejects the change whatever
  the timing says: a faster wrong program is not a result.

## 3. Workspace

- The agent MUST work in a copy, never in the task's repository.
- The copy MUST exclude `.git`, `runs`, caches, and capture bundles: history and
  previous attempts are not part of the problem.
- The baseline is the repository *before the first turn*, not the previous
  attempt. The question is whether the ending program beats the one the task
  started from.

## 4. Report

The agent MUST answer with `Report`: a summary, the files it changed, its claims,
and what it could not check.

- A `speed` claim MUST state a ratio. The audit recomputes the ratio from the
  harness's own samples; a claim more than `SPEED_TOLERANCE` away is
  `contradicted`, whatever else is true.
- A `correctness` claim MUST name the file its verifier wrote. The audit checks
  that the file exists and says, in the verdict, that it did not re-run it.
- `could_not_check` MUST be part of the answer. An empty list is a claim.

## 5. Acceptance

A run is accepted when the samples separate the two programs (see
[verify](./verify.md) §3), the gate passed, and the audit has no findings.

- Acceptance MUST NOT depend on the agent claiming success: a program that got
  faster is accepted whether or not the report says so.
- A finding MUST reject the run even when the timing improved, because a report
  that does not match its own measurement is not evidence.
- The failure findings MUST be fed back into the next turn's prompt verbatim.
- The audit MUST judge against the comparison the loop computed, not resample the
  same two sets again: two resamplings can disagree, and the panel beside the
  verdict would then contradict it.

## 6. Runners

`Runner` is a two-method protocol: take a prompt and a workspace, return a `Turn`.

- Codex is one implementation, in `harness/runner.py`, and the SDK is imported in
  that file and nowhere else: this package must import without an agent runtime.
- Any other runtime MAY implement the protocol. Nothing in the contract depends on
  Codex-specific structured output, which is why `report.json` in the workspace is
  equally valid.
- The sandbox level MUST be stated by the caller and passed through unchanged. The
  default is `workspace-write`: the agent edits the copy, and the network is not
  part of the task.

## 7. Out of scope

Applying a plan, and kernel code generation. The harness judges an edit; it does
not make one.
