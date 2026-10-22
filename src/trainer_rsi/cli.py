import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from trainer_rsi.analyze.report import as_json, build, render
from trainer_rsi.analyze.timeline import timeline
from trainer_rsi.facts import installed, limits, load
from trainer_rsi.facts.document import Fact, HardwareDoc
from trainer_rsi.harness import CodexRunner, Drive, Task, drive
from trainer_rsi.harness import build as build_prompt
from trainer_rsi.harness.runner import environment
from trainer_rsi.ir.bundle import read_bundle
from trainer_rsi.plan import candidate_plans
from trainer_rsi.plan import report as plan_report
from trainer_rsi.units import quantity
from trainer_rsi.verify import Samples, compare

PROGRAM = "trsi"

WIDTH = 100
"""Fixed report width: a table read through a pipe is read by an agent, and a
width that follows an unset terminal truncates the numbers it is reporting."""


def facts_table(docs: dict[str, HardwareDoc]) -> Table:
    """Every installed device, with the two rates a bound divides by."""
    table = Table(box=None, pad_edge=False)
    table.add_column("device", style="cyan", no_wrap=True)
    table.add_column("architecture", style="dim", no_wrap=True)
    table.add_column("memory", justify="right", no_wrap=True)
    table.add_column("capacity", justify="right", no_wrap=True)
    table.add_column("link", justify="right", no_wrap=True)
    for doc in docs.values():
        nvlink = doc.maybe("interconnect.nvlink.bandwidth")
        table.add_row(
            doc.id,
            doc.architecture,
            quantity(doc.amount("memory.hbm.bandwidth"), "byte/s"),
            quantity(doc.size("memory.hbm.capacity"), "byte"),
            "none" if nvlink is None else f"{quantity(nvlink / 2, 'byte/s')} per direction",
        )
    return table


def facts_detail(doc: HardwareDoc, only: str | None) -> Table:
    """One document's facts, each with the provenance that makes it checkable."""
    table = Table(box=None, pad_edge=False, show_lines=True)
    table.add_column("fact", style="cyan", no_wrap=True)
    table.add_column("value", justify="right", no_wrap=True)
    table.add_column("provenance")
    selected = {only: doc.fact(only)} if only else doc.facts
    for path, fact in selected.items():
        if fact.status == "unavailable":
            value = "unavailable"
        elif isinstance(fact.value, int | float):
            value = quantity(fact.value, fact.unit)
        else:
            value = str(fact.value)
        provenance = Text(fact.origin, style="bold")
        provenance.append(f"\n{fact.source}", style="blue")
        provenance.append(f"\n{fact.conditions}", style="dim")
        table.add_row(path, value, provenance)
    return table


def facts_json(doc: HardwareDoc, only: str | None) -> dict[str, object]:
    """One document's facts as data."""
    selected: dict[str, Fact] = {only: doc.fact(only)} if only else doc.facts
    return {
        "id": doc.id,
        "label": doc.label,
        "architecture": doc.architecture,
        "digest": doc.digest,
        "facts": {path: fact.jsonable() for path, fact in selected.items()},
    }


def write_json(path: Path, payload: object) -> None:
    """Write a JSON payload, creating the parent directory it names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_facts(args: argparse.Namespace, console: Console) -> int:
    """List installed devices, or print one device's facts."""
    if args.device is None:
        docs = installed()
        if args.json:
            print(
                json.dumps(
                    {d.id: facts_json(d, None) for d in docs.values()}, indent=2, sort_keys=True
                )
            )
        else:
            console.print(facts_table(docs))
        return 0
    doc = load(args.device)
    if args.json:
        print(json.dumps(facts_json(doc, args.fact), indent=2, sort_keys=True))
    else:
        console.print(facts_detail(doc, args.fact))
    return 0


SNIPPET = """import trainer_rsi.capture as capture

with capture.record(
    device_id="{device}",
    out="{out}",
    step="{step}",
    steps={steps},
    module=model,              # the module whose graph should be priced
    example=(batch,),          # one example input for it
):
    for _ in range({steps}):
        train_step()           # whatever your loop already does
"""


def command_capture(args: argparse.Namespace, console: Console) -> int:
    """Print the capture snippet for a device, with no dependency on the model's stack."""
    doc = load(args.device)
    snippet = SNIPPET.format(device=doc.id, out=args.out, step=args.step, steps=args.steps)
    if args.json:
        print(
            json.dumps(
                {"device": doc.id, "label": doc.label, "out": args.out, "snippet": snippet},
                indent=2,
            )
        )
        return 0
    console.print(
        Panel(
            Group(
                Text(f"{doc.label}  ({doc.id})", style="bold"),
                Text(
                    "Paste this where your loop already is. The `with` block is the measured window: "
                    "open it after warmup, and the report describes exactly those steps.",
                    style="dim",
                ),
                Syntax(snippet, "python", theme="ansi_dark", padding=0),
                Text(
                    f"then: {PROGRAM} analyze " + args.out,
                    style="green",
                ),
            ),
            title=f"{PROGRAM} capture",
            border_style="blue",
        )
    )
    return 0


def command_analyze(args: argparse.Namespace, console: Console) -> int:
    """Analyse one capture bundle against the device it states."""
    report = build(read_bundle(Path(args.bundle)))
    if args.json:
        write_json(Path(args.json), as_json(report))
    else:
        render(report, console)
    return 0


def command_plan(args: argparse.Namespace, console: Console) -> int:
    """Turn a captured step into ranked rewrites, each priced against the same device."""
    bundle = read_bundle(Path(args.bundle))
    report = build(bundle)
    if report.roofline is None:
        print(
            f"{PROGRAM}: error: this bundle carries no graph, so there is nothing to rewrite",
            file=sys.stderr,
        )
        return 1
    step = timeline(bundle.trace) if bundle.trace is not None else None
    plans = candidate_plans(
        graph=bundle.graph,
        limits=limits(load(report.device_id)),
        priced=report.roofline,
        step=step,
    )
    shown = plans if args.all else plans[: args.top]
    if args.json:
        write_json(
            Path(args.json),
            plan_report.as_json(
                shown, measured_us=report.measured_us, floor_us=report.roofline.floor_us
            ),
        )
    else:
        plan_report.render(
            console, shown, measured_us=report.measured_us, floor_us=report.roofline.floor_us
        )
    return 0


def read_timings(path: Path) -> Samples:
    """Read one timing set, refusing a file that does not state what it measured.

    Raises:
        ValueError: The file is not a JSON object with a name and a list of timings.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "us" not in payload or "name" not in payload:
        raise ValueError(f'{path}: expected {{"name": ..., "us": [...]}}')
    return Samples(name=str(payload["name"]), us=tuple(float(value) for value in payload["us"]))


def command_compare(args: argparse.Namespace, console: Console) -> int:
    """Compare two timing sets and state only what their difference supports."""
    comparison = compare(read_timings(Path(args.before)), read_timings(Path(args.after)))
    if args.json:
        print(json.dumps(comparison.jsonable(), indent=2, sort_keys=True))
        return 0
    style = {"faster": "green", "slower": "red", "indistinguishable": "yellow"}[comparison.claim]
    table = Table(box=None, pad_edge=False)
    table.add_column("", style="dim", justify="right")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_row("samples", str(comparison.before.count), str(comparison.after.count))
    table.add_row(
        "median", f"{comparison.before.median_us:.1f} us", f"{comparison.after.median_us:.1f} us"
    )
    table.add_row(
        "spread", f"{comparison.before.spread_us:.1f} us", f"{comparison.after.spread_us:.1f} us"
    )
    table.add_row(
        "p10",
        f"{comparison.before.percentile(0.1):.1f} us",
        f"{comparison.after.percentile(0.1):.1f} us",
    )
    table.add_row(
        "p90",
        f"{comparison.before.percentile(0.9):.1f} us",
        f"{comparison.after.percentile(0.9):.1f} us",
    )
    console.print(table)
    console.print(
        Text(f"{comparison.claim}", style=style)
        + Text(f"  x{comparison.speedup:.3f}  ", style="bold")
        + Text(
            f"[{comparison.low:.3f}, {comparison.high:.3f}] over {comparison.resamples} resamples",
            style="dim",
        )
    )
    if comparison.claim == "indistinguishable":
        console.print(
            Text("the intervals overlap, so this run does not separate the two", style="yellow")
        )
    return 0


def command_drive(args: argparse.Namespace, console: Console) -> int:
    """Hand one step to a coding agent, then measure what it did."""
    task = Task(
        repo=str(Path(args.repo).resolve()),
        device=load(args.device).id,
        entry=args.entry,
        measure=args.measure,
        gate=args.gate,
        goal=args.goal,
        attempts=args.attempts,
        sandbox=args.sandbox,
        warmup=args.warmup,
        reps=args.reps,
    )
    if args.print_prompt:
        console.print(build_prompt(task, Path(task.repo) / "runs" / "workspace", ()))
        return 0
    runner = CodexRunner(
        model=args.model,
        provider=args.provider,
        sandbox=args.sandbox,
        approval=args.approval,
        config=tuple(args.config),
        env=environment(args.env) or None,
    )
    runner.check()
    result = drive(task, runner)
    if args.json:
        write_json(Path(args.json), result.jsonable())
    else:
        render_drive(console, result)
    return 0 if result.accepted else 1


def render_drive(console: Console, result: Drive) -> None:
    """What the agent did, what the harness measured, and what it accepted."""
    comparison = result.comparison
    lines = [Text(f"{len(result.turns)} turn(s), artifacts in {result.root}", style="dim")]
    if comparison is not None:
        lines.append(
            Text()
            + Text(
                f"{comparison.before.median_us:.1f} us -> {comparison.after.median_us:.1f} us  ",
                style="bold",
            )
            + Text(
                f"x{comparison.speedup:.3f} ",
                style="green" if comparison.claim == "faster" else "yellow",
            )
            + Text(
                f"[{comparison.low:.3f}, {comparison.high:.3f}]  {comparison.claim}", style="dim"
            )
        )
    if result.gate is not None:
        lines.append(
            Text(f"gate: exit {result.gate.exit_code}  {result.gate.command}", style="dim")
        )
    if result.report is not None:
        lines.append(Text(f"agent: {result.report.summary}", style="dim"))
        for item in result.report.could_not_check:
            lines.append(Text(f"  not checked: {item}", style="yellow"))
    for verdict in () if result.audit is None else result.audit.verified:
        style = {"supported": "green", "checked": "green"}.get(verdict.verdict, "red")
        lines.append(
            Text(
                f"  claim {verdict.verdict}: {verdict.claim.statement} — {verdict.detail}",
                style=style,
            )
        )
    for finding in () if result.audit is None else result.audit.findings:
        lines.append(Text(f"  finding: {finding}", style="red"))
    lines.append(
        Text(
            "accepted" if result.accepted else "not accepted",
            style="green" if result.accepted else "red",
        )
    )
    console.print(
        Panel(Group(*lines), title="trsi drive", border_style="blue" if result.accepted else "red")
    )


def facts_commands(commands: argparse._SubParsersAction) -> None:
    """Register the hardware-fact commands."""
    facts = commands.add_parser("facts", help="hardware facts this build can price against")
    facts.add_argument(
        "device", nargs="?", help="installed device id; omit to list what is installed"
    )
    facts.add_argument("fact", nargs="?", help="one dotted fact path, e.g. throughput.bf16")
    facts.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of a table"
    )
    facts.set_defaults(run=command_facts)


def capture_commands(commands: argparse._SubParsersAction) -> None:
    """Register the commands that produce a bundle or read one."""
    capture = commands.add_parser(
        "capture", help="print the snippet that records a step into a bundle"
    )
    capture.add_argument(
        "--device", required=True, help="installed device id the capture will run on"
    )
    capture.add_argument(
        "--out", default="step.bundle.json.gz", help="bundle path the snippet writes"
    )
    capture.add_argument("--step", default="train_step", help="label for the captured step")
    capture.add_argument(
        "--steps", type=int, default=3, help="iterations the measured window contains"
    )
    capture.add_argument("--json", action="store_true", help="print the snippet as JSON")
    capture.set_defaults(run=command_capture)

    analyze = commands.add_parser(
        "analyze", help="report where a captured step loses time, largest first"
    )
    analyze.add_argument("bundle", help="path to a .bundle.json.gz written by a capture")
    analyze.add_argument(
        "--json", metavar="PATH", help="write the report to PATH instead of stdout"
    )
    analyze.set_defaults(run=command_analyze)

    plan = commands.add_parser(
        "plan", help="ranked rewrites for a captured step, priced before anything is run"
    )
    plan.add_argument("bundle", help="path to a .bundle.json.gz")
    plan.add_argument("--top", type=int, default=5, help="how many plans to show")
    plan.add_argument("--all", action="store_true", help="show every plan, not just the best few")
    plan.add_argument("--json", metavar="PATH", help="write the plans to PATH instead of stdout")
    plan.set_defaults(run=command_plan)

    compare = commands.add_parser("compare", help="does a change actually move the number")
    compare.add_argument("before", help='JSON written as {"name": ..., "us": [...]}')
    compare.add_argument("after", help="the same shape for the changed program")
    compare.add_argument("--json", action="store_true", help="print machine-readable JSON")
    compare.set_defaults(run=command_compare)

    drive = commands.add_parser(
        "drive", help="hand one step to a coding agent, then measure what it did"
    )
    drive.add_argument("--repo", default=".", help="repository to optimize")
    drive.add_argument("--device", required=True, help="installed device id this machine runs")
    drive.add_argument("--entry", required=True, help="one line saying how the step runs")
    drive.add_argument(
        "--measure", required=True, help="command printing one us=<float> per iteration"
    )
    drive.add_argument(
        "--gate", default="", help="command that must exit zero for a change to be accepted"
    )
    drive.add_argument("--goal", default=Task.model_fields["goal"].default, help="what to achieve")
    drive.add_argument("--attempts", type=int, default=2, help="turns the agent gets")
    drive.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=("read-only", "workspace-write", "full-access"),
    )
    drive.add_argument("--approval", default="auto_review", choices=("auto_review", "deny_all"))
    drive.add_argument(
        "--model", default=None, help="model id; omit to use the runtime's configured one"
    )
    drive.add_argument(
        "--provider", default=None, help="model provider id the runtime was configured with"
    )
    drive.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="runtime override passed as `codex --config KEY=VALUE`, e.g. model_providers.x.base_url=...",
    )
    drive.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment for the runtime process, e.g. MY_API_KEY=...",
    )
    drive.add_argument(
        "--warmup", type=int, default=2, help="measure runs discarded before sampling"
    )
    drive.add_argument("--reps", type=int, default=3, help="measure runs that are kept")
    drive.add_argument(
        "--print-prompt", action="store_true", help="print the prompt and run nothing"
    )
    drive.add_argument("--json", metavar="PATH", help="write the drive record to PATH")
    drive.set_defaults(run=command_drive)


def parser() -> argparse.ArgumentParser:
    """The command surface an agent drives, discoverable from its own help."""
    root = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Full-pipeline optimizer for training programs: capture, attribute, rewrite, verify.",
    )
    commands = root.add_subparsers(dest="command")
    facts_commands(commands)
    capture_commands(commands)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit status."""
    console = Console(width=WIDTH)
    root = parser()
    args = root.parse_args(argv)
    if args.command is None:
        root.print_help()
        return 0
    try:
        return int(args.run(args, console))
    except (KeyError, ValueError, OSError, TypeError, RuntimeError) as error:
        print(f"{PROGRAM}: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
