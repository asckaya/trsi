from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trainer_rsi.analyze.gaps import Finding, detect
from trainer_rsi.analyze.roofline import Roofline, roofline
from trainer_rsi.analyze.timeline import Timeline, timeline
from trainer_rsi.base import Value
from trainer_rsi.facts import limits, load
from trainer_rsi.ir.bundle import Bundle, Capture, Environment


class Report(Value):
    """Everything one analysis concluded, as data.

    The text rendering is a view of this object and carries no fact the object
    does not: an agent reading `--json` and a human reading the table are
    looking at the same claims.
    """

    bundle_digest: str
    device_id: str
    device_label: str
    limits_digest: str
    capture: Capture
    environment: Environment
    roofline: Roofline | None
    timeline: Timeline | None
    findings: tuple[Finding, ...]
    unmodelled: tuple[str, ...]

    @property
    def measured_us(self) -> float | None:
        """The step time the report is judged against, if a trace covered it."""
        return None if self.roofline is None else self.roofline.measured_us

    @property
    def headroom(self) -> float | None:
        """How many times the per-op floor the step took: 1.0 means every kernel dense."""
        if self.roofline is None or not self.roofline.ideal_us or self.measured_us is None:
            return None
        return self.measured_us / self.roofline.ideal_us


def build(bundle: Bundle) -> Report:
    """Analyse one bundle against the hardware document it states.

    Raises:
        KeyError: The bundle names a device no installed document describes.
    """
    document = load(bundle.device_id)
    device_limits = limits(document)
    graph = bundle.graph
    priced = (
        None if graph is None else roofline(graph.ops, graph.by_id, device_limits, bundle.trace)
    )
    step = None if bundle.trace is None else timeline(bundle.trace)
    return Report(
        bundle_digest=bundle.digest(),
        device_id=device_limits.device_id,
        device_label=device_limits.label,
        limits_digest=device_limits.digest,
        capture=bundle.capture,
        environment=bundle.environment,
        roofline=priced,
        timeline=step,
        findings=detect(graph=graph, limits=device_limits, priced=priced, step=step)
        if priced is not None
        else (),
        unmodelled=()
        if priced is None
        else tuple(f"{op.target} x{op.scale}" for op, _ in priced.unpriced),
    )


def headline(report: Report) -> Table:
    """The three numbers a reader needs before any finding."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", justify="right")
    table.add_column()
    measured = report.measured_us
    if report.roofline is not None:
        table.add_row("measured", "no trace" if measured is None else f"{measured:.1f} us per step")
        table.add_row("per-op floor", f"{report.roofline.ideal_us:.1f} us")
        table.add_row("absolute floor", f"{report.roofline.floor_us:.1f} us")
        headroom = report.headroom
        table.add_row(
            "headroom",
            "unknown" if headroom is None else f"{headroom:.1f}x the per-op floor",
        )
        table.add_row("priced coverage", f"{report.roofline.coverage:.1%} of modelled traffic")
    if report.timeline is not None:
        table.add_row(
            "device busy",
            f"{report.timeline.busy_us:.1f} us of {report.timeline.span_us:.1f} us per step",
        )
        table.add_row(
            "kernels",
            f"{report.timeline.kernels} per step, mean {report.timeline.mean_kernel_us:.1f} us",
        )
    return table


def finding_panel(index: int, finding: Finding) -> Panel:
    """One finding as a panel: what was seen, what it could be worth, what it needs."""
    ceiling = "no ceiling" if finding.ceiling_us is None else f"up to {finding.ceiling_us:.1f} us"
    if finding.ceiling_share is not None:
        ceiling += f"  ({finding.ceiling_share:.0%} of the step)"
    body = Group(
        Text(finding.title, style="bold"),
        *(Text(f"· {line}", style="dim") for line in finding.evidence),
        Text(f"→ {finding.action}", style="green"),
        *(Text(f"requires {requirement}", style="yellow") for requirement in finding.requires),
    )
    return Panel(
        body, title=f"{index}. {finding.waste} · {ceiling}", border_style="cyan", title_align="left"
    )


def render(report: Report, console: Console) -> None:
    """Write the report to a console."""
    header = Panel(
        Group(
            Text(f"{report.device_label}  ({report.device_id})", style="bold"),
            Text(
                f"{report.capture.entry}  ·  {report.capture.iters} iterations, {report.capture.warmup} warmup"
                f"  ·  graph: {report.capture.graph_source}",
                style="dim",
            ),
            Text(
                f"bundle {report.bundle_digest[:16]}  ·  facts {report.limits_digest[:16]}",
                style="dim",
            ),
            headline(report),
        ),
        title="trainer-rsi",
        border_style="blue",
    )
    console.print(header)
    for index, finding in enumerate(report.findings, start=1):
        console.print(finding_panel(index, finding))
    if not report.findings:
        console.print("[yellow]no finding cleared its threshold[/yellow]")
    if report.unmodelled:
        console.print(
            Panel(
                "\n".join(report.unmodelled),
                title=f"{len(report.unmodelled)} ops have no cost rule",
                border_style="yellow",
            )
        )


def as_json(report: Report) -> dict[str, Any]:
    """The report as JSON-compatible data."""
    return report.jsonable()
