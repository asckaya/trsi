from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from trainer_rsi.plan.candidates import Plan, combination


def us(value: float | None) -> str:
    """A duration a reader can compare against the others on the page."""
    return "n/a" if value is None else f"{value:.1f} us"


def plan_panel(index: int, plan: Plan) -> Panel:
    """One rewrite with what it replaces, what it would owe, and what it needs first."""
    saving = "no saving" if not plan.delta_us else f"save {plan.delta_us:.1f} us"
    if plan.delta_share is not None:
        saving += f"  ({plan.delta_share:.1%} of the step)"
    body = Group(
        Text(plan.title, style="bold"),
        Text(f"· {us(plan.now_us)} -> {us(plan.then_us)} at the roofline", style="dim"),
        Text(f"· {plan.change}", style="dim"),
        Text(f"→ {saving}", style="green"),
        *(Text(f"requires {requirement}", style="yellow") for requirement in plan.requires),
    )
    return Panel(body, title=f"{index}. {plan.kind}", border_style="magenta", title_align="left")


def summary(plans: tuple[Plan, ...], measured_us: float | None, floor_us: float | None) -> Text:
    """What the plans add up to, and whether they could all be true at once."""
    taken = combination(plans)
    total = sum(plan.delta_us or 0.0 for plan in taken)
    text = Text()
    text.append(f"{len(taken)} disjoint plans ", style="bold")
    text.append(f"predict {total:.1f} us per step", style="green")
    if measured_us:
        text.append(
            f" out of {measured_us:.1f} us measured ({total / measured_us:.0%})", style="dim"
        )
        predicted = measured_us - total
        if floor_us is not None and predicted < floor_us:
            text.append(
                f"\nnot all of them can hold: they predict {predicted:.1f} us,"
                f" below the {floor_us:.1f} us floor this step cannot beat",
                style="red",
            )
    return text


def render(
    console: Console, plans: tuple[Plan, ...], *, measured_us: float | None, floor_us: float | None
) -> None:
    """Write the ranked plans and their summary."""
    for index, plan in enumerate(plans, start=1):
        console.print(plan_panel(index, plan))
    console.print(Panel(summary(plans, measured_us, floor_us), border_style="green", title="sum"))


def as_json(
    plans: tuple[Plan, ...], *, measured_us: float | None, floor_us: float | None
) -> dict[str, Any]:
    """The plans as data, with the same summary the text rendering states."""
    taken = combination(plans)
    return {
        "plans": [plan.jsonable() for plan in plans],
        "combination": [plan.id for plan in taken],
        "predicted_saving_us": sum(plan.delta_us or 0.0 for plan in taken),
        "measured_us": measured_us,
        "floor_us": floor_us,
    }
