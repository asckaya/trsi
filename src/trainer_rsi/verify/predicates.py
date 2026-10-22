from typing import Literal

import numpy as np

from trainer_rsi.base import Value

NEAR_ZERO = 1e-12
"""Below this a relative measure divides by nothing and stops meaning anything."""


class AllClose(Value):
    """Per-element agreement within an absolute and a relative slack.

    Per element, not on average: one wrong element is a total failure, and an
    aggregate over indices hides it.
    """

    kind: Literal["allclose"] = "allclose"
    atol: float
    rtol: float


class RelL2(Value):
    """Relative L2 error of the whole tensor against a bound."""

    kind: Literal["rell2"] = "rell2"
    bound: float


class Exact(Value):
    """Element-for-element equality, for indices and masks."""

    kind: Literal["exact"] = "exact"


type Bound = AllClose | RelL2 | Exact


class Verdict(Value):
    """What one predicate measured on one output, and whether it held."""

    output: str
    predicate: str
    metric: float
    reference_norm: float | None
    passed: bool
    note: str = ""


def is_discrete(values: np.ndarray) -> bool:
    """Whether an output carries indices or masks rather than approximate values."""
    return not np.issubdtype(values.dtype, np.floating)


def measure(
    output: str, bound: Bound, candidate: np.ndarray, reference: np.ndarray | None
) -> Verdict:
    """Judge one output against one bound.

    Args:
        output: Name of the output, as the program returned it.
        bound: The predicate and its bound.
        candidate: What the implementation produced.
        reference: What the reference produced, absent for a one-sided predicate.

    Returns:
        The verdict, carrying the measured metric and the reference's own scale.

    Raises:
        ValueError: A two-sided predicate ran without a reference, the shapes
            differ, or an aggregate ran on a discrete output.
    """
    if reference is not None and candidate.shape != reference.shape:
        raise ValueError(
            f"{output}: candidate {candidate.shape} and reference {reference.shape} differ"
        )
    match bound:
        case Exact():
            if reference is None:
                raise ValueError(f"{output}: exact comparison needs a reference")
            wrong = int(np.count_nonzero(candidate != reference))
            return Verdict(
                output=output,
                predicate="exact",
                metric=float(wrong),
                reference_norm=None,
                passed=wrong == 0,
                note=f"{wrong} elements differ",
            )
        case AllClose():
            if reference is None:
                raise ValueError(f"{output}: allclose needs a reference")
            if is_discrete(candidate):
                raise ValueError(f"{output}: allclose on a discrete output; use exact")
            slack = bound.atol + bound.rtol * np.abs(reference)
            violation = float(np.max(np.abs(candidate - reference) - slack))
            return Verdict(
                output=output,
                predicate="allclose",
                metric=violation,
                reference_norm=float(np.linalg.norm(reference)),
                passed=violation <= 0.0,
            )
        case RelL2():
            if reference is None:
                raise ValueError(f"{output}: relative error needs a reference")
            if is_discrete(candidate):
                raise ValueError(f"{output}: relative error on a discrete output; use exact")
            scale = float(np.linalg.norm(reference))
            if scale < NEAR_ZERO:
                raise ValueError(
                    f"{output}: reference norm {scale:g} is too near zero for a relative measure"
                )
            error = float(np.linalg.norm(candidate.astype(np.float64) - reference) / scale)
            return Verdict(
                output=output,
                predicate="rell2",
                metric=error,
                reference_norm=scale,
                passed=error <= bound.bound,
            )


def judge(
    produced: dict[str, np.ndarray],
    bounds: dict[str, Bound],
    reference: dict[str, np.ndarray] | None = None,
) -> tuple[Verdict, ...]:
    """Judge every output against the bound its caller stated.

    There is no default bound and no default tolerance: a number nobody stated
    is a number nobody can fail, which reads the same as a pass.

    Raises:
        ValueError: The judged set is empty, an output was left unjudged, a bound
            names an output that was not produced, or a two-sided bound has no
            reference.
    """
    if not produced:
        raise ValueError("nothing was produced, which is not agreement")
    if not bounds:
        raise ValueError("no bounds were stated, so nothing can pass")
    unjudged = sorted(set(produced) - set(bounds))
    if unjudged:
        raise ValueError(f"outputs left unjudged: {unjudged}")
    invented = sorted(set(bounds) - set(produced))
    if invented:
        raise ValueError(f"bounds name outputs that were not produced: {invented}")
    if reference is None:
        two_sided = sorted(name for name, bound in bounds.items() if not isinstance(bound, Exact))
        if two_sided:
            raise ValueError(f"no reference was supplied, and these need one: {two_sided}")
    return tuple(
        measure(name, bounds[name], produced[name], None if reference is None else reference[name])
        for name in sorted(produced)
    )
