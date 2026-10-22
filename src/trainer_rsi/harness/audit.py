from pathlib import Path

from trainer_rsi.harness.contract import Audit, Claim, Gate, Report, Verified
from trainer_rsi.verify.stats import Comparison

SPEED_TOLERANCE = 0.05
"""How far a claimed ratio may sit from the measured one before it is contradicted."""


def speed_verdict(claim: Claim, comparison: Comparison) -> Verified:
    """Judge one speed claim against the ratio the harness measured."""
    if claim.speedup is None:
        return Verified(
            claim=claim,
            verdict="unbacked",
            speedup=comparison.speedup,
            detail="a speed claim with no ratio cannot be checked against one",
        )
    off = abs(claim.speedup - comparison.speedup) / comparison.speedup
    if off > SPEED_TOLERANCE:
        return Verified(
            claim=claim,
            verdict="contradicted",
            speedup=comparison.speedup,
            low=comparison.low,
            high=comparison.high,
            detail=f"claimed x{claim.speedup:.3f}, measured x{comparison.speedup:.3f} "
            f"[{comparison.low:.3f}, {comparison.high:.3f}]",
        )
    if comparison.claim != "faster":
        return Verified(
            claim=claim,
            verdict="indistinguishable",
            speedup=comparison.speedup,
            low=comparison.low,
            high=comparison.high,
            detail=f"the samples do not separate the two: {comparison.claim}",
        )
    return Verified(
        claim=claim,
        verdict="supported",
        speedup=comparison.speedup,
        low=comparison.low,
        high=comparison.high,
        detail=f"x{comparison.speedup:.3f} [{comparison.low:.3f}, {comparison.high:.3f}]",
    )


def correctness_verdict(claim: Claim, workspace: Path) -> Verified:
    """Judge one correctness claim by whether the files it names exist.

    The harness does not re-run the agent's verifier: it states that it did not,
    rather than reporting a check it did not perform.
    """
    missing = [
        name
        for name in claim.evidence
        if not (workspace / name).resolve().is_relative_to(workspace.resolve())
        or not (workspace / name).exists()
    ]
    if not claim.evidence or missing:
        return Verified(
            claim=claim,
            verdict="unbacked",
            detail=f"no verifier output to read: {missing or 'none named'}",
        )
    return Verified(
        claim=claim,
        verdict="checked",
        detail="the named output exists; the harness did not re-run the verifier",
    )


def audit(
    report: Report | None,
    *,
    workspace: Path,
    comparison: Comparison,
    gate: Gate | None,
) -> Audit:
    """Check the agent's account against what the harness measured.

    Acceptance belongs to the measurement, not to the prose: a run is accepted
    when the samples separate the two programs and the gate passed. The report is
    judged for honesty, and a contradicted claim is a finding either way.

    The comparison is the loop's, not this function's: resampling the same two
    sets twice would let the audit disagree with the panel beside it.
    """
    findings: list[str] = []
    verified: list[Verified] = []
    if gate is not None and gate.exit_code != 0:
        findings.append(f"the gate exited {gate.exit_code}: {gate.output}")
    if report is None:
        findings.append("no report was produced, so nothing the agent says can be checked")
    else:
        for name in report.changed:
            if not (workspace / name).exists():
                findings.append(f"the report names a changed file that does not exist: {name}")
        for claim in report.claims:
            verdict = (
                speed_verdict(claim, comparison)
                if claim.kind == "speed"
                else correctness_verdict(claim, workspace)
            )
            verified.append(verdict)
            if verdict.verdict in ("contradicted", "unbacked"):
                findings.append(f"{verdict.verdict}: {claim.statement} — {verdict.detail}")
    return Audit(
        accepted=comparison.claim == "faster" and not findings,
        verified=tuple(verified),
        findings=tuple(findings),
    )
