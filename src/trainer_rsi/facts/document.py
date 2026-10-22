import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from trainer_rsi.base import Value

SCHEMA = "trainer_rsi.hardware/v1"

type Origin = Literal["vendor", "derived", "reference", "measured"]


class Fact(Value):
    """One figure with the provenance and the conditions that make it true.

    A figure the vendor does not publish is recorded as `status="unavailable"`
    rather than omitted: omission and unavailability are different answers, and
    an analysis that conflates them reports a bound it did not earn.
    """

    value: float | int | str | None = None
    unit: str
    origin: Origin
    source: str
    conditions: str
    status: Literal["stated", "unavailable"] = "stated"

    @model_validator(mode="after")
    def status_matches_value(self) -> Fact:
        """Reject a fact that both states a number and states it is unavailable."""
        if self.status == "unavailable" and self.value is not None:
            raise ValueError(f"fact states status=unavailable and value={self.value!r}")
        if self.status == "stated" and self.value is None:
            raise ValueError("fact states status=stated and no value")
        return self


class HardwareDoc(Value):
    """One device's installed facts, addressed by dotted path."""

    id: str
    label: str
    architecture: str
    facts: dict[str, Fact]
    digest: str
    """SHA-256 of the installed document, so a report names the exact figures it used."""
    path: str

    def fact(self, path: str) -> Fact:
        """Return one fact by dotted path.

        Raises:
            KeyError: No fact is installed at that path, listing what is.
        """
        try:
            return self.facts[path]
        except KeyError:
            raise KeyError(
                f"{self.id} has no fact {path!r}; installed: {sorted(self.facts)}"
            ) from None

    def amount(self, path: str) -> float:
        """Return one stated numeric fact.

        Raises:
            KeyError: The fact is absent or recorded as unavailable.
            TypeError: The fact is stated but is not a number.
        """
        fact = self.fact(path)
        if fact.status == "unavailable":
            raise KeyError(f"{self.id}.{path} is unavailable: {fact.conditions}")
        if not isinstance(fact.value, int | float):
            raise TypeError(f"{self.id}.{path} is {type(fact.value).__name__}, not a number")
        return float(fact.value)

    def maybe(self, path: str) -> float | None:
        """Return a stated numeric fact, or `None` when the document says unavailable."""
        if self.fact(path).status == "unavailable":
            return None
        return self.amount(path)

    def size(self, path: str) -> int:
        """Return one stated fact that counts things or bytes."""
        return int(self.amount(path))


def is_fact_table(node: dict[str, object]) -> bool:
    """Whether a toml table is a fact rather than a group of them."""
    return "unit" in node or "status" in node


def flatten_facts(node: dict[str, object], prefix: str) -> dict[str, Fact]:
    """Flatten nested fact tables into dotted paths, refusing anything else.

    Raises:
        TypeError: A leaf is not a fact table.
    """
    leaves: dict[str, Fact] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        if not isinstance(value, dict):
            raise TypeError(f"{path} is {type(value).__name__}, not a fact table")
        if is_fact_table(value):
            leaves[path] = Fact.model_validate(value)
        else:
            leaves |= flatten_facts(value, f"{path}.")
    return leaves


def load_document(path: Path) -> HardwareDoc:
    """Read and validate one installed hardware document.

    Raises:
        ValueError: The document states a schema or kind this build does not accept.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    spec = raw.get("spec", {})
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {spec.get('schema')!r} is not {SCHEMA!r}")
    if spec.get("kind") != "device":
        raise ValueError(f"{path}: kind {spec.get('kind')!r} is not 'device'")
    return HardwareDoc(
        id=spec["id"],
        label=spec["label"],
        architecture=raw["compatibility"]["architecture"],
        facts=flatten_facts(raw["facts"], ""),
        digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        path=str(path),
    )
