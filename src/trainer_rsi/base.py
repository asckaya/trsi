from typing import Any

from pydantic import BaseModel, ConfigDict


class Value(BaseModel):
    """Base for every data structure in this project.

    Frozen, strict, and closed: an instance cannot be edited after it is built,
    a field will not quietly coerce one type into another, and a misspelled key
    is an error rather than a silent extra. Every one of these models is also a
    wire format, so the same rules hold for JSON read back off a bundle or off
    an agent's structured output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def jsonable(self) -> dict[str, Any]:
        """The model as plain JSON-compatible data, for a report or an artifact."""
        return self.model_dump(mode="json")
