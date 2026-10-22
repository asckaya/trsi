from pathlib import Path

from trainer_rsi.facts.document import HardwareDoc, load_document


def installed() -> dict[str, HardwareDoc]:
    """Every installed hardware document, keyed by id."""
    root = Path(__file__).parent / "hardware"
    return {doc.id: doc for doc in (load_document(path) for path in sorted(root.glob("*.toml")))}


def load(device_id: str) -> HardwareDoc:
    """Return the installed document for *device_id*.

    Raises:
        KeyError: No such document is installed, listing the ids that are.
    """
    docs = installed()
    try:
        return docs[device_id]
    except KeyError:
        raise KeyError(f"no hardware document {device_id!r}; installed: {sorted(docs)}") from None
