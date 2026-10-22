import gzip
import hashlib
import time
from pathlib import Path

from pydantic import Field

from trainer_rsi.base import Value
from trainer_rsi.ir.graph import Graph
from trainer_rsi.ir.trace import Trace

SCHEMA = "trainer_rsi.bundle/v1"


class Capture(Value):
    """How the bundle was taken, so a reader can judge what it does not cover."""

    entry: str
    """The command or callable label the capture ran."""
    iters: int
    warmup: int
    activities: tuple[str, ...]
    graph_source: str = "none"
    """``torch.export``, ``manual``, or ``none`` when only a trace was taken."""


class Environment(Value):
    """The software and hardware the capture ran against."""

    device_id: str
    python: str
    torch: str
    device_name: str
    """The runtime's own name for the device, e.g. ``NVIDIA H200``."""
    torch_cuda: str = "none"
    driver: str = "none"
    git_commit: str = "none"
    git_dirty: bool = False
    git_root: str = ""


class Bundle(Value):
    """One capture: the graph, the trace, and the conditions they were taken under.

    A bundle is the unit of exchange between the machine that owns the model and
    the machine that reasons about it. The hardware document id travels with it,
    so a report can never be read against the wrong envelope, and a digest covers
    the content so a cached analysis can prove what it analysed.
    """

    schema_version: str = SCHEMA
    created_unix: int
    environment: Environment
    capture: Capture
    graph: Graph | None = None
    trace: Trace | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def device_id(self) -> str:
        """The hardware document both halves of this bundle ran on."""
        return self.environment.device_id

    def digest(self) -> str:
        """Content digest over the canonical serialisation."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def new_bundle(
    *,
    environment: Environment,
    capture: Capture,
    graph: Graph | None = None,
    trace: Trace | None = None,
    notes: tuple[str, ...] = (),
) -> Bundle:
    """Assemble a bundle stamped with the current time.

    Raises:
        ValueError: Neither a graph nor a trace was supplied.
    """
    if graph is None and trace is None:
        raise ValueError("a bundle needs a graph, a trace, or both")
    return Bundle(
        created_unix=int(time.time()),
        environment=environment,
        capture=capture,
        graph=graph,
        trace=trace,
        notes=notes,
    )


def write_bundle(path: Path, bundle: Bundle) -> str:
    """Write *bundle* to *path* as gzipped JSON and return the digest of its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.model_dump_json(indent=2).encode()
    path.write_bytes(gzip.compress(payload, mtime=0))
    return hashlib.sha256(payload).hexdigest()


def read_bundle(path: Path) -> Bundle:
    """Read a bundle written by [`write_bundle`][trainer_rsi.ir.bundle.write_bundle].

    Raises:
        ValueError: The file states a schema this build does not read.
    """
    bundle = Bundle.model_validate_json(gzip.decompress(path.read_bytes()))
    if bundle.schema_version != SCHEMA:
        raise ValueError(
            f"bundle schema {bundle.schema_version!r} is not {SCHEMA!r}; re-capture it"
        )
    return bundle
