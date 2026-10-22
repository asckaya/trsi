from typing import Literal

from pydantic import Field

from trainer_rsi.base import Value
from trainer_rsi.ir.dtype import DType

type Role = Literal[
    "parameter",
    "gradient",
    "optimizer_state",
    "activation",
    "input",
    "output",
    "temporary",
]

type Kind = Literal[
    "matmul",
    "attention",
    "conv",
    "norm",
    "softmax",
    "elementwise",
    "reduce",
    "movement",
    "view",
    "gather",
    "collective",
    "optimizer",
    "constant",
    "other",
]

type Phase = Literal["forward", "backward", "optimizer", "data"]


class Tensor(Value):
    """One materialised tensor: its shape, its dtype, and why it exists."""

    id: str
    shape: tuple[int, ...]
    dtype: DType
    role: Role
    strides: tuple[int, ...] | None = None

    @property
    def numel(self) -> int:
        """Element count, treating a rank-0 tensor as one element."""
        total = 1
        for extent in self.shape:
            total *= extent
        return total

    @property
    def nbytes(self) -> int:
        """Storage this tensor occupies, in bytes."""
        return self.dtype.nbytes(self.numel)


class Op(Value):
    """One operation: what it consumes, what it produces, and where it sits."""

    id: str
    target: str
    """Captured spelling, e.g. ``aten.linear.default``; the cost table keys on it."""
    kind: Kind
    phase: Phase
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    module_path: str = ""
    """Dotted module path from the model root, empty for host-side ops."""
    attrs: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @property
    def scale(self) -> int:
        """How many calls capture merged into this node."""
        return int(self.attrs.get("count", 1))


class Mesh(Value):
    """The parallel axes a step is spread over, each defaulting to one device."""

    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1

    @property
    def world(self) -> int:
        """Total ranks the step runs on."""
        return self.dp * self.tp * self.pp * self.cp * self.ep


class Graph(Value):
    """A captured step as dataflow over tensors, stating work and nothing about time.

    Timings live in `trainer_rsi.ir.trace`. Keeping the two apart is what makes
    a cost model falsifiable: the graph states an amount of work, the trace
    states a duration, and the analysis is the disagreement between them.

    Ops the capture layer could not map are kept as `kind="other"` rather than
    dropped, so a report can say what fraction of the step it could not speak
    about instead of silently narrowing its own scope.
    """

    tensors: tuple[Tensor, ...]
    ops: tuple[Op, ...]
    mesh: Mesh = Field(default_factory=Mesh)
    step: str = ""
    source_digest: str = ""

    @property
    def by_id(self) -> dict[str, Tensor]:
        """Tensors indexed for lookup during pricing."""
        return {tensor.id: tensor for tensor in self.tensors}

    def producers(self) -> dict[str, Op]:
        """Map each tensor to the single op that wrote it.

        Raises:
            ValueError: Two ops claim the same output, so this is not a graph.
        """
        written: dict[str, Op] = {}
        for op in self.ops:
            for out in op.outputs:
                if out in written:
                    raise ValueError(f"tensor {out!r} written by {written[out].id!r} and {op.id!r}")
                written[out] = op
        return written

    def consumers(self) -> dict[str, tuple[Op, ...]]:
        """Map each tensor to every op that reads it."""
        read: dict[str, list[Op]] = {}
        for op in self.ops:
            for name in op.inputs:
                read.setdefault(name, []).append(op)
        return {name: tuple(ops) for name, ops in read.items()}
