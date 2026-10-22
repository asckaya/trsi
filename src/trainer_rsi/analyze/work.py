from math import prod

from trainer_rsi.base import Value
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Op, Tensor

COLLECTIVE_FACTOR: dict[str, float] = {
    "all_reduce": 2.0,
    "reduce_scatter": 1.0,
    "all_gather": 1.0,
    "all_to_all": 1.0,
    "broadcast": 1.0,
    "send_recv": 1.0,
}
"""Bytes a rank moves per byte of payload on a ring.

All-reduce is the 2(n-1)/n term of a ring, which is at most two payloads; the
others move each payload once. A tree or two-shot algorithm changes the
constant, not the payload, and no installed document prices one.
"""

OPTIMIZER_FLOPS_PER_ELEMENT = 10
OPTIMIZER_READS = 4
OPTIMIZER_WRITES = 3
"""AdamW per element: m, v and the update at three multiply-adds each, one bias
correction, and seven element accesses (param twice, grad, m twice, v twice)."""


class Work(Value):
    """The arithmetic and traffic one op owes, independent of any machine."""

    flops: int
    bytes_in: int
    bytes_out: int
    dtype: DType
    """The type the arithmetic is issued in, which selects the peak rate."""
    priced: bool
    note: str = ""

    @property
    def bytes_total(self) -> int:
        """Traffic an unfused kernel for this op would cross memory with."""
        return self.bytes_in + self.bytes_out


def price(op: Op, tensors: dict[str, Tensor]) -> Work:
    """Price one op from its shapes and dtype.

    Elementwise arithmetic is counted at one flop per output element per input,
    which undercounts transcendentals. That is deliberate: those ops are priced
    to be memory bound, and inflating their flop count would move a bound the
    memory side already sets.
    """
    ins = tuple(tensors[name] for name in op.inputs)
    outs = tuple(tensors[name] for name in op.outputs)
    out = outs[0]
    match op.kind:
        case "matmul":
            first = ins[0]
            rows = prod(out.shape[:-1])
            flops = 2 * rows * out.shape[-1] * first.shape[-1]
            return Work(
                flops=flops,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=first.dtype,
                priced=True,
            )
        case "attention":
            batch, heads, queries = ins[0].shape[:3]
            head_dim = ins[0].shape[-1]
            keys = ins[1].shape[-2]
            flops = 4 * batch * heads * queries * keys * head_dim
            return Work(
                flops=flops,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=ins[0].dtype,
                priced=True,
            )
        case "conv":
            per_output = 2 * prod(ins[1].shape[1:])
            return Work(
                flops=out.numel * per_output,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=ins[0].dtype,
                priced=True,
            )
        case "norm":
            return Work(
                flops=5 * out.numel,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=True,
            )
        case "softmax":
            return Work(
                flops=4 * out.numel,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=True,
            )
        case "elementwise":
            return Work(
                flops=(len(ins) + 1) * out.numel,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=True,
            )
        case "reduce":
            return Work(
                flops=ins[0].numel,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=True,
            )
        case "movement":
            return Work(
                flops=0,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=True,
            )
        case "gather":
            return Work(
                flops=0,
                bytes_in=sum(t.nbytes for t in ins) + out.nbytes,
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=True,
            )
        case "collective":
            algorithm = str(op.attrs.get("algorithm", "all_reduce"))
            moved = int(sum(t.nbytes for t in ins) * COLLECTIVE_FACTOR[algorithm])
            return Work(
                flops=0,
                bytes_in=moved,
                bytes_out=0,
                dtype=out.dtype,
                priced=True,
                note=f"{algorithm} x{COLLECTIVE_FACTOR[algorithm]}",
            )
        case "optimizer":
            width = out.dtype.nbytes(out.numel)
            return Work(
                flops=OPTIMIZER_FLOPS_PER_ELEMENT * out.numel,
                bytes_in=OPTIMIZER_READS * width,
                bytes_out=OPTIMIZER_WRITES * width,
                dtype=out.dtype,
                priced=True,
            )
        case "constant" | "view":
            return Work(flops=0, bytes_in=0, bytes_out=0, dtype=out.dtype, priced=True)
        case "other":
            return Work(
                flops=0,
                bytes_in=sum(t.nbytes for t in ins),
                bytes_out=out.nbytes,
                dtype=out.dtype,
                priced=False,
                note=op.target,
            )
