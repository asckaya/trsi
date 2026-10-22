import operator

import torch
from torch.export import ExportedProgram
from torch.fx import Node

from trainer_rsi.capture.aten import PARAMETER_KINDS, kind_of
from trainer_rsi.ir.dtype import DType
from trainer_rsi.ir.graph import Graph, Mesh, Op, Role, Tensor

GETITEM = "operator.getitem"


def elements(node: Node) -> tuple[torch.Tensor, ...]:
    """The tensors a graph node's recorded value holds, in order."""
    value = node.meta["val"]
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(item for item in value if isinstance(item, torch.Tensor))


def node_refs(value: object) -> tuple[Node, ...]:
    """Every graph node reachable from one argument position, outermost first."""
    if isinstance(value, Node):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(node for item in value for node in node_refs(item))
    return ()


def roles_of(program: ExportedProgram) -> dict[str, Role]:
    """Which graph inputs are weights rather than activations."""
    roles: dict[str, Role] = {}
    for spec in program.graph_signature.input_specs:
        name = spec.arg.name
        if name is None:
            continue
        kind = spec.kind.name.lower()
        roles[name] = "parameter" if kind in PARAMETER_KINDS else "input"
    return roles


def tensor_of(name: str, value: torch.Tensor, role: Role) -> Tensor:
    """One graph tensor from an exported value."""
    return Tensor(
        id=name,
        shape=tuple(value.shape),
        dtype=DType.of_torch(str(value.dtype)),
        role=role,
        strides=tuple(value.stride()),
    )


def graph_of(program: ExportedProgram, *, mesh: Mesh, step: str, source_digest: str = "") -> Graph:
    """Turn an exported program into the program graph the analysis prices.

    Args:
        program: The output of `torch.export.export`.
        mesh: The parallel axes the captured run was spread over.
        step: Label for the step.
        source_digest: Digest of the code that produced the program.

    Returns:
        The graph, with one node per exported call and one tensor per value.

    Raises:
        ValueError: A node produced no tensor, so the graph would be incomplete.
    """
    roles = roles_of(program)
    tensors: dict[str, Tensor] = {}
    ops: list[Op] = []
    for node in program.graph_module.graph.nodes:
        if node.op == "placeholder":
            for value in elements(node):
                tensors[node.name] = tensor_of(node.name, value, roles[node.name])
            continue
        if node.op != "call_function":
            continue
        values = elements(node)
        if not values:
            raise ValueError(f"{node.name} ({node.target}) produced no tensor to price")
        is_getitem = node.target is operator.getitem
        target = GETITEM if is_getitem else str(node.target)
        if is_getitem:
            parent, index = node.args
            source = parent.name if index == 0 else f"{parent.name}#{index}"
            outputs = (node.name,)
            inputs = (source,)
            kind = "view"
        else:
            outputs = (node.name, *(f"{node.name}#{i}" for i in range(1, len(values))))
            inputs = tuple(
                reference.name for argument in node.args for reference in node_refs(argument)
            )
            kind = kind_of(target)
        for position, value in enumerate(values):
            tensors[outputs[position]] = tensor_of(outputs[position], value, "activation")
        stack = node.meta.get("nn_module_stack") or {}
        module_path = next(reversed(stack), "") if stack else ""
        ops.append(
            Op(
                id=node.name,
                target=target,
                kind=kind,
                phase="forward",
                inputs=inputs,
                outputs=outputs,
                module_path=module_path.removeprefix("L__self__").lstrip("."),
            )
        )
    return Graph(
        tensors=tuple(tensors.values()),
        ops=tuple(ops),
        mesh=mesh,
        step=step,
        source_digest=source_digest,
    )
