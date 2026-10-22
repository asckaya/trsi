import platform
import subprocess
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from dulwich import porcelain
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo
from torch.export import ExportedProgram

from trainer_rsi.capture.graph import graph_of
from trainer_rsi.capture.kineto import read_chrome
from trainer_rsi.ir.bundle import Bundle, Capture, Environment, new_bundle, write_bundle
from trainer_rsi.ir.graph import Graph, Mesh


def git_state(directory: Path) -> tuple[str, bool, str]:
    """The commit, whether the tree is dirty, and the repository root.

    A checkout with no commit yet reports ``none``: it is a repository, and the
    code it holds has no commit to name.
    """
    try:
        repo = Repo.discover(str(directory))
    except NotGitRepository:
        return "none", False, ""
    commit = repo.get_refs().get(b"HEAD", b"").decode() or "none"
    status = porcelain.status(repo)
    worktree = repo.get_worktree()
    root = repo.path if worktree is None else worktree.path
    return commit, bool(status.staged or status.unstaged or status.untracked), str(root)


def driver_version() -> str:
    """The installed NVIDIA driver version, as the driver itself reports it."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return query.stdout.strip().splitlines()[0]


def environment_of(device_id: str) -> Environment:
    """The software, hardware and checkout state a capture ran against."""
    commit, dirty, root = git_state(Path.cwd())
    accelerated = torch.cuda.is_available()
    return Environment(
        device_id=device_id,
        python=platform.python_version(),
        torch=torch.__version__,
        device_name=torch.cuda.get_device_name(0) if accelerated else "cpu",
        torch_cuda=torch.version.cuda or "none",
        driver=driver_version() if accelerated else "none",
        git_commit=commit,
        git_dirty=dirty,
        git_root=root,
    )


@contextmanager
def record(
    *,
    device_id: str,
    out: Path,
    step: str = "train_step",
    steps: int = 1,
    mesh: Mesh | None = None,
    module: torch.nn.Module | None = None,
    example: Sequence[object] | None = None,
    activities: tuple[str, ...] = ("cpu", "cuda"),
    notes: tuple[str, ...] = (),
) -> Generator[None]:
    """Measure the block it wraps and write a bundle describing it.

    The block is the measured window: open it after warmup and close it after the
    steps whose cost the report should describe. Nothing outside the window is
    profiled, so a slow first iteration cannot be reported as the steady state.

    Args:
        device_id: Installed hardware document id the run is on.
        out: Where to write the bundle.
        step: Label for the captured step.
        steps: Iterations the window contains, so per-step figures divide honestly.
        mesh: Parallel axes the run is spread over; single device by default.
        module: A module to export for the graph half, with *example* inputs.
        example: Positional example inputs for *module*.
        activities: Profiler activities.
        notes: Anything a reader should know about the capture.

    Raises:
        ValueError: A module was given without example inputs, or only one of them.
        RuntimeError: The activities name CUDA and no accelerator is visible.
    """
    if (module is None) != (example is None):
        raise ValueError("exporting the graph needs both a module and its example inputs")
    if "cuda" in activities and not torch.cuda.is_available():
        raise RuntimeError(
            "the activities name cuda and no accelerator is visible; "
            'capture on the device, or pass activities=("cpu",) for a graph-only bundle'
        )
    trace_path = Path(tempfile.mkdtemp(prefix="trainer-rsi-")) / "trace.json"
    with torch.profiler.profile(activities=profiler_activities(activities)) as profiler:
        yield
    profiler.export_chrome_trace(str(trace_path))
    trace = read_chrome(trace_path, device_id=device_id, steps=steps)
    if trace is None:
        notes = (
            *notes,
            "no device event in the trace: this bundle carries a graph and no timeline",
        )
    program: ExportedProgram | None = None
    if module is not None and example is not None:
        program = torch.export.export(module, tuple(example))
    graph: Graph | None = (
        None if program is None else graph_of(program, mesh=mesh or Mesh(), step=step)
    )
    bundle: Bundle = new_bundle(
        environment=environment_of(device_id),
        capture=Capture(
            entry=step,
            iters=steps,
            warmup=0,
            activities=activities,
            graph_source="none" if program is None else "torch.export",
        ),
        graph=graph,
        trace=trace,
        notes=notes,
    )
    write_bundle(out, bundle)


def profiler_activities(names: tuple[str, ...]) -> list[torch.profiler.ProfilerActivity]:
    """Map activity names to the profiler's enum.

    Raises:
        KeyError: A name the profiler has no activity for.
    """
    known = {
        "cpu": torch.profiler.ProfilerActivity.CPU,
        "cuda": torch.profiler.ProfilerActivity.CUDA,
    }
    return [known[name] for name in names]
