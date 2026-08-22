"""
Module for all functions useful for :
- partitioning
- managing path
- plotting
"""

import contextlib
import io

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from .distribution import generate, iid, noniid
from .path import DATA_PATH, EXP_PATH, create, iterate
from .plot import chart

__all__ = [
    "generate",
    "iid",
    "noniid",
    "create",
    "iterate",
    "tracker",
    "dataset",
    "toplot",
    "DATA_PATH",
    "EXP_PATH",
]


def tracker(
    dataname, nworkers, label_distrb, volume_distrb, minlabels=3, balanced=True, k=None
):
    d = dataname.lower()
    n = str(nworkers)
    v = "Vi" if volume_distrb == "iid" else "Vni"
    l = "Li" if label_distrb == "iid" else "Lni"
    b = "bal" if balanced else "unbal"
    m = "" if label_distrb == "iid" else str(minlabels)
    k = "" if k is None else "-" + str(k)
    bb = "-" + b if label_distrb == "noniid" else ""
    return "data-" + d + "-" + "".join((n, l, m, v)) + bb + k


def dataset(name: str, live: Live, panel: Panel, texts: list[Align]):
    path = DATA_PATH / name
    isavailable = path.exists()
    datasetfromtorch = hasattr(datasets, name)
    cifar10 = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if isavailable or datasetfromtorch:
        transform = cifar10 if name == "CIFAR10" else ToTensor()
        loader = getattr(datasets, name)

        if not isavailable:
            texts[-1] = Align.center(f"[yellow]Downloading {name!r} dataset ...[/]")
            panel.renderable = Group(*texts)
            live.refresh()

        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            datatrain = loader(
                root="data", train=True, download=not isavailable, transform=transform
            )

        datatest = loader(root="data", train=False, transform=transform)
        return datatrain, datatest
    else:
        raise RuntimeError("Dataset not found")


def toplot(global_accs):
    return chart(
        range(1, max(map(len, global_accs)) + 1),
        {"Training acc": global_accs[0], "Testing acc": global_accs[1]},
        title="Evolution of the average accuracy per round",
        x_title="Rounds",
        y_title="Accuracy (in %)",
        print_labels=True,
        margin_right=75,
        # style=DefaultStyle(label_font_size=8),
    )
