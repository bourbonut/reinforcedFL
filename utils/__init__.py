"""
Module for all functions useful for :
- partitioning
- managing path
- plotting
"""

from .path import *
from .distribution import *
from .plot import chart, topng
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from pygal.style import DefaultStyle


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


def dataset(name):
    path = DATA_PATH / name
    isavailable = path.exists()
    datasetfromtorch = hasattr(datasets, name)
    cifar10 = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if isavailable or datasetfromtorch:
        transform = cifar10 if name == "CIFAR10" else ToTensor()
        loader = getattr(datasets, name)
        datatrain = loader(
            root="data", train=True, download=not (isavailable), transform=transform
        )
        datatest = loader(root="data", train=False, transform=transform)
        return datatrain, datatest
    else:
        raise RuntimeError("Dataset not found")


def toplot(global_accs):
    return topng(
        chart(
            range(1, max(map(len, global_accs)) + 1),
            {"Training acc": global_accs[0], "Testing acc": global_accs[1]},
            title="Evolution of the average accuracy per round",
            x_title="Rounds",
            y_title="Accuracy (in %)",
            print_labels=True,
            margin_right=75,
            # style=DefaultStyle(label_font_size=8),
        )
    )
