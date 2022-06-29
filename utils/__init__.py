"""
Module for all functions useful for :
- partitioning
- managing path
- plotting
- parser
"""

from .path import *
from .distribution import *
from .plot import chart, topng
from torchvision import datasets
from torchvision.transforms import ToTensor

def tracker(dataname, nworkers, label_distrb, volume_distrb, minlabels=3, balanced=True):
    d = dataname.lower()
    n = str(nworkers)
    v = "Vi" if volume_distrb == "iid" else "Vni"
    l = "Li" if label_distrb == "iid" else "Lni"
    b = "bal" if balanced else "unbal"
    m = "" if label_distrb == "iid" else str(minlabels)
    return (
        "data-" + d + "-" + "".join((n, l, m, v)) + ("-" + b if label_distrb == "noniid" else "")
    )


def dataset(name):
    path = DATA_PATH / name
    isavailable = path.exists()
    datasetfromtorch = hasattr(datasets, name)
    if isavailable or datasetfromtorch:
        loader = getattr(datasets, name)
        datatrain = loader(
            root="data", train=True, download=not (isavailable), transform=ToTensor()
        )
        datatest = loader(root="data", train=False, transform=ToTensor())
        return datatrain, datatest
    else:
        raise RuntimeError("Dataset not found")
