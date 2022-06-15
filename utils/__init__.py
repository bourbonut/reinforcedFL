"""
Module for all functions useful for :
- partitioning
- managing path
- plotting
"""

from .path import *
from .distribution import *
from .plot import chart, topng, lineXY


def tracker(nworkers, label_distrb, volume_distrb, minlabels=3, balanced=True):
    n = str(nworkers)
    v = "Vi" if volume_distrb == "iid" else "Vni"
    l = "Li" if label_distrb == "iid" else "Lni"
    b = "bal" if balanced else "unbal"
    m = "" if label_distrb == "iid" else str(minlabels)
    return (
        "data-" + "".join((n, l, m, v)) + ("-" + b if label_distrb == "noniid" else "")
    )
