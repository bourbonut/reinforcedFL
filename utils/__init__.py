"""
Module for all functions useful for :
- partitioning
- managing path
- plotting
"""

from .path import *
from .distribution import *
from .plot import lineXY
from rich.panel import Panel

def tracker(nworkers, label_distrb, volume_distrb, minlabels=3, balanced=True):
    n = str(nworkers)
    v = "Vi" if volume_distrb == "iid" else "Vni"
    l = "Li" if label_distrb == "iid" else "Lni"
    b = "bal" if balanced else "unbal"
    m = "" if label_distrb == "iid" else str(minlabels)
    return "data-"+"".join((n,l,m,v)) + ("-" + b if label_distrb=="noniid" else "")
        

def panel(partition_type, nworkers, nbrounds, nbepoches, ongpu):
    msg = "[bold]Informations[/bold]"
    msg += "\n    Number of simulated devices: {}".format(nworkers)
    msg += "\n    Number of rounds: {}".format(nbrounds)
    msg += "\n    Number of epoches: {}".format(nbepoches)
    msg += "\n    Partitioning type: {}".format(partition_type)
    msg += "\n    On GPU ? {}".format(ongpu)
    return Panel(msg, title="Federated Learning with Reinforcement Learning")
