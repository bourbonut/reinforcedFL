"""
Module for all functions useful for :
- partitioning
- managing path
- plotting
"""

from .path import *
from .partition import generate_IID_parties
from .plot import lineXY
from rich.panel import Panel


def panel(partition_type, nbnodes, nbrounds, nbepoches, ongpu):
    msg = "[bold]Informations[/bold]"
    msg += "\n    Number of simulated devices: {}".format(nbnodes)
    msg += "\n    Number of rounds: {}".format(nbrounds)
    msg += "\n    Number of epoches: {}".format(nbepoches)
    msg += "\n    Partitioning type: {}".format(partition_type)
    msg += "\n    On GPU ? {}".format(ongpu)
    return Panel(msg, title="Federated Learning with Reinforcement Learning")
