"""
Function to manage paths
"""

from pathlib import Path

ROOT_PATH = Path().absolute()
EXP_PATH = ROOT_PATH / "experiments"
DATA_PATH = Path().absolute() / "data"


def create(path):
    """
    Create a full path recursively
    """
    if not (path.exists()):
        if not (path.parent.exists()):
            create(path.parent)
        path.mkdir()
        print("{} was created.".format(str(path)))


def iterate(path):
    """
    Find the path for a new experiment
    """
    folders = list(path.iterdir())
    index = 1
    while any(path.glob(f"experiment-{index}*")):
        index += 1

    return path / f"experiment-{index}"


def data_path_key(dataset_name, partition_type, k_nodes):
    """
    Get the path for distributed data
    """
    return EXP_PATH / "data-{}-{}-{}".format(dataset_name, partition_type, k_nodes)
