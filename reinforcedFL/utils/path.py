"""
Function to manage paths
"""

from pathlib import Path

ROOT_PATH = Path().absolute()
EXP_PATH = ROOT_PATH / "experiments"
DATA_PATH = Path().absolute() / "data"


def create(path: Path):
    """
    Create a full path recursively
    """
    if not path.exists():
        path.mkdir(parents=True)
        return path


def iterate(path):
    """
    Find the path for a new experiment
    """
    index = 1
    while any(path.glob(f"experiment-{index}*")):
        index += 1

    return path / f"experiment-{index}"
