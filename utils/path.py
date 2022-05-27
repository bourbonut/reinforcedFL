from pathlib import Path

ROOT_PATH = Path().absolute()
EXP_PATH = ROOT_PATH / "experiments"
DATA_PATH = Path().absolute() / "data"


def create(path):
    if not (path.exists()):
        if not (path.parent.exists()):
            create(path.parent)
        path.mkdir()


def iterate(path):
    folders = list(path.iterdir())
    index = 1
    while any(path.glob(f"experiment-{index}*")):
        index += 1

    return path / f"experiment-{index}"
