from pathlib import Path

ROOT_PATH = Path().absolute()
EXP_PATH = ROOT_PATH / "experiments"
DATA_PATH = Path().absolute() / "data"


def create(path):
    if not (path.exists()):
        if not (path.parent.exists()):
            create(path.parent)
        path.mkdir()
        print("{} was created.".format(str(path)))


def iterate(path):
    folders = list(path.iterdir())
    index = 1
    while any(path.glob(f"experiment-{index}*")):
        index += 1

    return path / f"experiment-{index}"


def data_path_key(dataset_name, partition_type, k_nodes):
    return EXP_PATH / "data-{}-{}-{}".format(dataset_name, partition_type, k_nodes)
