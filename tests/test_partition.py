from utils.path import create, iterate, EXP_PATH, DATA_PATH
from utils.partition import *
from torchvision import datasets
from torchvision.transforms import ToTensor
import pickle

create(EXP_PATH)
curr_path = EXP_PATH
nodes = 4
isdownloaded = not (DATA_PATH.exists())
mnist_dataset = {}
mnist_dataset["training"] = datasets.MNIST(
    root="data", train=True, download=isdownloaded, transform=ToTensor()
)
mnist_dataset["test"] = datasets.MNIST(root="data", train=False, transform=ToTensor())


def test_check_data():
    assert DATA_PATH.exists()


def test_dataset():
    assert "training" in mnist_dataset and "test" in mnist_dataset


def test_generate_IID():
    nodes_data_path = curr_path / "data-IID-4" / "nodes"
    create(nodes_data_path)
    generate_IID_parties(mnist_dataset, nodes, nodes_data_path)
    assert nodes_data_path.exists()
    assert len(list(nodes_data_path.iterdir())) == nodes
    ref = len(mnist_dataset["test"]) / len(mnist_dataset["training"])
    for filename in nodes_data_path.iterdir():
        with open(filename, "rb") as file:
            data = pickle.load(file)
        assert len(data[2]) / len(data[0]) == ref
        assert len(data[3]) / len(data[1]) == ref
