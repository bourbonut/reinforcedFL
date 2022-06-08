from utils.path import create, iterate, data_path_key, EXP_PATH, DATA_PATH
from utils.partition import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pickle, pytest

create(EXP_PATH)
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


@pytest.mark.slow
def test_generate_IID():
    nodes_data_path = data_path_key("MNIST", "IID", nodes) / "nodes"
    create(nodes_data_path)
    generate_IID_parties(mnist_dataset, nodes, nodes_data_path)
    assert nodes_data_path.exists()
    assert len(list(nodes_data_path.iterdir())) == nodes
    ref = len(mnist_dataset["test"]) / len(mnist_dataset["training"])
    sizes_train = []
    sizes_test = []
    for filename in nodes_data_path.iterdir():
        with open(filename, "rb") as file:
            data = pickle.load(file)
        assert len(data[1]) / len(data[0]) == ref
        sizes_train.append(len(data[0]))
        sizes_test.append(len(data[1]))
    assert sum(sizes_train) == len(mnist_dataset["training"])
    assert sum(sizes_test) == len(mnist_dataset["test"])


def test_open_dataset():
    nodes_data_path = data_path_key("MNIST", "IID", nodes) / "nodes"
    if nodes_data_path.exists():
        batch_size = 64
        filename = next(nodes_data_path.iterdir())
        with open(filename, "rb") as file:
            data = pickle.load(file)
        trainloader = DataLoader(data[0], batch_size=batch_size, num_workers=1)
        testloader = DataLoader(data[1], batch_size=batch_size, num_workers=1)
        sample_train, label_train = next(iter(trainloader))
        assert sample_train.shape == (batch_size, 1, 28, 28)
        assert label_train.shape == (batch_size,)
        sample_test, label_test = next(iter(testloader))
        assert sample_test.shape == (batch_size, 1, 28, 28)
        assert label_test.shape == (batch_size,)
