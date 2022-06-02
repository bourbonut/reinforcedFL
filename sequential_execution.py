#!/home/osboxes/anaconda3/envs/gym/bin/python
from utils import *
from core import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from multiprocessing.pool import Pool

# Parameters
PARTITION_TYPE = "IID"
NODES = 4

# Get the dataset
isdownloaded = not (DATA_PATH.exists())
mnist_dataset = {}
mnist_dataset["training"] = datasets.MNIST(
    root="data", train=True, download=isdownloaded, transform=ToTensor()
)
mnist_dataset["test"] = datasets.MNIST(root="data", train=False, transform=ToTensor())

nclasses = len(mnist_dataset["training"].classes)  # for the model
size_data = len(mnist_dataset["training"])  # for aggregation

# Get path of data for workers and generate them
nodes_data_path = data_path_key("MNIST", "IID", NODES) / "nodes"
if not (nodes_data_path.exists()):
    generate_IID_parties(mnist_dataset, NODES, nodes_data_path)

# Initialization of the server
server = FederatedAveraging(ModelMNIST(nclasses), size_data)

# Initialization of workers
models = (ModelMNIST(nclasses) for _ in range(4))
workers = tuple(
    Node(model, data_path / "nodes-{}.pkl".format(i + 1))
    for i, model in enumerate(models)
)
