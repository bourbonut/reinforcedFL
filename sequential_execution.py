#!/home/osboxes/anaconda3/envs/gym/bin/python
from utils import *
from core import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from model4FL.mnist import ModelMNIST, extras
from rich.progress import Progress
from threading import Thread
from rich import print
import pickle

# Parameters
PARTITION_TYPE = "IID"
NODES = 4
ROUNDS = 10
EPOCHES = 3

# Get the dataset
print("Opening the dataset", end="")
isdownloaded = not (DATA_PATH.exists())
mnist_dataset = {}
mnist_dataset["training"] = datasets.MNIST(
    root="data", train=True, download=isdownloaded, transform=ToTensor()
)
mnist_dataset["test"] = datasets.MNIST(root="data", train=False, transform=ToTensor())
print(" ->[bold green] OK")

nclasses = len(mnist_dataset["training"].classes)  # for the model
size_traindata = len(mnist_dataset["training"])  # for aggregation
size_testdata = len(mnist_dataset["test"])  # for aggregation

# Get path of data for workers and generate them
print("Generate data for workers", end="")
nodes_data_path = data_path_key("MNIST", "IID", NODES) / "nodes"
if not (nodes_data_path.exists()):
    create(nodes_data_path)
    generate_IID_parties(mnist_dataset, NODES, nodes_data_path)
    print(" ->[bold green] OK")
else:
    print(" ->[bold yellow] Already done")


# Initialization of the server
print("Initialization of the server", end="")
server = FederatedAveraging(ModelMNIST(nclasses), size_traindata, size_testdata)
print(" ->[bold green] OK")

# Initialization of workers
print("Initialization of the workers", end="")
models = (ModelMNIST(nclasses) for _ in range(4))
workers = tuple(
    Node(model, nodes_data_path / "nodes-{}.pkl".format(i + 1))
    for i, model in enumerate(models)
)
print(" ->[bold green] OK")

global_accs = []
# Main loop
with Progress(auto_refresh=False) as progress:
    task = progress.add_task("Training ...", total=ROUNDS * EPOCHES)
    for r in range(ROUNDS):
        for worker in workers:
            worker.communicatewith(server)
        accuracies = evaluate(workers)
        avg_acc = server.global_accuracy(accuracies)
        global_accs.append(avg_acc)
        for _ in range(EPOCHES):
            train(workers)
            progress.advance(task)
            progress.refresh()
        for worker in workers:
            server.communicatewith(worker)

with open("result.pkl", "wb") as file:
    pickle.dump(global_accs, file)
