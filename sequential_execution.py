from utils import *
from core import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from model4FL.mnist import ModelMNIST, extras
from rich.live import Live
from rich.table import Table
from threading import Thread
from rich import print
import pickle, torch

from itertools import starmap

# Parameters
PARTITION_TYPE = "IID"
NODES = 4
ROUNDS = 10
EPOCHES = 3
ON_GPU = False

if ON_GPU:
    from core.sequential_gpu import train, evaluate
else:
    from core.parallel import train, evaluate


def check(workers, aggregator):
    return all(
        all(
            starmap(
                torch.equal,
                zip(worker.model.parameters(), aggregator.global_model.parameters()),
            )
        )
        for worker in workers
    )


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


# Experiment path
exp_path = iterate(EXP_PATH)

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
table = Table()
table.add_column("Rounds")
table.add_column("Global accuracy")
# Main loop
with Live(table) as live_layout:
    for r in range(ROUNDS):
        # Workers download the global model
        for worker in workers:
            worker.communicatewith(server)

        # Workers evaluate accuracy of the global model
        # on their local data
        accuracies = evaluate(workers)
        avg_acc = server.global_accuracy(accuracies)
        global_accs.append(avg_acc)

        table.add_row(str(r), "{:.2%}".format(avg_acc))
        live_layout.refresh()

        # Training loop of workers
        for e in range(EPOCHES):
            curr_path = exp_path / "round{}".format(r) / "epoch{}".format(e)
            create(curr_path, verbose=False)
            train(workers, curr_path)

        # Server downloads all local updates
        for worker in workers:
            server.communicatewith(worker)
        server.update()

with open(exp_path / "result.pkl", "wb") as file:
    pickle.dump(global_accs, file)
