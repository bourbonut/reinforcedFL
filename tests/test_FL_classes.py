from core import *
from utils import *
from model4FL.mnist import ModelMNIST
from itertools import starmap

data_path = data_path_key("MNIST", "IID", 4) / "nodes"


def test_federated_averaging():
    models = tuple(ModelMNIST(10) for _ in range(4))
    workers = tuple(
        Node(model, data_path / "nodes-{}.pkl".format(i + 1))
        for i, model in enumerate(models)
    )
    aggregator = FederatedAveraging(ModelMNIST(10), 60000)
    for worker in workers:
        aggregator.communicatewith(worker)

    aggregator.update()
    assert len(aggregator.workers_updates) == 0
    for worker in workers:
        worker.communicatewith(aggregator)

    assert all(
        all(
            starmap(
                torch.equal,
                zip(worker.model.parameters(), aggregator.global_model.parameters()),
            )
        )
        for worker in workers
    )
