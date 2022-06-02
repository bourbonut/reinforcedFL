from core import *
from utils import *
from model4FL.mnist import ModelMNIST, extras
from itertools import starmap
from threading import Thread
import pytest

data_path = data_path_key("MNIST", "IID", 4) / "nodes"


def test_federated_averaging():
    models = tuple(ModelMNIST(10) for _ in range(4))
    workers = tuple(
        Node(model, data_path / "nodes-{}.pkl".format(i + 1), **extras)
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


@pytest.mark.slow
def test_worker_forward():
    worker = Node(ModelMNIST(10), data_path / "nodes-1.pkl", **extras)
    worker.forward()


@pytest.mark.slow
def test_multiprocessing():
    models = tuple(ModelMNIST(10) for _ in range(4))
    workers = tuple(
        Node(model, data_path / "nodes-{}.pkl".format(i + 1), **extras)
        for i, model in enumerate(models)
    )
    train = lambda worker: worker.forward()
    threads = [Thread(target=train, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
