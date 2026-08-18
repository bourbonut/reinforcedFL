from itertools import starmap
from threading import Thread

import pytest
import torch

from core.federated_learning.aggregation import FederatedAveraging
from core.federated_learning.worker import Worker
from model4FL.mnist import Model, extras
from utils import EXP_PATH, tracker

wk_data_path = EXP_PATH / tracker("MNIST", 4, "iid", "iid")


def test_federated_averaging():
    models = tuple(Model(10, None) for _ in range(4))
    workers = tuple(
        Worker(model, wk_data_path / f"worker-{i + 1}.pkl", 1, **extras)
        for i, model in enumerate(models)
    )
    aggregator = FederatedAveraging(Model(10, None), 60000, 10000)
    for worker in workers:
        aggregator.communicatewith(worker)

    aggregator.update([])
    assert len(aggregator.participants_updates) == 0
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


def test_federated_averaging_2():
    ref = Model(10, None)
    workers = tuple(
        Worker(ref, wk_data_path / f"worker-{i + 1}.pkl", 1, **extras) for i in range(4)
    )
    aggregator = FederatedAveraging(Model(10, None), 60000, 10000)
    for worker in workers:
        aggregator.communicatewith(worker)

    aggregator.update([])
    assert len(aggregator.participants_updates) == 0
    for worker in workers:
        worker.communicatewith(aggregator)

    assert all(
        all(
            starmap(
                torch.equal,
                zip(worker.model.parameters(), ref.parameters()),
            )
        )
        for worker in workers
    )


@pytest.mark.slow
def test_worker_train():
    worker = Worker(Model(10, None), wk_data_path / "nodes-1.pkl", **extras)
    worker.train()


@pytest.mark.slow
def test_multithreading():
    models = tuple(Model(10, None) for _ in range(4))
    workers = tuple(
        Worker(model, wk_data_path / "nodes-{}.pkl".format(i + 1), 1, **extras)
        for i, model in enumerate(models)
    )

    def local_train(worker):
        worker.train()

    threads = [Thread(target=local_train, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
