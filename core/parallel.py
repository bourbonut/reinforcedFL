"""
Functions to run for making parallel the training part of workers
"""


from threading import Thread


def train(workers, path=None):
    """
    Train workers on their local data in parallel
    """
    if path is None:
        train_worker = lambda worker: worker.train()
        threads = [Thread(target=train_worker, args=(worker,)) for worker in workers]
    else:

        def train_worker(worker, index, path):
            worker.train(path / "worker-{}.png".format(index))

        threads = [
            Thread(target=train_worker, args=(worker, i, path))
            for i, worker in enumerate(workers)
        ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def evaluate(workers):
    """
    Evaluate the global model on local data of workers in parallel
    """
    accuracies = [0] * len(workers)

    def eval_worker(worker, idx, accuracies):
        acc = worker.evaluate()
        accuracies[idx] = acc

    threads = [
        Thread(target=eval_worker, args=(worker, i, accuracies))
        for i, worker in enumerate(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return accuracies
