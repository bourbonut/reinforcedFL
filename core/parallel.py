"""
Functions to run for making parallel the training part of workers
"""


from threading import Thread


def train(workers, update, path=None, **kwargs):
    """
    Train workers on their local data in parallel
    """
    if path is None:

        def train_worker(worker):
            worker.train()
            update()

        threads = [Thread(target=train_worker, args=(worker,)) for worker in workers]
    else:

        def train_worker(worker, index, path):
            worker.train(path / f"worker-{index}.png")
            update()

        threads = [
            Thread(target=train_worker, args=(worker, i, path))
            for i, worker in enumerate(workers)
        ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def evaluate(workers, update, ontrain=False, perlabel=False):
    """
    Evaluate the global model on local data of workers in parallel
    """
    accuracies = [0] * len(workers)

    def eval_worker(worker, idx, ontrain, perlabel, accuracies):
        acc = worker.evaluate(ontrain, perlabel)
        accuracies[idx] = acc
        update()

    threads = [
        Thread(target=eval_worker, args=(worker, i, ontrain, perlabel, accuracies))
        for i, worker in enumerate(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return accuracies
