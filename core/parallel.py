from threading import Thread


def train(workers):
    train_worker = lambda worker: worker.train()
    threads = [Thread(target=train_worker, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def evaluate(workers):
    accuracies = [None] * len(workers)

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
