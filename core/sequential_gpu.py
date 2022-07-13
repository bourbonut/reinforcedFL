"""
Functions to run when a GPU is available
"""


def train(workers, update, path=None):
    """
    Train workers on their local data in sequential
    """
    l = len(workers) // 5
    for i, worker in enumerate(workers):
        worker.train(filename=path)
        if i % l == 0:
            update()


def evaluate(workers, update, ontrain=False, perlabel=False):
    """
    Evaluate the global model on local data of workers in sequential
    """
    l = len(workers) // 5

    def single_eval(worker, i):
        result = worker.evaluate(ontrain, perlabel)
        update()
        return result

    return [single_eval(worker, i) for i, worker in enumerate(workers)]
