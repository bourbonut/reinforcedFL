"""
Functions to run when a GPU is available
"""


def train(workers, *args):
    """
    Train workers on their local data in sequential
    """
    for worker in workers:
        worker.train()


def evaluate(workers, ontrain=False, perlabel=False):
    """
    Evaluate the global model on local data of workers in sequential
    """
    return [worker.evaluate(ontrain, perlabel) for worker in workers]
