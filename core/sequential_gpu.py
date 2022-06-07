"""
Functions to run when a GPU is available
"""


def train(workers):
    """
    Train workers on their local data in sequential
    """
    for worker in workers:
        worker.train()


def evaluate(workers):
    """
    Evaluate the global model on local data of workers in sequential
    """
    return [worker.evaluate() for worker in workers]
