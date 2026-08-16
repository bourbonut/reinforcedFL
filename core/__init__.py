"""
Module which deals with :
- Federated Learning classes such as aggregation class and worker class
- The evaluator class for better aggregation through Reinforcement Learning
- Main functions for training and evaluation
"""

from .federated_learning import aggregation, worker, participation


def train(workers, path=None):
    """
    Train workers on their local data in sequential
    """
    for worker in workers:
        worker.train(filename=path)


def evaluate(workers, ontrain=False, perlabel=False, full=False):
    """
    Evaluate the global model on local data of workers in sequential
    """
    return [worker.evaluate(ontrain, perlabel, full) for worker in workers]
