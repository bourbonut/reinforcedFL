"""
Functions to run when a GPU is available
"""

def train(workers):
    for worker in workers:
        worker.train()


def evaluate(workers):
    return [worker.evaluate() for worker in,workers]
