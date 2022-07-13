"""
Functions to run when a GPU is available
"""


def train(workers, update, path=None):
    """
    Train workers on their local data in sequential
    """
    for worker in workers:
        worker.train(filename=path)
        update()


def evaluate(workers, update, ontrain=False, perlabel=False):
    """
    Evaluate the global model on local data of workers in sequential
    """
    def single_eval(worker):
        result = worker.evaluate(ontrain, perlabel)
        update()
        return result
    
    return [single_eval(worker) for worker in workers]
