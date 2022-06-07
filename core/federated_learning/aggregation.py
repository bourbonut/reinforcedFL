from core.evaluator.model import ReinforceAgent
from functools import reduce
from operator import add

# WARNING: Class not finished
class EvaluatorServer(ReinforceAgent):
    """
    This class is based on the `REINFORCE` algorithm class from
    the evaluator module (`core.evaluator`) for a better aggregation
    (see the algorithm `FRCCE` - `arXiv:2102.13314v1`)
    """

    def __init__(self, global_model, *args, **kwargs):
        ReinforceAgent.__init__(self, *args, **kwargs)
        self.global_model = global_model
        self.workers_updates = []

    def send(self):
        """
        The server sends his global model parameters through this method
        """
        return self.global_model.state_dict()

    def receive(self, workers_update):
        """
        The server collects local model parameters through this method
        """
        self.workers_updates.append(workers_update)


class FederatedAveraging:
    """
    Class which plays the role of the server and aggregate through
    the algorithm `FedAvg` which is also named Federated Averaging
    (see `arXiv:1602.05629v3`)
    """

    def __init__(self, global_model, size_traindata, size_testdata, *args, **kwargs):
        self.global_model = global_model
        self.workers_updates = []
        self.n = size_traindata
        self.t = size_testdata

    def send(self):
        """
        The server sends his global model parameters through this method
        """
        return self.global_model.state_dict()

    def receive(self, workers_update):
        """
        The server collects local model parameters through this method
        """
        self.workers_updates.append(workers_update)

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        self.receive(worker.send())

    def update(self):
        """
        Update the global model based on Federated Averaging algorithm

        Note:
            Updates of workers must be multiplied by the number of local examples
            In other words, a worker update is : `[nk * w for w in weights]` where
            `nk` is the number of local examples.
        """
        new_weights = map(
            lambda layer: reduce(add, layer) / self.n, zip(*self.workers_updates)
        )
        for target_param, param in zip(self.global_model.parameters(), new_weights):
            target_param.data.copy_(param.data)
        self.workers_updates.clear()

    def global_accuracy(self, workers_accuracies):
        """
        Compute the global accuracy based on the Federated Averaging algorithm
        """
        return reduce(add, workers_accuracies) / self.t
