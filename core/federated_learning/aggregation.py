from core.evaluator.model import ReinforceAgent
from functools import reduce
from operator import add


class EvaluatorServer(ReinforceAgent):
    def __init__(self, global_model, *args, **kwargs):
        ReinforceAgent.__init__(self, *args, **kwargs)
        self.global_model = global_model
        self.workers_updates = []

    def send(self):
        return self.global_model.state_dict()

    def receive(self, workers_update):
        self.workers_updates.append(workers_update)


class FederatedAveraging:
    def __init__(self, global_model, size_traindata, size_testdata, *args, **kwargs):
        self.global_model = global_model
        self.workers_updates = []
        self.n = size_traindata
        self.t = size_testdata

    def send(self):
        return self.global_model.state_dict()

    def receive(self, workers_update):
        self.workers_updates.append(workers_update)

    def communicatewith(self, worker):
        self.receive(worker.send())

    def update(self):
        """
        Update the global model

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
        Compute the global accuracy based on the federated averaging algorithm
        """
        return reduce(add, workers_accuracies) / self.t
