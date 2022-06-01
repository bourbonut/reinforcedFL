from core.evaluator.model import ReinforceAgent


class EvaluatorServer(ReinforceAgent):
    def __init__(self, global_model, *args, **kwargs):
        ReinforceAgent.__init__(self, *args, **kwargs)
        self.global_model = global_model
        self.remote_updates = []

    def send(self):
        return self.global_model.state_dict()

    def receive(self, remote_update):
        self.remote_updates.append(remote_update)
