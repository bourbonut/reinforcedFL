from core.federated_learning.aggregation.base import EvaluatorServer
from rich.table import Table
from rich.align import Align
from rich.live import Live
from time import perf_counter
from itertools import compress
from copy import copy
import torch, pickle


class EvaluatorV1(EvaluatorServer):
    """
    This class is based on the `REINFORCE` algorithm class from
    the evaluator module (`core.evaluator`) for a better aggregation
    (inspired by the algorithm `FRCCE` - `arXiv:2102.13314v1`)

    The state is defined as the accuracies after aggregation 
    The reward is computed as the exponential moving average
    of the average of accuracies.
    """

    def __init__(
        self,
        global_model,
        size_traindata,
        size_testdata,
        size,
        capacity=3,
        gamma=0.99,
        optimizer=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the class

        Parameters:

            global_model (nn.Module):   the global model
            size_traindata (list):      the list of sizes of local data for training
                                        (supposed gotten by Federated Analytic)
            size_testdata (list):       the list of sizes of local data for testing
                                        (supposed gotten by Federated Analytic)
            size (int):                 the number of workers
            capacity (int):             the capacity of the moving batch size
            gamma (float):              the discount factor used for reward
            optimizer(Optimizer):       the optimizer for the agent
        """
        super(EvaluatorV1, self).__init__(
            global_model,
            size_traindata,
            size_testdata,
            size,
            capacity,
            gamma,
            optimizer,
        )
        self.delta = 0 # Window for moving average

    def collect_accuracy(self, worker_accuracy):
        """
        The server collects local model accuracy through this method
        """
        self.accuracies.append(worker_accuracy)

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        super().communicatewith(worker)
        self.collect_accuracy(worker.evaluate(train=True, perlabel=True))  

    def compute_glb_acc(self, workers_accuracies, train=False):
        """
        Compute the global accuracy based on the Federated Averaging algorithm
        """
        indices = range(len(workers_accuracies)) # all workers are participants
        return super().compute_glb_acc(workers_accuracies, indices, train)

    def update_batch(self, state, action):
        """
        Update MovingBatch class
        """
        self.batchs.rewards.append(self.discount_rewards())
        self.batchs.states.append(state)
        self.batchs.actions.append(action)
        self.batchs.update_size()

    def train_agent(self, action):
        """
        Train the agent for better aggregation
        """
        p = sum(self.curr_selection)
        curr_accuracy = sum(compress(self.accuracies, self.curr_selection)) / p
        reward = curr_accuracy - self.delta
        self.rewards.append(reward)
        self.tracking_rewards.append(reward)

        # Update batch array
        self.update_batch(copy(self.accuracies), action.T)
        self.delta = self.delta + self.alpha * (curr_accuracy - self.delta)
        self.accuracies.clear()

        return super().train_agent()
    
    def reset(self, filename=None):
        """
        Reset work values for the next round
        """
        self.delta = 0
        return super().reset(filename)

    def execute(
        self, nexp, rounds, workers, train, evaluate, path, model, *args, **kwargs
    ):
        self.path = path
        # Global accuracies : first list for training
        # second list for testing
        global_accs = []
        # Main loop
        for iexp in range(nexp):
            table = Table(
                "Round",
                "Training accuracies [%]",
                "Testing accuracies [%]",
                "Duration \[s]",
                "Losses",
                title=f"Experiment {iexp}",
            )
            align = Align.center(table)
            with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
                for r in range(rounds):
                    # Workers download the global model
                    for worker in workers:
                        worker.communicatewith(self)

                    # Workers evaluate accuracy of the global model
                    # on their local testing data
                    start = perf_counter()
                    accuracies = evaluate(workers)
                    tr_avg_acc = self.compute_glb_acc(accuracies)

                    # Training loop of workers
                    train(workers)

                    # Workers evaluate accuracy of global model
                    # on their local training data
                    accuracies = evaluate(workers, True)
                    te_avg_acc = self.compute_glb_acc(accuracies, True)

                    duration = perf_counter() - start

                    # Update the table for training average accuracy
                    table.add_row(
                        str(r + 1),
                        f"{tr_avg_acc:.2%}",
                        f"{te_avg_acc:.2%}",
                        f"{duration:.3f} s",
                        f"{self.batch_loss}",
                    )
                    live.refresh()

                    # Server downloads all local updates
                    for worker in workers:
                        self.communicatewith(worker)
                    action = self.update(torch.tensor(self.accuracies))
                    self.train_agent(action)

                    global_accs.append((tr_avg_acc, te_avg_acc))

            # Reset the self
            self.reset(path / "agent" / f"loss-rl-{iexp}.png")
            self.global_model = model()

            # Reset workers
            for worker in workers:
                worker.model = model()

            # Save results
            with open(path / f"global_accs-{iexp}.pkl", "wb") as file:
                pickle.dump(global_accs, file)

            global_accs.clear()
