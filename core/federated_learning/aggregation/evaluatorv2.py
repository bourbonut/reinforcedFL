from core.federated_learning.aggregation.base import EvaluatorServer
from rich.table import Table
from rich.align import Align
from rich.live import Live
from time import perf_counter
from math import log
import torch, pickle


class EvaluatorV2(EvaluatorServer):
    """
    This class is based on the `REINFORCE` algorithm class from
    the evaluator module (`core.evaluator`) for a better aggregation
    (inspired by the algorithm `FRCCE` - `arXiv:2102.13314v1`)

    The state is defined as the accuracies of workers after local
    training less the accuracies of the global model before aggregation
    
    The reward is defined with two different ways
    - with an exponential moving average
    - with an exponential function
    (see the code for more information)
    """

    TARGET = 0.99

    def __init__(
        self,
        global_model,
        size_traindata,
        size_testdata,
        size,
        capacity=3,
        gamma=0.99,
        optimizer=None,
        delta_method=True,
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
            optimizer (Optimizer):      the optimizer for the agent
            delta_method (bool):        for choosing the method used to compute the reward
        """
        super(EvaluatorV2, self).__init__(
            global_model,
            size_traindata,
            size_testdata,
            size,
            capacity,
            gamma,
            optimizer,
        )
        if delta_method:
            self.delta = 0 # Window for moving average
        else:
            self.speed = log(1e-6) / log(1 - 0.95) # exponent for reward
        self.delta_method = delta_method

    def compute_glb_acc(self, workers_accuracies, train=False):
        """
        Compute the global accuracy based on the Federated Averaging algorithm
        """
        indices = range(len(workers_accuracies))
        return super().compute_glb_acc(workers_accuracies, indices, train)

    def update_batch(self, state, action):
        """
        Update MovingBatch class
        """
        self.batchs.states.append(state)
        self.batchs.actions.append(action)

    def update_delta(self, accuracies):
        """
        Update the initial value on the exponential moving average
        """
        if self.delta_method:
            self.delta = self.compute_glb_acc(accuracies)

    def train_agent(self, accuracies):
        """
        Train the agent
        """
        curr_accuracy = self.compute_glb_acc(accuracies)
        if self.delta_method:
            reward = curr_accuracy - self.delta
        else:
            reward = (1 - (self.TARGET - curr_accuracy)) ** self.speed

        self.rewards.append(reward)
        self.tracking_rewards.append(reward)

        self.batchs.rewards.append(self.discount_rewards())
        self.batchs.update_size()
        if self.delta_method:
            self.delta = self.delta + self.alpha * (curr_accuracy - self.delta)
        self.accuracies.clear()

        return super().train_agent()

    def reset(self, filename=None):
        """
        Reset work values for next round
        """
        if self.delta_method:
            self.delta = 0
        return super().reset(filename)

    def execute(
        self, nexp, rounds, workers, train, evaluate, path, model, *args, **kwargs
    ):
        # Global accuracies : first list for training
        # second list for testing
        global_accs = []  # Main loop
        for iexp in range(nexp):
            table = Table(
                "Round",
                "Training accuracies [%]",
                "Testing accuracies [%]",
                "Duration \[s]",
                "Losses",
                title=f"Experiment {iexp}",
            )

            # Global initial accuracies
            pair = evaluate(workers, full=True)
            accuracies, singular_accuracies = zip(*pair)
            self.update_delta(accuracies)
            self.collects_global_accuracies(singular_accuracies)

            align = Align.center(table)
            with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
                for r in range(rounds):
                    start = perf_counter()

                    # Workers download the global model
                    for worker in workers:
                        worker.communicatewith(self)

                    # Training loop of workers
                    train(workers)

                    # Workers evaluate accuracy of global model
                    # on their local training data
                    pair = evaluate(workers, True, full=True)
                    accuracies, singular_accuracies = zip(*pair)
                    tr_avg_acc = self.compute_glb_acc(accuracies, True)
                    self.collects_training_accuracies(singular_accuracies)

                    # Server downloads all local updates
                    for worker in workers:
                        self.communicatewith(worker)

                    state = torch.tensor(self.accuracies) - torch.tensor(
                        self.global_accuracies
                    )
                    action = self.update(state)
                    self.update_batch(state.tolist(), action.T)

                    # Workers evaluate accuracy of the global model
                    # on their local testing data
                    pair = evaluate(workers, full=True)
                    accuracies, singular_accuracies = zip(*pair)
                    te_avg_acc = self.compute_glb_acc(accuracies)
                    self.collects_global_accuracies(singular_accuracies)
                    duration = perf_counter() - start

                    # Train the agent
                    self.train_agent(accuracies)

                    # Update the table
                    table.add_row(
                        str(r + 1),
                        f"{tr_avg_acc:.2%}",
                        f"{te_avg_acc:.2%}",
                        f"{duration:.3f} s",
                        f"{self.batch_loss}",
                    )
                    live.refresh()

                    # Update global_accs
                    global_accs.append((tr_avg_acc, te_avg_acc))

            # Reset the server
            self.reset(path / "agent" / f"loss-rl-{iexp}.png")
            self.global_model = model()

            # Reset workers
            for worker in workers:
                worker.model = model()

            # Save results
            with open(path / f"global_accs-{iexp}.pkl", "wb") as file:
                pickle.dump(global_accs, file)

            global_accs.clear()
