from core.federated_learning.aggregation.base import EvaluatorServer
from rich.table import Table
from rich.align import Align
from rich.live import Live
from time import perf_counter
from itertools import compress
import torch, pickle

class EvaluatorV1(EvaluatorServer):
    """
    Class based on `EvaluatorServer`
    """

    def __init__(
        self,
        global_model,
        size_traindata,
        size_testdata,
        ninput=None,
        noutput=None,
        capacity=3,
        gamma=0.99,
        optimizer=None,
        *args,
        **kwargs,
    ):
        super(EvaluatorV1, self).__init__(
            global_model,
            size_traindata,
            size_testdata,
            ninput,
            noutput,
            capacity,
            gamma,
            optimizer,
        )
        self.delta = 0

    def compute_glb_acc(self, workers_accuracies, train=False):
        indices = range(len(workers_accuracies))
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
        p = sum(self.curr_selection)
        curr_accuracy = sum(compress(self.accuracies, self.curr_selection)) / p
        reward = curr_accuracy - self.delta
        self.rewards.append(reward)
        self.tracking_rewards.append(reward) 

        # Update batch array
        self.update_batch(self.accuracies, action.T)
        self.delta = self.delta + self.alpha * (curr_accuracy - self.delta)
        self.accuracies.clear()

        return super().train_agent()

    def execute(
        self, nexp, rounds, workers, train, evaluate, path, model, *args, **kwargs
    ):
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

                    # # No save of loss evolution
                    # curr_path = exp_path / f"round{r}" / f"epoch{e}"
                    # create(curr_path, verbose=False)
                    # train(workers, curr_path)
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
