from pygal.util import compute_logarithmic_scale
from torch import rand
from torch.utils.data import communication
from core.federated_learning.aggregation.base import BaseServer
from rich.table import Table
from rich.align import Align
from rich.live import Live
from time import perf_counter
import pickle, statistics


class FederatedAveraging(BaseServer):
    """
    Class which plays the role of the self and aggregate through
    the algorithm `FedAvg` which is also named Federated Averaging
    (see `arXiv:1602.05629v3`)
    """

    def __init__(self, global_model, size_traindata, size_testdata, *args, **kwargs):
        """
        Initialize the class

        Parameters:

            global_model (nn.Module):   the global model
            size_traindata (list):      the list of sizes of local data for training
                                        (supposed gotten by Federated Analytic)
            size_testdata (list):       the list of sizes of local data for testing
                                        (supposed gotten by Federated Analytic)
        """
        super(FederatedAveraging, self).__init__(
            global_model, size_traindata, size_testdata
        )
        self.probabilities = []
        self.times = []  # used for tracking selection

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        self.receive(worker.send())

    def update(self, indices):
        """
        Update the global model based on Federated Averaging algorithm

        Note:
            Updates of workers must be multiplied by the number of local examples
            In other words, a worker update is : `[nk * w for w in weights]` where
            `nk` is the number of local examples.
        """
        p = self.local_size(indices, True)
        new_weights = map(lambda layer: sum(layer) / p, zip(*self.participants_updates))
        for target_param, param in zip(self.global_model.parameters(), new_weights):
            target_param.data.copy_(param.data)
        self.participants_updates.clear()

    def _update_times(self, workers):
        """
        Return the indices of workers with the lowest times
        """
        self.nworkers = len(workers)
        computation_times = [
            w.computation_speed * (len(w._train) // w.batch_size) * w.epochs
            for w in workers
        ]
        download_times = [w.NB_PARAMS * 1e-6 * 32 / w.network[0][0] for w in workers]
        upload_times = [w.NB_PARAMS * 1e-6 * 32 / w.network[1][0] for w in workers]
        z = zip(computation_times, download_times, upload_times)
        self.times = [statistics.mean(m) for m in z]

    def _update_probabilities(self, probabilities, random_round=False):
        """
        Update the list of participants in order to visualize
        the selection of the experiment
        """
        if len(probabilities) == 0:
            self.probabilities.append([0.0] * self.nworkers)
        elif random_round:
            self.probabilities.append(self.probabilities[-1])
        else:
            x = [(i, p) for i, p in zip(self.times, probabilities)]
            self.probabilities.append([e for _, e in sorted(x, key=lambda e: e[0])])

    def finish(self):
        path = self.path / "scheduler"
        with open(path / "probabilities.pkl", "wb") as file:
            pickle.dump(self.probabilities, file)

    # TODO : make one loop without first round
    def execute(
        self,
        nexp,
        rounds,
        workers,
        train,
        evaluate,
        path,
        model,
        scheduler,
        k=10,
        *args,
        **kwargs,
    ):
        # Global accuracies : first list for training
        # second list for testing
        self.path = path
        global_accs = []
        self._update_times(workers)
        for iexp in range(nexp):
            table = Table(
                "Round",
                "Training accuracies [%]",
                "Testing accuracies [%]",
                "Duration \[s]",
                "Losses",
                "Time (computation & communication) \[s]",
                title=f"Experiment {iexp}",
            )

            align = Align.center(table)
            with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
                # First round
                start = perf_counter()
                # Selection of future participants
                action, indices_participants = scheduler.select_next_partipants()
                participants = [workers[i] for i in indices_participants]
                for worker in participants:
                    worker.communicatewith(self)

                train(participants)
                pair = evaluate(participants, True, full=True)
                accuracies, _ = zip(*pair)
                tr_avg_acc = self.compute_glb_acc(
                    accuracies, indices_participants, True
                )

                for worker in participants:
                    self.communicatewith(worker)

                scheduler.update_state(workers, indices_participants)
                self.update(indices_participants)

                for worker in workers:
                    worker.communicatewith(self)
                pair = evaluate(workers, full=True)
                accuracies, _ = zip(*pair)
                te_avg_acc = self.compute_glb_acc(accuracies, list(range(len(workers))))
                duration = perf_counter() - start

                max_time = scheduler.max_time(action, False)
                table.add_row(
                    str(1),
                    f"{tr_avg_acc:.2%}",
                    f"{te_avg_acc:.2%}",
                    f"{duration:.3f} s",
                    f"No loss",
                    f"{max_time:.3f} s",
                )
                live.refresh()
                global_accs.append((tr_avg_acc, te_avg_acc))
                for r in range(1, rounds):
                    start = perf_counter()

                    # Selection of future participants
                    action, indices_participants = scheduler.select_next_partipants()
                    participants = [workers[i] for i in indices_participants]

                    # Workers download the global model
                    for worker in participants:
                        worker.communicatewith(self)

                    # Training loop of workers
                    train(participants)

                    # Workers evaluate accuracy of global model
                    # on their local training data
                    pair = evaluate(participants, True, full=True)
                    accuracies, _ = zip(*pair)
                    tr_avg_acc = self.compute_glb_acc(
                        accuracies, indices_participants, True
                    )

                    # Server downloads all local updates
                    for worker in participants:
                        self.communicatewith(worker)
                    scheduler.update_new_state(workers, indices_participants)

                    self.update(indices_participants)
                    scheduler.update(action)

                    for worker in workers:
                        worker.communicatewith(self)

                    # Workers evaluate accuracy of the global model
                    # on their local testing data
                    pair = evaluate(workers, full=True)
                    accuracies, _ = zip(*pair)
                    te_avg_acc = self.compute_glb_acc(
                        accuracies, list(range(len(workers)))
                    )
                    duration = perf_counter() - start

                    max_time = scheduler.max_time(action, True)
                    scheduler.copy_state()

                    # Update the table
                    table.add_row(
                        str(r + 1),
                        f"{tr_avg_acc:.2%}",
                        f"{te_avg_acc:.2%}",
                        f"{duration:.3f} s",
                        f"{scheduler.loss}",
                        f"{max_time:.3f} s",
                    )
                    live.refresh()

                    # Update global_accs
                    global_accs.append((tr_avg_acc, te_avg_acc))

                # Reset the server
                self.reset(path / "agent" / f"loss-rl-{iexp}.png")
                self.global_model = model()
                scheduler.reset()

                # Reset workers
                for worker in workers:
                    worker.model = model()

                # Save results
                with open(path / f"global_accs-{iexp}.pkl", "wb") as file:
                    pickle.dump(global_accs, file)

                global_accs.clear()
        scheduler.finish()
