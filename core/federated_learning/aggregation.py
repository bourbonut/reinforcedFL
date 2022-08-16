from core.federated_learning.base import BaseServer, EvaluatorServer
from rich.table import Table
from rich.align import Align
from rich.live import Live
from time import perf_counter
import torch, pickle


class FederatedAveraging(BaseServer):
    """
    Class which plays the role of the self and aggregate through
    the algorithm `FedAvg` which is also named Federated Averaging
    (see `arXiv:1602.05629v3`)
    """

    def __init__(self, global_model, size_traindata, size_testdata, *args, **kwargs):
        super(FederatedAveraging, self).__init__(
            global_model, size_traindata, size_testdata
        )
        self.batch_loss = []

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

    def execute(
        self,
        nexp,
        rounds,
        scheduler,
        workers,
        train,
        evaluate,
        path,
        model,
        k=10,
        *args,
        **kwargs,
    ):
        global_accs = []
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
                start = perf_counter()
                # Selection of future participants
                selection, indices_participants = scheduler.select_next_partipants()
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

                max_time = scheduler.max_time(selection, False)
                table.add_row(
                    str(1),
                    f"{tr_avg_acc:.2%}",
                    f"{te_avg_acc:.2%}",
                    f"{duration:.3f} s",
                    f"{0}",
                    f"{max_time:.3f} s",
                )
                live.refresh()
                global_accs.append((tr_avg_acc, te_avg_acc))
                for r in range(1, rounds):
                    start = perf_counter()

                    # Selection of future participants
                    selection, indices_participants = scheduler.select_next_partipants()
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
                    scheduler.update(selection)

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

                    max_time = scheduler.max_time(selection, True)
                    scheduler.copy_state()

                    # Update the table
                    table.add_row(
                        str(r + 1),
                        f"{tr_avg_acc:.2%}",
                        f"{te_avg_acc:.2%}",
                        f"{duration:.3f} s",
                        f"{scheduler.agent.losses}",
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
                    self.update(torch.tensor(self.accuracies))

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
