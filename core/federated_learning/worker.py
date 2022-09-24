from torch.utils.data import DataLoader
from torch import optim, nn, rand
import pickle, torch, statistics
from utils.plot import lineXY
import random

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Worker:
    """
    Class which represents a 'worker' and communicate with the server.
    It holds the local data and a model.
    """

    GROUPS = [0.5, 0.7, 1]  # second / batch
    STD = 0.02
    BANDWIDTHS = (((30, 5), (8, 2)), ((5, 1), (0.5, 0.2)))
    NB_PARAMS = 1199882  # number of parameters

    def __init__(
        self,
        model,
        data_path,
        epochs,
        batch_size=64,
        optimizer=optim.Adadelta,
        criterion=nn.CrossEntropyLoss,
        **kwargs,
    ):
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        self._train, self._test = data
        self.toloader = lambda data: DataLoader(
            data, batch_size=batch_size, num_workers=1
        )
        self.computation_speed = self.GROUPS[random.choice(range(len(self.GROUPS)))]
        self.speed = lambda: abs(random.normalvariate(self.computation_speed, self.STD))
        self.network = random.choice(self.BANDWIDTHS)
        self.bandwidth_download = lambda: abs(
            random.normalvariate(self.network[0][0], self.network[0][1])
        )
        self.bandwidth_upload = lambda: abs(
            random.normalvariate(self.network[1][0], self.network[1][1])
        )
        self.model = model
        self.device = self.model.device
        self.epochs = epochs
        self.optim_obj = optimizer
        self.optimizer = self.optim_obj(self.model.parameters())
        self.criterion = criterion()
        self.batch_size = batch_size

    def receive(self, parameters):
        """
        When the server 'uploads' the global model parameters, the node
        receives them through this method
        """
        self.model.load_state_dict(parameters)
        self.optimizer = self.optim_obj(self.model.parameters())

    def send(self, weighted=True):
        """
        The node sends his local model parameters through this method
        """
        if weighted:
            return [len(self._train) * weight for weight in self.model.parameters()]
        else:
            return self.model.parameters()

    def communicatewith(self, aggregator):
        """
        For convenience, this method is used for communication.
        """
        self.receive(aggregator.send())

    def compute_times(self):
        return [
            self.speed() * (len(self._train) // self.batch_size) * self.epochs,
            self.NB_PARAMS * 1e-6 * 32 / self.bandwidth_upload(),
            self.NB_PARAMS * 1e-6 * 32 / self.bandwidth_download(),
        ]

    def train(self, filename=None):
        """
        Train the model using local data
        """
        losses = []
        self.model.train()
        trainloader = self.toloader(self._train)
        for _ in range(self.epochs):
            for samples, labels in trainloader:
                predictions = self.model(samples)
                loss = self.criterion(predictions, labels.to(self.device))
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if filename is not None:
            attrbs = {"title": "Evolution of loss function"}
            attrbs.update({"xrange": (0, len(losses) - 1)})
            attrbs.update({"x_title": "Steps", "y_title": "Loss values"})
            lineXY({"Losses": losses}, filename, **attrbs)

    def _evaluate(self, train=False, label=None, full=False):
        """
        Compute the accuracy on the local data given the parameters
        See `evaluate` for global usage

        Parameters:

            train (bool):
                True for evaluation on data for training else
                for evaluation on data for testing
            label (int):
                Evaluate on data given the label else evaluate
                on all available data
            full (bool):
                Return two values if True, the weighted accuracy
                and the non weighted accuracy
        """
        data = self._train if train else self._test
        if label is None:
            dataloader = self.toloader(data)
            n = len(data)
        elif label in data.labels:
            dataloader = self.toloader(data.classified[label])
            n = 1
        else:
            return 0
        correct, total = 0, 0
        self.model.eval()
        for samples, labels in dataloader:
            predictions = self.model(samples)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum().item()
        accuracy = correct / total
        return [n * accuracy, accuracy] if full else n * accuracy

    def evaluate(self, train=False, perlabel=False, full=False):
        """
        Compute the accuracy on the local data given the parameters

        Parameters:

            train (bool):
                True for evaluation on data for training else
                for evaluation on data for testing
            perlabel (bool):
                The evaluation is done label per label if it is True
                which means the accuracy is computed as the average of
                the accuracies per label
            full (bool):
                Return a list of pair values if True, the weighted
                accuracy and the non weighted accuracy
        """
        if perlabel:
            labels = self._train.labels if train else self._test.labels
            return statistics.mean(
                [self._evaluate(train, label, full) for label in labels]
            )
        else:
            return self._evaluate(train, full=full)
