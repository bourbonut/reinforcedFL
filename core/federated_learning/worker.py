from torch.utils.data import DataLoader
from torch import optim, nn
import pickle, torch, statistics
from utils.plot import lineXY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Worker:
    """
    Class which represents a 'worker' and communicate with the server.
    It holds the local data and a model.
    """

    def __init__(
        self,
        model,
        data_path,
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
        self.model = model
        self.optim_obj = optimizer
        self.optimizer = self.optim_obj(self.model.parameters())
        self.criterion = criterion()

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

    def train(self, filename=None):
        """
        Train the model using local data
        """
        losses = []
        self.model.train()
        trainloader = self.toloader(self._train)
        for samples, labels in trainloader:
            predictions = self.model(samples)
            loss = self.criterion(predictions, labels.to(device))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if filename is not None:
            attrbs = {"title": "Evolution of loss function"}
            attrbs.update({"xrange": (0, len(losses) - 1)})
            attrbs.update({"x_title": "Steps", "y_title": "Loss values"})
            lineXY({"Losses": losses}, filename, **attrbs)

    def _evaluate(self, train=False, label=None):
        """
        Compute the accuracy on the local data given the parameters
        See `evaluate` for global usage

        Parameters

            train (bool):
                True for evaluation on data for training else
                for evaluation on data for testing
            label (int):
                Evaluate on data given the label else evaluate
                on all available data
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
            correct += (predicted == labels.to(device)).sum().item()
        return n * correct / total

    def evaluate(self, train=False, perlabel=False):
        """
        Compute the accuracy on the local data given the parameters

        Parameters

            train (bool):
                True for evaluation on data for training else
                for evaluation on data for testing
            perlabel (bool):
                The evaluation is done label per label if it is True
                which means the accuracy is computed as the average of
                the accuracies per label
        """
        if perlabel:
            labels = self._train.labels if train else self._test.labels
            return statistics.mean([self._evaluate(train, label) for label in labels])
        else:
            return self._evaluate(train)
