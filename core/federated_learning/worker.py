from torch.utils.data import DataLoader
from torch import optim, nn
import pickle, torch
from utils import lineXY


class Node:
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
        self.trainloader = DataLoader(data[0], batch_size=batch_size, num_workers=1)
        self.testloader = DataLoader(data[1], batch_size=batch_size, num_workers=1)
        self.nk = len(data[0])  # number of local examples
        self.nt = len(data[1])  # number of local tests
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

    def send(self):
        """
        The node sends his local model parameters through this method
        """
        return [self.nk * weight for weight in self.model.parameters()]

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
        for samples, labels in self.trainloader:
            predictions = self.model(samples)
            loss = self.criterion(predictions, labels)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if not (filename is None):
            attrbs = {"title": "Evolution of loss function"}
            attrbs.update({"xrange": (0, len(losses) - 1)})
            attrbs.update({"x_title": "Steps", "y_title": "Loss values"})
            lineXY({"Losses": losses}, filename, **attrbs)

    def evaluate(self):
        """
        Compute the accuracy on the local data for testing
        """
        correct, total = 0, 0
        self.model.eval()
        for samples, labels in self.testloader:
            predictions = self.model(samples)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return self.nt * 100 * correct / total
