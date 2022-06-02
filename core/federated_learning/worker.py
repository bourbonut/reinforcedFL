from torch.utils.data import DataLoader
from torch import optim, nn
import pickle, torch


class Node:
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
        self.nk = len(self.trainloader)  # number of local examples
        self.nt = len(self.testloader)  # number of local tests
        self.model = model
        self.optim_obj = optimizer
        self.optimizer = self.optim_obj(self.model.parameters())
        self.criterion = criterion()

    def receive(self, parameters):
        self.model.load_state_dict(parameters)
        self.optimizer = self.optim_obj(self.model.parameters())

    def send(self):
        return [self.nk * weight for weight in self.model.parameters()]

    def communicatewith(self, aggregator):
        self.receive(aggregator.send())

    def train(self):
        """
        Train the model
        """
        self.model.train()
        for samples, labels in self.trainloader:
            predictions = self.model(samples)
            loss = self.criterion(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        """
        Compute the accuracy on the data for testing
        """
        correct, total = 0, 0
        self.model.eval()
        for samples, labels in self.testloader:
            predictions = self.model(samples)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return self.nt * 100 * correct / total
