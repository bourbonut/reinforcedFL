from torch.utils.data import DataLoader
from torch import optim, nn
import pickle


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
        self.model = model
        self.optimizer = optimizer(self.model.parameters())
        self.criterion = criterion()

    def receive(self, parameters):
        self.model.load_state_dict(parameters)
        # self.optimizer = optimizer(self.model.parameters())

    def send(self):
        return [self.nk * weight for weight in self.model.parameters()]

    def communicatewith(self, aggregator):
        self.receive(aggregator.send())

    def forward(self):
        for sample, label in self.trainloader:
            prediction = self.model(sample)
            loss = self.criterion(prediction, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
