import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    Neural network for MNIST dataset
    """

    def __init__(self, nclasses):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.maxp2d = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.hidden_layer = nn.Linear(24 * 24 * 16, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(128, nclasses)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxp2d(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.hidden_layer(x))

        x = self.dropout2(x)
        return F.softmax(self.output(x), dim=-1)


extras = {
    "optimizer": lambda params: optim.Adadelta(params),
    "criterion": nn.CrossEntropyLoss,
    # accuracy : manually
}
