from torch import nn
from torch.nn import functional as F
from torch import optim

# https://github.com/kuangliu/pytorch-cifar

class Model(nn.Module):
    """
    Neural network for Cifar10 dataset
    """
    def __init__(self, nclasses, device):
        super(Model, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512, nclasses)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # def __init__(self, nclasses, device):
    #     super(Model, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)

    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, nclasses)

    #     self.device = device

    # def forward(self, x):
    #     x = x.to(self.device)
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)

    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))

    #     return self.fc3(x)


extras = {
    "optimizer": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    "criterion": nn.CrossEntropyLoss,
    # accuracy : manually
}
