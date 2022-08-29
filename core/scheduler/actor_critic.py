import torch
from torch import nn
from torch.nn import functional as F

class Actor(nn.Module):

    NHIDDEN = 128

    def __init__(self, ninput, noutput, device):
        super(Actor, self).__init__()
        self.action_space = noutput
        self.device = device

        # Layer 1
        self.linear1 = nn.Linear(ninput, self.NHIDDEN)
        # self.ln1 = nn.LayerNorm(self.NHIDDEN)

        # Layer 2
        self.linear2 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        # self.ln2 = nn.LayerNorm(self.NHIDDEN)

        # Layer 3
        self.linear3 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        # self.ln3 = nn.LayerNorm(self.NHIDDEN)

        # Output Layer
        self.mu = nn.Linear(self.NHIDDEN, noutput)

    def forward(self, inputs):
        x = inputs.to(self.device)

        # Layer 1
        x = self.linear1(x)
        # x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        # x = self.ln2(x)
        x = F.relu(x)

        # Layer 3
        x = self.linear3(x)
        # x = self.ln3(x)
        x = F.relu(x)

        # Output
        mu = (torch.tanh(self.mu(x)) + 1) * 0.5
        return mu


class Critic(nn.Module):

    NHIDDEN = 128

    def __init__(self, ninput, noutput, device):
        super(Critic, self).__init__()
        self.action_space = noutput
        self.device = device

        # Layer 1
        self.linear1 = nn.Linear(ninput, self.NHIDDEN)
        # self.ln1 = nn.LayerNorm(self.NHIDDEN)

        # Layer 2
        self.linear2 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        # self.ln2 = nn.LayerNorm(self.NHIDDEN)

        # Layer 3
        self.linear3 = nn.Linear(self.NHIDDEN + num_outputs, self.NHIDDEN)
        # self.ln3 = nn.LayerNorm(self.NHIDDEN)

        # Output layer (single value)
        self.V = nn.Linear(self.NHIDDEN, 1)


    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        # x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        # x = self.ln2(x)
        x = F.relu(x)

        # Layer 3
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear3(x)
        # x = self.ln3(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V
