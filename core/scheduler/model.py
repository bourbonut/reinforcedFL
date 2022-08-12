import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from torch.optim import Adam
from torch.distributions import Bernoulli


class Actor(nn.Module):
    NHIDDEN = 128

    def __init__(self, state_dim, action_num, device):
        super(Actor, self).__init__()
        self.input = nn.Linear(state_dim, self.NHIDDEN)
        self.hidden1 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        self.hidden2 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        self.output = nn.Linear(self.NHIDDEN, action_num)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return (torch.tanh(self.output(x)) + 1) * 0.5


class Critic(nn.Module):
    NHIDDEN = 128

    def __init__(self, state_dim, device):
        super(Critic, self).__init__()
        self.input = nn.Linear(state_dim, self.NHIDDEN)
        self.hidden1 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        self.hidden2 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        self.output = nn.Linear(self.NHIDDEN, 1)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)


class ActorCritic:
    def __init__(self, ninput, noutput, device, la=1e-3, lc=1e-2):
        self.device = device
        self.gamma = 0.99
        self.actor = Actor(ninput, noutput, device)
        self.critic = Critic(ninput, device)

        self.a_optim = Adam(self.actor.parameters(), lr=la)
        self.c_optim = Adam(self.critic.parameters(), lr=lc)

    def train_actor(self, state, action, td_error):
        probas = self.actor(state)
        action = torch.tensor(action, dtype=torch.int)
        logprob = torch.gather(torch.log(probas), 1, action)
        loss = -logprob * td_error

        self.a_optim.zero_grad()
        loss.backward()
        self.a_optim.step()

    def train_critic(self, state, reward, state_):
        v_ = self.critic(state_)
        v = self.critic(state)
        td_error = reward + self.gamma * v_ - v
        loss = torch.square(td_error)

        self.c_optim.zero_grad()
        loss.backward()
        self.c_optim.step()
        return td_error

    def get_action(self, state):
        return torch.bernoulli(self.actor(state)).type(torch.int).tolist()


# class Policy(nn.Module):
#     NHIDDEN = 128
#     def __init__(self, state_dim, action_dim, device):
#         super(Policy, self).__init__()
#         self.device = device
#         self.input = nn.Linear(state_dim, self.NHIDDEN)
#         self.hidden1 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
#         self.hidden2 = nn.Linear(self.NHIDDEN, self.NHIDDEN)

#         self.actor_output = nn.Linear(self.NHIDDEN, action_dim)
#         self.critic_output = nn.Linear(self.NHIDDEN, 1)

#     def forward(self, x):
#         x = x.to(self.device)
#         x = F.relu(self.input(x))
#         x = F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         a = torch.tanh(self.actor_output(x))
#         actor_probas = (a + 1) * 0.5
#         critic_value = self.critic_output(x)
#         return actor_probas, critic_value
