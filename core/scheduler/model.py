from rich import console
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
        x = torch.tanh(self.input(x))
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
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)


class ActorCritic:
    def __init__(self, ninput, noutput, device, la=1e-3, lc=1e-2):
        self.device = device
        self.gamma = 0.99
        # self.minp = int(0.1 * noutput)
        self.minp = 1
        self.actor = Actor(ninput, noutput, device).to(device)
        self.critic = Critic(ninput, device).to(device)

        self.a_optim = Adam(self.actor.parameters(), lr=la)
        self.c_optim = Adam(self.critic.parameters(), lr=lc)
        self.losses = [0, 0]
        self.probabilities= []

    def train_actor(self, state, action, td_error):
        probas = self.actor(state)
        action = torch.bernoulli(probas)
        logprob = torch.log(probas) * action
        loss = -logprob.sum() * td_error
        self.losses[1] = loss.item()

        self.a_optim.zero_grad()
        loss.backward()
        # for param in self.actor.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.a_optim.step()

    def train_critic(self, state, reward, state_):
        v_ = self.critic(state_).detach()
        v = self.critic(state)
        td_error = reward + self.gamma * v_ - v
        loss = torch.square(td_error)
        self.losses[0] = loss.item()

        self.c_optim.zero_grad()
        loss.backward()
        # for param in self.critic.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.c_optim.step()
        return td_error.detach()

    def get_action(self, state, debug=None):
        # print(state.size())
        probas = self.actor(state)
        if debug is not None:
            x = [(s, p) for s, p in zip(debug, probas.tolist())]
            sx = sorted(x, key=lambda e: e[0])
            self.probabilities.append([b for a, b in sx])
            # string = ""
            # for i in range(10):
            #     data = sx[10 * i : 10 * (i + 1)]
            #     string += ", ".join((f"{a:>8.3f}" + ":" + f"{b:.2%}" for a, b in data)) + "\n"
            # print(string)

        # mean = probas.mean()
        # action = [1 if x >= mean else 0 for x in probas.tolist()]
        # selection = sorted(list(zip(range(100), probas.tolist())), key=lambda x:x[1])[-5:]
        # indices = [x for x, _ in selection]
        # action = [1 if i in indices else 0 for i in range(100)]
        # p = 0
        # while p < self.minp:
        #     try:
        #         action = torch.bernoulli(probas).type(torch.int).tolist()
        #     except:
        #         print(state)
        #         action = None
        #     p = sum(action)
        action = torch.bernoulli(probas).type(torch.int).tolist()
        return action


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
