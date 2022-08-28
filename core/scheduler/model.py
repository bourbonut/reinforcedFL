import torch
from torch import log, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.distributions import Bernoulli
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):

    NHIDDEN = 64

    def __init__(self, ninput, noutput, device):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(ninput, self.NHIDDEN)
        self.affine2 = nn.Linear(self.NHIDDEN, self.NHIDDEN)
        self.affine3 = nn.Linear(self.NHIDDEN, self.NHIDDEN)

        # actor's layer
        self.action_head = nn.Linear(self.NHIDDEN, noutput)

        # critic's layer
        self.value_head = nn.Linear(self.NHIDDEN, 1)

        # action & reward buffer
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.sigmoid(self.affine1(x))
        x = F.sigmoid(self.affine2(x))
        x = F.sigmoid(self.affine3(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = (torch.tanh(self.action_head(x)) + 1.) * 0.5

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

class ActorCritic:

    def __init__(self, ninput, noutput, device, lr=1e-3):
        self.agent = Policy(ninput, noutput, device).to(device)
        self.device = device
        self.saved_actions = []
        self.rewards = []
        self.optimizer = Adam(self.agent.parameters(), lr=lr)
        self.gamma = 0.99
        self.losses = []
        self.probabilities = []

    def get_action(self, state, debug=None):
        probs, state_value = self.agent(state)
        self.probabilities.append(probs.to("cpu").tolist())
        m = Bernoulli(probs)

        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.tolist()
    
    def train_agent(self):
        eps = 1e-6
        returns = []
        policy_losses = []
        value_losses = []

        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        self.rewards.clear()
        self.saved_actions.clear()
