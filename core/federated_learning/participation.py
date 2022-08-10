from core.scheduler.model import Policy
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log


class Scheduler:
    def __init__(self, ninput, noutput, device, **kwargs):
        self.agent = Policy(ninput, noutput, device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=3e-2)
        self.speed = log(1e-6) / log(1 - 0.95)
        self.gamma = 0.99
        self.rewards = []
        self.critic_value = None
        self.action = None
        self.log_prob = None

    def discount_reward(self):
        R = 0
        # returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            # returns.insert(0, R)
        # returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        # return returns[0]
        return R

    def select_next_partipants(self, state):
        actor_probas, self.critic_value = self.agent(state)
        m = Bernoulli(actor_probas)
        self.action = m.sample()
        self.log_prob = m.log_prob(self.action)

    def compute_reward(self, current_accuracy):
        reward = (1 - (0.95 - current_accuracy)) ** self.speed
        self.rewards.append(reward)

    def update(self):
        R = self.discount_reward()
        advantage = R - self.critic_value.item()
        policy_loss = -self.log_prob * advantage
        value_loss = F.smooth_l1_loss(self.critic_value, torch.tensor([R]))

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
