from core.scheduler.model import ActorCritic
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log
import random, pickle


# def grouped(list_, k):
#     for i in range(len(list_) // k):
#         yield list_[k * i : k * (i + 1)]


class Scheduler:
    def __init__(self, ninput, noutput, device, path, k=1, **kwargs):
        self.agent = ActorCritic(ninput * k, noutput, device, la=1e-3, lc=1e-2).to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=5e-3)
        self.device = device
        self.rewards = []
        self.action_dim = noutput
        self.action = None
        self.path = path
        self.i = 0
        self.loss = 0
        self.k = k

    def normalize(self, state):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        state = state / torch.norm(state, dim=0)
        return state.flatten()

    def select_next_partipants(self, state):
        if state == []:
            population = list(range(self.action_dim))
            k = self.action_dim // 10
            sample = random.sample(population, k)
            return [int(i in sample) for i in range(self.action_dim)]
        return self.agent.get_action(state)

    def grouped(self, list_):
        k = self.k
        for i in range(len(list_) // k):
            yield list_[k * i : k * (i + 1)]

    def compute_reward(self, action, new_state):
        action = torch.tensor(action)
        normalized_state = self.normalize(new_state)
        reward = torch.max(normalized_state.sum(1) * action).item()
        self.rewards.append(reward)
        return reward

    def update(self, state, action, reward, new_state):
        normalized_state = self.normalize(state)
        normalized_new_state = self.normalize(new_state)
        td_error = self.agent.train_critic(normalized_state, reward, normalized_new_state)
        self.agent.train_actor(normalized_state, action, td_error)

    def reset(self):
        with open(self.path / f"rewards-{self.i}.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        self.rewards.clear()
        self.action = None
        self.i += 1
        self.loss = 0
