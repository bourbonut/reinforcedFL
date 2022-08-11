from core.scheduler.model import Policy
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log
import random, pickle


def grouped(list_, k):
    for i in range(len(list_) // k):
        yield list_[k * i : k * (i + 1)]


class Scheduler:
    def __init__(self, ninput, noutput, device, path, **kwargs):
        self.agent = Policy(ninput, noutput, device).to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=3e-2)
        self.device = device
        self.speed = log(1e-6) / log(1 - 0.95)
        self.gamma = 0.99
        self.rewards = []
        self.critic_value = None
        self.action = None
        self.log_prob = None
        self.action_dim = noutput
        self.path = path
        self.i = 0
        self.loss = 0
        self.k = ninput // noutput

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
        if state == []:
            return sorted(random.sample(list(range(self.action_dim)), self.action_dim // 10))
            #return sorted(random.sample(list(range(self.action_dim)), self.action_dim))
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        state = state / torch.norm(state, dim=0)
        state = state.flatten()
        actor_probas, self.critic_value = self.agent(state)
        #print(f"{actor_probas = }")
        #print(f"{self.critic_value = }")
        m = Bernoulli(actor_probas)
        self.action = m.sample()
        self.log_prob = m.log_prob(self.action)
        action = self.action.type(torch.int).tolist()
        return [i for i in range(len(action)) if action[i]]
    
    def grouped(self, list_):
        k = self.k
        for i in range(len(list_) // k):
            yield list_[k * i : k * (i + 1)]

    def compute_reward(self, worker_times):
        # iterator = zip(action, self.grouped(worker_times))
        # reward = -max((a * sum(time) for a, time in iterator))
        action = self.action.clone().to("cpu")
        worker_times = torch.tensor(worker_times).view(-1, self.k)
        worker_times = worker_times / torch.norm(worker_times, dim=0)
        reward = torch.max(worker_times.sum(1) * action).item()
        #print(f"{reward = }")
        self.rewards.append(reward)

    # def compute_reward(self, current_accuracy):
    #     reward = (1 - (0.95 - current_accuracy)) ** self.speed
    #     self.rewards.append(reward)

    def update(self):
        R = self.discount_reward()
        advantage = R - self.critic_value.item()
        policy_loss = -self.log_prob.sum() * advantage
        value_loss = F.smooth_l1_loss(self.critic_value, torch.tensor([R]).to(self.device))

        #print(f"{policy_loss = }")
        #print(f"{value_loss = }")
        loss = policy_loss + value_loss
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        with open(self.path / f"rewards-{self.i}.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        self.rewards.clear()
        self.critic_value = None
        self.action = None
        self.log_prob = None
        self.i += 1
        self.loss = 0
