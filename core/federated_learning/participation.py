from core.scheduler.model import ActorCritic
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log
import random, pickle
from itertools import compress


# def grouped(list_, k):
#     for i in range(len(list_) // k):
#         yield list_[k * i : k * (i + 1)]


class Scheduler:
    def __init__(self, ninput, noutput, device, path, k=1, **kwargs):
        self.agent = ActorCritic(ninput * k, noutput, device, la=1e-3, lc=1e-2)
        self.device = device
        self.rewards = []
        self.action_dim = noutput
        self.action = None
        self.path = path
        self.i = 0
        self.k = k
        self.participants = []

    def normalize(self, state, selection, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        cstate = torch.tensor(list(compress(state.tolist(), selection)))
        mean = torch.cat([cstate.mean(0).unsqueeze(0)] * state.size(0))
        std = torch.cat([cstate.std(0).unsqueeze(0)] * state.size(0))
        state = (state - mean) / std
        #state = state / torch.norm(state, dim=0)
        return state.flatten() if flatten else state

    def minmax(self, times, selection):
        times = torch.tensor(times, dtype=torch.float).view(-1, self.k)
        ctimes = torch.tensor(list(compress(times.tolist(), selection)))
        min_ = torch.cat([ctimes.min(0)[0].unsqueeze(0)] * times.size(0))
        max_ = torch.cat([ctimes.max(0)[0].unsqueeze(0)] * times.size(0))
        return (times - min_) / (max_ - min_)

    def select_next_partipants(self, state, old_action):
        if state == []:
            population = list(range(self.action_dim))
            k = self.action_dim // 10
            sample = random.sample(population, k)
            self.participants.append([])
            return [int(i in sample) for i in range(self.action_dim)]
        participants = self.agent.get_action(self.normalize(state, old_action))
        self.participants.append(participants)
        return participants

    def grouped(self, list_):
        k = self.k
        for i in range(len(list_) // k):
            yield list_[k * i : k * (i + 1)]

    def compute_reward(self, action, new_state):
        action = torch.tensor(action)
        scaled_times = self.minmax(new_state, action)
        reward = -torch.max(scaled_times.sum(1) * action).item()
        self.rewards.append(reward)
        return reward

    def update(self, old_action, state, action, reward, new_state):
        normalized_state = self.normalize(state, old_action)
        normalized_new_state = self.normalize(new_state, action)
        td_error = self.agent.train_critic(normalized_state, reward, normalized_new_state)
        self.agent.train_actor(normalized_state, action, td_error)

    def reset(self):
        with open(self.path / f"rewards-{self.i}.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        self.rewards.clear()
        self.action = None
        self.i += 1
        self.agent.losses[0] = 0
        self.agent.losses[1] = 0

    def finish(self):
        with open(self.path / f"selections.pkl", "wb") as file:
            pickle.dump(self.participants, file)
        torch.save(self.agent.actor.state_dict(), self.path / "actor.pt")
        torch.save(self.agent.critic.state_dict(), self.path / "critic.pt")
