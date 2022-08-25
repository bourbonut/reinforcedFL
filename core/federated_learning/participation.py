from core.scheduler.model import ActorCritic
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log, exp
import random, pickle
from itertools import compress


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
        self.old_time = 0
        self.delta = 0

    def normalize(self, state, selection, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        cstate = torch.tensor(list(compress(state.tolist(), selection)))
        if cstate.shape[0] == 1:
            return state / torch.cat([cstate] * state.size(0))
        mean = torch.cat([cstate.mean(0).unsqueeze(0)] * state.size(0))
        std = torch.cat([cstate.std(0).unsqueeze(0)] * state.size(0))
        state = (state - mean) / std
        return state.flatten() if flatten else state

    def norml2(self, state, selection, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        cstate = torch.tensor(list(compress(state.tolist(), selection)))
        state = state / torch.norm(cstate, dim=0)
        return state.flatten() if flatten else state

    def minmax(self, times, selection, _):
        times = torch.tensor(times, dtype=torch.float).view(-1, self.k)
        ctimes = torch.tensor(list(compress(times.tolist(), selection)))
        if ctimes.shape[0] == 1:
            return times / torch.cat([ctimes] * times.size(0))
        min_ = torch.cat([ctimes.min(0)[0].unsqueeze(0)] * times.size(0))
        max_ = torch.cat([ctimes.max(0)[0].unsqueeze(0)] * times.size(0))
        return (times - min_) / (max_ - min_)

    def select_next_partipants(self, state, old_action, debug=None):
        if state == []:
            population = list(range(self.action_dim))
            k = self.action_dim // 10
            sample = random.sample(population, k)
            self.participants.append([])
            return [int(i in sample) for i in range(self.action_dim)]
        participants = self.agent.get_action(
            self.normalize(state, old_action), debug=debug
        )
        self.participants.append(participants)
        return participants

    def grouped(self, list_):
        k = self.k
        for i in range(len(list_) // k):
            yield list_[k * i : k * (i + 1)]

    def compute_reward(self, action, new_state):
        k = sum(action)
        action = torch.tensor(action)
        times = torch.tensor(new_state).view(-1, self.k)
        scaled_times = self.minmax(new_state, action, False)
        not_scaled_times = times.sum(1) * action
        # x = scaled_times.sum(1) * action
        # index = torch.argmax(x).item()
        # time = times.sum(1)[index].item()
        # reward = -exp(-torch.max(x).item() * k / 100)
        # reward = -exp((-torch.max(x).item() - torch.min(x).item()) * k / 100)
        # print("Delta avant:", self.delta)
        # rt = -(torch.max(not_scaled_times).item() + torch.min(not_scaled_times).item())
        # print("Rt:", rt)
        # reward = rt - self.delta
        # self.delta = self.delta + 0.9 * (reward - self.delta)
        # print("Delta après:", self.delta)
        # reward = torch.max(not_scaled_times).item() + torch.min(not_scaled_times).item()
        # reward = (reward - not_scaled_times.mean()) / not_scaled_times.std() * k / 100

        time_action = scaled_times.sum(1) * action
        reward =  2 * scaled_times.sum(1).mean() - (torch.max(time_action) + torch.min(time_action))
        # reward = reward * 2


        # self.old_time = reward
        # reward = -torch.mean(x).item()

        # if value > self.old_time:
        #     reward = -1
        #     self.old_time = value
        # elif value < self.old_time:
        #     reward = 1
        #     self.old_time = value
        # else:
        #     reward = 0
        #     self.old_time = value

        # if time > self.old_time:
        #     reward = -1 # * k
        #     self.old_time = time
        # elif time < self.old_time:
        #     reward = 1 # * k
        #     self.old_time = time
        # else:
        #     reward = 0
        #     self.old_time = time

        print("Reward:", reward)
        self.rewards.append(reward)
        return reward

    def update(self, old_action, state, action, reward, new_state):
        normalized_state = self.normalize(state, old_action)
        normalized_new_state = self.normalize(new_state, action)
        td_error = self.agent.train_critic(
            normalized_state, reward, normalized_new_state
        )
        self.agent.train_actor(normalized_state, action, td_error)

    def reset(self):
        with open(self.path / f"rewards-{self.i}.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        self.rewards.clear()
        self.action = None
        self.i += 1
        self.old_time = 0
        self.delta = 0
        self.agent.losses[0] = 0
        self.agent.losses[1] = 0

    def finish(self):
        with open(self.path / "selections.pkl", "wb") as file:
            pickle.dump(self.participants, file)
        with open(self.path / "rewards.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        torch.save(self.agent.actor.state_dict(), self.path / "actor.pt")
        torch.save(self.agent.critic.state_dict(), self.path / "critic.pt")
