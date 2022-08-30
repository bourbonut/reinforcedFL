from core.scheduler.model import ActorCritic
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log, exp
import random, pickle
from itertools import compress
from collections import deque
from functools import reduce, partial
from math import copysign


class MovingBatch:
    """
    Class which deals with a "moving array"

    Example with a capacity of 4
    `|` represents a value

      n  n + 1  n + 2  n + 3  n + 4
    [ |    |      |      |    ]
    [ |    |      |      |    ]  round t
    --------------------------------------------
        [  |      |      |      |  ]
    --->[  |      |      |      |  ] round t + 1
    """

    def __init__(self, capacity, device):
        self.states = deque([], maxlen=capacity)
        self.actions = deque([], maxlen=capacity)
        self.size = 0
        self.capacity = capacity
        self.device = device

    def totorch(self):
        """
        Return three tensors namely rewards and the log of probabilities
        """
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        return states, actions

    def update_size(self):
        """
        Add 1 to the size attribute (maximum: capacity)
        """
        self.size = min(self.capacity, self.size + 1)

    def isfull(self):
        """
        Return True if the batch is full (to start the
        learning process)
        """
        return self.capacity == self.size

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.size = 0


class Scheduler:
    def __init__(
        self, ninput, noutput, device, path, k=1, mean=None, std=None, **kwargs
    ):
        self.agent = ActorCritic(ninput * k, noutput, device, la=1e-3, lc=1e-2)
        self.device = device
        self.rewards = []
        self.action_dim = noutput
        self.action = None
        self.path = path
        self.i = 0
        self.k = k
        self.participants = []
        self.batchs = MovingBatch(4, device)
        self.old_time = 0
        self.mean = mean
        self.std = std

    def norm(self, state, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        state = (state - self.mean) / self.std
        return state.flatten() if flatten else state

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

    def normalize_all(self, state, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        mean = torch.cat([state.mean(0).unsqueeze(0)] * state.size(0))
        std = torch.cat([state.std(0).unsqueeze(0)] * state.size(0))
        state = (state - mean) / std
        return state.flatten() if flatten else state

    def norml2_all(self, state, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        state = state / torch.norm(state, dim=0)
        return state.flatten() if flatten else state

    def minmax_all(self, times, _):
        times = torch.tensor(times, dtype=torch.float).view(-1, self.k)
        min_ = torch.cat([times.min(0)[0].unsqueeze(0)] * times.size(0))
        max_ = torch.cat([times.max(0)[0].unsqueeze(0)] * times.size(0))
        return (times - min_) / (max_ - min_)


    def select_next_partipants(self, state, old_action, debug=None):
        if state == []:
            population = list(range(self.action_dim))
            k = self.action_dim // 10
            sample = random.sample(population, k)
            self.participants.append([])
            if len(self.agent.probabilities)>0:
                self.agent.probabilities.append(self.agent.probabilities[-1])
            else:
                self.agent.probabilities.append([0.] * self.action_dim)
            return [int(i in sample) for i in range(self.action_dim)]
        normalized_state = self.normalize_all(state)
        participants = self.agent.get_action(normalized_state, debug=debug)

        # Update batch
        self.batchs.states.append(normalized_state.view(-1, self.k).sum(1))
        self.batchs.actions.append([i for i, x in enumerate(participants) if x])
        self.batchs.update_size()

        self.participants.append(participants)
        return participants

    def grouped(self, list_):
        k = self.k
        for i in range(len(list_) // k):
            yield list_[k * i : k * (i + 1)]

    def compute_reward(self, action, new_state):
        action = torch.tensor(action)
        normalized_new_state = self.normalize_all(new_state, False)
        time = -torch.max(normalized_new_state.mean(1) * action).item()
        reward = (time - self.old_time)
        self.old_time = time
        # new_state = torch.tensor(new_state).view(-1, self.k)
        # reward = -torch.max(new_state.sum(1) * action).item()
        self.rewards.append(reward)
        return reward

    def update(self, old_action, state, action, reward, new_state):
        if reward is not None:
            normalized_state = self.norm(state, old_action)
            normalized_new_state = self.norm(new_state, action)
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
        # self.delta = 0
        self.agent.losses[0] = 0
        self.agent.losses[1] = 0
        self.batchs.clear()

    def finish(self):
        with open(self.path / "selections.pkl", "wb") as file:
            pickle.dump(self.participants, file)
        with open(self.path / "probabilities.pkl", "wb") as file:
            pickle.dump(self.agent.probabilities, file)
        torch.save(self.agent.actor.state_dict(), self.path / "actor.pt")
        torch.save(self.agent.critic.state_dict(), self.path / "critic.pt")
