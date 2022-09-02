from core.scheduler.model import DDPG, ReplayMemory, Transition
from torch.distributions import Bernoulli
from torch.nn import functional as F
import torch
from math import log, exp
import random, pickle
from itertools import compress
from collections import deque
from functools import reduce, partial
from math import copysign
import uuid


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
        self, ninput, noutput, device, path, k=1, mean=None, std=None, path_model=None, **kwargs
    ):
        self.agent = DDPG(ninput * k, noutput, device)
        self.device = device
        self.rewards = []
        self.action_dim = noutput
        self.action = None
        self.path = path
        self.i = 0
        self.k = k
        self.participants = []
        self.batchs = MovingBatch(2, device)
        self.mean = mean
        self.memory = ReplayMemory(50)
        self.std = std
        self.old_time = 0
        self.delta = 0
        self.losses = [0, 0]
        if path_model is not None:
            actor_path, critic_path = path_model
            self.agent.actor.load_state_dict(torch.load(str(actor_path)))
            self.agent.critic.load_state_dict(torch.load(str(critic_path)))
            self.agent.set_eval()

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
            return [int(i in sample) for i in range(self.action_dim)], None
        # print(state)
        normalized_state = self.normalize_all(state)
        # print(normalized_state)
        action = self.agent.get_action(normalized_state, debug=debug)
        participants = torch.bernoulli(action).tolist()

        # Update batch
        # self.batchs.states.append(normalized_state.view(-1, self.k).sum(1))
        # self.batchs.actions.append([i for i, x in enumerate(participants) if x])
        # self.batchs.update_size()

        self.participants.append(participants)
        return participants, action.tolist()

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
        
        # if self.batchs.isfull():
        #     states = self.batchs.states
        #     actions = self.batchs.actions
        #     selected = reduce(lambda x, y: x.union(y), map(set, actions))
        #     action = [1 if x in selected else 0 for x in selected]
        #     speed = states[-1] - states[0]
        #     sign = partial(copysign, 1)
        #     select = lambda x : 1 if x==1 else -1
        #     rt = lambda sp, ac: select(ac) * (sign(sp) if sp != 0 else 0) # * abs(sp)
        #     iterator = zip(speed, action)
        #     reward = reduce(lambda x, y: x + rt(*y), iterator, rt(*next(iterator))) / sum(action)
        #     reward = reward * 10
        #     # reward = (reward / 1000).item()
        #     # print("Reward:", reward)
        #     self.rewards.append(reward)
        #     # return reward
        # # else:
        #     # return None

    def update(self):
        if len(self.memory) > 10:
            transitions = self.memory.sample(10)
            batch = Transition(*zip(*transitions))
            
            # if len(self.memory)>19:
            #     print(batch)
            value_loss, policy_loss = self.agent.update_params(batch)
            self.losses = [value_loss, policy_loss]


    def reset(self):
        with open(self.path / f"rewards-{self.i}.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        torch.save(self.agent.actor.state_dict(), self.path / f"actor-{self.i}.pt")
        torch.save(self.agent.critic.state_dict(), self.path / f"critic-{self.i}.pt")
        self.rewards.clear()
        self.action = None
        self.i += 1
        self.old_time = 0
        self.delta = 0
        self.losses[0] = 0
        self.losses[1] = 0
        self.batchs.clear()

    def finish(self):
        idd = uuid.uuid4().hex # id
        with open(self.path / f"selections{idd}.pkl", "wb") as file:
            pickle.dump(self.participants, file)
        with open(self.path / f"probabilities{idd}.pkl", "wb") as file:
            pickle.dump(self.agent.probabilities, file)
        torch.save(self.agent.actor.state_dict(), self.path / "actor.pt")
        torch.save(self.agent.critic.state_dict(), self.path / "critic.pt")
