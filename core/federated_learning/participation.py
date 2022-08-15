from core.scheduler.model import ActorCritic
from math import log
from copy import copy
import random, pickle, torch

class BaseScheduler:
    def __init__(self, size, ratio=1):
        self.size = size
        self.state = []
        self.new_state = []
        self.k = ratio

    def normalize(self, state, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        state = state / torch.norm(state, dim=0)
        return state.flatten() if flatten else state

    def grouped(self, new=True):
        k = self.k
        list_ = self.new_state if new else self.state
        for i in range(len(list_) // k):
            yield list_[k * i : k * (i + 1)]

    def update_state(self, workers, indices):
        self.state.clear()
        for i, worker in enumerate(workers):
            if i in indices:
                self.state.extend(worker.compute_times())
            else:
                self.state.extend([0.0, 0.0, 0.0])

    def update_new_state(self, workers, indices):
        self.new_state.clear()
        for i, worker in enumerate(workers):
            if i in indices:
                self.new_state.extend(worker.compute_times())
            else:
                self.new_state.extend(self.state[3 * i : 3 * (i + 1)])

    def copy_state(self):
        self.state = copy(self.new_state)

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


class Scheduler(BaseScheduler):

    K = 3
    PORTION = 10

    def __init__(self, ninput, noutput, device, path, **kwargs):
        super(Scheduler, self).__init__(noutput, self.K)
        self.agent = ActorCritic(ninput * self.K, noutput, device, la=1e-3, lc=1e-2)
        self.device = device
        self.rewards = []
        self.path = path
        self.i = 0
        self.state = []
        self.new_state = []

    def select_next_partipants(self):
        if self.state == []:
            population = list(range(self.size))
            k = self.size // self.PORTION
            sample = random.sample(population, k)
            selection = [int(i in sample) for i in range(self.size)]
        else:
            selection = self.agent.get_action(self.normalize(self.state))
        indices = [i for i in range(len(self.size)) if selection[i]]
        return selection, indices

    def compute_reward(self, action):
        action = torch.tensor(action)
        normalized_state = self.normalize(self.new_state, False)
        reward = torch.max(normalized_state.sum(1) * action).item()
        self.rewards.append(reward)
        return reward

    def update(self, action):
        reward = self.compute_reward(action)
        normalized_state = self.normalize(self.state)
        normalized_new_state = self.normalize(self.new_state)
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
        self.agent.losses.clear()


class RandomScheduler(BaseScheduler):

    PORTION = 10

    def __init__(self, size):
        super(RandomScheduler, self).__init__(size)

    def select_next_partipants(self):
        population = list(range(self.size))
        k = self.size // self.PORTION
        sample = random.sample(population, k)
        selection = [int(i in sample) for i in range(self.size)]
        indices = [i for i in range(len(self.size)) if selection[i]]
        return selection, indices
