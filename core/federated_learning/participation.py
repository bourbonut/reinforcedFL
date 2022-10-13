from core.scheduler.model import DDPG, ReplayMemory, Transition
from math import log
from copy import copy
from itertools import compress
import random, pickle, torch, statistics


class BaseScheduler:
    def __init__(self, size, ratio=1):
        self.size = size
        self.state = []
        self.new_state = []
        self.k = ratio
        self.loss = "No loss"

    def normalize(self, state, flatten=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.k)
        mean = torch.cat([state.mean(0).unsqueeze(0)] * state.size(0))
        std = torch.cat([state.std(0).unsqueeze(0)] * state.size(0))
        state = (state - mean) / std 
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

        # Note here, this is an approximation of the mean of each time
        means = [statistics.mean(filter(lambda x: x!=0., self.state[i::3])) for i in range(3)]
        self.state = [(random.random() * 0.2 + 0.9) * means[i%3] if s==0. else s for i, s in enumerate(self.state)]

    def update_new_state(self, workers, indices):
        self.new_state.clear()
        for i, worker in enumerate(workers):
            if i in indices:
                self.new_state.extend(worker.compute_times())
            else:
                self.new_state.extend(self.state[3 * i : 3 * (i + 1)])

    def max_time(self, selection, new=True):
        return max(compress(map(sum, self.grouped(new)), selection))

    def copy_state(self):
        self.state = copy(self.new_state)

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class Scheduler(BaseScheduler):

    K = 3
    PORTION = 0.1

    def __init__(self, size, device, path, **kwargs):
        super(Scheduler, self).__init__(size, self.K)
        self.agent = DDPG(size * self.K, size, device)
        self.device = device
        self.rewards = []
        self.path = path
        self.memory = ReplayMemory(50)
        self.state = []
        self.new_state = []
        self.participants = []
        self.loss = []
        self.i = 0
        self.old_time = 0

    def select_next_partipants(self):
        if self.state == []:
            population = list(range(self.size))
            k = int(self.size * self.PORTION)
            sample = random.sample(population, k)
            action = participants = [int(i in sample) for i in range(self.size)]
            self.participants.append([])
        else:
            action = self.agent.get_action(self.normalize(self.state))
            participants = torch.bernoulli(action).tolist()
            action = action.tolist()
            self.participants.append(participants)
        indices = [i for i in range(self.size) if participants[i]]
        return action, indices

    def compute_reward(self, action):
        action = torch.tensor(action)
        new_state = self.normalize(self.new_state, False)
        time = -torch.max(new_state.mean(1) * action).item()
        reward = time - self.old_time
        self.old_time = time
        self.rewards.append(reward)
        return reward

    def update(self, action):
        reward = self.compute_reward(action)
        state = self.normalize(self.state).tolist()
        new_state = self.normalize(self.new_state).tolist()
        self.memory.push(state, action, 0, new_state, reward)
        if len(self.memory) > 10:
            transitions = self.memory.sample(10)
            batch = Transition(*zip(*transitions))
            value_loss, policy_loss = self.agent.update_params(batch)
            self.loss = [value_loss, policy_loss]

    def reset(self):
        with open(self.path / f"rewards-{self.i}.pkl", "wb") as file:
            pickle.dump(self.rewards, file)
        self.rewards.clear()
        self.action = None
        self.i += 1
        self.loss = []
        self.old_time = 0
        self.state.clear()
        self.new_state.clear()

    def finish(self):
        with open(self.path / "selections.pkl", "wb") as file:
            pickle.dump(self.participants, file)
        torch.save(self.agent.actor.state_dict(), self.path / "actor.pt")
        torch.save(self.agent.critic.state_dict(), self.path / "critic.pt")


class RandomScheduler(BaseScheduler):

    PORTION = 0.1

    def __init__(self, size, *args, **kwargs):
        super(RandomScheduler, self).__init__(size)

    def select_next_partipants(self):
        population = list(range(self.size))
        k = max(1, int(self.size * self.PORTION))
        sample = random.sample(population, k)
        selection = [int(i in sample) for i in range(self.size)]
        indices = [i for i in range(self.size) if selection[i]]
        return selection, indices


class FullScheduler(BaseScheduler):
    def __init__(self, size, ratio=1, *args, **kwargs):
        super().__init__(size, ratio)

    def select_next_partipants(self):
        return [1] * self.size, list(range(self.size))
