from core.evaluator.model import ReinforceAgent
from collections import deque
from itertools import compress
from utils.plot import lineXY
import torch, pickle
import statistics

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MovingBatch:
    """
    Class which deals with a "moving array"

    Example with a capacity of 4
    `|` represents a value

      n  n + 1  n + 2  n + 3  n + 4
    [ |    |      |      |    ]
    [ |    |      |      |    ]  round t
    [ |    |      |      |    ]
    --------------------------------------------
        [  |      |      |      |  ]
    --->[  |      |      |      |  ] round t + 1
        [  |      |      |      |  ]
    """

    def __init__(self, capacity, device):
        self.states = deque([], maxlen=capacity)
        self.rewards = deque([], maxlen=capacity)
        self.actions = deque([], maxlen=capacity)
        self.size = 0
        self.capacity = capacity
        self.device = device

    def totorch(self):
        """
        Return three tensors namely rewards and the log of probabilities
        """
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.stack(tuple(self.actions)).to(self.device)
        return states, rewards, actions

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
        self.rewards.clear()
        self.actions.clear()
        self.size = 0


class EvaluatorServer:
    """
    This class is based on the `REINFORCE` algorithm class from
    the evaluator module (`core.evaluator`) for a better aggregation
    (inspired by the algorithm `FRCCE` - `arXiv:2102.13314v1`)
    """

    def __init__(
        self,
        global_model,
        size_traindata,
        size_testdata,
        ninput=None,
        noutput=None,
        capacity=3,
        gamma=0.99,
        optimizer=None,
        *args,
        **kwargs
    ):
        assert (
            ninput is not None and noutput is not None
        ), "`ninput` and `noutput` must be specified in configuration file"
        self.global_model = global_model
        self.n = size_traindata
        self.t = size_testdata
        device = self.global_model.device
        self.agent = ReinforceAgent(ninput, noutput, device).to(device)
        self.optimizer = (
            torch.optim.Adam(self.agent.parameters(), lr=0.01)
            if optimizer is None
            else optimizer(self.agent.parameters())
        )
        self.nworkers = ninput
        self.workers_updates = []
        self.gamma = gamma
        self.delta = 0  # Window for moving average
        self.accuracies = []  # accuracies during training of task model
        self.global_accuracies = [0] * ninput  # accuracies during testing of global task model
        self.alpha = 0.9  # window for exponential moving average
        self.capacity = capacity
        self.batchs = MovingBatch(capacity, device)
        self.tracking_rewards = []
        self.rewards = []
        self.losses = []
        self.batch_loss = []
        self.selections = []  # selections over time for analysis

    def send(self):
        """
        The server sends his global model parameters through this method
        """
        return self.global_model.state_dict()

    def receive(self, worker_update):
        """
        The server collects local model parameters through this method
        """
        self.workers_updates.append(worker_update)

    def collects_training_accuracies(self, accuracies):
        """
        The server collects local model accuracy
        on local worker data through this method
        """
        self.accuracies.extend(accuracies)

    def collects_global_accuracies(self, accuracies):
        """
        The server collects global model accuracy
        on local worker data through this method
        """
        self.global_accuracies.extend(accuracies)

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        self.receive(worker.send(False))

    def compute_glb_acc(self, workers_accuracies, train=False):
        """
        Compute the global accuracy based on the Federated Averaging algorithm
        """
        size = self.n if train else self.t
        return sum(workers_accuracies) / size

    def discount_rewards(self):
        """
        Compute the discount reward of the current step
        """
        tstep = torch.arange(len(self.rewards))
        rewards = torch.tensor(self.rewards)
        r = rewards * self.gamma**tstep
        r = r.flip(0).cumsum(0).flip(0)
        return (r / self.gamma**tstep)[-1]

    def update_batch(self, state, action):
        """
        Update MovingBatch class
        """
        self.batchs.rewards.append(self.discount_rewards())
        self.batchs.states.append(state)
        self.batchs.actions.append(action)
        self.batchs.update_size()

    def update_delta(self):
        """
        Update the initial value on the exponential moving average
        """
        self.delta = self.compute_glb_acc(self.global_accuracies)

    def update(self):
        """
        Update the global model and the agent policy
        """
        # Selection of gradients which are going
        # to participate to the next aggregation
        state = torch.tensor(self.accuracies) - torch.tensor(self.global_accuracies)
        probas = self.agent.forward(state)
        p = 0  # number of participants
        minp = self.nworkers // 10
        with torch.no_grad():
            while p <= minp:
                action = torch.bernoulli(probas)
                selection = action[:, 0].tolist()
                p = sum(selection)
        self.selections.append(selection)
        participants = compress(self.workers_updates, selection)

        # Update batch array
        self.update_batch(state, action.T)

        # Update the global model
        new_weights = map(lambda layer: sum(layer) / p, zip(*participants))
        for target_param, param in zip(self.global_model.parameters(), new_weights):
            target_param.data.copy_(param.data)
        self.workers_updates.clear()
        self.global_accuracies.clear()

    def train_agent(self):
        """
        Train the agent
        """
        # Compute the reward
        curr_accuracy = self.compute_glb_acc(self.global_accuracies)
        reward = curr_accuracy - self.delta
        self.rewards.append(reward)
        self.tracking_rewards.append(reward)

        # Optimization if batch is complete
        if self.batchs.isfull():
            self.optimizer.zero_grad()
            states, rewards, actions = self.batchs.totorch()

            # Compute the loss value
            probas = self.agent.forward(states)
            log_prob = (torch.log(probas) * actions).sum(1)
            batch_loss = (-log_prob * rewards.unsqueeze(1)).sum(1)
            self.batch_loss = batch_loss.tolist()
            loss = batch_loss.mean()
            self.losses.append(loss.item())

            loss.backward()  # Compute gradients
            self.optimizer.step()  # Apply gradients

        # Update the exponential moving average
        self.delta = self.delta + self.alpha * (curr_accuracy - self.delta)
        self.accuracies.clear()

    def reset(self, filename=None):
        """
        Reset working attributes
        """
        self.batchs.clear()
        self.workers_updates.clear()
        self.rewards.clear()
        self.accuracies.clear()
        self.global_accuracies.clear()
        self.delta = 0
        if filename is not None:
            attrbs = {"title": "Evolution of loss function"}
            attrbs.update({"xrange": (0, len(self.losses) - 1)})
            attrbs.update({"x_title": "Steps", "y_title": "Loss values"})
            lineXY({"Losses": self.losses}, filename, **attrbs)
            pkl = str(filename).replace("png", "pkl")
            with open(pkl, "wb") as file:
                pickle.dump(self.losses, file)
        self.losses.clear()

    def finish(self, path):
        """
        Save results at the end of the training for the agent
        """
        filename = path / "rl_rewards.png"
        attrbs = {"title": "Evolution of reward function"}
        attrbs.update({"xrange": (0, len(self.tracking_rewards) - 1)})
        attrbs.update({"x_title": "Steps", "y_title": "Rewards values"})
        lineXY({"Rewards": self.tracking_rewards}, filename, **attrbs)
        with open(path / "rl_rewards.pkl", "wb") as file:
            pickle.dump(self.tracking_rewards, file)
        with open(path / "selections.pkl", "wb") as file:
            pickle.dump(self.selections, file)
        torch.save(self.agent.state_dict(), path / "agent.pt")


class FederatedAveraging:
    """
    Class which plays the role of the server and aggregate through
    the algorithm `FedAvg` which is also named Federated Averaging
    (see `arXiv:1602.05629v3`)
    """

    def __init__(self, global_model, size_traindata, size_testdata, *args, **kwargs):
        self.global_model = global_model
        self.workers_updates = []
        self.n = size_traindata
        self.t = size_testdata

    def send(self):
        """
        The server sends his global model parameters through this method
        """
        return self.global_model.state_dict()

    def receive(self, worker_update):
        """
        The server collects local model parameters through this method
        """
        self.workers_updates.append(worker_update)

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        self.receive(worker.send())

    def update(self, workers):
        """
        Update the global model based on Federated Averaging algorithm

        Note:
            Updates of workers must be multiplied by the number of local examples
            In other words, a worker update is : `[nk * w for w in weights]` where
            `nk` is the number of local examples.
        """
        new_weights = map(lambda layer: sum(layer) / self.n, zip(*self.workers_updates))
        for target_param, param in zip(self.global_model.parameters(), new_weights):
            target_param.data.copy_(param.data)
        self.workers_updates.clear()

    def compute_glb_acc(self, workers_accuracies, train=False):
        """
        Compute the global accuracy based on the Federated Averaging algorithm
        """
        size = self.n if train else self.t
        return sum(workers_accuracies) / size

    def reset(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass
