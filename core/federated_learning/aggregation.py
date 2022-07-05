from core.evaluator.model import ReinforceAgent
from collections import deque
from itertools import compress
from utils.plot import lineXY
import torch


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

    def __init__(self, capacity):
        self.rewards = deque([], maxlen=capacity)
        self.states = deque([], maxlen=capacity)
        self.actions = deque([], maxlen=capacity)
        self.size = 0
        self.capacity = capacity

    def totorch(self):
        """
        Return three tensors namely states, rewards and actions
        """
        states = torch.FloatTensor(self.states)
        rewards = torch.FloatTensor(self.rewards)
        actions = torch.LongTensor([self.actions]).T
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
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.size = 0

class EvaluatorServer:
    """
    This class is based on the `REINFORCE` algorithm class from
    the evaluator module (`core.evaluator`) for a better aggregation
    (see the algorithm `FRCCE` - `arXiv:2102.13314v1`)
    """

    def __init__(
        self,
        global_model,
        ninput,
        noutput,
        capacity=3,
        gamma=0.99,
        window=3,
        optimizer=None,
        *args,
        **kwargs
    ):
        self.global_model = global_model
        self.agent = ReinforceAgent(ninput, noutput)
        self.optimizer = (
            torch.optim.Adam(self.agent.parameters(), lr=0.01)
            if optimizer is None
            else optimizer(self.agent.parameters())
        )
        self.workers_updates = []
        self.gamma = gamma
        self.delta = 0  # Window for moving average
        self.accuracies = []  # accuracies during training of task model
        self.window = window  # window for weighting moving average
        self.rewards = []
        self.losses = []
        self.capacity = capacity
        self.batchs = MovingBatch(capacity)
        self.total_rewards = []

    def send(self):
        """
        The server sends his global model parameters through this method
        """
        return self.global_model.state_dict()

    def receive(self, workers_update):
        """
        The server collects local model parameters through this method
        """
        self.workers_updates.append(workers_update)

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        self.receive(worker.send(False))

    def discount_rewards(self):
        """
        Compute the discount reward of the current step
        """
        r = torch.tensor([self.gamma**i * rw for i, rw in enumerate(self.rewards)])
        r = r.flip(0).cumsum(0).flip(0)
        return (r - r.mean())[-1]

    def update_batch(self, state, action):
        """
        Update MovingBatch class
        """
        self.batchs.rewards.append(self.discount_rewards())
        self.batchs.states.append(state)
        self.batchs.actions.append(action)
        self.batchs.update_size()

    def update(self, state, done):
        """
        Update the global model and the agent policy
        """
        # Selection of gradients which are going
        # to participate to the next aggregation
        probas = self.agent.forward(state)
        selection = action = torch.bernouilli(probas).tolist()
        p = sum(selection)  # number of participants
        participants = compress(self.workers_updates, selection)

        # Update the global model
        new_weights = map(lambda layer: sum(layer) / p, zip(*participants))
        for target_param, param in zip(self.global_model.parameters(), new_weights):
            target_param.data.copy_(param.data)
        self.workers_updates.clear()

        # Compute the reward
        curr_accuracy = sum(state) / p
        reward = curr_accuracy - self.delta
        self.rewards.append(reward)

        # Update batch array
        self.update_batch(state, action)

        # Optimization if batch is complete
        if self.batchs.isfull():
            self.optimizer.zero_grad()
            states, rewards, actions = self.batchs.totorch()

            # Calculate loss
            logprob = torch.log(self.agent.forward(states))

            selected_logprobs = rewards * torch.gather(logprob, 1, actions).squeeze()
            loss = -selected_logprobs.mean()
            self.losses.append(loss.item())

            loss.backward()  # Compute gradients
            self.optimizer.step()  # Apply gradients

            # Update the moving average
            self.delta = (curr_accuracy + (self.window - 1) * self.delta) / self.window

    def reset(self, filename=None):
        """
        Reset working attributes
        """
        self.batchs.clear()
        self.workers_updates.clear()
        self.rewards.clear()
        if filename is not None:
            attrbs = {"title": "Evolution of loss function"}
            attrbs.update({"xrange": (0, len(self.losses) - 1)})
            attrbs.update({"x_title": "Steps", "y_title": "Loss values"})
            lineXY({"Losses": self.losses}, filename, **attrbs)
        self.losses.clear()

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

    def receive(self, workers_update):
        """
        The server collects local model parameters through this method
        """
        self.workers_updates.append(workers_update)

    def communicatewith(self, worker):
        """
        For convenience, this method is used for communication.
        """
        self.receive(worker.send())

    def update(self):
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

    def global_accuracy(self, workers_accuracies, train=False):
        """
        Compute the global accuracy based on the Federated Averaging algorithm
        """
        size = self.n if train else self.t
        return sum(workers_accuracies) / size

    def reset(self, *args, **kwargs):
        pass
