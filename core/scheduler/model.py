import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from core.scheduler.actor_critic import Actor, Critic
from collections import namedtuple
import random

Transition = namedtuple(
    "Transition", ("state", "action", "done", "next_state", "reward")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, done, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        done = torch.tensor([done])
        reward = torch.tensor([reward])
        next_state = torch.tensor([next_state])
        self.memory[self.position] = Transition(state, action, done, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    NHIDDEN = 128

    def __init__(self, ninput, noutput, device):
        self.gamma = 0.99
        self.tau = 0.05
        self.action_space = [0., 1.]
        self.device = device

        # Define the actor
        self.actor = Actor(ninput, noutput, device).to(device)
        self.actor_target = Actor(ninput, noutput, device).to(device)

        # Define the critic
        self.critic = Critic(ninput, noutput, device).to(device)
        self.critic_target = Critic(ninput, noutput, device).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=1e-4
        )  # optimizer for the actor network
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=1e-3, weight_decay=1e-2
        )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def get_action(self, state, action_noise=None):
        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(self.device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space[0], self.action_space[0])

        return mu

    def update_params(self, batch):
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch.detach()
        )

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = (
            reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values
        )

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def set_eval(self):
        """
        Sets the model in evaluation mode
        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode
        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
