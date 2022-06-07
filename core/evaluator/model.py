# https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# WARNING: Not finished, only works on CartPole (see `gym` library)
class ReinforceAgent(nn.Module):

    """
    Neural network for REINFORCE algorithm
    """

    NHIDDEN = 128

    def __init__(self, ninput, noutput, *args, **kwargs):
        super(ReinforceAgent, self).__init__()
        self.input = nn.Linear(ninput, self.NHIDDEN)
        self.output = nn.Linear(self.NHIDDEN, noutput)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.input(x))
        return F.softmax(self.output(x), dim=-1)


def discount_rewards(rewards, gamma=0.99):
    """
    Return the discount reward
    """
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=2000, batch_size=10, gamma=0.99):
    """
    Function which follows the REINFORCE algorithm
    """
    # Set up lists to hold results
    results = {"ep": [], "avg_rewards": []}
    totorch = lambda x: torch.from_numpy(x)
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.parameters(), lr=0.01)

    action_space = np.arange(env.action_space.n)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            tensor_state = totorch(s_0)
            action_probs = policy_estimator.forward(tensor_state).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, done, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(np.array(batch_states))
                    reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(np.array([batch_actions])).T

                    # Calculate loss
                    logprob = torch.log(policy_estimator.forward(state_tensor))

                    selected_logprobs = (
                        reward_tensor
                        * torch.gather(logprob, 1, action_tensor).squeeze()
                    )
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                # print(
                #     "Ep: {} - Average of rewards of last 100: {:.2f}".format(
                #         ep + 1, avg_rewards
                #     )
                # )
                if (ep + 1) % 100 == 0:
                    print(ep + 1)
                results["ep"].append(ep + 1)
                results["avg_rewards"].append(avg_rewards)
                ep += 1

    return results
