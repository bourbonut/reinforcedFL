from evaluator.model import *
import gym

env = gym.make("CartPole-v0")
Estimator.NHIDDEN = 16
ninput = env.observation_space.shape[0]
noutput = env.action_space.n
agent = Estimator(ninput, noutput).to(device)


def func(x):
    return x + 1


def test_cannary():
    assert func(3) == 4


def test_neural_network():
    assert agent.input.in_features == 4
    assert agent.input.out_features == 16
    assert agent.output.in_features == 16
    assert agent.output.out_features == 2


def test_reinforce():
    results = reinforce(env, agent)
    assert results["ep"][-1] == 2000
    assert len(results["ep"]) == 2000
    assert max(results["avg_rewards"]) > 180
