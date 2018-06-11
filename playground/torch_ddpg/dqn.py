import torch

from gym.spaces import Discrete, Box
from torch import nn
import torch_helpers as h
from params_proto import cli_parse


@cli_parse
class G:
    env_name = "CartPole-v1"
    parallel_envs = 10
    n_rollouts = 10
    gamma = 0.99


def Q(ob_space, act_space):
    assert type(ob_space) == Box, "observation space should have Box.shape"
    assert type(act_space) == Discrete, "only Discrete action space is allowed"
    ob_dim = ob_space.shape
    act_dim = act_space.n
    return nn.Sequential(
        nn.Linear(sum(ob_dim), 10),
        nn.ReLU(),
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, act_dim),
        nn.Tanh(),
        nn.Softmax()
    )


def forward(self, obs):
    return self.model(obs)


def train():
    import gym
    env = gym.make(G.env_name)
    q = Q(env.observation_space, env.action_space)
    q_target = q.copy()

    def update_target():
        for p_, p in zip(q_target.parameters(), q.parameters()):
            p_.data = p.data.detach()

    def td_error(rewards, terminal, state, action):
        

    # Let's collect data
    trajectories = []
    actions = []
    rewards = []
    obs = env.reset()
    with h.Eval(q) and torch.no_grad():
        for i in range(G.n_rollouts):
            a_prob = q(h.const([obs]))
            a = torch.distributions.Categorical(a_prob).sample().detach().item()
            obs, reward, info, done = env.step(a)
            trajectories.append(obs)
            actions.append(a)
            rewards.append(reward)
    import numpy as np
    values = np.zeros(len(rewards))
    gammas = G.gamma ** np.arange(len(rewards))[::-1]
    # todo: can linearize
    for ind, r in enumerate(rewards):
        values[:-ind - 1] += r * gammas[:-ind - 1]
    print(values)

    # compute TD error
    # td error: 
    td = q(trajectories) * actions
    # gradient descent on Q function
    # update target Q function
    # Plappert et.al. Parameter Space Exploration


if __name__ == "__main__":
    train()
