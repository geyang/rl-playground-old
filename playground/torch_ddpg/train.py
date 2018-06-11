from gym.spaces import Discrete, Box
from torch import nn
from params_proto import cli_parse


@cli_parse
class G:
    env_name = "CartPole-v1"


def Critic(act_space):
    if type(act_space) == Discrete:
        act_dim = act_space.n
    elif type(act_space) == Box:
        act_dim = sum(act_space.shape)
    else:
        raise NotImplementedError('Action space is not supported.')
    return nn.Sequential(
        nn.Linear(sum(act_dim), 10),
        nn.ReLU(),
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
        nn.ReLU(),
    )


def MlpPolicy(ob_space, act_space):
    assert type(ob_space) == Box, "observation space should have Box.shape"
    ob_dim = ob_space.shape
    if type(act_space) == Discrete:
        act_dim = act_space.n
    elif type(act_space) == Box:
        act_dim = sum(act_space.shape)
    else:
        raise NotImplementedError('Action space is not supported.')
    return nn.Sequential(
        nn.Linear(sum(ob_dim), 10),
        nn.ReLU(),
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, act_dim),
        nn.ReLU(),
    )


def forward(self, obs):
    return self.model(obs)


def train():
    import gym
    env = gym.make(G.env_name)
    policy = MlpPolicy(env.observation_space, env.action_space)
    critic = Critic(env.action_space)

    # Let's collect data
    trajectories = []
    obs, reward, info, done = env.reset()
    for i in range(G.n_rollouts):
        a = policy(obs)
        obs, reward, info, done = env.reset(a)
        trajectories.append(obs)

    

    

if __name__ == "__main__":
    train()
