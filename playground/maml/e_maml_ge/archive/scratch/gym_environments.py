import gym
from gym import spaces
from moleskin import Moleskin
import numpy as np
from pprint import pformat
from tqdm import trange

from rl_algs.common.vec_env.subproc_vec_env import SubprocVecEnv

M = Moleskin()

env_names = [entry.id for entry in gym.envs.registry.all()]
M.green(pformat(env_names))


class G:
    env_name = 'HalfCheetah-v1'
    # env_name = 'CartPole-v0'
    n_envs = 10
    start_seed = 42
    log_directory = '../test_runs/demo_envs/{env_name}-{seed}'


# @M.timeit
def env_factory(*, seed=G.start_seed, env_name=G.env_name, monitor=False):
    env = gym.make(env_name)
    env.seed(seed)
    if monitor:
        env = gym.wrappers.Monitor(env, RUN.log_directory.format(env_name=env_name, seed=seed), force=True)
    return env


@M.timeit
def batch_env():
    # note: this takes hundreds of mili-seconds to run. Env instantiation is *slow*.
    return SubprocVecEnv([lambda: env_factory(seed=G.start_seed + s) for s in range(G.n_envs)])


envs = batch_env()

envs.reset()
M.green(envs.action_space)


@M.timeit
def run():
    for i in trange(1000):
        obs, rews, dones, infos = envs.step(np.array([[1, 1, 1, 1, 1, 1]] * 10))


run()

M.green(envs.action_space)
assert isinstance(envs.action_space, spaces.Discrete) is False, "Box should not be Discrete"
assert isinstance(envs.action_space, spaces.Box)
