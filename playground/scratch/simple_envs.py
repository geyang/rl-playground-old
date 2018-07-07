import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from rllab.core.serializable import Serializable


logger = logging.getLogger(__name__)

class SquareEnv(gym.Env, Serializable):
    """Agent controls a point in a square."""

    _goal = np.array([-1, -1.])

    def __init__(self, noise=1.0, border_terminate=False):
        Serializable.quick_init(self, locals())
        self._action_max = 1.0
        self._obs_max = 10.0
        self._noise = noise
        self._border_terminate = border_terminate
        self._set_spaces()
        self._seed()
        self.viewer = None

    def _set_spaces(self):
        bound = np.ones(2)
        self.action_space = spaces.Box(-self._action_max * bound, self._action_max * bound)
        self.observation_space = spaces.Box(-self._obs_max * bound, self._obs_max * bound)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return np.array(self._state)

    def step(self, action):
        self._prev_state = self._state.copy()
        self._state += action
        done = False
        if self._border_terminate and np.any(np.abs(self._state) >= self._obs_max):
            done = True
        self._state = np.clip(self._state, -self._obs_max, self._obs_max)
        reward = -np.linalg.norm(self._state - self._goal)
        obs = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        self._state = self.np_random.uniform(low=-self._noise, high=self._noise, size=(2,))
        self._state = np.clip(self._state, -self._obs_max, self._obs_max)
        self._prev_state = self._state.copy()
        return self._get_obs()

    def log_diagnostics(self, paths):
        pass
