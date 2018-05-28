from gym import Env, spaces, utils
from gym.envs import register

from .krazy_grid_no_pygame import EASY_GRID_KWARGS, HARD_GRID_KWARGS, KrazyGridWorld, MEDIUM_GRID_KWARGS


class KrazyGridWorldEnv(Env, utils.EzPickle):
    """
    Half cheetah environment with a randomly generated goal path.
    todo: test seeding
    """

    def __init__(self, **kwargs):
        screen_height = kwargs['screen_height']
        del kwargs['screen_height']
        self.env = game = KrazyGridWorld(screen_height=screen_height, **kwargs)
        self.action_space = spaces.Discrete(4)
        # note: this low and high is not good.
        self.observation_space = spaces.Box(low=-100, high=100, shape=(game.get_obs().shape[0],))
        self.ob_space = self.observation_space
        self.ac_space = self.action_space

        # todo: double check with Bradly: which to use: reset x0, reset, reset_board?

        # bind methods
        self._step = game.step
        self._reset = game.reset
        self.reset_board = game.reset_board
        self.change_colors = game.change_colors
        self.change_dynamics = game.change_dynamics

        # super(KrazyGridWorldEnv, self).__init__()
        utils.EzPickle.__init__(self)

    def _seed(self, seed=None):
        position_seed, task_seed = seed
        self.env.seed(position_seed, task_seed)

    def _render(self, mode='human', close=False):
        # todo: @bstadie need to take care of other render modes
        return self.env.render()


GRID_WORLDS = {
    "EasyWorld-v0": EASY_GRID_KWARGS,
    "MediumWorld-v0": MEDIUM_GRID_KWARGS,
    "HardWorld-v0": HARD_GRID_KWARGS
}

for env_id, kwargs in GRID_WORLDS.items():
    register(
        env_id,
        # entry_point=lambda: KrazyGridWorldEnv(**kwargs),
        entry_point="custom_vendor.krazy_worlds.krazy_world_envs:KrazyGridWorldEnv",
        kwargs=kwargs,
        max_episode_steps=24,
        reward_threshold=50.0
    )
