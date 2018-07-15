# not tested
import gym
gym.logger.set_level(40)  # set logging level to avoid annoying warning.

from .patches import *
# from .krazy_worlds.krazy_world_envs import *
from .maze_env import *
from .half_cheetah_goal_velocity import *
from .half_cheetah_goal_direction import *

IS_PATCHED = True
