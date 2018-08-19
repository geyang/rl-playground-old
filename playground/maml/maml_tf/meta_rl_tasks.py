import numpy

from playground.maml.maml_tf import config
from playground.maml.maml_tf.custom_vendor import IS_PATCHED, MAZES  # GRID_WORLDS
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from playground.maml.maml_tf.vec_env_normalize import vec_normalize

assert IS_PATCHED, "need to use patch for new env and proper monitor wraps"

MAZE_KEYS = MAZES.keys()
# GRID_WORLD_KEYS = GRID_WORLDS.keys()
ALLOWED_ENVS = "HalfCheetah-v2", "HalfCheetahGoalVel-v0", "HalfCheetahGoalDir-v0", *MAZE_KEYS  # *GRID_WORLD_KEYS


class MetaRLTasks:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.envs.close()

    def __init__(self, *, env_name, batch_size, start_seed, log_directory=None, max_steps=None):
        """
        use log_directory/{seed}/ to dynamically generate movies with individual seeds.
        """
        import gym
        gym.logger.set_level(40)  # set logging level to avoid annoying warning.

        assert env_name in ALLOWED_ENVS, \
            "environment {} is not supported. Need to be one of {}".format(env_name, ALLOWED_ENVS)

        # keep the env_name for sampling logic. Can be removed if made more general.
        self.env_name = env_name

        def make_env(env_seed, env_name, monitor_log_directory=None, wrap=None):
            def _f():
                nonlocal max_steps
                env = gym.make(env_name)
                # Note: gym seed does not allow task_seed. Use constructor instead.
                # if self.env_name in GRID_WORLD_KEYS:
                #     env.seed(seed=(seed, task_seed))
                # else:
                env.seed(seed=env_seed)
                if max_steps:  # 0, None, False are null values.
                    # see issue #410: https://github.com/openai/gym/issues/410 the TimeLimit wrapper is now used as a
                    # standard wrapper, and the _max_episode_steps is used inside TimeLimit wrapper for episode step-out
                    # limit.
                    # Note: should not override the default when reporting.
                    env._max_episode_steps = max_steps

                # setting numpy.random.seed is a very bad pattern.
                numpy.random.seed(env_seed)
                if monitor_log_directory is not None:
                    env = gym.wrappers.Monitor(env, monitor_log_directory.format(seed=env_seed), force=True)
                    # todo: use bench Montior
                    # from rl_algs.bench import Monitor
                    # env = Monitor(env, monitor_log_directory.format(seed=seed), force=True)
                if wrap:
                    env = wrap(env)
                return env

            return _f

        # if self.env_name in MAZE_KEYS:
        #     # Maze configuration logic
        #     path = MAZES[self.env_name]
        #     self.envs = MazeEnv(batch_size=batch_size, path=path, seed=start_seed, n_episodes=1, num_envs=batch_size)
        # else:
        for s in range(batch_size):
            print(f"seed should be: {start_seed + s}")

        self.envs = SubprocVecEnv(
            [make_env(env_seed=start_seed + s, env_name=env_name, monitor_log_directory=log_directory) for s in
             range(batch_size)])

        if config.G.normalize_env:
            self.envs = vec_normalize(self.envs)

        # This is used in the reporting logic, to respect the standard reporting for episode length etc..
        self.spec = gym.envs.registry.spec(env_name)

    def sample(self, identical_batch=True):
        """has to set the goals by batch at least once. Otherwise the initial goals are different depending on the
        random seed."""
        envs = self.envs  # type: gym.Env
        if self.env_name == "HalfCheetahGoalVel-v0":
            new_goal = numpy.random.uniform(0, 2.0)
            envs.set_goal_velocity(new_goal if identical_batch else None)
        elif self.env_name == "HalfCheetahGoalDir-v0":
            new_direction = 1 if numpy.random.rand() > 0.5 else -1
            envs.set_goal_direction(new_direction if identical_batch else None)
        elif self.env_name in MAZE_KEYS:
            envs.reset_task()
        elif self.env_name.startswith("ReacherMultiTask"):
            new_goal_index = numpy.random.randint(0, 4)
            envs.sample_task(index=new_goal_index if identical_batch else None)
        # elif self.env_name in GRID_WORLD_KEYS:
        #     if config.G.change_colors:
        #         envs.change_colors()
        #     if config.G.change_dynamics:
        #         envs.change_dynamics()
        #     envs.reset_board()
        #     envs.reset()
        else:
            """For other envs, do nothing."""
        # algorithm always resets, so no need to reset here.
        return envs

    @property
    def task_spec(self):
        if self.env_name == "ReacherMultiTask-v0":
            index, *_ = self.envs.get_goal_index()
            return dict(index=index)
        return None


if __name__ == "__main__":
    class TestGlobals:
        # env_name = 'HalfCheetah-v1'
        env_name = 'HalfCheetahGoalVel-v0'
        # env_name = 'MediumWorld-v0'
        n_envs = 10
        start_seed = 42
        log_directory = '../test_runs/demo_envs/{env_name}'


    # Example Usages:
    tasks = MetaRLTasks(env_name=TestGlobals.env_name, batch_size=TestGlobals.n_envs, start_seed=TestGlobals.start_seed,
                        # log_directory=TestGlobals.log_directory.format(env_name=TestGlobals.env_name) + "/{seed}",
                        max_steps=10)

    envs = tasks.sample()
    envs.reset()

    if TestGlobals.env_name == "HalfCheetahGoalVel-v0":
        goal_velocities = envs.get_goal_velocity()
        print('Goal velocities are:', end=" ")
        print(', '.join([str(g) for g in goal_velocities]))
        assert goal_velocities[0] == goal_velocities[2], "goal_velocities are different"
        print('✓', end=" ")
        print('They are identical!', end=" ")
    # elif TestGlobals.env_name in GRID_WORLDS.keys():
    #     # change_colors
    #     colors = envs.change_colors()
    #     assert colors[0]['goal'] == colors[1]['goal'], 'goal color should be identical'
    #     print(colors)
    #     # change_dynamics
    #     assert TestGlobals.env_name == "MediumWorld-v0"
    #     old_dynamics = dynamics = envs.change_dynamics()
    #     assert dynamics[0]['r'] == dynamics[0]['r'], 'right action should be identical'
    #     for i in range(2):
    #         dynamics = envs.change_dynamics()
    #         assert old_dynamics[0]['r'] != dynamics[0]['r'], 'dynamics should be different'
    #         old_dynamics = dynamics
    #     envs.reset_board()
    elif TestGlobals.env_name == "HalfCheetah-v1":
        envs.reset()
