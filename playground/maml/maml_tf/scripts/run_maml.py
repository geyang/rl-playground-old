import datetime
from collections import defaultdict, deque
from contextlib import ExitStack
import tensorflow as tf

import baselines.common.tf_util as U
from playground.maml.maml_tf.meta_rl_tasks import ALLOWED_ENVS, MetaRLTasks
import numpy as np

from ml_logger import logger
from playground.maml.maml_tf import config
from playground.maml.maml_tf.e_maml_ge import E_MAML
from playground.maml.maml_tf.trainer import Trainer

ind = 0
cache = defaultdict(lambda: deque(maxlen=config.Reporting.plot_smoothing))
batch_smoothed = defaultdict(list)
# batch_data = defaultdict(list)
batch_indices = defaultdict(list)


def make_plot_fn(dash):
    def plot_fn(dump=False):
        global ind
        ind += 1
        slice = logger.Logger.CURRENT.name2val
        keys = [k.replace('_', " ") for k in slice.keys()]

        for key, v in zip(keys, slice.values()):
            cache[key].append(v)
            batch_smoothed[key].append(np.mean(cache[key]))
            # batch_data[key].append(v)
            batch_indices[key].append(ind)

        if dump or ind % config.Reporting.plot_interval == config.Reporting.plot_interval - 1:
            Y = np.array([vs for vs in batch_smoothed.values()]).T
            X = np.array([vs for vs in batch_indices.values()]).T
            dash.append(config.RUN.log_directory, 'line', Y, X=X, opts=dict(legend=keys))
            batch_smoothed.clear()
            # batch_data.clear()
            batch_indices.clear()

    return plot_fn


from playground.maml.maml_tf.trainer import comet_logger


def run_e_maml(_G=None):
    if _G is not None:
        config.G.update(_G)

    for k, v in [*vars(config.RUN).items(), *vars(config.G).items(), *vars(config.Reporting).items(),
                 *vars(config.DEBUG).items()]:
        comet_logger.log_parameter(k, v)

    # todo: let's take the control of the log director away from the train script. It should all be set from outside.
    logger.configure(log_directory=config.RUN.log_dir, prefix=f"run_maml-{config.G.seed}")
    logger.log_params(
        RUN=vars(config.RUN),
        G=vars(config.G),
        Reporting=vars(config.Reporting),
        DEBUG=vars(config.DEBUG)
    )
    logger.log_file(__file__)

    import sys
    print(" ".join(sys.argv))

    tasks = MetaRLTasks(env_name=config.G.env_name, batch_size=config.G.n_parallel_envs,
                        start_seed=config.G.start_seed,
                        task_seed=config.G.task_seed,
                        log_directory=(config.RUN.log_directory + "/{seed}") if config.G.render else None,
                        max_steps=config.G.env_max_timesteps)

    test_tasks = MetaRLTasks(env_name=config.G.env_name, batch_size=config.G.n_parallel_envs,
                             start_seed=config.G.test_start_seed,
                             task_seed=config.G.test_task_seed,
                             log_directory=(config.RUN.log_directory + "/{seed}") if config.G.render else None,
                             max_steps=config.G.env_max_timesteps) if config.G.eval_test_interval \
        else ExitStack()

    sess_config = tf.ConfigProto(log_device_placement=True)
    with tf.Session(config=sess_config), tf.device('/gpu:0'), tasks, test_tasks:
        # with U.make_session(num_cpu=config.G.n_cpu), tasks, test_tasks:
        maml = E_MAML(ob_space=tasks.envs.observation_space, act_space=tasks.envs.action_space)
        comet_logger.set_model_graph(tf.get_default_graph())
        trainer = Trainer()
        U.initialize()
        trainer.train(tasks=tasks, maml=maml, test_tasks=test_tasks)
        logger.flush()

    tf.reset_default_graph()


def launch(cuda_id: int = 0, **_G):
    import os
    import traceback
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

    try:
        run_e_maml(_G=_G)
    except Exception as e:
        tb = traceback.format_exc()
        logger.print(tb)
        logger.print(U.ALREADY_INITIALIZED)
        raise e


if __name__ == '__main__':
    ps = [1e-1, 1e-2, 1e-3, 1e-4]
    i = 0
    launch(cuda_id=(i + 1) % 4, alpha=ps[i])
