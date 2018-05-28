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


def run_e_maml():
    # print(config.RUN.log_directory)
    # if config.G.run_mode == "e_maml":
    #     print('{G.inner_alg} E-MAML'.format(G=config.G))
    # elif config.G.run_mode == "maml":
    #     print('{G.inner_alg} Vanilla MAML'.format(G=config.G))

    # todo: let's take the control of the log director away from the train script. It should all be set from outside.
    logger.configure(log_directory=config.RUN.log_directory, prefix=f"run_maml-{config.G.seed}")
    logger.log_params(
        RUN=vars(config.RUN),
        G=vars(config.G),
        Reporting=vars(config.Reporting),
        DEBUG=vars(config.DEBUG)
    )

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

    # with Dashboard(config.RUN.prefix, server=config.Reporting.plot_server,
    #                port=config.Reporting.plot_server_port) as dash, U.single_threaded_session(), tasks, test_tasks:
    with U.make_session(num_cpu=config.G.n_cpu), tasks, test_tasks:
        # logger.on_dumpkvs(make_plot_fn(dash))
        maml = E_MAML(ob_space=tasks.envs.observation_space, act_space=tasks.envs.action_space)
        summary = tf.summary.FileWriter(config.RUN.log_directory, tf.get_default_graph())
        summary.flush()
        trainer = Trainer()
        U.initialize()
        trainer.train(tasks=tasks, maml=maml, test_tasks=test_tasks)
        # logger.clear_callback()

    tf.reset_default_graph()


def launch(**_G):
    from datetime import datetime
    now = datetime.now()
    config.G.update(_G)
    config.RUN.log_dir = "http://54.71.92.65:8081"
    config.RUN.log_prefix = f"ge_maml/{now:%Y-%m-%d}"


if __name__ == '__main__':
    import traceback

    try:
        run_e_maml()
    except Exception as e:
        tb = traceback.format_exc()
        logger.print(tb)
        logger.print(U.ALREADY_INITIALIZED)
        raise e
