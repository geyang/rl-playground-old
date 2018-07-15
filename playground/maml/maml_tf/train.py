import tensorflow as tf
from ml_logger import logger
from playground.maml.maml_tf.meta_rl_tasks import MetaRLTasks
from playground.maml.maml_tf import config
from playground.maml.maml_tf.e_maml_ge import E_MAML
from playground.maml.maml_tf.trainer import Trainer


# from playground.maml.maml_tf.trainer import comet_logger


def run_e_maml(_G=None):
    import baselines.common.tf_util as U
    if _G is not None:
        config.G.update(_G)

    # for k, v in [*vars(config.RUN).items(), *vars(config.G).items(), *vars(config.Reporting).items(),
    #              *vars(config.DEBUG).items()]:
    #     comet_logger.log_parameter(k, v)

    # todo: let's take the control of the log director away from the train script. It should all be set from outside.
    logger.configure(log_directory=config.RUN.log_dir, prefix=config.RUN.log_prefix)
    logger.log_params(
        RUN=vars(config.RUN),
        G=vars(config.G),
        Reporting=vars(config.Reporting),
        DEBUG=vars(config.DEBUG)
    )
    logger.log_file(__file__)

    tasks = MetaRLTasks(env_name=config.G.env_name, batch_size=config.G.n_parallel_envs,
                        start_seed=config.G.start_seed,
                        log_directory=(config.RUN.log_directory + "/{seed}") if config.G.render else None,
                        max_steps=config.G.env_max_timesteps)

    # sess_config = tf.ConfigProto(log_device_placement=config.Reporting.log_device_placement)
    # with tf.Session(config=sess_config), tf.device('/gpu:0'), tasks:
    graph = tf.Graph()
    with graph.as_default(), U.make_session(num_cpu=config.G.n_cpu), tasks:
        maml = E_MAML(ob_space=tasks.envs.observation_space, act_space=tasks.envs.action_space)
        # comet_logger.set_model_graph(tf.get_default_graph())

        # writer = tf.summary.FileWriter(logdir='/opt/project/debug-graph', graph=graph)
        # writer.flush()
        # exit()

        trainer = Trainer()
        U.initialize()
        trainer.train(tasks=tasks, maml=maml)
        logger.flush()

    tf.reset_default_graph()


# comet_logger = None


def launch(**_G):
    import baselines.common.tf_util as U
    import traceback
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    from playground.maml.maml_tf.train import run_e_maml

    try:
        config.config_run(**_G)
        run_e_maml(_G=_G)
    except Exception as e:
        tb = traceback.format_exc()
        logger.print(tb)
        logger.print(U.ALREADY_INITIALIZED)
        raise e


if __name__ == '__main__':
    config.RUN.log_prefix = "alpha-0-check"
    launch()

    # from playground.maml.maml_torch.experiments.run import run
    # run(launch, log_prefix="quick-test", _as_daemon=False)
